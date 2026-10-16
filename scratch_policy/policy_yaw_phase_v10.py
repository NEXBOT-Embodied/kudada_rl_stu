"""Owned actor with a zero-initialized command/gyro/phase latent bypass.

All features come from the existing oldest-first raw 924 sensor history. The
ordinary five-frame MLP remains fully trainable and keeps its parent's exact
initial function. This branch has no action-feedback memory or external state.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

import torch
from torch import nn

from .policy import ScratchActor, load_actor as load_parent_actor
from .policy import state_digest, validate_checkpoint as validate_parent_checkpoint
from .policy_gyro_v4 import nominal_dc_filter

ARCHITECTURE = "scratch_yaw_command_phase_bypass_v10"
METHOD = "PPO_yaw_command_phase_bypass_v10"
GAIN_KEY = "yaw_phase.weight"
MIGRATION_SCHEMA = "scratch_owned_yaw_phase_zero_branch_migration_v10"
BUFFER_NAMES = ("yaw_command_raw_scale", "gyro_raw_scale", "gyro_fir", "turn_rate_normalizer")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def yaw_phase_feature_config() -> dict[str, Any]:
    return dict(schema="scratch_yaw_command_gyro_phase_features_v10",
        feature_names=["yaw_command_norm", "gyro_dc_norm", "yaw_command_sin_phase",
                       "yaw_command_cos_phase", "gyro_dc_sin_phase", "gyro_dc_cos_phase"],
        gain_shape=[10, 6], trainable_bias=False, initial_gain_zero=True,
        command_index=5, gyro_index=10, phase_sin_index=6, phase_cos_index=7,
        yaw_command_raw_scale=4., gyro_raw_scale=4., turn_rate_normalizer=.1,
        gyro_history_frames=21, core_observation_frames=5,
        nominal_interval_seconds=.02, notch_frequencies_hz=[2., 4.],
        filter_weights=nominal_dc_filter().tolist(),
        feature_order="u,r,u*sin,u*cos,r*sin,r*cos",
        right_half_cycle="sin_and_cos_change_sign;independent_action_rows_learn_leg_mapping",
        history_order="oldest_first_newest_last", external_filter_state=False,
        uses_previous_action_residual=False, second_harmonics=False,
        normalization_scope="raw_physical_command_and_gyro;independent_of_core_feature_scale",
        limitations=["20ms FIR has leakage at other inference intervals",
                     "zero-padded startup history changes DC transient",
                     "gait phase does not establish actual foot contact",
                     "finite sensor history does not supply world heading or lateral position"])


class YawPhaseActor(ScratchActor):
    def __init__(self, config: dict[str, Any]):
        if config.get('architecture') != ARCHITECTURE:
            raise ValueError("YawPhaseActor requires the explicit v10 architecture")
        if config.get('yaw_phase_bypass') != yaw_phase_feature_config():
            raise ValueError("Yaw-phase feature declaration differs from the fixed v10 specification")
        if config.get('input_size') != 220 or config.get('hidden') != [256, 128, 64]:
            raise ValueError("Yaw-phase v10 retains the ordinary full five-frame parent MLP")
        if config['feature_scale'][5] not in (1., 8.):
            raise ValueError("The ordinary parent yaw feature scale must be one or eight")
        super().__init__(config)
        spec = config['yaw_phase_bypass']
        for name in BUFFER_NAMES:
            key = 'filter_weights' if name == 'gyro_fir' else name
            self.register_buffer(name, torch.tensor(spec[key], dtype=torch.float32))
        self.yaw_phase = nn.Linear(6, 10, bias=False)
        nn.init.zeros_(self.yaw_phase.weight)

    def sensor_features(self, observations: torch.Tensor) -> torch.Tensor:
        latest = observations[:, -44:]
        command = latest[:, 5] * self.yaw_command_raw_scale / self.turn_rate_normalizer
        gyro = observations[:, 10::44] * self.gyro_raw_scale
        dc = torch.matmul(gyro, self.gyro_fir) / self.turn_rate_normalizer
        sine, cosine = latest[:, 6], latest[:, 7]
        return torch.stack((command, dc, command*sine, command*cosine,
                            dc*sine, dc*cosine), dim=-1)

    def latent_components(self, observations: torch.Tensor):
        """The rollout uses these already-computed components for diagnostics."""
        features = self.sensor_features(observations)
        core = self.network(self.features(observations))
        branch = self.yaw_phase(features)
        return core + branch, core, branch, features

    def latent(self, observations: torch.Tensor) -> torch.Tensor:
        return self.latent_components(observations)[0]


def validate_checkpoint(checkpoint: dict[str, Any]) -> None:
    validate_parent_checkpoint(checkpoint)
    if checkpoint['provenance'].get('uses_policy_demonstrations') is not False:
        raise ValueError("Yaw-phase v10 requires a demonstration-free own lineage")
    config = checkpoint.get('actor_config', {})
    if config.get('architecture') != ARCHITECTURE or config.get('yaw_phase_bypass') != yaw_phase_feature_config():
        raise ValueError("Checkpoint lacks the fixed yaw-phase v10 architecture/features")


def load_actor(checkpoint: dict[str, Any]) -> YawPhaseActor:
    validate_checkpoint(checkpoint)
    actor = YawPhaseActor(checkpoint['actor_config']).cpu().eval()
    actor.load_state_dict(checkpoint['actor_state_dict'], strict=True)
    if any(not torch.isfinite(value).all() for value in actor.state_dict().values()):
        raise ValueError("Nonfinite yaw-phase actor tensors")
    expected = YawPhaseActor(checkpoint['actor_config']).state_dict()
    for name in ('feature_scale', 'action_scale', 'action_bias', 'action_lower', 'action_upper', *BUFFER_NAMES):
        if not torch.equal(expected[name], actor.state_dict()[name]):
            raise ValueError(f"Actor transform {name} differs from the yaw-phase declaration")
    if tuple(actor.yaw_phase.weight.shape) != (10, 6):
        raise ValueError("Yaw-phase branch must map six fixed features to ten latent actions")
    return actor


def augment_owned_checkpoint(parent: dict[str, Any]):
    """Copy every old tensor exactly; add fixed buffers and sixty zero weights."""
    validate_parent_checkpoint(parent)
    if parent['provenance'].get('uses_policy_demonstrations') is not False:
        raise ValueError("Augmentation requires a demonstration-free own parent")
    if 'architecture' in parent['actor_config']:
        raise ValueError("Warm augmentation requires an ordinary parent; use resume for an existing v10 actor")
    load_parent_actor(parent)
    config = deepcopy(parent['actor_config'])
    config.update(architecture=ARCHITECTURE, yaw_phase_bypass=yaw_phase_feature_config())
    actor = YawPhaseActor(config).cpu()
    state = {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
    state.update({name: value.detach().cpu().clone() for name, value in parent['actor_state_dict'].items()})
    evidence = dict(schema=MIGRATION_SCHEMA,
        source_actor_config_sha256=canonical_sha256(parent['actor_config']),
        target_actor_config_sha256=canonical_sha256(config),
        source_actor_state_sha256=state_digest(parent['actor_state_dict']),
        target_initial_actor_state_sha256=state_digest(state),
        source_critic_state_sha256=state_digest(parent['critic_state_dict']),
        target_initial_critic_state_sha256=state_digest(parent['critic_state_dict']),
        old_actor_tensors_preserved=sorted(parent['actor_state_dict']),
        old_critic_tensors_preserved=sorted(parent['critic_state_dict']),
        new_actor_tensors=sorted(set(state) - set(parent['actor_state_dict'])),
        gain_key=GAIN_KEY, gain_shape=[10, 6], initial_gain_zero=True,
        parent_core_yaw_feature_scale=float(config['feature_scale'][5]),
        core_feature_conditioning_reapplied=False,
        old_actor_and_critic_tensors_exact=True,
        function_preservation='zero_additive_latent_branch_for_valid_raw_sensor_inputs_with_finite_features',
        numerical_parity='requires_independent_CPU_and_fixed_ONNX_checks',
        feature_config=deepcopy(config['yaw_phase_bypass']),
        core_observation_frames=5, gyro_observation_frames=21,
        retained_external_input_size=924, retained_external_output_size=11,
        uses_teacher=False, uses_policy_demonstrations=False,
        physical_steps=0, optimizer_updates=0)
    augmented = deepcopy(parent)
    augmented.update(actor_config=config, actor_state_dict=state, yaw_phase_migration=deepcopy(evidence))
    return augmented, evidence
