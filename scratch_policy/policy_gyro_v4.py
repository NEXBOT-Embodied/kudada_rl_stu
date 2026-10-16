"""Owned scratch actor with a zero-initialized trainable sensor gyro branch.

The external interface remains 21 oldest-first frames of 44 sensor values.
The original MLP keeps its own five- or twenty-one-frame input and tensors.
Both branches contribute to the same latent actions before shared limits.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

import numpy as np
import torch
from torch import nn

from .policy import ScratchActor, load_actor as load_parent_actor
from .policy import state_digest, validate_checkpoint as validate_parent_checkpoint

ARCHITECTURE = "scratch_sensor_gyro_skip_v4"
FEATURE_SCHEMA = "scratch_sensor_gyro_features_v4"
MIGRATION_SCHEMA = "scratch_owned_gyro_skip_function_migration_v4"
GAIN_KEY = "gyro_skip.weight"
GYRO_BUFFER_NAMES = ("gyro_raw_scale", "gyro_fir", "gyro_feature_scale")
ANATOMICAL_DIRECTIONS = {
    "yaw_common": [1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    "roll_common": [0., 1., 0., 0., 0., 0., 1., 0., 0., 0.],
    "hip_pitch_difference": [0., 0., 1., 0., 0., 0., 0., -1., 0., 0.],
    "knee_difference": [0., 0., 0., 1., 0., 0., 0., 0., -1., 0.],
    "ankle_difference": [0., 0., 0., 0., 1., 0., 0., 0., 0., -1.],
}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def nominal_dc_filter() -> np.ndarray:
    """Minimum-L2 DC filter rejecting nominal 20 ms 2/4 Hz sine and cosine.

    It has no simulator state and uses only the supplied sensor history. Its
    response at other sample intervals or frequencies is not a hard notch.
    """
    time = (np.arange(21, dtype=np.float64) - 20.) * .02
    rows = [np.ones(21, dtype=np.float64)]
    for frequency in (2., 4.):
        phase = 2. * np.pi * frequency * time
        rows.extend((np.sin(phase), np.cos(phase)))
    constraints = np.stack(rows)
    target = np.array([1., 0., 0., 0., 0.], dtype=np.float64)
    weights = constraints.T @ np.linalg.solve(constraints @ constraints.T, target)
    return weights.astype(np.float32)


def gyro_feature_config() -> dict[str, Any]:
    return dict(schema=FEATURE_SCHEMA, gyro_index=10, history_frames=21,
                raw_gyro_scale=4., nominal_interval_seconds=.02,
                notch_frequencies_hz=[2., 4.],
                feature_names=["latest_gyro_z_rad_s", "fir_gyro_z_rad_s"],
                feature_scale=[2., 50.], filter_weights=nominal_dc_filter().tolist(),
                feature_order="latest_then_nominal_DC_FIR",
                history_order="oldest_first_newest_last", external_filter_state=False,
                trainable_bias=False)


class GyroSkipActor(ScratchActor):
    """A complete learned actor; the sensor branch has twenty trainable weights."""

    def __init__(self, config: dict[str, Any]):
        if config.get("architecture") != ARCHITECTURE:
            raise ValueError("GyroSkipActor requires the explicit gyro_v4 architecture")
        if config.get("gyro_skip") != gyro_feature_config():
            raise ValueError("Gyro feature declaration differs from the fixed sensor-only v4 specification")
        if int(config.get("input_size", 220)) not in (220, 924):
            raise ValueError("Gyro v4 preserves an owned five- or twenty-one-frame MLP")
        super().__init__(config)
        gyro = config["gyro_skip"]
        self.register_buffer("gyro_raw_scale", torch.tensor(gyro["raw_gyro_scale"], dtype=torch.float32))
        self.register_buffer("gyro_fir", torch.tensor(gyro["filter_weights"], dtype=torch.float32))
        self.register_buffer("gyro_feature_scale", torch.tensor(gyro["feature_scale"], dtype=torch.float32))
        self.gyro_skip = nn.Linear(2, 10, bias=False)
        nn.init.zeros_(self.gyro_skip.weight)

    def sensor_features(self, observations: torch.Tensor) -> torch.Tensor:
        gyro = observations[:, 10::44] * self.gyro_raw_scale
        latest = gyro[:, -1:]
        dc = torch.matmul(gyro, self.gyro_fir).unsqueeze(-1)
        return torch.cat((latest, dc), dim=-1) * self.gyro_feature_scale

    def latent(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(self.features(observations)) + self.gyro_skip(self.sensor_features(observations))


def validate_checkpoint(checkpoint: dict[str, Any]) -> None:
    validate_parent_checkpoint(checkpoint)
    config = checkpoint.get("actor_config", {})
    if config.get("architecture") != ARCHITECTURE or config.get("gyro_skip") != gyro_feature_config():
        raise ValueError("Checkpoint does not declare the fixed gyro_v4 architecture and features")
    if checkpoint["provenance"].get("uses_policy_demonstrations") is not False:
        raise ValueError("Gyro v4 requires a demonstration-free own scratch lineage")


def load_actor(checkpoint: dict[str, Any]) -> GyroSkipActor:
    validate_checkpoint(checkpoint)
    actor = GyroSkipActor(checkpoint["actor_config"]).cpu().eval()
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    if any(not torch.isfinite(value).all() for value in actor.state_dict().values()):
        raise ValueError("Nonfinite gyro actor weights or transforms")
    expected = GyroSkipActor(checkpoint["actor_config"]).state_dict()
    for name in ("feature_scale", "action_scale", "action_bias", "action_lower", "action_upper", *GYRO_BUFFER_NAMES):
        if not torch.equal(expected[name], actor.state_dict()[name]):
            raise ValueError(f"Actor transform {name} disagrees with the declared architecture")
    if tuple(actor.gyro_skip.weight.shape) != (10, 2):
        raise ValueError("Gyro skip must have ten actions by two sensor features")
    return actor


def augment_owned_checkpoint(parent: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Preserve every old tensor and add zero gyro weights plus fixed buffers.

    The caller creates a new training stage, its optimizer and source snapshot;
    this function does not claim that any optimization or physics occurred.
    """
    validate_parent_checkpoint(parent)
    if parent["provenance"].get("uses_policy_demonstrations") is not False:
        raise ValueError("Augmentation requires a demonstration-free own parent")
    if parent["actor_config"].get("architecture") == ARCHITECTURE:
        raise ValueError("A gyro_v4 parent needs explicit continuation rather than a second augmentation")
    load_parent_actor(parent)
    config = deepcopy(parent["actor_config"])
    if int(config["input_size"]) not in (220, 924):
        raise ValueError("Augmentation preserves an existing own five- or twenty-one-frame MLP")
    config.update(architecture=ARCHITECTURE, gyro_skip=gyro_feature_config())
    actor = GyroSkipActor(config).cpu().eval()
    state = {name: tensor.detach().cpu().clone() for name, tensor in actor.state_dict().items()}
    state.update({name: tensor.detach().cpu().clone() for name, tensor in parent["actor_state_dict"].items()})
    actor.load_state_dict(state, strict=True)
    evidence = dict(schema=MIGRATION_SCHEMA, source_actor_config=deepcopy(parent["actor_config"]),
                    target_actor_config=deepcopy(config),
                    source_actor_config_sha256=canonical_sha256(parent["actor_config"]),
                    target_actor_config_sha256=canonical_sha256(config),
                    source_actor_state_sha256=state_digest(parent["actor_state_dict"]),
                    target_initial_actor_state_sha256=state_digest(state),
                    source_critic_state_sha256=state_digest(parent["critic_state_dict"]),
                    target_initial_critic_state_sha256=state_digest(parent["critic_state_dict"]),
                    old_actor_tensors_preserved=sorted(parent["actor_state_dict"]),
                    old_critic_tensors_preserved=sorted(parent["critic_state_dict"]),
                    new_actor_tensors=sorted(set(state) - set(parent["actor_state_dict"])),
                    gain_key=GAIN_KEY, gain_shape=[10, 2], initial_gain_zero=True,
                    core_observation_frames=int(config["input_size"]) // 44,
                    gyro_observation_frames=21, retained_external_input_size=924,
                    retained_external_output_size=11, gyro_feature_config=deepcopy(config["gyro_skip"]),
                    function_preserved_before_optimization=True, uses_teacher=False,
                    uses_policy_demonstrations=False, physical_steps=0, optimizer_updates=0)
    augmented = deepcopy(parent)
    augmented.update(actor_config=config, actor_state_dict=state, gyro_skip_migration=deepcopy(evidence))
    augmented["provenance"]["gyro_skip_migration"] = deepcopy(evidence)
    load_actor(augmented)
    return augmented, evidence


def anatomical_gain_weight(parameters: Any, directions: list[str] | None = None) -> torch.Tensor:
    """Bake feature-major CEM gains into the complete [actions,features] tensor."""
    names = list(ANATOMICAL_DIRECTIONS) if directions is None else list(directions)
    if not names or len(set(names)) != len(names) or any(name not in ANATOMICAL_DIRECTIONS for name in names):
        raise ValueError("Need unique known anatomical directions")
    gains = np.asarray(parameters, dtype=np.float32)
    if gains.shape != (2 * len(names),) or not np.isfinite(gains).all():
        raise ValueError("Need finite feature-major gains: latest directions, then FIR directions")
    basis = np.asarray([ANATOMICAL_DIRECTIONS[name] for name in names], dtype=np.float32)
    weight = basis.T @ gains.reshape(2, len(names)).T
    return torch.from_numpy(np.ascontiguousarray(weight))
