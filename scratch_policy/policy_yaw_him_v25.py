"""Owned yaw actor with detached sensor-only HIM estimates in its core.

The parent sensor features and gyro/command/phase branch are exact. The core
adds predicted scaled body velocity and a unit 16-vector after its original
220 sensor features. The added first-layer columns start at zero. A separate
training head owns the target network and prototypes; neither is deployed.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .policy import state_digest, validate_checkpoint as validate_base_checkpoint
from .policy_yaw_phase_v10 import (
    ARCHITECTURE as PARENT_ARCHITECTURE, BUFFER_NAMES, GAIN_KEY, YawPhaseActor,
    validate_checkpoint as validate_owned_parent_checkpoint,
    yaw_phase_feature_config,
)

ARCHITECTURE = "scratch_him_yaw_command_phase_bypass_v25"
METHOD = "PPO_him_yaw_row_adam_v25"
MIGRATION_SCHEMA = "scratch_owned_him_zero_extension_migration_v25"
HIM_SCHEMA = "scratch_sensor_only_him_estimator_v25"
ESTIMATOR_SEED = 7380
SENSOR_WIDTH = 220
ESTIMATOR_INPUT_WIDTH = 924
ESTIMATE_WIDTH = 19
CORE_FEATURE_WIDTH = 239
PRIVILEGED_WIDTH = 13
ACTOR_FIRST_LAYER_KEY = "network.0.weight"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def him_feature_config(estimator_seed: int = ESTIMATOR_SEED) -> dict[str, Any]:
    if estimator_seed != ESTIMATOR_SEED:
        raise ValueError("V25 uses the explicit independent estimator seed7380")
    return dict(schema=HIM_SCHEMA, estimator_seed=estimator_seed,
        input_width=924, input_frames=21, frame_width=44,
        external_history_order="oldest_first_newest_last",
        encoder_history_order="newest_first_oldest_last", history_reversals=1,
        encoder_widths=[924, 128, 64, 19], target_widths=[44, 128, 64, 16],
        activation="ELU", initialization="torch_nn_Linear_default_with_local_CPU_seed7380",
        prediction_order="scaled_body_velocity3_then_unit_latent16",
        velocity_label_scale=2.0, latent_size=16, latent_normalization="L2_dim_last",
        core_sensor_width=220, core_estimate_width=19, core_feature_width=239,
        actor_estimate_detached=True, critic_sensor_width=220, critic_input_width=233,
        prototype_count=32, prototype_width=16, temperature=3.0,
        sinkhorn_epsilon=0.05, sinkhorn_iterations=3, estimator_max_grad_norm=10.0,
        target_label="concat(pre_autoreset_next_raw44[:,3:44],pre_autoreset_next_body_velocity3*2)",
        estimator_optimizer="Adam", estimator_learning_rate=3e-4,
        estimator_learning_rate_adapted_by_PPO_KL=False,
        update_order="one_estimator_step_after_each_accepted_PPO_minibatch",
        deployed_estimator_components=["him_encoder"],
        training_only_estimator_components=["target", "proto"],
        privileged_actor_inputs=False, target_network_deployed=False,
        prototypes_deployed=False, external_state=False)


def _seeded_him_parts(estimator_seed: int):
    """Independent deterministic initializer shared by migration and trainer.

    Construct the encoder, target and prototypes in original HIM order under a
    local CPU RNG. This leaves the caller's PPO/environment RNG unchanged.
    """
    him_feature_config(estimator_seed)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(estimator_seed)
        encoder = nn.Sequential(nn.Linear(924, 128), nn.ELU(),
                                nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 19))
        target = nn.Sequential(nn.Linear(44, 128), nn.ELU(),
                               nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 16))
        proto = nn.Embedding(32, 16)
    return encoder, target, proto


@torch.no_grad()
def sinkhorn(out: torch.Tensor, eps: float = 0.05, iters: int = 3) -> torch.Tensor:
    """Original HIM balanced assignments; no added normalization convention."""
    Q = torch.exp(out / eps).T
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()
    for _ in range(iters):
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    return (Q * B).T


class HimTrainingHead(nn.Module):
    """Target/prototypes live in training state separately from actor tensors."""
    def __init__(self, estimator_seed: int = ESTIMATOR_SEED):
        super().__init__()
        _, self.target, self.proto = _seeded_him_parts(estimator_seed)
        self.temperature = 3.0
        self.max_grad_norm = 10.0
        self.estimator_seed = estimator_seed

    @torch.no_grad()
    def normalize_prototypes_(self) -> None:
        self.proto.weight.copy_(F.normalize(self.proto.weight.detach().clone(), dim=-1, p=2))

    def losses(self, actor: "HimYawActor", raw_history: torch.Tensor,
               target44: torch.Tensor, scaled_velocity3: torch.Tensor):
        """Original velocity MSE and symmetric prototype swapped assignment loss.

        The trainer owns zero_grad/backward/clip/Adam and records their actual
        executions. This method does not step or adapt an optimizer.
        """
        predicted_velocity, z_s = actor.encode_estimate(raw_history.detach())
        z_t = F.normalize(self.target(target44.detach()), dim=-1, p=2)
        self.normalize_prototypes_()
        score_s = z_s @ self.proto.weight.T
        score_t = z_t @ self.proto.weight.T
        with torch.no_grad():
            q_s, q_t = sinkhorn(score_s), sinkhorn(score_t)
        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)
        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
        estimation_loss = F.mse_loss(predicted_velocity, scaled_velocity3.detach())
        return estimation_loss, swap_loss


class HimYawActor(YawPhaseActor):
    def __init__(self, config: dict[str, Any]):
        if (config.get("architecture") != ARCHITECTURE or config.get("input_size") != SENSOR_WIDTH
                or config.get("core_feature_size") != CORE_FEATURE_WIDTH
                or config.get("hidden") != [256, 128, 64]
                or config.get("yaw_phase_bypass") != yaw_phase_feature_config()
                or config.get("him_estimator") != him_feature_config()):
            raise ValueError("V25 requires the exact parent220 plus detached HIM19 core")
        parent_config = deepcopy(config)
        parent_config["architecture"] = PARENT_ARCHITECTURE
        super().__init__(parent_config)
        self.config = deepcopy(config)
        self.core_feature_size = CORE_FEATURE_WIDTH
        old = self.network[0]
        # Only these nineteen actor-core columns are added; the critic remains233.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            first = nn.Linear(CORE_FEATURE_WIDTH, old.out_features)
        with torch.no_grad():
            first.weight[:, :SENSOR_WIDTH].copy_(old.weight)
            first.weight[:, SENSOR_WIDTH:].zero_()
            first.bias.copy_(old.bias)
        self.network[0] = first
        self.him_encoder, _, _ = _seeded_him_parts(ESTIMATOR_SEED)

    def critic_sensor_features(self, observations: torch.Tensor) -> torch.Tensor:
        # Inherited features() remains exactly the original scaled sensor220.
        return self.features(observations)

    def estimator_input(self, observations: torch.Tensor) -> torch.Tensor:
        # External deployment IO remains oldest-first; reverse once internally.
        return observations.reshape(-1, 21, 44).flip(1).reshape(-1, 924)

    def encode_estimate(self, observations: torch.Tensor):
        encoded = self.him_encoder(self.estimator_input(observations.detach()))
        return encoded[:, :3], F.normalize(encoded[:, 3:], dim=-1, p=2)

    def estimate_features(self, observations: torch.Tensor) -> torch.Tensor:
        velocity, latent = self.encode_estimate(observations)
        return torch.cat((velocity, latent), dim=-1).detach()

    def core_features(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.features(observations), self.estimate_features(observations)), dim=-1)

    def latent_components_with_estimate(self, observations: torch.Tensor):
        estimate = self.estimate_features(observations)
        branch_features = self.sensor_features(observations)
        core = self.network(torch.cat((self.features(observations), estimate), dim=-1))
        branch = self.yaw_phase(branch_features)
        return core + branch, core, branch, branch_features, estimate

    def latent_components(self, observations: torch.Tensor):
        return self.latent_components_with_estimate(observations)[:4]


def validate_checkpoint(checkpoint: dict[str, Any]) -> None:
    validate_base_checkpoint(checkpoint)
    config = checkpoint.get("actor_config", {})
    if (checkpoint["provenance"].get("uses_policy_demonstrations") is not False
            or config.get("architecture") != ARCHITECTURE
            or config.get("input_size") != SENSOR_WIDTH
            or config.get("core_feature_size") != CORE_FEATURE_WIDTH
            or config.get("yaw_phase_bypass") != yaw_phase_feature_config()
            or config.get("him_estimator") != him_feature_config()
            or checkpoint.get("him_migration", {}).get("schema") != MIGRATION_SCHEMA):
        raise ValueError("Checkpoint lacks the fixed V25 HIM architecture/migration")
    if (checkpoint.get("critic_privileged_size") != PRIVILEGED_WIDTH
            or tuple(checkpoint["critic_state_dict"]["0.weight"].shape) != (256, 233)):
        raise ValueError("V25 must retain the parent's exact233 critic interface")
    if any(key.startswith(("target.", "proto.", "him_target.", "him_proto."))
           for key in checkpoint["actor_state_dict"]):
        raise ValueError("Training-only target/prototypes must not enter actor_state_dict")


def load_actor(checkpoint: dict[str, Any]) -> HimYawActor:
    validate_checkpoint(checkpoint)
    actor = HimYawActor(checkpoint["actor_config"]).cpu().eval()
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    if any(not torch.isfinite(value).all() for value in actor.state_dict().values()):
        raise ValueError("Nonfinite V25 actor tensors")
    expected = HimYawActor(checkpoint["actor_config"]).state_dict()
    for name in ("feature_scale", "action_scale", "action_bias", "action_lower", "action_upper", *BUFFER_NAMES):
        if not torch.equal(expected[name], actor.state_dict()[name]):
            raise ValueError(f"Actor transform {name} differs from the V25 declaration")
    if tuple(actor.yaw_phase.weight.shape) != (10, 6):
        raise ValueError("V25 must retain the parent's six-feature yaw branch")
    return actor


def extend_owned_checkpoint(parent: dict[str, Any], estimator_seed: int = ESTIMATOR_SEED):
    """Exact parent tensors plus a seeded encoder and19 zero core columns.

    No Actor or Critic is constructed and no network forward is performed.
    Constructing the new HIM initialization is separate from policy execution.
    """
    validate_owned_parent_checkpoint(parent)
    config = parent["actor_config"]
    old_actor, old_critic = parent["actor_state_dict"], parent["critic_state_dict"]
    if (config.get("input_size") != SENSOR_WIDTH or config.get("hidden") != [256, 128, 64]
            or parent.get("critic_hidden") != [256, 128, 64]
            or parent.get("critic_privileged_size") != PRIVILEGED_WIDTH
            or tuple(old_actor[ACTOR_FIRST_LAYER_KEY].shape) != (256, 220)
            or tuple(old_critic["0.weight"].shape) != (256, 233)):
        raise ValueError("V25 migration requires the exact own220/233 parent")
    if any(not torch.isfinite(value).all() for value in (*old_actor.values(), *old_critic.values())):
        raise ValueError("Nonfinite owned parent tensors")
    spec = him_feature_config(estimator_seed)
    encoder, target, proto = _seeded_him_parts(estimator_seed)
    actor_state = {key: value.detach().cpu().clone() for key, value in old_actor.items()}
    critic_state = {key: value.detach().cpu().clone() for key, value in old_critic.items()}
    first = old_actor[ACTOR_FIRST_LAYER_KEY].detach().cpu()
    actor_state[ACTOR_FIRST_LAYER_KEY] = torch.cat((first, first.new_zeros((256, 19))), dim=1)
    actor_state.update({"him_encoder."+key: value.detach().cpu().clone()
                        for key, value in encoder.state_dict().items()})
    target_config = deepcopy(config)
    target_config.update(architecture=ARCHITECTURE, core_feature_size=239, him_estimator=spec)
    exact_actor = sorted(set(old_actor) - {ACTOR_FIRST_LAYER_KEY})
    if (not torch.equal(actor_state[ACTOR_FIRST_LAYER_KEY][:, :220], first)
            or torch.count_nonzero(actor_state[ACTOR_FIRST_LAYER_KEY][:, 220:239]).item() != 0
            or any(not torch.equal(actor_state[key], old_actor[key].detach().cpu()) for key in exact_actor)
            or any(not torch.equal(critic_state[key], old_critic[key].detach().cpu()) for key in old_critic)):
        raise ValueError("V25 failed the exact parent tensor mapping")
    training_state = {"target."+key: value.detach().cpu().clone() for key, value in target.state_dict().items()}
    training_state.update({"proto."+key: value.detach().cpu().clone() for key, value in proto.state_dict().items()})
    evidence = dict(schema=MIGRATION_SCHEMA,
        parent_architecture=PARENT_ARCHITECTURE, target_architecture=ARCHITECTURE,
        source_actor_config_sha256=canonical_sha256(config), target_actor_config_sha256=canonical_sha256(target_config),
        source_actor_state_sha256=state_digest(old_actor), target_initial_actor_state_sha256=state_digest(actor_state),
        source_critic_state_sha256=state_digest(old_critic), target_initial_critic_state_sha256=state_digest(critic_state),
        actor_first_layer=dict(key=ACTOR_FIRST_LAYER_KEY, source_shape=[256, 220], target_shape=[256, 239],
            retained_sensor_column_range=[0, 220], inserted_zero_column_range=[220, 239],
            retained_columns_exact_parent=True, inserted_columns_exact_zero=True,
            inserted_zero_value_count=4864, inserted_zero_nonzero_count=0,
            parent_sensor_sha256=state_digest({"value":first}),
            retained_sensor_columns_sha256=state_digest({"value":actor_state[ACTOR_FIRST_LAYER_KEY][:, :220]}),
            inserted_zero_columns_sha256=state_digest({"value":actor_state[ACTOR_FIRST_LAYER_KEY][:, 220:239]})),
        other_actor_state_keys_exact_parent=exact_actor, critic_state_keys_exact_parent=sorted(old_critic),
        other_actor_state_sha256=state_digest({key:actor_state[key] for key in exact_actor}),
        critic_all_tensors_exact_parent=True, critic_input_width=233, critic_privileged_width=13,
        feature_scale_exact_parent=True, action_transforms_exact_parent=True,
        branch_gain_key=GAIN_KEY, branch_gain_exact_parent=True, branch_buffers_exact_parent=True,
        branch_feature_math_unchanged=True, parent_yaw_phase_migration=deepcopy(parent["yaw_phase_migration"]),
        estimator_seed=estimator_seed, him_feature_config=deepcopy(spec),
        new_actor_state_keys=sorted(set(actor_state)-set(old_actor)),
        initial_encoder_state_sha256=state_digest(encoder.state_dict()),
        initial_training_head_state_sha256=state_digest(training_state),
        new_core_column_parameter_count=4864, new_encoder_parameter_count=127891,
        all_added_core_and_encoder_parameters_trainable=True,
        actor_sensor_frames=5, estimator_sensor_frames=21, external_history_order="oldest_first_newest_last",
        retained_external_input_size=924, retained_external_output_size=11,
        function_preservation="exact_parent220_core_columns_then19_zero_estimate_columns_with_exact_existing_yaw_branch",
        critic_function_preservation="entire_parent233_critic_and_parent_sensor220_features_exact",
        numerical_parity="requires_unique_independent_V25_CPU_and_fixed_ONNX_proof",
        source_parent_state_tensors_modified=False, old_mathematical_proofs_reexecuted=False,
        uses_teacher=False, uses_policy_demonstrations=False, actor_constructed=False,
        critic_constructed=False, actor_forward_calls=0, physical_steps=0, optimizer_updates=0)
    migrated = deepcopy(parent)
    migrated.update(actor_config=target_config, actor_state_dict=actor_state,
                    critic_state_dict=critic_state, him_migration=deepcopy(evidence))
    return migrated, evidence
