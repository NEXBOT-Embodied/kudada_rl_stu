"""Export the complete sensor-only HIM yaw actor with fixed motion_ctrl IO."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import onnx
import torch

from .export_fixed import build_fixed_model, verify_policy_fixed
from .ort_runtime import activate_runtime
from .policy_yaw_him_v25 import (
    ARCHITECTURE, CORE_FEATURE_WIDTH, METHOD, SOURCE_FILES, canonical_sha256,
    load_actor, source_hashes, state_digest,
)

EXPORT_SCHEMA = "scratch_complete_fixed_him_yaw_export_v25"


def verify_source_binding(checkpoint: dict[str, Any]) -> dict[str, Any]:
    recorded = checkpoint.get("source_sha256", {})
    actual = source_hashes()
    for relative in SOURCE_FILES:
        if recorded.get(relative) != actual[relative]:
            raise ValueError(f"HIM V25 export source is absent or changed: {relative}")
    return dict(schema="scratch_him_yaw_export_source_binding_v25", passed=True,
                required_sources=actual, required_source_sha256=canonical_sha256(actual),
                checkpoint_source_sha256=canonical_sha256(recorded))


def export_checkpoint_fixed(checkpoint_path: Path, output_path: Path,
                            observations_path: Path | None = None) -> dict[str, Any]:
    from .export import observation_cases
    checkpoint_path, output_path = Path(checkpoint_path).resolve(), Path(output_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor = load_actor(checkpoint)
    source_binding = verify_source_binding(checkpoint)
    actor_sha = state_digest(actor.state_dict())
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    contract_sha = canonical_sha256(checkpoint["hardware_contract"])
    yaw_migration_sha = canonical_sha256(checkpoint["yaw_phase_migration"])
    him_migration_sha = canonical_sha256(checkpoint["him_migration"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    metadata = dict(export_schema=EXPORT_SCHEMA, actor_architecture=ARCHITECTURE,
        training_method=METHOD, policy_origin="independent_random_initialization", uses_teacher="false",
        history_order="oldest_first_newest_last", single_observation_size="44",
        observation_frames="21", used_observation_frames="21", core_observation_frames="5",
        core_feature_width=str(CORE_FEATURE_WIDTH), gyro_observation_frames="21", estimator_observation_frames="21",
        estimator_internal_history_order="newest_first_oldest_last", estimator_history_reversals="1",
        estimator_features="predicted_scaled_body_velocity3_then_unit_latent16",
        estimator_detached_from_PPO="true", true_body_velocity_actor_input="false",
        target_and_prototypes_deployed="false", external_filter_state="false",
        yaw_phase_features=json.dumps(checkpoint["actor_config"]["yaw_phase_bypass"], sort_keys=True),
        him_estimator=json.dumps(checkpoint["actor_config"]["him_estimator"], sort_keys=True),
        actor_config=json.dumps(checkpoint["actor_config"]),
        actions="first_10_joint_actions_with_embedded_limits", reserved_output="index_10_zero_ignored_by_current_motion_ctrl",
        initial_actor_sha256=checkpoint["provenance"]["initial_actor_sha256"],
        target_initial_actor_sha256=checkpoint["him_migration"]["target_initial_actor_state_sha256"],
        checkpoint_sha256=checkpoint_sha, actor_sha256=actor_sha, hardware_contract_sha256=contract_sha, batch_size="1",
        source_checkpoint_sha256=checkpoint["parent_binding"]["checkpoint_sha256"],
        source_actor_sha256=checkpoint["parent_binding"]["actor_sha256"],
        yaw_phase_migration_sha256=yaw_migration_sha, him_migration_sha256=him_migration_sha,
        structural_intervention="parent220_sensor_features_plus19_detached_HIM_predictions_core239_critic233_exact",
        source_binding_sha256=canonical_sha256(source_binding))
    with tempfile.TemporaryDirectory(prefix="scratch-fixed-him-yaw-v25-onnx-", dir=output_path.parent) as directory:
        staged = Path(directory) / "policy.onnx"
        model = build_fixed_model(actor, staged, metadata)
        onnx.save(model, str(staged), save_as_external_data=False)
        report = verify_policy_fixed(staged, actor, observation_cases(observations_path))
        staged.replace(output_path)
    report.update(export_schema=EXPORT_SCHEMA, actor_architecture=ARCHITECTURE, training_method=METHOD,
        checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_sha,
        policy=str(output_path), policy_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(),
        actor_sha256=actor_sha, actor_config=checkpoint["actor_config"], core_observation_frames=5,
        core_feature_width=239, critic_input_width=233, gyro_observation_frames=21,
        estimator_observation_frames=21, used_observation_frames=21,
        completed_updates=checkpoint["completed_updates"], total_transitions=checkpoint["total_transitions"],
        provenance=checkpoint["provenance"], hardware_contract=checkpoint["hardware_contract"],
        hardware_contract_sha256=contract_sha, source_checkpoint_sha256=metadata["source_checkpoint_sha256"],
        source_actor_sha256=metadata["source_actor_sha256"],
        yaw_phase_migration=checkpoint["yaw_phase_migration"], yaw_phase_migration_sha256=yaw_migration_sha,
        him_migration=checkpoint["him_migration"], him_migration_sha256=him_migration_sha,
        him_evidence=checkpoint.get("him_evidence"),
        structural_intervention=metadata["structural_intervention"], source_binding=source_binding,
        estimator_detached_from_PPO=True, true_body_velocity_actor_input=False,
        target_network_deployed=False, prototypes_deployed=False,
        self_contained=True, uses_original_onnx=False, external_filter_state=False,
        export_wall_seconds=time.perf_counter()-started,
        verification_scope="Fixed IO, complete deployed encoder/core/branch source binding and numerical parity; no physical performance is established")
    output_path.with_suffix(".verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--observations", type=Path)
    args = parser.parse_args()
    activate_runtime(args.runtime_dir)
    print(json.dumps(export_checkpoint_fixed(args.checkpoint, args.output, args.observations), ensure_ascii=False))


if __name__ == "__main__":
    main()
