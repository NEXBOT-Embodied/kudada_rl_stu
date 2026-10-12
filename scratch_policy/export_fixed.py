"""Export complete scratch actors with the robot's fixed batch-one interface."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
import onnx
from onnx import helper
import torch

from .ort_runtime import activate_runtime, runtime_evidence, session_options
from .policy import HardwareActor, ScratchActor, load_actor, state_digest


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def graph_digests(path_or_model: Path | str | onnx.ModelProto) -> dict[str, str]:
    model = onnx.load(str(path_or_model)) if isinstance(path_or_model, (str, Path)) else path_or_model
    graph = deepcopy(model.graph)
    parameters = hashlib.sha256()
    for tensor in sorted(graph.initializer, key=lambda item: item.name):
        parameters.update(tensor.SerializeToString(deterministic=True))
    for tensor in graph.initializer:
        for field in ("raw_data", "float_data", "int32_data", "string_data", "int64_data", "double_data", "uint64_data"):
            tensor.ClearField(field)
    return dict(graph_sha256=hashlib.sha256(model.graph.SerializeToString(deterministic=True)).hexdigest(),
                graph_structure_sha256=hashlib.sha256(graph.SerializeToString(deterministic=True)).hexdigest(),
                graph_parameters_sha256=parameters.hexdigest())


def build_fixed_model(actor: ScratchActor, staged_path: Path,
                      metadata: dict[str, str] | None = None) -> onnx.ModelProto:
    """Build the complete deployment graph; audit callers can reconstruct it."""
    staged_path = Path(staged_path)
    torch.onnx.export(HardwareActor(actor), torch.zeros(1, 924), str(staged_path),
                      input_names=["input"], output_names=["output"], opset_version=17, dynamo=False)
    model = onnx.load(str(staged_path))
    model.ir_version = 8
    if metadata is not None:
        helper.set_model_props(model, metadata)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    return model


def verify_policy_fixed(path: Path, actor: ScratchActor, observations: np.ndarray) -> dict[str, Any]:
    import onnxruntime as ort
    runtime = runtime_evidence(ort)
    session = ort.InferenceSession(str(path), sess_options=session_options(ort), providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or inputs[0].name != "input" or inputs[0].shape != [1, 924]
            or inputs[0].type != "tensor(float)"):
        raise ValueError("Fixed ONNX input must be float32 input [1,924]")
    if (len(outputs) != 1 or outputs[0].name != "output" or outputs[0].shape != [1, 11]
            or outputs[0].type != "tensor(float)"):
        raise ValueError("Fixed ONNX output must be float32 output [1,11]")
    cases = np.ascontiguousarray(observations, dtype=np.float32)
    if cases.ndim != 2 or cases.shape[1] != 924 or not len(cases) or not np.isfinite(cases).all():
        raise ValueError("Numerical parity needs nonempty finite raw924 observations")
    maximum = 0.
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(len(cases)):
            sample = np.ascontiguousarray(cases[index:index+1])
            expected = actor.hardware_output(torch.from_numpy(sample)).numpy()
            actual = session.run(["output"], {"input": sample})[0]
            if actual.shape != (1, 11) or not np.isfinite(actual).all() or actual[0, 10] != 0.:
                raise ValueError("Fixed ONNX output is invalid or the reserved output is nonzero")
            if np.any(actual[:, :10] < actor.action_lower.numpy()-1.e-6) or np.any(actual[:, :10] > actor.action_upper.numpy()+1.e-6):
                raise ValueError("ONNX output exceeds embedded per-joint limits")
            maximum = max(maximum, float(np.max(np.abs(actual-expected))))
    if maximum >= 1.e-4:
        raise ValueError(f"Fixed ONNX parity max error {maximum} exceeds 1e-4")
    model = onnx.load(str(path))
    if model.ir_version != 8 or [(item.domain, item.version) for item in model.opset_import] != [("", 17)]:
        raise ValueError("Deployment graph must use standard opset17 and IR8")
    return dict(passed=True, cases=len(cases), batch_sizes=[1]*len(cases), max_abs_error=maximum,
                input_name="input", input_shape=[1, 924], output_name="output", output_shape=[1, 11],
                dtype="float32", opset=17, ir_version=8, reserved_output_zero=True,
                onnxruntime_validation_version=ort.__version__, inference_runtime=runtime,
                batch_mode="one_row_per_session_call", physical_batch_size=1,
                numerical_session_run_calls=len(cases), numerical_session_creation_count=1,
                wall_seconds=time.perf_counter()-started, **graph_digests(model))


def export_checkpoint_fixed(checkpoint_path: Path, output_path: Path,
                            observations_path: Path | None = None) -> dict[str, Any]:
    """Export the provided independently trained actor, including its transforms."""
    # Import only after the command/driver selected its isolated runtime.
    from .export import observation_cases
    checkpoint_path, output_path = Path(checkpoint_path).resolve(), Path(output_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor = load_actor(checkpoint)
    actor_sha = state_digest(actor.state_dict())
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    contract_sha = _canonical_hash(checkpoint["hardware_contract"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    metadata = dict(policy_origin="independent_random_initialization", uses_teacher="false",
                    history_order="oldest_first_newest_last", single_observation_size="44", observation_frames="21",
                    used_observation_frames=str(actor.input_size//44), actor_config=json.dumps(checkpoint["actor_config"]),
                    actions="first_10_joint_actions_with_embedded_limits", reserved_output="index_10_zero_ignored_by_current_motion_ctrl",
                    initial_actor_sha256=checkpoint["provenance"]["initial_actor_sha256"], checkpoint_sha256=checkpoint_sha,
                    actor_sha256=actor_sha, hardware_contract_sha256=contract_sha, batch_size="1",
                    source_checkpoint_sha256=checkpoint.get("source_checkpoint_sha256", checkpoint_sha),
                    source_actor_sha256=checkpoint.get("source_actor_sha256", actor_sha))
    with tempfile.TemporaryDirectory(prefix="scratch-fixed-onnx-", dir=output_path.parent) as directory:
        staged = Path(directory)/"policy.onnx"
        model = build_fixed_model(actor, staged, metadata)
        onnx.save(model, str(staged), save_as_external_data=False)
        report = verify_policy_fixed(staged, actor, observation_cases(observations_path))
        staged.replace(output_path)
    report.update(checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_sha,
                  policy=str(output_path), policy_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(),
                  actor_sha256=actor_sha, actor_config=checkpoint["actor_config"],
                  completed_updates=checkpoint["completed_updates"], total_transitions=checkpoint["total_transitions"],
                  provenance=checkpoint["provenance"], hardware_contract=checkpoint["hardware_contract"],
                  hardware_contract_sha256=contract_sha,
                  source_checkpoint_sha256=metadata["source_checkpoint_sha256"], source_actor_sha256=metadata["source_actor_sha256"],
                  self_contained=True, uses_original_onnx=False, export_wall_seconds=time.perf_counter()-started,
                  verification_scope="Fixed interface and numerical parity only; no physics is executed by export")
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
