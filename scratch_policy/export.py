"""Export a complete independent policy to the motion_ctrl ONNX interface."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import onnx
from onnx import helper
import onnxruntime as ort
import torch

from .policy import HardwareActor, ScratchActor, load_actor, state_digest


def observation_cases(observations_path: Path | None = None) -> np.ndarray:
    generator = np.random.default_rng(9184)
    frames = []
    frames.append(np.zeros((21, 44), dtype=np.float32))
    for command in (0., .05, .1, .2, .3, -.1):
        for noise in (0., .01, .05, .15):
            history = np.zeros((21, 44), dtype=np.float32)
            history[:, 3] = command*2.
            phase = (np.arange(21)*.02/.5) % 1. if command else np.zeros(21)
            history[:, 6] = np.sin(2.*np.pi*phase)
            history[:, 7] = np.cos(2.*np.pi*phase)
            history[:, 13] = -1.
            history[:, 8:11] = generator.normal(0., noise, (21, 3))
            history[:, 11:13] = generator.normal(0., noise*.1, (21, 2))
            history[:, 14:24] = generator.normal(0., noise, (21, 10))
            history[:, 24:34] = generator.normal(0., noise*.3, (21, 10))
            history[:, 34:44] = generator.normal(0., noise*2., (21, 10))
            frames.append(history)
    samples = np.stack(frames).reshape(-1, 924)
    if observations_path is not None:
        with np.load(observations_path, allow_pickle=False) as archive:
            actual = np.asarray(archive["observations"], dtype=np.float32).reshape(-1, 924)
        if not len(actual) or not np.isfinite(actual).all():
            raise ValueError("Recorded observations must be nonempty and finite")
        samples = np.concatenate((samples, actual[:4096]))
    return np.ascontiguousarray(samples, dtype=np.float32)


def verify_policy(path: Path, actor: ScratchActor, observations: np.ndarray) -> dict[str, Any]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or inputs[0].name != "input" or inputs[0].shape[-1] != 924
            or inputs[0].type != "tensor(float)"):
        raise ValueError("ONNX input must be float32 input [batch,924]")
    if (len(outputs) != 1 or outputs[0].name != "output" or outputs[0].shape[-1] != 11
            or outputs[0].type != "tensor(float)"):
        raise ValueError("ONNX output must be float32 output [batch,11]")
    maximum = 0.
    batches = []
    cursor = 0
    while cursor < len(observations):
        size = (1, 7, 64)[len(batches)%3]
        sample = np.ascontiguousarray(observations[cursor:cursor+size], dtype=np.float32)
        with torch.inference_mode():
            expected = actor.hardware_output(torch.from_numpy(sample)).numpy()
        actual = session.run(["output"], {"input": sample})[0]
        if actual.shape != (len(sample), 11) or not np.isfinite(actual).all():
            raise ValueError("ONNX output is invalid")
        if np.any(actual[:, 10] != 0.):
            raise ValueError("Reserved eleventh output must be zero")
        if np.any(actual[:, :10] < actor.action_lower.numpy()-1.e-6) or np.any(actual[:, :10] > actor.action_upper.numpy()+1.e-6):
            raise ValueError("ONNX output exceeds embedded per-joint limits")
        maximum = max(maximum, float(np.max(np.abs(actual-expected))))
        batches.append(len(sample))
        cursor += len(sample)
    if maximum >= 1.e-4:
        raise ValueError(f"ONNX parity max error {maximum} exceeds 1e-4")
    return dict(passed=True, cases=len(observations), batch_sizes=batches, max_abs_error=maximum,
                input_name="input", input_shape=["batch",924], output_name="output", output_shape=["batch",11],
                dtype="float32", opset=17, ir_version=8, reserved_output_zero=True,
                onnxruntime_validation_version=ort.__version__)


def export_checkpoint(checkpoint_path: Path, output_path: Path,
                      observations_path: Path | None = None) -> dict[str, Any]:
    """Export without loading any model other than this scratch checkpoint."""
    checkpoint_path, output_path = Path(checkpoint_path).resolve(), Path(output_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor = load_actor(checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scratch-onnx-", dir=output_path.parent) as directory:
        staged = Path(directory)/"policy.onnx"
        torch.onnx.export(HardwareActor(actor), torch.zeros(1,924), str(staged),
                          input_names=["input"], output_names=["output"], opset_version=17,
                          dynamic_axes={"input":{0:"batch"}, "output":{0:"batch"}}, dynamo=False)
        model = onnx.load(str(staged))
        model.ir_version = 8
        helper.set_model_props(model, {
            "policy_origin":"independent_random_initialization",
            "uses_teacher":"false", "history_order":"oldest_first_newest_last",
            "single_observation_size":"44", "observation_frames":"21",
            "used_observation_frames":str(actor.input_size//44),
            "actor_config":json.dumps(checkpoint["actor_config"]),
            "actions":"first_10_joint_actions_with_embedded_limits",
            "reserved_output":"index_10_zero_ignored_by_current_motion_ctrl",
            "initial_actor_sha256":checkpoint["provenance"]["initial_actor_sha256"],
            "checkpoint_sha256":hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        })
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model, full_check=True)
        onnx.save(model, str(staged), save_as_external_data=False)
        report = verify_policy(staged, actor, observation_cases(observations_path))
        staged.replace(output_path)
    report.update(dict(checkpoint=str(checkpoint_path), checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                       policy=str(output_path), policy_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(),
                       actor_sha256=state_digest(actor.state_dict()), actor_config=checkpoint["actor_config"],
                       completed_updates=checkpoint["completed_updates"], total_transitions=checkpoint["total_transitions"],
                       provenance=checkpoint["provenance"], hardware_contract=checkpoint["hardware_contract"],
                       self_contained=True, uses_original_onnx=False,
                       verification_scope="Interface, standard ONNX operators and numerical parity; physical performance evaluated separately"))
    output_path.with_suffix(".verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observations", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.checkpoint, args.output, args.observations), indent=2))


if __name__ == "__main__":
    main()
