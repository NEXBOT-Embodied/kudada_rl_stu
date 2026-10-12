"""Select the robot's isolated ONNX Runtime before importing inference modules."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Any

REQUIRED_VERSION = "1.21.0"


def activate_runtime(runtime_dir: Path | str) -> Any:
    directory = Path(runtime_dir).resolve()
    if not (directory / "onnxruntime" / "__init__.py").is_file():
        raise ValueError("runtime-dir must contain an installed onnxruntime package")
    existing = sys.modules.get("onnxruntime")
    if existing is not None:
        module_path = Path(existing.__file__).resolve()
        if existing.__version__ != REQUIRED_VERSION or not module_path.is_relative_to(directory):
            raise ValueError("A different ONNX Runtime is already loaded; use a fresh process")
    sys.path.insert(0, str(directory))
    runtime = importlib.import_module("onnxruntime")
    module_path = Path(runtime.__file__).resolve()
    if runtime.__version__ != REQUIRED_VERSION or not module_path.is_relative_to(directory):
        raise ValueError(f"Expected isolated ONNX Runtime {REQUIRED_VERSION}, found {runtime.__version__} at {module_path}")
    if "CPUExecutionProvider" not in runtime.get_available_providers():
        raise ValueError("The selected runtime has no CPUExecutionProvider")
    return runtime


def session_options(runtime: Any) -> Any:
    options = runtime.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = runtime.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = runtime.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options


def runtime_evidence(runtime: Any) -> dict[str, Any]:
    if runtime.__version__ != REQUIRED_VERSION:
        raise ValueError("Training and fixed export require actual ONNX Runtime 1.21.0")
    return dict(onnxruntime_version=runtime.__version__, module_file=str(Path(runtime.__file__).resolve()),
                providers=["CPUExecutionProvider"], intra_op_num_threads=1, inter_op_num_threads=1,
                execution_mode="ORT_SEQUENTIAL", graph_optimization_level="ORT_ENABLE_ALL",
                input_shape=[1, 924], output_shape=[1, 11])
