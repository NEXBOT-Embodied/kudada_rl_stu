# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np

from motrix_env_core.numba.kernel_data import KernelDataScope


@dataclass(frozen=True)
class InputSlotLayout:
    index: int
    source: str
    scope: KernelDataScope


@dataclass(frozen=True)
class SimInputLayout:
    slot: int
    key: str
    shape: tuple[int, ...]
    consumers: tuple[str, ...]


@dataclass(frozen=True)
class ActionTermLayout:
    name: str
    input_slice: slice
    actuator_names: tuple[str, ...] | None


@dataclass(frozen=True)
class ObservationTermLayout:
    name: str
    output_slice: slice


@dataclass(frozen=True)
class ObservationGroupLayout:
    name: str
    terms: tuple[ObservationTermLayout, ...]
    size: int


@dataclass(frozen=True)
class ScalarTermLayout:
    name: str
    index: int


@dataclass(frozen=True)
class MetricFieldLayout:
    name: str
    index: int
    dtype: np.dtype


@dataclass(frozen=True)
class ManagerLayout:
    inputs: tuple[InputSlotLayout, ...]
    sim_inputs: tuple[SimInputLayout, ...]
    actions: tuple[ActionTermLayout, ...]
    observations: dict[str, ObservationGroupLayout]
    rewards: tuple[ScalarTermLayout, ...]
    terminations: tuple[ScalarTermLayout, ...]
    metrics: tuple[MetricFieldLayout, ...]
    plan_keys: tuple[str, str, str]
    generated_filenames: tuple[str, str, str]

    def dump(self) -> str:
        lines = [
            f"plan_keys (evaluate, observe, reset): {', '.join(key[:12] for key in self.plan_keys)}",
            f"generated_filenames: {', '.join(self.generated_filenames)}",
            "inputs:",
        ]
        lines.extend(f"  [{slot.index}] scope={slot.scope.value} source={slot.source}" for slot in self.inputs)
        lines.append("sim_inputs:")
        lines.extend(
            f"  [{sim_input.slot}] {sim_input.key}: shape={sim_input.shape} consumers={sim_input.consumers}"
            for sim_input in self.sim_inputs
        )
        lines.append("actions:")
        lines.extend(
            f"  {term.name}: input=[{term.input_slice.start}:{term.input_slice.stop}] actuators={term.actuator_names}"
            for term in self.actions
        )
        lines.append("observations:")
        lines.extend(
            f"  {group.name}.{term.name}: [{term.output_slice.start}:{term.output_slice.stop}]"
            for group in self.observations.values()
            for term in group.terms
        )
        lines.append("rewards:")
        lines.extend(f"  [{term.index}] {term.name}" for term in self.rewards)
        lines.append("terminations:")
        lines.extend(f"  [{term.index}] {term.name}" for term in self.terminations)
        lines.append("metrics:")
        lines.extend(f"  [{metric.index}] dtype={metric.dtype.name} {metric.name}" for metric in self.metrics)
        return "\n".join(lines)


__all__ = [
    "ActionTermLayout",
    "InputSlotLayout",
    "ManagerLayout",
    "MetricFieldLayout",
    "ObservationGroupLayout",
    "ObservationTermLayout",
    "ScalarTermLayout",
    "SimInputLayout",
]
