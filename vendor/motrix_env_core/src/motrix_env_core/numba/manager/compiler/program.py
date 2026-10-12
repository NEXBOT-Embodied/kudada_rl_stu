# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

from motrix_env_core.numba.manager.env import CompiledManagerProgram, KernelInputSource


@dataclass(frozen=True)
class PreparedInvocation:
    """Runtime metadata required to generate one compiled manager term."""

    context: str
    kind: str
    dispatcher: Any
    receiver_key: str | None
    args_expressions: tuple[str, ...] = ()
    args_values: tuple[Any, ...] = ()
    args_prepared_indices: tuple[int | None, ...] = ()
    output_size: int = 0


@dataclass(frozen=True)
class ResolvedManagerContext:
    prepared_index: int
    expression: str


@dataclass(frozen=True)
class ResolvedSimReset:
    invocation: PreparedInvocation
    sim_writes_type: str
    output_offset: int
    output_count: int


@dataclass(frozen=True)
class _CompiledManagerProgram(CompiledManagerProgram):
    invocations: tuple[PreparedInvocation, ...]
    prepared_terms: tuple[KernelInputSource, ...]
    input_offsets: tuple[int, ...]
    context: ResolvedManagerContext


__all__ = [
    "PreparedInvocation",
    "ResolvedManagerContext",
    "ResolvedSimReset",
    "_CompiledManagerProgram",
]
