# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any

from motrix_env_core.numba.manager.compiler.plan import ManagerLayout

if TYPE_CHECKING:
    from motrix_env_core.numba.manager.compiler.compiler import NumbaKernelCompiler

__all__ = ["ManagerLayout", "NumbaKernelCompiler"]


def __getattr__(name: str) -> Any:
    if name == "NumbaKernelCompiler":
        from motrix_env_core.numba.manager.compiler.compiler import NumbaKernelCompiler

        return NumbaKernelCompiler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
