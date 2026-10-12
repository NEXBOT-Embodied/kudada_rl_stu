# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any, TypeVar

_Dispatchable = TypeVar("_Dispatchable", bound=Callable[..., Any])
_DISPATCH_MARKER = "__motrix_manager_dispatch__"


def dispatch(function: _Dispatchable) -> _Dispatchable:
    """Mark a manager fused-kernel entry method for compiler dispatch.

    The decorator is intentionally transparent: it preserves the original Python
    function so the manager compiler can inspect its signature and annotations.
    Numba compilation remains owned by the manager compiler.
    """
    setattr(function, _DISPATCH_MARKER, True)
    return function


def is_dispatched(function: object) -> bool:
    """Return whether a function carries the manager dispatch marker."""
    return bool(getattr(function, _DISPATCH_MARKER, False))


__all__ = ["dispatch", "is_dispatched"]
