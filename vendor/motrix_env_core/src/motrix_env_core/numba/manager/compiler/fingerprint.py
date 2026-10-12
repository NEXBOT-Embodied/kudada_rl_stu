# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Plan-key and term-source fingerprinting for the manager kernel compiler."""

import hashlib
import inspect
from collections.abc import Callable
from typing import Any

import numba
import numpy as np


def plan_key(parts: list[Any]) -> str:
    return hashlib.sha256(repr(parts).encode()).hexdigest()


def type_name(value_type: type[Any]) -> str:
    return f"{value_type.__module__}.{value_type.__qualname__}"


def function_fingerprint(function: Callable[..., Any]) -> str:
    """Fingerprint a dispatch entry and every function it transitively references.

    The fused kernel inlines the dispatch body plus any ``@njit(inline="always")``
    helpers it calls from its module globals. Hashing only the entry's own source
    would leave helper edits invisible to the plan key, so stale compiled kernels
    would be silently reused (issue #54). Referenced functions are resolved through
    ``co_names`` and hashed recursively; module-level non-function constants are
    hashed by value when cheaply representable.
    """
    hasher = hashlib.sha256()
    seen: set[int] = set()
    queue = [function]
    while queue:
        current = queue.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, numba.core.dispatcher.Dispatcher):
            current = current.py_func
        if not inspect.isfunction(current):
            continue
        code = current.__code__
        try:
            source = inspect.getsource(current).encode()
        except (OSError, TypeError):
            source = code.co_code
        hasher.update(f"{current.__module__}.{current.__qualname__}\0".encode())
        hasher.update(source)
        hasher.update(b"\0")
        module_globals = getattr(current, "__globals__", {})
        for name in code.co_names:
            if name not in module_globals:
                continue
            referenced = module_globals[name]
            if isinstance(referenced, (numba.core.dispatcher.Dispatcher,)) or inspect.isfunction(referenced):
                queue.append(referenced)
            elif isinstance(referenced, (int, float, bool, str, complex, np.generic)):
                hasher.update(f"{name}={referenced!r}\0".encode())
    return hasher.hexdigest()


__all__ = ["function_fingerprint", "plan_key", "type_name"]
