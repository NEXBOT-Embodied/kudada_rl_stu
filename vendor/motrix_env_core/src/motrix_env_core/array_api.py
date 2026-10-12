# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral array types and operations shared across runtimes."""

from __future__ import annotations

from types import ModuleType
from typing import Any, TypeAlias, TypeGuard, Union

import numpy as np
import numpy.typing as npt
from array_api_compat import array_namespace as _array_namespace
from array_api_compat import is_numpy_array, is_torch_array

Array: TypeAlias = Union[npt.NDArray[Any], "torch.Tensor"]  # noqa: F821


def array_namespace(*arrays: Array) -> ModuleType:
    """Return the NumPy or Torch Array API namespace shared by ``arrays``."""

    for array in arrays:
        if not is_array(array):
            raise TypeError(
                f"array_namespace only supports NumPy arrays and Torch tensors, got {type(array).__name__}."
            )

    return _array_namespace(*arrays)


def is_array(value: object) -> TypeGuard[Array]:
    """Return whether ``value`` is a supported NumPy array or Torch tensor."""

    return (is_numpy_array(value) and isinstance(value, np.ndarray)) or is_torch_array(value)


__all__ = [
    "Array",
    "array_namespace",
    "is_array",
]
