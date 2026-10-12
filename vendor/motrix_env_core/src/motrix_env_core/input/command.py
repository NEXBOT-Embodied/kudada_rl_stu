# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Common high-level command value types."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class PlanarVelocityCommand:
    """Batch-first body-frame ``[vx, vy, yaw_rate]`` command."""

    values: FloatArray

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float32, copy=True)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 3:
            raise ValueError("values must have shape (batch_size, 3), with batch_size > 0")
        if not np.all(np.isfinite(values)):
            raise ValueError("values must contain only finite numbers")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    @property
    def linear_velocity_x_mps(self) -> FloatArray:
        return self.values[:, 0]

    @property
    def linear_velocity_y_mps(self) -> FloatArray:
        return self.values[:, 1]

    @property
    def yaw_rate_rad_s(self) -> FloatArray:
        return self.values[:, 2]


__all__ = ["PlanarVelocityCommand"]
