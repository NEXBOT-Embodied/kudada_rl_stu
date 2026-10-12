# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_env_core.config.decorate import configclass


@configclass
class SimCfg:
    """Physics simulation settings consumed by a scene compiler."""

    dt: float = 0.01
    solver_iterations: int | None = None
    solver_tolerance: float | None = None
    gravity: tuple[float, float, float] | None = None

    def validate(self) -> None:
        if self.dt <= 0.0:
            raise ValueError(f"sim.dt must be positive, got {self.dt}")
        if self.solver_iterations is not None and self.solver_iterations <= 0:
            raise ValueError(f"sim.solver_iterations must be positive, got {self.solver_iterations}")
        if self.solver_tolerance is not None and self.solver_tolerance <= 0.0:
            raise ValueError(f"sim.solver_tolerance must be positive, got {self.solver_tolerance}")
        if self.gravity is not None and len(self.gravity) != 3:
            raise ValueError(f"sim.gravity must contain 3 values, got {self.gravity!r}")
