# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_env_core.config import configclass
from motrix_env_core.config.scene._utils import Vec3, optional_vec
from motrix_env_core.config.scene.base import SceneObjCfg


@configclass
class LightCfg(SceneObjCfg):
    """A world-level directional light in the generated scene."""

    position: Vec3 = (0.0, 0.0, 1.5)
    direction: Vec3 = (-1.0, -1.0, -1.0)
    color: Vec3 = (1.0, 1.0, 1.0)
    illuminance: float = 5_000.0
    cast_shadows: bool = True

    def validate(self, name: str) -> None:
        super().validate(name)
        optional_vec("LightCfg.position", self.position, 3)
        optional_vec("LightCfg.direction", self.direction, 3)
        optional_vec("LightCfg.color", self.color, 3)
        if self.illuminance < 0.0:
            raise ValueError(f"LightCfg.illuminance must be non-negative, got {self.illuminance}")
