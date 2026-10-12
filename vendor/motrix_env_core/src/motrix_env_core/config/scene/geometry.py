# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_env_core.config import configclass
from motrix_env_core.config.scene._utils import Vec3, optional_vec
from motrix_env_core.config.scene.base import SceneObjCfg


@configclass(kw_only=True)
class GeomCfg(SceneObjCfg):
    """Base config for generated scene geometry."""

    material: str | None = None
    friction: Vec3 = (1.0, 0.005, 0.0001)
    condim: int = 3
    contype: int = 1
    conaffinity: int = 1
    priority: int = 1

    def validate(self, name: str) -> None:
        super().validate(name)
        optional_vec("GeomCfg.friction", self.friction, 3)
        if self.material == "":
            raise ValueError("GeomCfg.material must not be empty")


@configclass
class FlatTerrainCfg(GeomCfg):
    """An infinite flat terrain in the generated scene."""

    height: float = 0.0


@configclass
class HFieldTerrainCfg(GeomCfg):
    """A terrain geometry backed by a configured height-field asset."""

    hfield: str

    def validate(self, name: str) -> None:
        super().validate(name)
        if not self.hfield:
            raise ValueError("HFieldTerrainCfg.hfield must not be empty")
