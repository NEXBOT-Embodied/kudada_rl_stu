# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from motrix_env_core.config.scene.base import SceneCfg
from motrix_env_core.config.sim import SimCfg

SceneModelT = TypeVar("SceneModelT", covariant=True)


class SceneCompiler(ABC, Generic[SceneModelT]):
    """Compile a backend-independent scene configuration into a backend model."""

    @abstractmethod
    def compile(self, scene: SceneCfg, sim: SimCfg) -> SceneModelT:
        """Validate and compile ``scene`` using the supplied simulation settings."""
