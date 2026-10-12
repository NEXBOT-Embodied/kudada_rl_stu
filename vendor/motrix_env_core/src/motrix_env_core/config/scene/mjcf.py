# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_env_core.config import configclass
from motrix_env_core.config.scene.base import ModelFileCfg


@configclass(kw_only=True)
class MjcfFileCfg(ModelFileCfg):
    """An MJCF model loaded from a file."""
