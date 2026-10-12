# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Grouped simulator data and model query declarations."""

from dataclasses import field
from typing import Any

from motrix_env_core.config import configclass
from motrix_env_core.sim.model import ModelQuery
from motrix_env_core.sim.read import SimDataQuery


@configclass
class SimQueriesCfg:
    """Simulator queries grouped by runtime data and static model sources."""

    data: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate data and model query declarations without modifying them."""
        if not isinstance(self.data, dict):
            raise TypeError(f"Simulator data queries must be a dict, got {type(self.data).__name__}.")
        for name, query in self.data.items():
            if not isinstance(query, SimDataQuery) or type(query) is SimDataQuery:
                raise TypeError(
                    f"Simulator data query {name!r} must be a concrete SimDataQuery, got {type(query).__name__}."
                )
        for name, query in self.model.items():
            if not isinstance(query, ModelQuery) or type(query) is ModelQuery:
                raise TypeError(f"Model query {name!r} must be a concrete ModelQuery, got {type(query).__name__}.")


__all__ = ["SimQueriesCfg"]
