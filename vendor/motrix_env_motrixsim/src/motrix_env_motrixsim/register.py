# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Entry-point registration for the ``motrix_env.sim_backends`` group.

Importing this module must stay cheap: factories are lazy so registry
discovery never imports the native simulator.
"""

from motrix_env_core.sim.registry import register_sim_backend


def _sim_backend():
    from motrix_env_motrixsim.runtime import MotrixSimBackend

    return MotrixSimBackend


def register() -> None:
    """Register the MotrixSim backend as the default simulator."""
    register_sim_backend("motrixsim", _sim_backend, default=True)


__all__ = ["register"]
