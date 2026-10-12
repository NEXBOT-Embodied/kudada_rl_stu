# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Stateful per-environment observation noise for Numba manager terms."""

import numba
import numpy as np

from motrix_env_core.numba.manager.rand import next_uniform


@numba.njit(inline="always")
def add_uniform_noise(out: np.ndarray, amplitude: np.float32, rng_state: np.ndarray) -> None:
    """Consume one random value per output component and add scaled uniform noise in place."""
    if amplitude != 0.0:
        for component in range(out.shape[0]):
            out[component] += next_uniform(rng_state) * amplitude


@numba.njit(cache=True, parallel=True)
def add_uniform_noise_batch(
    out: np.ndarray,
    amplitude: float,
    rng_states: np.ndarray,
    env_ids: np.ndarray,
) -> None:
    """Advance selected environment states while adding noise to a reset batch."""
    if amplitude != 0.0:
        for row in numba.prange(out.shape[0]):
            add_uniform_noise(out[row], amplitude, rng_states[env_ids[row]])


__all__ = [
    "add_uniform_noise",
    "add_uniform_noise_batch",
]
