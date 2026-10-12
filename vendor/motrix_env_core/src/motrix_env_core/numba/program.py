# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np

from motrix_env_core.numba.kernel import CPUDispatcher


@dataclass(frozen=True)
class NumbaTaskProgram:
    """Fused reset/evaluate/observe kernels paired with environment-local reward weights."""

    evaluate_kernel: CPUDispatcher
    observe_kernel: CPUDispatcher
    reset_kernel: CPUDispatcher
    reward_weights: np.ndarray
