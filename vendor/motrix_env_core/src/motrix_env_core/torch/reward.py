# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import math

import torch

_DEFAULT_VALUE_AT_MARGIN = 0.1


def _sigmoids(x: torch.Tensor, value_at_1: float, sigmoid: str) -> torch.Tensor:
    if sigmoid in ("cosine", "linear", "quadratic"):
        if not 0 <= value_at_1 < 1:
            raise ValueError(f"`value_at_1` must be nonnegative and smaller than 1, got {value_at_1}.")
    elif not 0 < value_at_1 < 1:
        raise ValueError(f"`value_at_1` must be strictly between 0 and 1, got {value_at_1}.")

    if sigmoid == "gaussian":
        scale = math.sqrt(-2.0 * math.log(value_at_1))
        return torch.exp(-0.5 * (x * scale).square())
    if sigmoid == "hyperbolic":
        scale = math.acosh(1.0 / value_at_1)
        return 1.0 / torch.cosh(x * scale)
    if sigmoid == "long_tail":
        scale = math.sqrt(1.0 / value_at_1 - 1.0)
        return 1.0 / ((x * scale).square() + 1.0)
    if sigmoid == "reciprocal":
        scale = 1.0 / value_at_1 - 1.0
        return 1.0 / (torch.abs(x) * scale + 1.0)
    if sigmoid == "linear":
        scaled_x = x * (1.0 - value_at_1)
        return torch.where(torch.abs(scaled_x) < 1.0, 1.0 - scaled_x, torch.zeros_like(x))
    if sigmoid == "quadratic":
        scaled_x = x * math.sqrt(1.0 - value_at_1)
        return torch.where(torch.abs(scaled_x) < 1.0, 1.0 - scaled_x.square(), torch.zeros_like(x))
    if sigmoid == "tanh_squared":
        scale = math.atanh(math.sqrt(1.0 - value_at_1))
        return 1.0 - torch.tanh(x * scale).square()
    raise ValueError(f"Unknown sigmoid type {sigmoid!r}.")


def tolerance(
    x: torch.Tensor,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: str = "gaussian",
    value_at_margin: float = _DEFAULT_VALUE_AT_MARGIN,
) -> torch.Tensor:
    lower, upper = bounds
    if lower > upper:
        raise ValueError("lower bound must be less than upper")
    if margin < 0:
        raise ValueError("margin must be non-negative")

    in_bounds = torch.logical_and(lower <= x, x <= upper)
    if margin == 0:
        return torch.where(in_bounds, torch.ones_like(x), torch.zeros_like(x))
    distance = torch.where(x < lower, lower - x, x - upper) / margin
    return torch.where(in_bounds, torch.ones_like(x), _sigmoids(distance, value_at_margin, sigmoid))
