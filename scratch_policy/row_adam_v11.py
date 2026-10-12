"""Explicit Adam row learning rates for the unchanged sixty-weight yaw branch.

Gradients enter the ordinary Adam moment equations without row scaling. Only
the final parameter update uses the ten actual row learning rates. This helper
has no simulator, policy-forward, ONNX or deployment-controller dependencies.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim import Optimizer


HIP_YAW_ROWS = (0, 5)
CORE_BASE_LR = 5.e-6
BRANCH_YAW_BASE_LR = 1.e-4
ROW_ADAM_SCHEMA = "scratch_branch_explicit_row_adam_v11"


def base_row_lrs_v11() -> list[float]:
    return [BRANCH_YAW_BASE_LR if row in HIP_YAW_ROWS else CORE_BASE_LR
            for row in range(10)]


class RowAdamV11(Optimizer):
    """One dense [10,6] parameter, with exact per-row Adam update rates.

The state uses the standard ``step``, ``exp_avg`` and ``exp_avg_sq`` names.
There is one shared moment/step history for all sixty entries. Changing the
adaptive multiplier affects parameter updates and never changes gradients.
"""

    def __init__(self, parameter: torch.Tensor,
                 base_row_lrs: Iterable[float] | None = None,
                 lr_multiplier: float = 1.,
                 betas: tuple[float, float] = (.9, .999), eps: float = 1.e-8):
        if parameter.shape != (10, 6) or not parameter.is_leaf or not parameter.requires_grad:
            raise ValueError("RowAdamV11 requires one trainable leaf [10,6] tensor")
        rates = list(base_row_lrs_v11() if base_row_lrs is None else base_row_lrs)
        if rates != base_row_lrs_v11():
            raise ValueError("V11 fixes hip-yaw row rates at 1e-4 and other rows at 5e-6")
        if not math.isfinite(lr_multiplier) or lr_multiplier <= 0.:
            raise ValueError("The shared adaptive multiplier must be finite and positive")
        if not (0. <= betas[0] < 1. and 0. <= betas[1] < 1.) or not math.isfinite(eps) or eps <= 0.:
            raise ValueError("Invalid Adam betas or epsilon")
        super().__init__([parameter], dict(base_row_lrs=rates,
            lr_multiplier=float(lr_multiplier), betas=betas, eps=float(eps)))

    def actual_row_lrs(self) -> list[float]:
        group = self.param_groups[0]
        return [float(rate)*group['lr_multiplier'] for rate in group['base_row_lrs']]

    def set_lr_multiplier(self, multiplier: float) -> None:
        if not math.isfinite(multiplier) or multiplier <= 0.:
            raise ValueError("The shared adaptive multiplier must be finite and positive")
        self.param_groups[0]['lr_multiplier'] = float(multiplier)

    def load_state_dict(self, state_dict) -> None:
        groups = state_dict.get('param_groups', [])
        if len(groups) != 1 or len(groups[0].get('params', [])) != 1:
            raise ValueError("RowAdamV11 keeps one sixty-weight parameter group")
        saved_group = groups[0]
        if saved_group.get('base_row_lrs') != base_row_lrs_v11():
            raise ValueError("Resumed RowAdamV11 base rates differ from the fixed specification")
        multiplier = saved_group.get('lr_multiplier', float('nan'))
        if not math.isfinite(multiplier) or multiplier <= 0.:
            raise ValueError("Resumed RowAdamV11 multiplier must be finite and positive")
        expected_group = self.param_groups[0]
        saved_betas = saved_group.get('betas', ())
        if len(saved_betas) != 2 or tuple(saved_betas) != tuple(expected_group['betas']):
            raise ValueError("Resumed RowAdamV11 betas differ from the initialized specification")
        if saved_group.get('eps') != expected_group['eps']:
            raise ValueError("Resumed RowAdamV11 epsilon differs from the initialized specification")
        parameter = expected_group['params'][0]
        saved_states = state_dict.get('state', {})
        parameter_key = saved_group['params'][0]
        if any(key != parameter_key for key in saved_states):
            raise ValueError("Resumed RowAdamV11 state contains an unrelated parameter")
        saved_state = saved_states.get(parameter_key, {})
        if saved_state:
            if set(saved_state) != {'step', 'exp_avg', 'exp_avg_sq'}:
                raise ValueError("Resumed RowAdamV11 state must contain step and both moments")
            step = saved_state['step']
            if (not isinstance(step, torch.Tensor) or step.shape != torch.Size([])
                    or step.dtype != torch.float32 or not bool(torch.isfinite(step))):
                raise ValueError("Resumed RowAdamV11 step must be a finite scalar float32 tensor")
            step_value = float(step.item())
            if step_value < 0. or not step_value.is_integer():
                raise ValueError("Resumed RowAdamV11 step must be a nonnegative integer")
            for name in ('exp_avg', 'exp_avg_sq'):
                moment = saved_state[name]
                if (not isinstance(moment, torch.Tensor) or moment.shape != parameter.shape
                        or moment.dtype != parameter.dtype or not bool(torch.isfinite(moment).all())):
                    raise ValueError(f"Resumed RowAdamV11 {name} must match parameter shape/dtype and be finite")
            if bool((saved_state['exp_avg_sq'] < 0.).any()):
                raise ValueError("Resumed RowAdamV11 second moment must be nonnegative")
        super().load_state_dict(state_dict)
        self.set_lr_multiplier(multiplier)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        group = self.param_groups[0]
        parameter = group['params'][0]
        gradient = parameter.grad
        if gradient is None:
            return loss
        if gradient.is_sparse:
            raise RuntimeError("RowAdamV11 does not accept sparse gradients")
        state = self.state[parameter]
        if not state:
            state['step'] = torch.tensor(0., dtype=torch.float32)
            state['exp_avg'] = torch.zeros_like(parameter)
            state['exp_avg_sq'] = torch.zeros_like(parameter)
        state['step'].add_(1.)
        step = int(state['step'].item())
        beta1, beta2 = group['betas']
        moment, second = state['exp_avg'], state['exp_avg_sq']
        moment.mul_(beta1).add_(gradient, alpha=1.-beta1)
        second.mul_(beta2).addcmul_(gradient, gradient, value=1.-beta2)
        denominator = second.sqrt()/math.sqrt(1.-beta2**step)
        denominator.add_(group['eps'])
        row_rates = parameter.new_tensor(self.actual_row_lrs()).unsqueeze(-1)
        parameter.add_(-row_rates*(moment/(1.-beta1**step))/denominator)
        return loss


def normal_kl_by_joint_v11(old_mean: torch.Tensor, old_logstd: torch.Tensor,
                          new_mean: torch.Tensor, new_logstd: torch.Tensor) -> torch.Tensor:
    """Mean Normal KL per joint, solely for recorded minibatch diagnostics.

The original sampled global KL controls PPO stopping and adaptation. This
closed-form statistic neither changes that gate nor calls a policy forward.
"""
    variance_ratio = torch.exp(2.*(old_logstd-new_logstd))
    scaled_mean_square = (old_mean-new_mean).square()*torch.exp(-2.*new_logstd)
    values = new_logstd-old_logstd+.5*(variance_ratio+scaled_mean_square-1.)
    return values.mean(dim=0)
