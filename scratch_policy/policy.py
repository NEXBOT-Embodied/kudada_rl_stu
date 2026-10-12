"""Full actor network and transforms shared by PPO and ONNX deployment."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from torch import nn

CHECKPOINT_FORMAT = "scratch_walking_ppo_v1"


def mlp(inputs: int, hidden: list[int], outputs: int) -> nn.Sequential:
    sizes = [inputs, *hidden, outputs]
    if any(int(size) < 1 for size in sizes):
        raise ValueError("All network widths must be positive")
    layers: list[nn.Module] = []
    for index, (source, target) in enumerate(zip(sizes[:-1], sizes[1:])):
        layer = nn.Linear(source, target)
        nn.init.orthogonal_(layer.weight, gain=.01 if index == len(sizes)-2 else 2.**.5)
        nn.init.zeros_(layer.bias)
        layers.append(layer)
        if index < len(sizes)-2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def _vector(value: Any, width: int, name: str) -> torch.Tensor:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = np.full(width, array, dtype=np.float32)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise ValueError(f"{name} needs {width} finite values or a scalar")
    return torch.from_numpy(array.copy())


class ScratchActor(nn.Module):
    """Use available sensor history only; privileged data never enters actor.

    PPO samples a Gaussian in latent coordinates and applies the same tanh,
    scaling and limits used by deterministic deployment. Storing latent actions
    avoids evaluating the density of clipped actions in the PPO likelihood ratio.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = dict(config)
        self.input_size = int(config.get("input_size", 220))
        if self.input_size < 44 or self.input_size > 924 or self.input_size % 44:
            raise ValueError("input_size must be 44 to 924 and a multiple of 44")
        frame_scale = _vector(config.get("feature_scale", [1.]*44), 44, "feature_scale")
        self.register_buffer("feature_scale", frame_scale.repeat(self.input_size//44))
        scale = _vector(config.get("action_scale", 2.), 10, "action_scale")
        if torch.any(scale <= 0):
            raise ValueError("action_scale must be positive")
        self.register_buffer("action_scale", scale)
        self.register_buffer("action_bias", _vector(config.get("action_bias", 0.), 10, "action_bias"))
        bounds = config["action_bounds"]
        lower, upper = _vector(bounds["lower"], 10, "lower"), _vector(bounds["upper"], 10, "upper")
        if torch.any(lower >= upper) or torch.any(lower < -20.) or torch.any(upper > 20.):
            raise ValueError("Action bounds must be ordered and within controller [-20,20]")
        self.register_buffer("action_lower", lower)
        self.register_buffer("action_upper", upper)
        self.network = mlp(self.input_size, [int(n) for n in config.get("hidden", [256, 128, 64])], 10)

    def features(self, observations: torch.Tensor) -> torch.Tensor:
        return observations[:, -self.input_size:] * self.feature_scale

    def latent(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(self.features(observations))

    def actions_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        actions = torch.tanh(latent) * self.action_scale + self.action_bias
        return torch.minimum(torch.maximum(actions, self.action_lower), self.action_upper)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actions_from_latent(self.latent(observations))

    def hardware_output(self, observations: torch.Tensor) -> torch.Tensor:
        actions = self(observations)
        # The inspected motion_ctrl ignores index 10. Zero preserves its shape
        # without needing the original network's auxiliary classifier.
        return torch.cat((actions, torch.zeros_like(actions[:, :1])), dim=-1)


class HardwareActor(nn.Module):
    def __init__(self, actor: ScratchActor):
        super().__init__()
        self.actor = actor

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor.hardware_output(observations)


def state_digest(state: dict[str, torch.Tensor]) -> str:
    """Hash tensor contents independent of torch.save archive timestamps."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_checkpoint(checkpoint: dict[str, Any]) -> None:
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Only independent scratch_walking_ppo_v1 checkpoints are accepted")
    if checkpoint.get("provenance", {}).get("uses_teacher") is not False:
        raise ValueError("Checkpoint must declare teacher-free independent training")
    if checkpoint.get("provenance", {}).get("weight_origin") != "seeded_random_initialization":
        raise ValueError("Checkpoint does not establish random initialization lineage")
    if not checkpoint.get("provenance", {}).get("initial_actor_sha256"):
        raise ValueError("Checkpoint is missing its random initialization hash")
    if any("teacher" in name.lower() for name in checkpoint if name != "provenance"):
        raise ValueError("Teacher-linked checkpoint fields are not accepted")


def load_actor(checkpoint: dict[str, Any]) -> ScratchActor:
    validate_checkpoint(checkpoint)
    actor = ScratchActor(checkpoint["actor_config"]).cpu().eval()
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    if any(not torch.isfinite(value).all() for value in actor.state_dict().values()):
        raise ValueError("Nonfinite actor weights or transforms")
    # Config transforms must match serialized buffers, rather than silently
    # trusting config when a checkpoint has been edited.
    expected = ScratchActor(checkpoint["actor_config"]).state_dict()
    for name in ("feature_scale", "action_scale", "action_bias", "action_lower", "action_upper"):
        if not torch.equal(expected[name], actor.state_dict()[name]):
            raise ValueError(f"Actor transform {name} disagrees with actor_config")
    return actor
