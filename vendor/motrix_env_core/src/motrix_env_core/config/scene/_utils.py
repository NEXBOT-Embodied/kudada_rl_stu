# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Vec4 = tuple[float, float, float, float]


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def optional_vec(name: str, value: tuple[float, ...] | None, length: int) -> tuple[float, ...] | None:
    if value is not None and len(value) != length:
        raise ValueError(f"{name} must contain {length} values, got {value!r}")
    return value


def validate_range(name: str, value: Vec2 | None) -> None:
    optional_vec(name, value, 2)
    if value is not None and value[0] > value[1]:
        raise ValueError(f"{name} lower bound must not exceed upper bound, got {value!r}")
