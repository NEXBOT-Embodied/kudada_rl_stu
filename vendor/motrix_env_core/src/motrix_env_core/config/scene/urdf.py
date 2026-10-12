# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_env_core.config import configclass
from motrix_env_core.config.scene._utils import Vec2, Vec3, Vec4, optional_vec, resolve_path, validate_range
from motrix_env_core.config.scene.base import ModelFileCfg


@configclass(kw_only=True)
class JointCfg:
    """Simulation properties applied to a joint in an imported robot."""

    joint_name: str
    armature: float | None = None
    friction_loss: float | None = None

    def validate(self) -> None:
        if not self.joint_name:
            raise ValueError("JointCfg.joint_name must not be empty")
        if self.armature is not None and self.armature < 0.0:
            raise ValueError(f"JointCfg.armature must be non-negative, got {self.armature}")
        if self.friction_loss is not None and self.friction_loss < 0.0:
            raise ValueError(f"JointCfg.friction_loss must be non-negative, got {self.friction_loss}")


@configclass(kw_only=True)
class UrdfGeomCfg:
    """Contact properties applied to a named geometry in an imported URDF robot."""

    geom_name: str
    friction: Vec3 | None = None
    condim: int | None = None
    priority: int | None = None

    def validate(self) -> None:
        if not self.geom_name:
            raise ValueError("UrdfGeomCfg.geom_name must not be empty")
        optional_vec("UrdfGeomCfg.friction", self.friction, 3)
        if self.friction is not None and any(value < 0.0 for value in self.friction):
            raise ValueError(f"UrdfGeomCfg.friction must be non-negative, got {self.friction!r}")
        if self.condim is not None and self.condim not in (1, 3, 4, 6):
            raise ValueError(f"UrdfGeomCfg.condim must be one of 1, 3, 4, or 6, got {self.condim}")
        if self.priority is not None and self.priority < 0:
            raise ValueError(f"UrdfGeomCfg.priority must be non-negative, got {self.priority}")


@configclass(kw_only=True)
class SiteCfg:
    """A simulation-only reference site attached to an imported URDF link."""

    name: str
    parent_link_name: str
    position: Vec3 = (0.0, 0.0, 0.0)
    rotation: Vec4 = (0.0, 0.0, 0.0, 1.0)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("SiteCfg.name must not be empty")
        if not self.parent_link_name:
            raise ValueError("SiteCfg.parent_link_name must not be empty")
        optional_vec("SiteCfg.position", self.position, 3)
        optional_vec("SiteCfg.rotation", self.rotation, 4)


@configclass(kw_only=True)
class ActuatorCfg:
    """An actuator added to an imported robot before scene composition."""

    joint_name: str
    name: str | None = None
    ctrl_range: Vec2 | None = None
    force_range: Vec2 | None = None

    @property
    def actuator_name(self) -> str:
        return self.joint_name if self.name is None else self.name

    def validate(self) -> None:
        if not self.joint_name:
            raise ValueError("ActuatorCfg.joint_name must not be empty")
        if not self.actuator_name:
            raise ValueError("ActuatorCfg.name must not be empty")
        validate_range("ActuatorCfg.ctrl_range", self.ctrl_range)
        validate_range("ActuatorCfg.force_range", self.force_range)


@configclass(kw_only=True)
class PositionActuatorCfg(ActuatorCfg):
    """A joint position servo using absolute damping coefficient ``kv``."""

    kp: float
    kv: float = 0.0
    inherit_joint_range: bool = False

    def validate(self) -> None:
        super().validate()
        if self.inherit_joint_range and self.ctrl_range is not None:
            raise ValueError("PositionActuatorCfg.inherit_joint_range is mutually exclusive with ctrl_range")
        if self.kp <= 0.0:
            raise ValueError(f"PositionActuatorCfg.kp must be positive, got {self.kp}")
        if self.kv < 0.0:
            raise ValueError(f"PositionActuatorCfg.kv must be non-negative, got {self.kv}")


@configclass(kw_only=True)
class UrdfFileCfg(ModelFileCfg):
    """A URDF model augmented with simulation-only sites, joints, and actuators."""

    geoms: list[UrdfGeomCfg] = []
    sites: list[SiteCfg] = []
    joints: list[JointCfg] = []
    actuators: list[ActuatorCfg] = []

    def validate(self) -> None:
        super().validate()
        path = resolve_path(self.file)
        if path.suffix.lower() != ".urdf":
            raise ValueError(f"UrdfFileCfg.file must be a URDF file, got {path.suffix!r}")

        for geom in self.geoms:
            geom.validate()
        for site in self.sites:
            site.validate()
        for joint in self.joints:
            joint.validate()
        for actuator in self.actuators:
            actuator.validate()
