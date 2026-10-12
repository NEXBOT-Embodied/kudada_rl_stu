# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from motrix_env_core.config import configclass
from motrix_env_core.config.scene.base import SceneSensorCfg


class ContactReportField(str, Enum):
    """Fields reported by contact sensors."""

    found = "found"
    force = "force"
    torque = "torque"
    dist = "dist"
    pos = "pos"
    normal = "normal"
    tangent = "tangent"


class ContactSensorReduce(str, Enum):
    """Reduction modes supported by contact sensors."""

    none = "none"
    mindist = "mindist"
    maxforce = "maxforce"
    netforce = "netforce"


@configclass(kw_only=True)
class ContactSensorCfg(SceneSensorCfg):
    """Contact sensor matching a pair of named geometries."""

    geom1: str
    geom2: str
    num: int = 1
    data: list[ContactReportField] = [ContactReportField.found]
    reduce: ContactSensorReduce = ContactSensorReduce.none

    def validate(self, name: str) -> None:
        super().validate(name)
        if not self.geom1 or not self.geom2:
            raise ValueError("ContactSensorCfg.geom1 and geom2 must not be empty")
        if self.num <= 0:
            raise ValueError(f"ContactSensorCfg.num must be positive, got {self.num}")
        if not self.data:
            raise ValueError("ContactSensorCfg.data must contain at least one report field")
        unsupported_fields = [field for field in self.data if not isinstance(field, ContactReportField)]
        if unsupported_fields:
            raise ValueError(f"ContactSensorCfg.data contains unsupported fields: {unsupported_fields}")


class FrameObjectKind(str, Enum):
    """Kinds of named scene objects supported by frame sensors."""

    site = "site"
    geom = "geom"
    link = "link"
    link_inertia = "link_inertia"


class FrameRefKind(str, Enum):
    """Reference frames supported by frame sensors."""

    local = "local"
    world = "world"
    object = "object"


class FrameSensorType(str, Enum):
    """Frame quantities supported by frame sensors."""

    framepos = "framepos"
    framequat = "framequat"
    framelinvel = "framelinvel"
    frameangvel = "frameangvel"
    framelinacc = "framelinacc"
    xaxis = "xaxis"
    yaxis = "yaxis"
    zaxis = "zaxis"


@configclass(kw_only=True)
class FrameSensorCfg(SceneSensorCfg):
    """Configuration for sampling a frame quantity from a named scene object.

    The target object must already exist in the assembled scene. The sampled value is expressed in
    the reference frame selected by ``ref_kind``.

    Attributes:
        object_type: Kind of target object: ``site``, ``geom``, ``link``, or ``link_inertia``.
        object_name: Name of the target object whose frame is sampled.
        sensor_type: Quantity to sample. ``framepos`` is position; ``framequat`` is orientation;
            ``framelinvel`` and ``frameangvel`` are linear and angular velocity; ``framelinacc`` is
            linear acceleration; and ``xaxis``, ``yaxis``, and ``zaxis`` are the target frame's axis
            directions.
        ref_kind: Reference frame used to express the sampled value. ``world`` uses the world frame,
            ``local`` uses the target object's local frame, and ``object`` uses another named object's
            frame.
        ref_object_type: Kind of reference object. Required only when ``ref_kind`` is ``object``.
        ref_object_name: Name of the reference object. Required only when ``ref_kind`` is ``object``.
    """

    object_type: FrameObjectKind
    object_name: str
    sensor_type: FrameSensorType
    ref_kind: FrameRefKind = FrameRefKind.world
    ref_object_type: FrameObjectKind | None = None
    ref_object_name: str | None = None

    def validate(self, name: str) -> None:
        super().validate(name)
        if not self.object_name:
            raise ValueError("FrameSensorCfg.object_name must not be empty")
        if self.ref_kind is FrameRefKind.object:
            if self.ref_object_type is None or not self.ref_object_name:
                raise ValueError(
                    "FrameSensorCfg.ref_kind='object' requires ref_object_type and a non-empty ref_object_name"
                )
        elif self.ref_object_type is not None or self.ref_object_name is not None:
            raise ValueError(
                f"FrameSensorCfg.ref_kind={self.ref_kind!r} takes no reference object; "
                "leave ref_object_type and ref_object_name unset"
            )
