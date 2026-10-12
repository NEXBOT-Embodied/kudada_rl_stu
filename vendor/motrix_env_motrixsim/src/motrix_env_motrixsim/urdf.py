# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from typing import Any

import motrixsim as mtx

from motrix_env_core.config.scene.urdf import (
    ActuatorCfg,
    JointCfg,
    PositionActuatorCfg,
    SiteCfg,
    UrdfFileCfg,
    UrdfGeomCfg,
)


def _iter_links(link: Any) -> Iterator[Any]:
    yield link
    for child in link.children:
        yield from _iter_links(child)


def _apply_joint_cfg(cfg: JointCfg, joint: Any) -> None:
    if cfg.armature is not None:
        joint.armature = cfg.armature
    if cfg.friction_loss is not None:
        joint.friction_loss = cfg.friction_loss


def _apply_geom_cfg(cfg: UrdfGeomCfg, geom: Any) -> None:
    if cfg.friction is not None:
        geom.physics_material.friction = list(cfg.friction)
    if cfg.condim is not None:
        geom.physics_material.condim = cfg.condim
    if cfg.priority is not None:
        geom.priority = cfg.priority


def _build_site(cfg: SiteCfg) -> mtx.msd.Site:
    site = mtx.msd.Site()
    site.name = cfg.name
    site.position = cfg.position
    site.orientation = cfg.rotation
    return site


def _build_actuator(cfg: ActuatorCfg) -> Any:
    actuator = mtx.msd.Actuator()
    actuator.name = cfg.actuator_name
    actuator.target = mtx.msd.ActuatorTarget.joint(cfg.joint_name)
    if isinstance(cfg, PositionActuatorCfg):
        param = mtx.msd.PositionParam()
        param.kp = cfg.kp
        param.damping_value = cfg.kv
        param.damping_type = mtx.msd.DampingType.Kv
        actuator.actuator_type = mtx.msd.ActuatorType.position(param)
    else:
        raise TypeError(f"MotrixSim does not support actuator config {type(cfg).__name__}")
    if cfg.ctrl_range is not None:
        actuator.ctrlrange = mtx.msd.Range(*cfg.ctrl_range)
    if cfg.force_range is not None:
        actuator.forcerange = mtx.msd.Range(*cfg.force_range)
    return actuator


def apply_urdf_cfgs(world: mtx.msd.World, cfg: UrdfFileCfg) -> None:
    """Apply declarative URDF augmentations to a MotrixSim MSD world."""
    cfg_by_geom: dict[str, UrdfGeomCfg] = {}
    for geom_cfg in cfg.geoms:
        geom_cfg.validate()
        if geom_cfg.geom_name in cfg_by_geom:
            raise ValueError(f"Multiple geom configs target geom {geom_cfg.geom_name!r}")
        cfg_by_geom[geom_cfg.geom_name] = geom_cfg

    cfg_by_site: dict[str, SiteCfg] = {}
    for site_cfg in cfg.sites:
        site_cfg.validate()
        if site_cfg.name in cfg_by_site:
            raise ValueError(f"Duplicate configured site name {site_cfg.name!r}")
        cfg_by_site[site_cfg.name] = site_cfg

    cfg_by_joint: dict[str, JointCfg] = {}
    for joint_cfg in cfg.joints:
        joint_cfg.validate()
        if joint_cfg.joint_name in cfg_by_joint:
            raise ValueError(f"Multiple joint configs target joint {joint_cfg.joint_name!r}")
        cfg_by_joint[joint_cfg.joint_name] = joint_cfg

    target_joints: set[str] = set()
    actuator_names: set[str] = set()
    for actuator_cfg in cfg.actuators:
        actuator_cfg.validate()
        if actuator_cfg.joint_name in target_joints:
            raise ValueError(f"Multiple actuator configs target joint {actuator_cfg.joint_name!r}")
        if actuator_cfg.actuator_name in actuator_names:
            raise ValueError(f"Duplicate configured actuator name {actuator_cfg.actuator_name!r}")
        target_joints.add(actuator_cfg.joint_name)
        actuator_names.add(actuator_cfg.actuator_name)

    configured_geoms: set[str] = set()
    configured_joints: set[str] = set()
    links_by_name: dict[str, Any] = {}
    joints_by_name: dict[str, Any] = {}
    existing_site_names = {site.name for site in world.hierarchy.sites if site.name is not None}

    def apply_geom(geom: Any) -> None:
        if geom.name not in cfg_by_geom:
            return
        _apply_geom_cfg(cfg_by_geom[geom.name], geom)
        configured_geoms.add(geom.name)

    for geom in world.hierarchy.geoms:
        apply_geom(geom)

    for body in world.hierarchy.bodies:
        for link in _iter_links(body.link):
            if link.name is not None:
                links_by_name[link.name] = link
            existing_site_names.update(site.name for site in link.sites if site.name is not None)
            for geom in link.geoms:
                apply_geom(geom)
            for joint in link.joints:
                if joint.name is not None:
                    joints_by_name[joint.name] = joint
                if joint.name not in cfg_by_joint:
                    continue
                _apply_joint_cfg(cfg_by_joint[joint.name], joint)
                configured_joints.add(joint.name)

    unknown_geoms = sorted(set(cfg_by_geom).difference(configured_geoms))
    if unknown_geoms:
        raise ValueError(f"Configured geoms do not exist in model: {unknown_geoms}")

    unknown_joints = sorted(set(cfg_by_joint).difference(configured_joints))
    if unknown_joints:
        raise ValueError(f"Configured joints do not exist in model: {unknown_joints}")

    duplicate_site_names = sorted(existing_site_names.intersection(cfg_by_site))
    if duplicate_site_names:
        raise ValueError(f"Configured site names already exist in model: {duplicate_site_names}")

    parent_link_names = {site_cfg.parent_link_name for site_cfg in cfg.sites}
    unknown_site_links = sorted(parent_link_names.difference(links_by_name))
    if unknown_site_links:
        raise ValueError(f"Configured site parent links do not exist in model: {unknown_site_links}")

    unknown_actuator_joints = sorted(target_joints.difference(joints_by_name))
    if unknown_actuator_joints:
        raise ValueError(f"Configured actuator joints do not exist in model: {unknown_actuator_joints}")

    existing_names = {actuator.name for actuator in world.actuators if actuator.name is not None}
    duplicate_names = sorted(existing_names.intersection(actuator_names))
    if duplicate_names:
        raise ValueError(f"Configured actuator names already exist in model: {duplicate_names}")

    existing_joint_targets = {
        actuator.target.value for actuator in world.actuators if actuator.target.variant == "joint"
    }
    duplicate_targets = sorted(existing_joint_targets.intersection(target_joints))
    if duplicate_targets:
        raise ValueError(f"Configured joints already have actuators in model: {duplicate_targets}")

    for site_cfg in cfg.sites:
        links_by_name[site_cfg.parent_link_name].sites.append(_build_site(site_cfg))

    for actuator_cfg in cfg.actuators:
        actuator = _build_actuator(actuator_cfg)
        if isinstance(actuator_cfg, PositionActuatorCfg) and actuator_cfg.inherit_joint_range:
            actuator.ctrlrange = joints_by_name[actuator_cfg.joint_name].pos_limit
        world.actuators.append(actuator)
