"""Additional active legacy Kudada rewards, vectorized over environments."""

import numpy as np

from kudada_motrix.constants import DEFAULT
from kudada_motrix.contract import rotate_inverse

EXTRA_REWARD_SCALES = {
    "track_vel_hard": .05, "feet_air_time": 1., "foot_clearance": .5,
    "foot_normal_reward": .1, "feet_stumble": -.2, "swing_foot_orient": .1,
    "base_acceleration": -1e-4, "default_joint_pos": .15, "dof_vel": -1e-7,
    "dof_acc": -1e-7, "joint_power": -2e-5, "feet_distance": .2, "low_speedx": .01,
}


def extra_reward_terms(env, phase):
    data = env.sim_data
    force = data["contacts"][:, 1:3, :3]
    contact = force[:, :, 2] > 5.
    filtered = contact | env.last_contacts
    first_contact = (env.feet_air_time > 0.) & filtered
    env.last_contacts[:] = contact
    env.feet_air_time += env.cfg.ctrl_dt
    air_time = np.sum(np.maximum(env.feet_air_time-.15, 0.)*first_contact, axis=1)
    air_time *= np.linalg.norm(env.commands[:, :2], axis=1) > .1
    env.feet_air_time *= ~filtered

    quat = np.repeat(data["base_quat"], 2, axis=0)
    feet_pos = rotate_inverse(quat, (data["feet_pos"]-data["base_pos"][:, None]).reshape(-1, 3)).reshape(-1, 2, 3)
    feet_vel = rotate_inverse(quat, (data["feet_vel"]-data["world_lin_vel"][:, None]).reshape(-1, 3)).reshape(-1, 2, 3)
    no_contact = force[:, :, 2] < 5.
    clearance = np.maximum(feet_pos[:, :, 2]+.22, 0.)*np.linalg.norm(feet_vel[:, :, :2], axis=2)*no_contact
    magnitude = np.linalg.norm(force, axis=2)+1e-6
    mask = magnitude > 5.
    normal = np.sum((force[:, :, 2]/magnitude)**2*mask, axis=1)/np.maximum(np.sum(mask, axis=1), 1)
    swing = np.stack((phase >= .6, (phase+.5) % 1. >= .6), axis=1)
    ankle = data["q"][:, [4, 9]]
    diff = data["q"]-DEFAULT
    yaw_roll = np.maximum(np.linalg.norm(diff[:, :2], axis=1)+np.linalg.norm(diff[:, 5:7], axis=1)-.1, 0.)
    distance = np.linalg.norm(data["feet_pos"][:, 0, :2]-data["feet_pos"][:, 1, :2], axis=1)
    dmin, dmax = np.clip(distance-.13, -.5, 0.), np.clip(distance-.5, 0., .5)
    elin = np.linalg.norm(env.commands[:, :2]-env.base_lin_vel[:, :2], axis=1)
    eang = np.abs(env.commands[:, 2]-env.base_ang_vel[:, 2])
    speed, command = np.abs(env.base_lin_vel[:, 0]), np.abs(env.commands[:, 0])
    low, high = speed < .5*command, speed > 1.2*command
    low_speed = np.where(low, -1., np.where(high, 0., 2.))
    low_speed[np.sign(env.base_lin_vel[:, 0]) != np.sign(env.commands[:, 0])] = -2.
    return {
        "track_vel_hard": (np.exp(-10*elin)+np.exp(-10*eang))/2.-.2*(elin+eang),
        "feet_air_time": air_time, "foot_clearance": np.sum(clearance, axis=1),
        "foot_normal_reward": normal,
        "feet_stumble": np.any(np.linalg.norm(force[:, :, :2], axis=2) > 3*np.abs(force[:, :, 2]), axis=1),
        "swing_foot_orient": np.sum(np.exp(-20*ankle**2)*swing*no_contact, axis=1)/2.,
        "base_acceleration": np.sum(env.base_lin_acc**2, axis=1),
        "default_joint_pos": np.exp(-np.minimum(yaw_roll, 50.)*100),
        "dof_vel": np.sum(data["qvel"]**2, axis=1),
        "dof_acc": np.sum(((env.last_dof_vel-data["qvel"])/env.cfg.ctrl_dt)**2, axis=1),
        "joint_power": np.sum(np.abs(data["qvel"])*np.abs(env.torques), axis=1),
        "feet_distance": (np.exp(-np.abs(dmin)*100)+np.exp(-np.abs(dmax)*100))/2.,
        "low_speedx": low_speed,
    }
