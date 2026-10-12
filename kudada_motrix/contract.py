"""Vectorized legacy observations, action decoding and failure conditions."""

import numpy as np

from kudada_motrix.constants import ACTION_SCALE, CLIP, COMMAND_SCALE, DEFAULT


def rotate_inverse(quat_xyzw, vectors):
    """Rotate world vectors into the body frame using unit xyzw quaternions."""
    xyz = quat_xyzw[:, :3]
    uv = np.cross(xyz, vectors)
    return vectors - 2 * quat_xyzw[:, 3:4] * uv + 2 * np.cross(xyz, uv)


def decode_actions(actions):
    clipped = np.clip(actions, -CLIP, CLIP).astype(np.float32)
    return clipped, DEFAULT + ACTION_SCALE * clipped


def observations(*, q, qvel, commands, phase, actions, base_lin_vel, base_ang_vel,
                 projected_gravity, base_height, foot_forces, terrain_heights, disturbance, actor_noise=None):
    moving = (np.linalg.norm(commands, axis=1) > .1)[:, None]
    angle = (2 * np.pi * phase).astype(np.float32)
    gait = np.stack((np.sin(angle), np.cos(angle)), axis=-1) * moving
    actor = np.concatenate((
        np.zeros_like(base_ang_vel), commands * COMMAND_SCALE, gait,
        base_ang_vel * .25, projected_gravity, q - DEFAULT, qvel * .05, actions,
    ), axis=-1)
    if actor_noise is not None:
        actor += actor_noise
    heights = np.clip(base_height[:, None] - .5 - terrain_heights, -1., 1.) * 5.
    critic = np.concatenate((actor, base_lin_vel * 2., disturbance, heights,
                             (foot_forces.reshape(-1, 6) - 100.) * .01), axis=-1)
    return np.clip(actor, -CLIP, CLIP).astype(np.float32), np.clip(critic, -CLIP, CLIP).astype(np.float32)


def termination_reasons(base_height, projected_gravity, base_contact_force):
    return {
        "height": base_height < .15,
        "tilt": np.linalg.norm(projected_gravity[:, :2], axis=1) > .8,
        "base_contact": np.linalg.norm(base_contact_force, axis=1) > 1.,
    }
