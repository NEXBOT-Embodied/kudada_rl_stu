# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from . import utils


def mul(q1, q2):
    """
    Multiply two quaternions.

    Quaternion format: (x, y, z, w)

    Args:
        q1: First quaternion(s). Shape: (..., 4)
        q2: Second quaternion(s). Shape: (..., 4)

    Returns:
        Product quaternion(s). Shape: (..., 4)
    """
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    # Standard quaternion multiplication formula
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.stack([x, y, z, w], axis=-1)


def conjugate(q):
    """
    Compute the conjugate of a quaternion.

    For q = (x, y, z, w), conjugate is (-x, -y, -z, w)

    Args:
        q: Input quaternion(s). Shape: (..., 4)

    Returns:
        Conjugate quaternion(s). Shape: (..., 4)
    """
    return q * np.array([-1, -1, -1, 1], dtype=q.dtype)


def inverse(q):
    """
    Compute the inverse of a quaternion.

    For unit quaternions, this is equal to the conjugate. For non-unit
    quaternions, divide the conjugate by the squared norm.

    Args:
        q: Input quaternion(s). Shape: (..., 4)

    Returns:
        Inverse quaternion(s). Shape: (..., 4)
    """
    q = np.asarray(q)
    norm_sq = np.sum(np.square(q), axis=-1, keepdims=True)
    return conjugate(q) / np.maximum(norm_sq, 1e-12)


def from_euler(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray):
    """
    Euler convert to quaternion, with [x, y, z, w] format
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.stack([qx, qy, qz, qw], dtype=np.float32, axis=-1)


def from_angle_axis(angle, axis):
    """
    Create quaternion from angle-axis representation.

    Args:
        angle: Rotation angle in radians. Shape: (batch,) or scalar
        axis: Rotation axis (will be normalized). Shape: (batch, 3) or (3,)

    Returns:
        Quaternion in (x, y, z, w) format. Shape: (batch, 4) or (4,)
    """
    # Ensure angle has proper shape for broadcasting
    if np.isscalar(angle):
        angle = np.array([angle])

    # Normalize axis
    axis = utils.normalize(axis)

    # Compute half angle
    half_angle = angle / 2.0

    # Compute quaternion components
    sin_half = np.sin(half_angle)
    cos_half = np.cos(half_angle)

    # Handle broadcasting
    if axis.ndim == 1:
        # Single axis
        w = cos_half
        xyz = axis * sin_half[..., np.newaxis]
    else:
        # Multiple axes
        w = cos_half
        xyz = axis * sin_half[..., np.newaxis]

    return utils.normalize(np.concatenate([xyz, w[..., np.newaxis]], axis=-1))


def rotate_vector(quats: np.ndarray, v: np.ndarray):
    """Rotate vectors ``v`` by quaternions ``quats`` (``[x, y, z, w]`` format).

    Both inputs broadcast on their leading dims:

    - ``quats``: shape ``(..., 4)``.
    - ``v``:     shape ``(..., 3)`` or ``(3,)``.

    Returns shape ``broadcast(quats.batch, v.batch) + (3,)``. So a single
    quaternion against a batch of vectors (or vice versa), and any
    higher-rank combination, all work without manual flatten/reshape.
    """
    batch = np.broadcast_shapes(quats.shape[:-1], v.shape[:-1])
    q = np.broadcast_to(quats, batch + (4,))
    vec = np.broadcast_to(v, batch + (3,))

    w = q[..., -1]
    im = q[..., :3]

    t = 2.0 * np.cross(im, vec)
    return vec + w[..., None] * t + np.cross(im, t)


def rotate_inverse(quats, v):
    """Inverse-rotate vectors ``v`` by quaternions ``quats``.

    Same broadcasting contract as :func:`rotate_vector`.
    """
    batch = np.broadcast_shapes(quats.shape[:-1], v.shape[:-1])
    q = np.broadcast_to(quats, batch + (4,))
    vec = np.broadcast_to(v, batch + (3,))

    w = q[..., -1]
    im = q[..., :3]

    # v' = v + 2 * r × (r × v - w * v)
    term = np.cross(im, vec) - w[..., None] * vec
    return vec + 2.0 * np.cross(im, term)


def similarity(q_current, q_target):
    """
    Use NumPy to compute attitude alignment reward between two batches of quaternions.

    Parameters:
        q_current (np.ndarray): Quaternion of current pose, shape (num_envs, 4).
        q_target (np.ndarray): Quaternion of target pose, shape (num_envs, 4) or (4,).
                                If (4,), it will be broadcast to all environments.

    Returns:
        np.ndarray: Reward value for each environment, shape (num_envs,). Reward value range is [-1, 1].
    """
    # Ensure input is float array
    q_current = q_current.astype(np.float32)
    q_target = q_target.astype(np.float32)

    # If q_target is a single quaternion, broadcast to all environments
    if q_target.ndim == 1:
        # Use np.tile for broadcasting
        q_target = np.tile(q_target, (q_current.shape[0], 1))

    # Step 1: Compute conjugate of q_current
    # Conjugate of quaternion (x, y, z, w) is (-x, -y, -z, w)
    q_current_conj = np.copy(q_current)
    q_current_conj[..., :3] *= -1  # Negate x, y, z components

    # Step 2: Compute relative quaternion q_rel = q_target * q_current_conj
    # Unpack quaternion components for computation
    x1, y1, z1, w1 = q_target[..., 0], q_target[..., 1], q_target[..., 2], q_target[..., 3]
    x2, y2, z2, w2 = q_current_conj[..., 0], q_current_conj[..., 1], q_current_conj[..., 2], q_current_conj[..., 3]

    # Apply quaternion multiplication formula
    w_rel = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    # For numerical stability, clamp w_rel to [-1.0, 1.0] range
    w_rel_clamped = np.clip(w_rel, -1.0, 1.0)

    # Step 3: Compute rotation angle theta
    theta = 2.0 * np.arccos(w_rel_clamped)

    # Step 4: Compute reward
    reward = np.cos(theta)

    return reward


def rotation_distance(q1, q2):
    """
    Compute the rotation distance between two quaternions in radians.

    Args:
        q1: First quaternion(s). Shape: (..., 4)
        q2: Second quaternion(s). Shape: (..., 4)

    Returns:
        Rotation distance in radians. Shape: (...)
    """
    quat_diff = mul(q1, conjugate(q2))
    # Extract imaginary part (x, y, z) and compute norm
    imaginary_norm = np.linalg.norm(quat_diff[..., :3], axis=-1)
    # Clamp to valid range for arcsin
    imaginary_norm = np.clip(imaginary_norm, 0.0, 1.0)
    return 2.0 * np.arcsin(imaginary_norm)


def get_euler_xyz(q: np.ndarray):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        q: Quaternion(s) in (x, y, z, w) format. Shape: (..., 4)

    Returns:
        Tuple of (roll, pitch, yaw) in radians. Each has shape: (...)
    """
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.copysign(np.pi / 2.0, sinp), np.arcsin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def get_yaw(quat: np.ndarray) -> np.ndarray:
    _, _, yaw = get_euler_xyz(quat)
    return yaw


def from_yaw(yaw: np.ndarray) -> np.ndarray:
    """Build a yaw-only (z-axis) quaternion from a yaw angle.

    Args:
        yaw: yaw angle(s) in radians, any shape (including scalar).

    Returns:
        Quaternion(s) of shape ``yaw.shape + (4,)`` in ``[x, y, z, w]`` format,
        with x=y=0 and (z, w) = (sin(yaw/2), cos(yaw/2)). Always unit-norm.
    """
    yaw = np.asarray(yaw, dtype=np.float32)
    half = yaw * 0.5
    out = np.zeros(yaw.shape + (4,), dtype=np.float32)
    out[..., 2] = np.sin(half)
    out[..., 3] = np.cos(half)
    return out


def to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) to row-major 3x3 rotation matrix(ices).

    Uses the Shepperd-style closed form with ``2/|q|^2`` normalization, so
    non-unit inputs are handled too. Output shape: ``q.shape[:-1] + (3, 3)``.
    ``R @ v`` rotates a column vector ``v`` by ``q``.
    """
    q = np.asarray(q)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    two_s = 2.0 / np.sum(q * q, axis=-1)
    return np.stack(
        (
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def generate_random_shoemake(size):
    """
    Generate uniformly distributed random quaternions using Shoemake's method.

    This ensures uniform distribution over SO(3) rotation space.

    Args:
        size: Number of quaternions to generate (int or tuple)

    Returns:
        Random quaternions in (x, y, z, w) format. Shape: (size, 4) or (*size, 4)

    Reference:
        K. Shoemake, "Uniform Random Rotations", Graphics Gems III, 1992
    """
    # Convert size to tuple if it's a scalar (handles numpy int64)
    if np.isscalar(size):
        size = (int(size),)
    elif not isinstance(size, tuple):
        size = tuple(size)

    # Generate three uniform random numbers
    u1, u2, u3 = np.random.uniform(0, 1, size=(3, *size)).astype(np.float32)

    # Shoemake's method
    sqrt1_u1 = np.sqrt(1 - u1)
    sqrtu1 = np.sqrt(u1)

    w = sqrt1_u1 * np.sin(2 * np.pi * u2)
    x = sqrt1_u1 * np.cos(2 * np.pi * u2)
    y = sqrtu1 * np.sin(2 * np.pi * u3)
    z = sqrtu1 * np.cos(2 * np.pi * u3)

    return np.stack([x, y, z, w], axis=-1)
