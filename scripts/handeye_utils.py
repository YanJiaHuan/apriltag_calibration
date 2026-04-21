#!/usr/bin/env python3
"""
Shared math utilities for hand-eye calibration.

Provides coordinate conversion, rotation, and homogeneous transform helpers
used by both eye-in-hand and eye-to-hand calibration pipelines.

Assumptions:
- Translation units are millimeters (mm).
- Rotation units are radians (rad), Euler XYZ convention (R = Rz @ Ry @ Rx).
- OpenCV camera frame: X+ right, Y+ down, Z+ forward.
- RealMan robot frame: X+ right, Y+ forward (out), Z+ up.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def euler_xyz_to_rot(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Convert Euler XYZ angles to rotation matrix (R = Rz @ Ry @ Rx).

    Args:
        rx (float): Rotation about X in radians.
        ry (float): Rotation about Y in radians.
        rz (float): Rotation about Z in radians.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz_m @ ry_m @ rx_m


def rot_to_euler_xyz(r: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler XYZ angles in radians.

    Args:
        r (np.ndarray): 3x3 rotation matrix (list or numpy array).

    Returns:
        tuple[float, float, float]: (rx, ry, rz) in radians.
    """
    r = np.asarray(r, dtype=float)
    r00, r10, r20 = float(r[0, 0]), float(r[1, 0]), float(r[2, 0])
    r21, r22 = float(r[2, 1]), float(r[2, 2])
    sy = math.sqrt(r00 * r00 + r10 * r10)
    if sy < 1e-9:
        rx = math.atan2(-float(r[1, 2]), float(r[1, 1]))
        ry = math.atan2(-r20, sy)
        rz = 0.0
    else:
        rx = math.atan2(r21, r22)
        ry = math.atan2(-r20, sy)
        rz = math.atan2(r10, r00)
    return rx, ry, rz


def make_transform(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Build a 4x4 homogeneous transform from rotation matrix and translation.

    Args:
        r (np.ndarray): 3x3 rotation matrix.
        t (np.ndarray): Translation vector (3,) or (3,1) in mm.

    Returns:
        np.ndarray: 4x4 homogeneous transform matrix.
    """
    t = np.asarray(t, dtype=float).reshape(3, 1)
    tmat = np.eye(4)
    tmat[:3, :3] = np.asarray(r, dtype=float)
    tmat[:3, 3] = t[:, 0]
    return tmat


def invert_transform(tmat: np.ndarray) -> np.ndarray:
    """
    Invert a 4x4 homogeneous transform using R^T (exact, no numerical inverse).

    Args:
        tmat (np.ndarray): 4x4 homogeneous transform.

    Returns:
        np.ndarray: 4x4 inverted transform.
    """
    tmat = np.asarray(tmat, dtype=float)
    r = tmat[:3, :3]
    t = tmat[:3, 3]
    r_inv = r.T
    out = np.eye(4)
    out[:3, :3] = r_inv
    out[:3, 3] = -r_inv @ t
    return out


def opencv_to_robot_rotation() -> np.ndarray:
    """
    Return the rotation matrix that maps OpenCV camera axes to RealMan robot axes.

    OpenCV: X+ right, Y+ down, Z+ forward.
    Robot:  X+ right, Y+ forward (out), Z+ up.
    Mapping: X_robot=X_opencv, Y_robot=Z_opencv, Z_robot=-Y_opencv.
    det=1 (pure rotation, not a reflection).

    Args:
        None.

    Returns:
        np.ndarray: 3x3 rotation matrix R_map such that v_robot = R_map @ v_opencv.
    """
    return np.array(
        [[1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0],
         [0.0, -1.0, 0.0]],
        dtype=float,
    )


def convert_opencv_pose_to_robot(
    r_target2cam: np.ndarray, t_target2cam: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a target-in-camera pose from OpenCV axes to robot camera axes.

    Args:
        r_target2cam (np.ndarray): 3x3 rotation matrix in OpenCV camera axes.
        t_target2cam (np.ndarray): Translation (3,) in OpenCV camera axes (mm).

    Returns:
        tuple[np.ndarray, np.ndarray]: (R, t) expressed in robot camera axes.
    """
    r_map = opencv_to_robot_rotation()
    r_out = r_map @ np.asarray(r_target2cam, dtype=float)
    t_out = r_map @ np.asarray(t_target2cam, dtype=float).reshape(3)
    return r_out, t_out


def rot_to_quat(r: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion [w, x, y, z].

    Args:
        r (np.ndarray): 3x3 rotation matrix.

    Returns:
        np.ndarray: Quaternion [w, x, y, z].
    """
    r = np.asarray(r, dtype=float)
    m00, m01, m02 = r[0, 0], r[0, 1], r[0, 2]
    m10, m11, m12 = r[1, 0], r[1, 1], r[1, 2]
    m20, m21, m22 = r[2, 0], r[2, 1], r[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w, x = 0.25 * s, (m21 - m12) / s
        y, z = (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w, x = (m21 - m12) / s, 0.25 * s
        y, z = (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w, x = (m02 - m20) / s, (m01 + m10) / s
        y, z = 0.25 * s, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w, x = (m10 - m01) / s, (m02 + m20) / s
        y, z = (m12 + m21) / s, 0.25 * s
    return np.array([w, x, y, z], dtype=float)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to rotation matrix.

    Args:
        q (np.ndarray): Quaternion [w, x, y, z].

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ],
        dtype=float,
    )


def average_rotations(rotations: List[np.ndarray]) -> np.ndarray:
    """
    Compute mean rotation matrix via quaternion averaging.

    Args:
        rotations (list[np.ndarray]): List of 3x3 rotation matrices.

    Returns:
        np.ndarray: Averaged 3x3 rotation matrix.
    """
    if not rotations:
        raise ValueError("no rotations to average")
    quats = []
    ref = rot_to_quat(rotations[0])
    for r in rotations:
        q = rot_to_quat(r)
        if np.dot(q, ref) < 0.0:
            q = -q
        quats.append(q)
    mean_q = np.mean(np.stack(quats, axis=0), axis=0)
    return quat_to_rot(mean_q)
