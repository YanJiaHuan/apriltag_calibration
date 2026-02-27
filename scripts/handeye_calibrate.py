#!/usr/bin/env python3
"""
Hand-eye calibration solver for eye-in-hand using robot + AprilTag data.

This module reads a robot-side CSV that contains robot end pose (end|base)
and AprilTag pose (tag|camera) in the OpenCV camera frame. It filters out
invalid detections, solves for the camera pose relative to the robot end
(camera|end), and estimates the fixed tag pose relative to the robot base
(tag|base).

Assumptions:
- Robot pose is end in base (x,y,z in mm, rx,ry,rz in rad; Euler XYZ).
- Tag pose is tag in camera (x,y,z in mm, rx,ry,rz in rad; Euler XYZ).
- OpenCV camera frame (x right, y down, z forward) is preserved.
- Output translation is mm and rotation is Euler XYZ in rad.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - handled by runtime check
    np = None


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        None.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Hand-eye calibration solver")
    parser.add_argument("--input-csv", default="data/robot_eye_in_hand.csv")
    parser.add_argument("--method", default="tsai")
    parser.add_argument("--max-reproj-error", type=float, default=2.0)
    parser.add_argument("--min-samples", type=int, default=3)
    return parser.parse_args()


def read_csv_rows(path: str) -> Tuple[List[str], List[List[str]]]:
    """
    Read a CSV file into header and rows.

    Args:
        path (str): CSV file path.

    Returns:
        tuple[list[str], list[list[str]]]: Header and rows.
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def expand_header_if_needed(header: List[str], row_len: int) -> List[str]:
    """
    Expand header for rows that include quality columns.

    Args:
        header (list[str]): Base header list.
        row_len (int): Length of a data row.

    Returns:
        list[str]: Expanded header if needed, otherwise base header.
    """
    if "quality_ok" in header:
        return header
    if "tag_found" not in header:
        return header
    if row_len == len(header) + 2:
        idx = header.index("tag_found") + 1
        return header[:idx] + ["quality_ok", "max_reproj_error"] + header[idx:]
    return header


def str_to_bool(value: Any) -> bool:
    """
    Convert a string-like value to bool.

    Args:
        value (Any): Input value.

    Returns:
        bool: True/False.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def euler_xyz_to_rot(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Convert Euler XYZ angles to rotation matrix.

    Args:
        rx (float): Rotation about X in radians.
        ry (float): Rotation about Y in radians.
        rz (float): Rotation about Z in radians.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    if np is None:
        raise RuntimeError("numpy is required for euler_xyz_to_rot")

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])

    return rz_m @ ry_m @ rx_m


def rot_to_euler_xyz(r: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler XYZ (roll, pitch, yaw) in radians.

    Args:
        r (np.ndarray): 3x3 rotation matrix.

    Returns:
        tuple[float, float, float]: (rx, ry, rz) in radians.
    """
    if np is None:
        raise RuntimeError("numpy is required for rot_to_euler_xyz")

    r = np.asarray(r, dtype=float)
    r00 = float(r[0, 0])
    r10 = float(r[1, 0])
    r20 = float(r[2, 0])
    r21 = float(r[2, 1])
    r22 = float(r[2, 2])

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
    Create a 4x4 homogeneous transform from R and t.

    Args:
        r (np.ndarray): 3x3 rotation matrix.
        t (np.ndarray): 3x1 translation vector.

    Returns:
        np.ndarray: 4x4 transform matrix.
    """
    if np is None:
        raise RuntimeError("numpy is required for make_transform")

    t = np.asarray(t, dtype=float).reshape(3, 1)
    tmat = np.eye(4)
    tmat[:3, :3] = r
    tmat[:3, 3] = t[:, 0]
    return tmat


def invert_transform(tmat: np.ndarray) -> np.ndarray:
    """
    Invert a 4x4 homogeneous transform.

    Args:
        tmat (np.ndarray): 4x4 transform matrix.

    Returns:
        np.ndarray: Inverted 4x4 transform.
    """
    if np is None:
        raise RuntimeError("numpy is required for invert_transform")

    tmat = np.asarray(tmat, dtype=float)
    r = tmat[:3, :3]
    t = tmat[:3, 3]
    r_inv = r.T
    t_inv = -r_inv @ t
    out = np.eye(4)
    out[:3, :3] = r_inv
    out[:3, 3] = t_inv
    return out


def rot_to_quat(r: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion (w, x, y, z).

    Args:
        r (np.ndarray): 3x3 rotation matrix.

    Returns:
        np.ndarray: Quaternion [w, x, y, z].
    """
    if np is None:
        raise RuntimeError("numpy is required for rot_to_quat")

    r = np.asarray(r, dtype=float)
    m00, m01, m02 = r[0, 0], r[0, 1], r[0, 2]
    m10, m11, m12 = r[1, 0], r[1, 1], r[1, 2]
    m20, m21, m22 = r[2, 0], r[2, 1], r[2, 2]

    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return np.array([w, x, y, z], dtype=float)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (w, x, y, z) to rotation matrix.

    Args:
        q (np.ndarray): Quaternion [w, x, y, z].

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    if np is None:
        raise RuntimeError("numpy is required for quat_to_rot")

    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def average_rotations(rotations: List[np.ndarray]) -> np.ndarray:
    """
    Average rotation matrices using quaternion mean.

    Args:
        rotations (list[np.ndarray]): List of 3x3 rotation matrices.

    Returns:
        np.ndarray: Average 3x3 rotation matrix.
    """
    if np is None:
        raise RuntimeError("numpy is required for average_rotations")

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


def load_samples(
    csv_path: str,
    max_reproj_error: float,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[str],
]:
    """
    Load valid samples from CSV and build rotation/translation lists.

    Args:
        csv_path (str): Input CSV path.
        max_reproj_error (float): Max allowed reprojection error.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str]]:
            Rotation/translation lists and warnings.
    """
    if np is None:
        return [], [], [], [], ["numpy_missing"]

    header, rows = read_csv_rows(csv_path)
    warnings: List[str] = []
    if not header:
        return [], [], [], [], ["empty_csv"]

    valid_r_gripper2base: List[np.ndarray] = []
    valid_t_gripper2base: List[np.ndarray] = []
    valid_r_target2cam: List[np.ndarray] = []
    valid_t_target2cam: List[np.ndarray] = []

    for idx, row in enumerate(rows, start=2):
        row_header = expand_header_if_needed(header, len(row))
        if len(row) != len(row_header):
            warnings.append(f"row_len_mismatch_line_{idx}")
            continue
        data = dict(zip(row_header, row))

        tag_found = str_to_bool(data.get("tag_found", False))
        quality_ok = data.get("quality_ok", "")
        quality_ok = str_to_bool(quality_ok) if quality_ok != "" else None
        reproj_error = data.get("reproj_error", "")

        if not tag_found:
            continue
        if quality_ok is None:
            try:
                reproj_value = float(reproj_error)
            except Exception:
                continue
            if reproj_value > max_reproj_error:
                continue
        else:
            if not quality_ok:
                continue

        try:
            end_x = float(data["end_x_mm"])
            end_y = float(data["end_y_mm"])
            end_z = float(data["end_z_mm"])
            end_rx = float(data["end_rx_rad"])
            end_ry = float(data["end_ry_rad"])
            end_rz = float(data["end_rz_rad"])

            tag_tx = float(data["tag_t_x_mm"])
            tag_ty = float(data["tag_t_y_mm"])
            tag_tz = float(data["tag_t_z_mm"])
            tag_rx = float(data["tag_r_x_rad"])
            tag_ry = float(data["tag_r_y_rad"])
            tag_rz = float(data["tag_r_z_rad"])
        except Exception:
            warnings.append(f"parse_error_line_{idx}")
            continue

        r_gripper2base = euler_xyz_to_rot(end_rx, end_ry, end_rz)
        t_gripper2base = np.array([end_x, end_y, end_z], dtype=float)

        r_target2cam = euler_xyz_to_rot(tag_rx, tag_ry, tag_rz)
        t_target2cam = np.array([tag_tx, tag_ty, tag_tz], dtype=float)

        valid_r_gripper2base.append(r_gripper2base)
        valid_t_gripper2base.append(t_gripper2base)
        valid_r_target2cam.append(r_target2cam)
        valid_t_target2cam.append(t_target2cam)

    if not valid_r_gripper2base:
        return [], [], [], [], warnings

    return (
        [np.array(r) for r in valid_r_gripper2base],
        [np.array(t) for t in valid_t_gripper2base],
        [np.array(r) for r in valid_r_target2cam],
        [np.array(t) for t in valid_t_target2cam],
        warnings,
    )


def resolve_method(method: str) -> int:
    """
    Resolve a method name to OpenCV hand-eye constant.

    Args:
        method (str): Method name.

    Returns:
        int: OpenCV method constant.
    """
    import cv2

    name = method.strip().lower()
    mapping = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    if name not in mapping:
        raise ValueError(f"unsupported method: {method}")
    return mapping[name]


def solve_handeye(
    r_gripper2base: List[np.ndarray],
    t_gripper2base: List[np.ndarray],
    r_target2cam: List[np.ndarray],
    t_target2cam: List[np.ndarray],
    method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve hand-eye calibration using OpenCV.

    Args:
        r_gripper2base (list[np.ndarray]): Rotation matrices (gripper to base).
        t_gripper2base (list[np.ndarray]): Translation vectors (gripper to base).
        r_target2cam (list[np.ndarray]): Rotation matrices (target to camera).
        t_target2cam (list[np.ndarray]): Translation vectors (target to camera).
        method (str): Hand-eye method name.

    Returns:
        tuple[np.ndarray, np.ndarray]: (R_cam2gripper, t_cam2gripper).
    """
    import cv2

    method_flag = resolve_method(method)
    r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        r_gripper2base,
        t_gripper2base,
        r_target2cam,
        t_target2cam,
        method=method_flag,
    )
    return r_cam2gripper, t_cam2gripper


def estimate_tag_in_base(
    r_base_end_list: List[np.ndarray],
    t_base_end_list: List[np.ndarray],
    r_end_cam: np.ndarray,
    t_end_cam: np.ndarray,
    r_cam_tag_list: List[np.ndarray],
    t_cam_tag_list: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate fixed tag pose in base by averaging per-sample estimates.

    Args:
        r_base_end_list (list[np.ndarray]): Base to end rotations.
        t_base_end_list (list[np.ndarray]): Base to end translations.
        r_end_cam (np.ndarray): End to camera rotation.
        t_end_cam (np.ndarray): End to camera translation.
        r_cam_tag_list (list[np.ndarray]): Camera to tag rotations.
        t_cam_tag_list (list[np.ndarray]): Camera to tag translations.

    Returns:
        tuple[np.ndarray, np.ndarray]: (R_base_tag, t_base_tag).
    """
    rotations = []
    translations = []
    for r_be, t_be, r_ct, t_ct in zip(
        r_base_end_list, t_base_end_list, r_cam_tag_list, t_cam_tag_list
    ):
        t_be = t_be.reshape(3, 1)
        t_ec = t_end_cam.reshape(3, 1)
        t_ct = t_ct.reshape(3, 1)

        t_base_end = make_transform(r_be, t_be)
        t_end_cam = make_transform(r_end_cam, t_ec)
        t_cam_tag = make_transform(r_ct, t_ct)

        t_base_tag = t_base_end @ t_end_cam @ t_cam_tag
        rotations.append(t_base_tag[:3, :3])
        translations.append(t_base_tag[:3, 3])

    r_mean = average_rotations(rotations)
    t_mean = np.mean(np.stack(translations, axis=0), axis=0)
    return r_mean, t_mean


def main() -> int:
    """
    Entry point.

    Args:
        None.

    Returns:
        int: Process exit code.
    """
    args = parse_args()

    if not os.path.exists(args.input_csv):
        print(f"input csv not found: {args.input_csv}")
        return 1

    try:
        import cv2  # noqa: F401
    except Exception as exc:
        print(f"OpenCV not available: {exc}")
        return 1

    r_g2b, t_g2b, r_t2c, t_t2c, warnings = load_samples(
        args.input_csv, args.max_reproj_error
    )
    if warnings:
        for w in warnings:
            print(f"warning: {w}")

    if len(r_g2b) < args.min_samples:
        print(f"not enough valid samples: {len(r_g2b)} < {args.min_samples}")
        return 1

    r_cam2gripper, t_cam2gripper = solve_handeye(
        r_g2b, t_g2b, r_t2c, t_t2c, args.method
    )

    t_cam2gripper = np.array(t_cam2gripper, dtype=float).reshape(3, 1)
    t_gripper_cam = invert_transform(make_transform(r_cam2gripper, t_cam2gripper))

    r_end_cam = t_gripper_cam[:3, :3]
    t_end_cam = t_gripper_cam[:3, 3]

    r_base_tag, t_base_tag = estimate_tag_in_base(
        r_g2b, t_g2b, r_end_cam, t_end_cam, r_t2c, t_t2c
    )

    cam_rx, cam_ry, cam_rz = rot_to_euler_xyz(r_end_cam)
    tag_rx, tag_ry, tag_rz = rot_to_euler_xyz(r_base_tag)

    print("valid_samples:", len(r_g2b))
    print("camera|end (T_end_cam):")
    print("  t_mm:", [float(t_end_cam[0]), float(t_end_cam[1]), float(t_end_cam[2])])
    print("  rpy_rad:", [cam_rx, cam_ry, cam_rz])
    print("apriltag|base (T_base_tag):")
    print("  t_mm:", [float(t_base_tag[0]), float(t_base_tag[1]), float(t_base_tag[2])])
    print("  rpy_rad:", [tag_rx, tag_ry, tag_rz])

    return 0


if __name__ == "__main__":
    sys.exit(main())
