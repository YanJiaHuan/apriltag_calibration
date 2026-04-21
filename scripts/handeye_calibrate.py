#!/usr/bin/env python3
"""
Hand-eye calibration solver for eye-in-hand using robot + AprilTag data.

This module reads a robot-side CSV containing robot end pose (end|base)
and AprilTag pose (tag|camera) in the OpenCV camera frame. It filters out
invalid detections, converts the OpenCV camera frame to the robot camera
frame, solves for the camera pose relative to the robot end (camera|end),
and estimates the fixed tag pose relative to the robot base (tag|base).

Assumptions:
- Robot pose is end in base (x,y,z in mm, rx,ry,rz in rad; Euler XYZ).
- Tag pose is tag in camera (x,y,z in mm, rx,ry,rz in rad; Euler XYZ).
- OpenCV camera frame (x right, y down, z out) is converted to
  robot frame (x right, y out, z up) before solving.
- Output translation is mm and rotation is Euler XYZ in rad.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from scripts.handeye_utils import (
    average_rotations,
    convert_opencv_pose_to_robot,
    euler_xyz_to_rot,
    invert_transform,
    make_transform,
    opencv_to_robot_rotation,
    quat_to_rot,
    rot_to_euler_xyz,
    rot_to_quat,
)


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
        r_target2cam, t_target2cam = convert_opencv_pose_to_robot(
            r_target2cam, t_target2cam
        )

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
    r_target2cam_list: List[np.ndarray],
    t_target2cam_list: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate fixed tag pose in base by averaging per-sample estimates.

    Args:
        r_base_end_list (list[np.ndarray]): Base to end rotations.
        t_base_end_list (list[np.ndarray]): Base to end translations.
        r_end_cam (np.ndarray): End to camera rotation.
        t_end_cam (np.ndarray): End to camera translation.
        r_target2cam_list (list[np.ndarray]): Target to camera rotations (cam <- target).
        t_target2cam_list (list[np.ndarray]): Target to camera translations (cam <- target).

    Returns:
        tuple[np.ndarray, np.ndarray]: (R_base_tag, t_base_tag).
    """
    rotations = []
    translations = []

    t_ec = np.asarray(t_end_cam, dtype=float).reshape(3, 1)
    t_end_cam_mat = make_transform(r_end_cam, t_ec)

    for r_be, t_be, r_t2c, t_t2c in zip(
        r_base_end_list, t_base_end_list, r_target2cam_list, t_target2cam_list
    ):
        t_be = np.asarray(t_be, dtype=float).reshape(3, 1)
        t_t2c = np.asarray(t_t2c, dtype=float).reshape(3, 1)

        t_base_end = make_transform(r_be, t_be)
        t_cam_tag = make_transform(r_t2c, t_t2c)

        t_base_tag = t_base_end @ t_end_cam_mat @ t_cam_tag
        rotations.append(t_base_tag[:3, :3])
        translations.append(t_base_tag[:3, 3])

    r_mean = average_rotations(rotations)
    t_mean = np.mean(np.stack(translations, axis=0), axis=0)
    return r_mean, t_mean


def solve_eye_in_hand(
    csv_path: str,
    method: str = "tsai",
    max_reproj_error: float = 2.0,
    min_samples: int = 3,
) -> dict:
    """
    Load CSV data and solve eye-in-hand hand-eye calibration.

    Args:
        csv_path (str): Path to robot_eye_in_hand.csv.
        method (str): Calibration method: tsai, park, horaud, andreff, daniilidis.
        max_reproj_error (float): Max reprojection error filter (pixels).
        min_samples (int): Minimum valid samples required.

    Returns:
        dict: {
            "r_cam2end": np.ndarray (3x3),   camera rotation in end frame
            "t_cam2end": np.ndarray (3,) mm, camera translation in end frame
            "r_tag2base": np.ndarray (3x3),  tag rotation in base frame
            "t_tag2base": np.ndarray (3,) mm,tag translation in base frame
            "n_samples": int,
        }

    Raises:
        ValueError: If CSV not found or not enough valid samples.
    """
    if not os.path.exists(csv_path):
        raise ValueError(f"csv not found: {csv_path}")

    r_g2b, t_g2b, r_t2c, t_t2c, warnings = load_samples(csv_path, max_reproj_error)
    for w in warnings:
        print(f"warning: {w}")

    if len(r_g2b) < min_samples:
        raise ValueError(f"not enough valid samples: {len(r_g2b)} < {min_samples}")

    r_cam2gripper, t_cam2gripper = solve_handeye(r_g2b, t_g2b, r_t2c, t_t2c, method)
    t_cam2gripper = np.array(t_cam2gripper, dtype=float).reshape(3)

    r_base_tag, t_base_tag = estimate_tag_in_base(
        r_g2b, t_g2b, r_cam2gripper, t_cam2gripper, r_t2c, t_t2c
    )

    return {
        "r_cam2end": np.array(r_cam2gripper),
        "t_cam2end": t_cam2gripper,
        "r_tag2base": np.array(r_base_tag),
        "t_tag2base": np.array(t_base_tag),
        "n_samples": len(r_g2b),
    }


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

    # For eye-in-hand, calibrateHandEye returns T_gripper<-cam (camera in gripper frame)
    # which is exactly what we need - no inversion required
    t_cam2gripper = np.array(t_cam2gripper, dtype=float).reshape(3, 1)
    r_end_cam = r_cam2gripper
    t_end_cam = t_cam2gripper.reshape(3)

    r_base_tag, t_base_tag = estimate_tag_in_base(
        r_g2b, t_g2b, r_end_cam, t_end_cam, r_t2c, t_t2c
    )

    cam_rx, cam_ry, cam_rz = rot_to_euler_xyz(r_end_cam)
    tag_rx, tag_ry, tag_rz = rot_to_euler_xyz(r_base_tag)

    print("valid_samples:", len(r_g2b))
    r_map = opencv_to_robot_rotation()
    print("opencv_to_robot_det:", float(np.linalg.det(r_map)))
    print("camera|end (T_end_cam):")
    print("  t_mm:", [float(t_end_cam[0]), float(t_end_cam[1]), float(t_end_cam[2])])
    print("  rpy_rad:", [cam_rx, cam_ry, cam_rz])
    print("apriltag|base (T_base_tag):")
    print("  t_mm:", [float(t_base_tag[0]), float(t_base_tag[1]), float(t_base_tag[2])])
    print("  rpy_rad:", [tag_rx, tag_ry, tag_rz])

    return 0


if __name__ == "__main__":
    sys.exit(main())
