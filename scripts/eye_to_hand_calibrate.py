#!/usr/bin/env python3
"""
Eye-to-hand calibration solver.

Reads the same CSV format as eye-in-hand (robot end pose + AprilTag detection)
and solves the AX=ZB robot-world hand-eye problem to find simultaneously:
  - T[cam|base]: camera pose relative to robot base (primary result)
  - T[tag|end]:  tag pose relative to end-effector (tag placement result)

Math:
  T[cam|base] @ T[tag|cam]_i = T[end|base]_i @ T[tag|end]

Solved by cv2.calibrateRobotWorldHandEye.

Note on OpenCV parameter naming (misleading but correct):
  Input  R_world2end  = T[end|base] rotations
  Input  R_base2cam   = T[tag|cam]  rotations  (naming is confusing in OpenCV)
  Output R_end2world  = T[tag|end]  rotation
  Output R_cam2base   = T[cam|base] rotation

Assumptions:
- Camera is fixed relative to robot base (eye-to-hand setup).
- AprilTag is mounted on robot end-effector at unknown pose T[tag|end].
- CSV format identical to eye-in-hand (robot_eye_in_hand.csv).
- Units: mm for translation, rad for Euler XYZ rotation.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from scripts.handeye_calibrate import load_samples
from scripts.handeye_utils import rot_to_euler_xyz


def resolve_method(method: str) -> int:
    """
    Resolve method name to OpenCV robot-world hand-eye method constant.

    Args:
        method (str): Method name: "shah" or "li".

    Returns:
        int: OpenCV method constant.

    Raises:
        ValueError: If method name is not recognized.
    """
    import cv2

    mapping = {
        "shah": cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
        "li": cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI,
    }
    name = method.strip().lower()
    if name not in mapping:
        raise ValueError(f"unsupported method: {method!r}. Choices: shah, li")
    return mapping[name]


def solve_eye_to_hand_raw(
    r_gripper2base: List[np.ndarray],
    t_gripper2base: List[np.ndarray],
    r_target2cam: List[np.ndarray],
    t_target2cam: List[np.ndarray],
    method: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Call cv2.calibrateRobotWorldHandEye and return the four result arrays.

    Args:
        r_gripper2base (list[np.ndarray]): T[end|base] rotations (3x3 each).
        t_gripper2base (list[np.ndarray]): T[end|base] translations (3,) mm each.
        r_target2cam (list[np.ndarray]): T[tag|cam] rotations in robot axes (3x3).
        t_target2cam (list[np.ndarray]): T[tag|cam] translations in robot axes (3,) mm.
        method (str): "shah" or "li".

    Returns:
        tuple: (R_cam2base, t_cam2base, R_tag2end, t_tag2end)
            R_cam2base (3x3), t_cam2base (3,1) mm,
            R_tag2end (3x3),  t_tag2end (3,1) mm.
    """
    import cv2

    method_flag = resolve_method(method)
    r_tag2end, t_tag2end, r_cam2base, t_cam2base = cv2.calibrateRobotWorldHandEye(
        r_gripper2base,
        [np.asarray(t, dtype=float).reshape(3, 1) for t in t_gripper2base],
        r_target2cam,
        [np.asarray(t, dtype=float).reshape(3, 1) for t in t_target2cam],
        method=method_flag,
    )
    return r_cam2base, t_cam2base, r_tag2end, t_tag2end


def solve_eye_to_hand(
    csv_path: str,
    method: str = "shah",
    max_reproj_error: float = 2.0,
    min_samples: int = 3,
) -> dict:
    """
    Load CSV and solve eye-to-hand calibration.

    Args:
        csv_path (str): Path to CSV (same format as eye-in-hand CSV).
        method (str): "shah" or "li".
        max_reproj_error (float): Max reprojection error filter (pixels).
        min_samples (int): Minimum valid samples required.

    Returns:
        dict: {
            "r_cam2base": np.ndarray (3x3),   camera rotation in base frame
            "t_cam2base": np.ndarray (3,) mm, camera translation in base frame
            "r_tag2end":  np.ndarray (3x3),   tag rotation in end frame
            "t_tag2end":  np.ndarray (3,) mm, tag translation in end frame
            "n_samples":  int,
        }

    Raises:
        ValueError: If CSV not found or insufficient valid samples.
    """
    if not os.path.exists(csv_path):
        raise ValueError(f"csv not found: {csv_path}")

    r_g2b, t_g2b, r_t2c, t_t2c, warnings = load_samples(csv_path, max_reproj_error)
    for w in warnings:
        print(f"warning: {w}")

    if "numpy_missing" in warnings:
        raise ValueError("numpy is not available; cannot solve calibration")
    if "empty_csv" in warnings:
        raise ValueError(f"csv is empty or has no header: {csv_path}")

    if len(r_g2b) < min_samples:
        raise ValueError(f"not enough valid samples: {len(r_g2b)} < {min_samples}")

    r_cam2base, t_cam2base, r_tag2end, t_tag2end = solve_eye_to_hand_raw(
        r_g2b, t_g2b, r_t2c, t_t2c, method
    )

    return {
        "r_cam2base": np.array(r_cam2base),
        "t_cam2base": np.array(t_cam2base, dtype=float).reshape(3),
        "r_tag2end": np.array(r_tag2end),
        "t_tag2end": np.array(t_tag2end, dtype=float).reshape(3),
        "n_samples": len(r_g2b),
    }


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        None.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Eye-to-hand calibration solver")
    parser.add_argument("--input-csv", default="data/eye_to_hand.csv")
    parser.add_argument("--method", default="shah", choices=["shah", "li"])
    parser.add_argument("--max-reproj-error", type=float, default=2.0)
    parser.add_argument("--min-samples", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    """
    Entry point.

    Args:
        None.

    Returns:
        int: Exit code.
    """
    args = parse_args()

    try:
        import cv2  # noqa: F401
    except ImportError as exc:
        print(f"OpenCV not available: {exc}")
        return 1

    try:
        result = solve_eye_to_hand(
            args.input_csv,
            method=args.method,
            max_reproj_error=args.max_reproj_error,
            min_samples=args.min_samples,
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 1

    t_cb = result["t_cam2base"]
    t_te = result["t_tag2end"]
    cam_rx, cam_ry, cam_rz = rot_to_euler_xyz(result["r_cam2base"])
    tag_rx, tag_ry, tag_rz = rot_to_euler_xyz(result["r_tag2end"])

    print(f"valid_samples: {result['n_samples']}")
    print("camera|base (T_cam2base):")
    print(f"  t_mm: [{t_cb[0]:.3f}, {t_cb[1]:.3f}, {t_cb[2]:.3f}]")
    print(f"  rpy_rad: [{cam_rx:.6f}, {cam_ry:.6f}, {cam_rz:.6f}]")
    print("tag|end (T_tag2end):")
    print(f"  t_mm: [{t_te[0]:.3f}, {t_te[1]:.3f}, {t_te[2]:.3f}]")
    print(f"  rpy_rad: [{tag_rx:.6f}, {tag_ry:.6f}, {tag_rz:.6f}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
