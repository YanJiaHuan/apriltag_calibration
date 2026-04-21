#!/usr/bin/env python3
"""
Hand-eye calibration validation script.

This script validates the eye-in-hand calibration by moving the robot arm to
different poses and verifying that the calculated AprilTag position in the base
frame remains constant (since both the tag and base are static).

Usage:
    python3 scripts/validate_handeye_calibration.py \
        --robot-ip <ip> \
        --robot-port <port> \
        --cam-in-end-x <x> --cam-in-end-y <y> --cam-in-end-z <z> \
        --cam-in-end-rx <rx> --cam-in-end-ry <ry> --cam-in-end-rz <rz>

The camera|end pose (T_end_cam) should be obtained from the calibration result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from typing import Any, Dict, List, Tuple

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    import numpy as np
except ImportError:
    np = None

from scripts.handeye_utils import (
    convert_opencv_pose_to_robot,
    euler_xyz_to_rot,
    make_transform,
    rot_to_euler_xyz,
)


def load_json(path: str) -> Dict[str, Any]:
    """
    Load a JSON file from disk.

    Args:
        path (str): JSON file path.

    Returns:
        dict: Parsed JSON dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def meters_to_mm(value_m: float) -> float:
    """
    Convert meters to millimeters.

    Args:
        value_m (float): Value in meters.

    Returns:
        float: Value in millimeters.
    """
    return value_m * 1000.0


def connect_robot(ip: str, port: int) -> Any:
    """
    Connect to the Realman robot using RM_API2.

    Args:
        ip (str): Robot IP address.
        port (int): Robot port.

    Returns:
        Any: RoboticArm instance.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    rm_path = os.path.join(repo_root, "third_party", "RM_API2", "Python")
    if rm_path not in sys.path:
        sys.path.insert(0, rm_path)

    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

    arm = RoboticArm(rm_thread_mode_e.RM_SINGLE_MODE_E)
    arm.rm_create_robot_arm(ip, port)
    return arm


def get_robot_pose_mm(arm: Any) -> List[float]:
    """
    Get robot end-effector pose in base frame, convert meters to millimeters.

    Args:
        arm (Any): RoboticArm instance.

    Returns:
        list[float]: [x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad].
    """
    ret, state = arm.rm_get_current_arm_state()
    if ret != 0:
        raise RuntimeError(f"rm_get_current_arm_state failed: {ret}")

    pose = state["pose"]
    return [
        meters_to_mm(float(pose[0])),
        meters_to_mm(float(pose[1])),
        meters_to_mm(float(pose[2])),
        float(pose[3]),
        float(pose[4]),
        float(pose[5]),
    ]


def calculate_tag_in_base(
    robot_pose: List[float],
    tag_pose_opencv: List[float],
    cam_in_end: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate AprilTag pose in base frame given robot pose, tag detection, and calibration.

    Args:
        robot_pose (list[float]): [x, y, z, rx, ry, rz] end in base (mm, rad).
        tag_pose_opencv (list[float]): [x, y, z, rx, ry, rz] tag in camera OpenCV (mm, rad).
        cam_in_end (list[float]): [x, y, z, rx, ry, rz] camera in end (mm, rad).

    Returns:
        tuple[np.ndarray, np.ndarray]: (R_base_tag, t_base_tag) as 3x3 and 3-element arrays.
    """
    # Parse robot pose (end|base)
    r_base_end = euler_xyz_to_rot(robot_pose[3], robot_pose[4], robot_pose[5])
    t_base_end = np.array(robot_pose[:3], dtype=float)

    # Parse tag pose in OpenCV camera frame (tag|camera_opencv)
    r_tag2cam_opencv = euler_xyz_to_rot(tag_pose_opencv[3], tag_pose_opencv[4], tag_pose_opencv[5])
    t_tag2cam_opencv = np.array(tag_pose_opencv[:3], dtype=float)

    # Convert tag pose to robot camera frame (tag|camera_robot)
    r_tag2cam_robot, t_tag2cam_robot = convert_opencv_pose_to_robot(r_tag2cam_opencv, t_tag2cam_opencv)

    # Parse camera in end (cam|end)
    r_end_cam = euler_xyz_to_rot(cam_in_end[3], cam_in_end[4], cam_in_end[5])
    t_end_cam = np.array(cam_in_end[:3], dtype=float)

    # Build transforms
    t_base_end_mat = make_transform(r_base_end, t_base_end)
    t_end_cam_mat = make_transform(r_end_cam, t_end_cam)
    t_cam_tag_mat = make_transform(r_tag2cam_robot, t_tag2cam_robot)

    # Compose: base <- end <- cam <- tag
    t_base_tag_mat = t_base_end_mat @ t_end_cam_mat @ t_cam_tag_mat

    r_base_tag = t_base_tag_mat[:3, :3]
    t_base_tag = t_base_tag_mat[:3, 3]

    return r_base_tag, t_base_tag


async def capture_tag_pose(websocket, max_reproj_error: float) -> Dict[str, Any]:
    """
    Capture a single AprilTag detection from the WebSocket server.

    Args:
        websocket: WebSocket connection.
        max_reproj_error (float): Maximum allowed reprojection error.

    Returns:
        dict: Detection result with tag pose or error.
    """
    req_id = uuid.uuid4().hex
    await websocket.send(json.dumps({"type": "capture", "request_id": req_id}))

    resp = json.loads(await websocket.recv())

    if not resp.get("tag_found", False):
        return {"success": False, "error": "tag_not_found"}

    reproj_error = resp.get("reproj_error", None)
    if reproj_error is not None and float(reproj_error) > max_reproj_error:
        return {"success": False, "error": f"high_reproj_error: {reproj_error}"}

    t_mm = resp.get("t_mm") or [None, None, None]
    rpy = resp.get("rpy_rad") or [None, None, None]

    if None in t_mm or None in rpy:
        return {"success": False, "error": "incomplete_pose"}

    return {
        "success": True,
        "pose": [float(t_mm[0]), float(t_mm[1]), float(t_mm[2]),
                 float(rpy[0]), float(rpy[1]), float(rpy[2])],
        "reproj_error": reproj_error,
    }


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        None.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Validate hand-eye calibration")
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--robot-port", type=int, required=True)
    parser.add_argument("--websocket-config", default="config/websocket.json")
    parser.add_argument("--cam-in-end-x", type=float, required=True, help="Camera X in end frame (mm)")
    parser.add_argument("--cam-in-end-y", type=float, required=True, help="Camera Y in end frame (mm)")
    parser.add_argument("--cam-in-end-z", type=float, required=True, help="Camera Z in end frame (mm)")
    parser.add_argument("--cam-in-end-rx", type=float, required=True, help="Camera RX in end frame (rad)")
    parser.add_argument("--cam-in-end-ry", type=float, required=True, help="Camera RY in end frame (rad)")
    parser.add_argument("--cam-in-end-rz", type=float, required=True, help="Camera RZ in end frame (rad)")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to collect")
    parser.add_argument("--max-reproj-error", type=float, default=2.0)
    parser.add_argument("--manual", action="store_true", help="Wait for Enter key between samples")
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    """
    Main validation loop.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        int: Exit code.
    """
    import websockets

    ws_cfg = load_json(args.websocket_config)
    server_uri = ws_cfg.get("server_uri")
    if not server_uri:
        raise ValueError("server_uri missing in websocket config")

    arm = connect_robot(args.robot_ip, args.robot_port)

    cam_in_end = [
        args.cam_in_end_x, args.cam_in_end_y, args.cam_in_end_z,
        args.cam_in_end_rx, args.cam_in_end_ry, args.cam_in_end_rz,
    ]

    print("=" * 80)
    print("Hand-Eye Calibration Validation")
    print("=" * 80)
    print(f"Camera|End (T_end_cam):")
    print(f"  t_mm: [{cam_in_end[0]:.3f}, {cam_in_end[1]:.3f}, {cam_in_end[2]:.3f}]")
    print(f"  rpy_rad: [{cam_in_end[3]:.6f}, {cam_in_end[4]:.6f}, {cam_in_end[5]:.6f}]")
    print(f"\nCollecting {args.num_samples} samples...")
    print("(Keep the AprilTag and robot base static)")
    print("=" * 80)

    tag_positions = []
    tag_rotations = []

    try:
        async with websockets.connect(server_uri, ping_interval=None, ping_timeout=None) as ws:
            for i in range(args.num_samples):
                if args.manual:
                    input(f"\nSample {i+1}/{args.num_samples}: Move robot to new pose, then press Enter...")
                else:
                    if i > 0:
                        print(f"\nSample {i+1}/{args.num_samples}: Move robot to new pose manually...")
                        await asyncio.sleep(2)
                    else:
                        print(f"\nSample {i+1}/{args.num_samples}: Starting...")

                # Get current robot pose
                robot_pose = get_robot_pose_mm(arm)
                print(f"  Robot pose: [{robot_pose[0]:.1f}, {robot_pose[1]:.1f}, {robot_pose[2]:.1f}] mm")

                # Capture tag detection
                result = await capture_tag_pose(ws, args.max_reproj_error)

                if not result["success"]:
                    print(f"  ❌ Failed: {result['error']}")
                    continue

                tag_pose_opencv = result["pose"]
                print(f"  Tag detected, reproj_error: {result.get('reproj_error', 'N/A')}")

                # Calculate tag in base
                r_base_tag, t_base_tag = calculate_tag_in_base(robot_pose, tag_pose_opencv, cam_in_end)
                rx, ry, rz = rot_to_euler_xyz(r_base_tag)

                print(f"  AprilTag|Base: [{t_base_tag[0]:.3f}, {t_base_tag[1]:.3f}, {t_base_tag[2]:.3f}] mm")

                tag_positions.append(t_base_tag)
                tag_rotations.append([rx, ry, rz])

    finally:
        try:
            arm.rm_delete_robot_arm()
        except Exception:
            pass

    # Analyze results
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    if len(tag_positions) < 2:
        print("❌ Not enough valid samples collected!")
        return 1

    tag_positions = np.array(tag_positions)
    tag_rotations = np.array(tag_rotations)

    mean_pos = np.mean(tag_positions, axis=0)
    std_pos = np.std(tag_positions, axis=0)
    max_dev_pos = np.max(np.abs(tag_positions - mean_pos), axis=0)

    mean_rot = np.mean(tag_rotations, axis=0)
    std_rot = np.std(tag_rotations, axis=0)
    max_dev_rot = np.max(np.abs(tag_rotations - mean_rot), axis=0)

    print(f"\nCollected {len(tag_positions)} valid samples\n")

    print("AprilTag|Base (position in mm):")
    print(f"  Mean:    [{mean_pos[0]:8.3f}, {mean_pos[1]:8.3f}, {mean_pos[2]:8.3f}]")
    print(f"  Std Dev: [{std_pos[0]:8.3f}, {std_pos[1]:8.3f}, {std_pos[2]:8.3f}]")
    print(f"  Max Dev: [{max_dev_pos[0]:8.3f}, {max_dev_pos[1]:8.3f}, {max_dev_pos[2]:8.3f}]")

    print(f"\nAprilTag|Base (rotation in rad):")
    print(f"  Mean:    [{mean_rot[0]:8.6f}, {mean_rot[1]:8.6f}, {mean_rot[2]:8.6f}]")
    print(f"  Std Dev: [{std_rot[0]:8.6f}, {std_rot[1]:8.6f}, {std_rot[2]:8.6f}]")
    print(f"  Max Dev: [{max_dev_rot[0]:8.6f}, {max_dev_rot[1]:8.6f}, {max_dev_rot[2]:8.6f}]")

    # Evaluation criteria
    max_pos_std = np.max(std_pos)
    max_pos_dev = np.max(max_dev_pos)

    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    # Good calibration should have position std < 5mm and max deviation < 10mm
    if max_pos_std < 5.0 and max_pos_dev < 10.0:
        print("✅ PASSED: Calibration is good!")
        print(f"   Max position std dev: {max_pos_std:.3f} mm (< 5 mm)")
        print(f"   Max position deviation: {max_pos_dev:.3f} mm (< 10 mm)")
        return 0
    elif max_pos_std < 10.0 and max_pos_dev < 20.0:
        print("⚠️  WARNING: Calibration is acceptable but could be improved")
        print(f"   Max position std dev: {max_pos_std:.3f} mm")
        print(f"   Max position deviation: {max_pos_dev:.3f} mm")
        print("   Consider recalibrating with more diverse poses")
        return 0
    else:
        print("❌ FAILED: Calibration is poor!")
        print(f"   Max position std dev: {max_pos_std:.3f} mm (should be < 5 mm)")
        print(f"   Max position deviation: {max_pos_dev:.3f} mm (should be < 10 mm)")
        print("   Please recalibrate with better data")
        return 1


def main() -> int:
    """
    Entry point.

    Args:
        None.

    Returns:
        int: Exit code.
    """
    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
