#!/usr/bin/env python3
"""
Eye-to-hand calibration data collection (local camera, no WebSocket).

Captures AprilTag detections from a locally attached RealSense camera and
simultaneously reads the robot end-effector pose. Both are written to a CSV
compatible with eye_to_hand_calibrate.py.

This module assumes the camera and robot controller are on the same machine.

Assumptions:
- Camera is fixed externally (eye-to-hand setup).
- AprilTag is mounted on the robot end-effector.
- Robot pose is end-effector in base frame (end|base).
- Tag pose from AprilTag library is in OpenCV camera frame (tag|camera).
- Units: mm for translation, rad for Euler XYZ rotation.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from typing import Any, Dict, List

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.handeye_utils import rot_to_euler_xyz


def now_iso() -> str:
    """
    Return current local timestamp in ISO-8601 with milliseconds.

    Args:
        None.

    Returns:
        str: ISO-8601 timestamp string.
    """
    return dt.datetime.now().isoformat(timespec="milliseconds")


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


def write_csv_row(csv_path: str, header: List[str], row: List[Any]) -> None:
    """
    Append a row to a CSV file, writing the header if the file is new.

    Args:
        csv_path (str): CSV file path.
        header (list[str]): CSV header columns.
        row (list[Any]): Row data.

    Returns:
        None.

    Side Effects:
        Creates/updates the CSV file on disk.
    """
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        writer.writerow(row)


def connect_robot(ip: str, port: int) -> Any:
    """
    Connect to the Realman robot using RM_API2.

    Args:
        ip (str): Robot IP address.
        port (int): Robot port.

    Returns:
        Any: RoboticArm instance.
    """
    rm_path = os.path.join(_repo_root, "third_party", "RM_API2", "Python")
    if rm_path not in sys.path:
        sys.path.insert(0, rm_path)
    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

    arm = RoboticArm(rm_thread_mode_e.RM_SINGLE_MODE_E)
    arm.rm_create_robot_arm(ip, port)
    return arm


def get_robot_pose_mm(arm: Any) -> List[float]:
    """
    Get robot end-effector pose in base frame, converting meters to mm.

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


def init_realsense(realsense_cfg_path: str):
    """
    Initialize RealSense pipeline and return (pipeline, intrinsics).

    Args:
        realsense_cfg_path (str): Path to realsense config JSON.

    Returns:
        tuple: (pipeline, intrinsics dict with fx, fy, cx, cy).
    """
    import pyrealsense2 as rs

    cfg = load_json(realsense_cfg_path)
    width = int(cfg["quality"]["width"])
    height = int(cfg["quality"]["height"])
    fps = int(cfg["quality"]["fps"])

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)

    sensor = profile.get_device().first_color_sensor()
    if sensor is not None and "rgb_sensor" in cfg:
        for opt_name, opt_value in cfg["rgb_sensor"].items():
            try:
                opt_enum = getattr(rs.option, opt_name)
                if sensor.supports(opt_enum):
                    sensor.set_option(opt_enum, float(opt_value))
            except Exception:
                continue

    stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = stream.get_intrinsics()
    intrinsics = {
        "fx": float(intr.fx), "fy": float(intr.fy),
        "cx": float(intr.ppx), "cy": float(intr.ppy),
    }
    return pipeline, intrinsics


def build_detector(tag_family: str) -> Any:
    """
    Build an AprilTag detector for the given tag family.

    Args:
        tag_family (str): Tag family name (e.g., tag36h11, tagStandard41h12).

    Returns:
        Any: AprilTag detector instance.
    """
    from apriltag import apriltag

    return apriltag(tag_family, threads=4, decimate=1.0, refine_edges=True)


def capture_and_detect(
    pipeline: Any,
    detector: Any,
    tag_id: int,
    tag_size_mm: float,
    intrinsics: Dict[str, float],
) -> Dict[str, Any]:
    """
    Capture one RGB frame and run AprilTag detection + pose estimation.

    Pose is returned in OpenCV camera frame (X right, Y down, Z forward).
    Translation is in mm.

    Args:
        pipeline (Any): RealSense pipeline.
        detector (Any): AprilTag detector.
        tag_id (int): Target tag ID.
        tag_size_mm (float): Physical tag size in mm.
        intrinsics (dict): Camera intrinsics with fx, fy, cx, cy.

    Returns:
        dict: {tag_found, tag_id, rpy_rad, t_mm, reproj_error, image} or
              {tag_found: False, error: str}.
    """
    import cv2
    import numpy as np

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return {"tag_found": False, "error": "no_color_frame"}

    color = np.asanyarray(color_frame.get_data())
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    target_det = None
    for det in detections:
        if int(det.get("id", -1)) == int(tag_id):
            target_det = det
            break
    if target_det is None:
        return {"tag_found": False, "error": "tag_id_not_found"}

    pose = detector.estimate_tag_pose(
        target_det,
        float(tag_size_mm) / 1000.0,
        intrinsics["fx"], intrinsics["fy"],
        intrinsics["cx"], intrinsics["cy"],
    )

    r_mat = np.array(pose["R"])
    rx, ry, rz = rot_to_euler_xyz(r_mat)
    t = pose["t"]

    return {
        "tag_found": True,
        "tag_id": int(target_det.get("id", -1)),
        "rpy_rad": [rx, ry, rz],
        "t_mm": [
            meters_to_mm(float(t[0][0])),
            meters_to_mm(float(t[1][0])),
            meters_to_mm(float(t[2][0])),
        ],
        "reproj_error": float(pose.get("error", 0.0)),
        "image": color,
    }


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        None.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Eye-to-hand data collection (local camera, no WebSocket)"
    )
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--robot-port", type=int, required=True)
    parser.add_argument("--realsense-config", default="config/realsense_D435.json")
    parser.add_argument("--vision-config", default="config/vision_config.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--csv-name", default="eye_to_hand.csv")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--max-reproj-error", type=float, default=2.0)
    parser.add_argument(
        "--manual", action="store_true",
        help="Wait for Enter key before each capture (recommended for calibration)",
    )
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
    vision_cfg = load_json(args.vision_config)
    tag_id = int(vision_cfg["tag"]["id"])
    tag_size_mm = float(vision_cfg["tag"]["size_mm"])
    tag_family = str(vision_cfg["tag"]["family"])

    os.makedirs(args.data_dir, exist_ok=True)
    csv_path = os.path.join(args.data_dir, args.csv_name)
    image_dir = os.path.join(args.data_dir, "eye_to_hand_images")
    os.makedirs(image_dir, exist_ok=True)

    header = [
        "robot_timestamp_iso", "camera_timestamp_iso",
        "robot_ip", "robot_port",
        "end_x_mm", "end_y_mm", "end_z_mm",
        "end_rx_rad", "end_ry_rad", "end_rz_rad",
        "tag_found", "quality_ok", "max_reproj_error",
        "tag_id", "tag_family", "tag_size_mm", "camera_frame",
        "image_path", "cam_fx", "cam_fy", "cam_cx", "cam_cy",
        "tag_t_x_mm", "tag_t_y_mm", "tag_t_z_mm",
        "tag_r_x_rad", "tag_r_y_rad", "tag_r_z_rad",
        "reproj_error", "error",
    ]

    pipeline, intrinsics = init_realsense(args.realsense_config)
    detector = build_detector(tag_family)
    arm = connect_robot(args.robot_ip, args.robot_port)

    collected = 0
    try:
        for i in range(args.count):
            if args.manual:
                input(
                    f"[{i + 1}/{args.count}] Move robot to a new pose, "
                    "then press Enter to capture... "
                )
            else:
                print(f"[{i + 1}/{args.count}] Capturing...")

            result = capture_and_detect(
                pipeline, detector, tag_id, tag_size_mm, intrinsics
            )
            timestamp = now_iso()

            reproj_error = result.get("reproj_error", None)
            quality_ok = (
                result.get("tag_found", False)
                and reproj_error is not None
                and float(reproj_error) <= args.max_reproj_error
            )

            if not result.get("tag_found", False):
                print(f"  tag_not_found: {result.get('error', '')}")
                continue

            if not quality_ok:
                print(
                    f"  low_quality: reproj_error={reproj_error:.3f}"
                    f" > {args.max_reproj_error}"
                )
                continue

            robot_pose = get_robot_pose_mm(arm)

            image_path = ""
            if "image" in result:
                import cv2

                fname = f"tag_{tag_id}_{timestamp.replace(':', '-')}.png"
                image_path = os.path.join(image_dir, fname)
                cv2.imwrite(image_path, result["image"])

            t_mm = result["t_mm"]
            rpy = result["rpy_rad"]

            row = [
                timestamp, timestamp,
                args.robot_ip, args.robot_port,
                robot_pose[0], robot_pose[1], robot_pose[2],
                robot_pose[3], robot_pose[4], robot_pose[5],
                True, True, args.max_reproj_error,
                tag_id, tag_family, tag_size_mm, "opencv",
                image_path,
                intrinsics["fx"], intrinsics["fy"],
                intrinsics["cx"], intrinsics["cy"],
                t_mm[0], t_mm[1], t_mm[2],
                rpy[0], rpy[1], rpy[2],
                reproj_error, "",
            ]
            write_csv_row(csv_path, header, row)
            collected += 1
            print(
                f"  OK: reproj_error={reproj_error:.3f},"
                f" t_mm={[round(x, 1) for x in t_mm]}"
            )

    finally:
        try:
            arm.rm_delete_robot_arm()
        except Exception:
            pass
        pipeline.stop()

    print(f"\nCollected {collected} valid samples -> {csv_path}")
    if collected < 3:
        print("WARNING: need at least 3 samples for calibration")
    return 0


if __name__ == "__main__":
    sys.exit(main())
