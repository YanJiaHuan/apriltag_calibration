#!/usr/bin/env python3
"""
Robot-side client for eye-in-hand calibration data collection.

This module runs on the robot controller side. It connects to the remote
RealSense AprilTag server via WebSocket, requests a capture, and records the
returned tag pose together with the robot end-effector pose (end|base) into
a CSV file in /data. It does not save images locally.

Assumptions:
- Robot pose is returned by RM_API2 in meters and radians (Euler).
- Output must be millimeters and radians (Euler).
- Tag pose is in the OpenCV camera frame and must not be mixed with robot axes.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
import uuid
from typing import Any, Dict


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


def write_csv_row(csv_path: str, header: list[str], row: list[Any]) -> None:
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
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    rm_path = os.path.join(repo_root, "third_party", "RM_API2", "Python")
    if rm_path not in sys.path:
        sys.path.insert(0, rm_path)

    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

    arm = RoboticArm(rm_thread_mode_e.RM_SINGLE_MODE_E)
    arm.rm_create_robot_arm(ip, port)
    return arm


def get_robot_pose_mm(arm: Any) -> list[float]:
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


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        None.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Robot-side eye-in-hand logger")
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--robot-port", type=int, required=True)
    parser.add_argument("--websocket-config", default="config/websocket.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--csv-name", default="robot_eye_in_hand.csv")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--interval-s", type=float, default=0.0)
    parser.add_argument("--manual", action="store_true")
    return parser.parse_args()


def main() -> int:
    """
    Entry point for the robot-side logger.

    Args:
        None.

    Returns:
        int: Process exit code.
    """
    import asyncio
    import websockets

    args = parse_args()
    ws_cfg = load_json(args.websocket_config)
    server_uri = ws_cfg.get("server_uri")
    if not server_uri:
        raise ValueError("server_uri missing in websocket config")

    os.makedirs(args.data_dir, exist_ok=True)
    csv_path = os.path.join(args.data_dir, args.csv_name)

    header = [
        "robot_timestamp_iso",
        "camera_timestamp_iso",
        "robot_ip",
        "robot_port",
        "end_x_mm",
        "end_y_mm",
        "end_z_mm",
        "end_rx_rad",
        "end_ry_rad",
        "end_rz_rad",
        "tag_found",
        "tag_id",
        "tag_family",
        "tag_size_mm",
        "camera_frame",
        "image_path",
        "cam_fx",
        "cam_fy",
        "cam_cx",
        "cam_cy",
        "tag_t_x_mm",
        "tag_t_y_mm",
        "tag_t_z_mm",
        "tag_r_x_rad",
        "tag_r_y_rad",
        "tag_r_z_rad",
        "reproj_error",
        "error",
    ]

    arm = connect_robot(args.robot_ip, args.robot_port)

    async def run_session() -> None:
        """
        Run capture loop and log CSV.

        Args:
            None.

        Returns:
            None.
        """
        async with websockets.connect(server_uri) as ws:
            for i in range(args.count):
                if args.manual:
                    input("Press Enter to capture... ")

                req_id = uuid.uuid4().hex
                await ws.send(json.dumps({"type": "capture", "request_id": req_id}))
                resp = json.loads(await ws.recv())

                robot_time = now_iso()
                robot_pose = get_robot_pose_mm(arm)

                intr = resp.get("intrinsics", {}) or {}
                t_mm = resp.get("t_mm") or ["", "", ""]
                rpy = resp.get("rpy_rad") or ["", "", ""]

                row = [
                    robot_time,
                    resp.get("timestamp_iso", ""),
                    args.robot_ip,
                    args.robot_port,
                    robot_pose[0],
                    robot_pose[1],
                    robot_pose[2],
                    robot_pose[3],
                    robot_pose[4],
                    robot_pose[5],
                    bool(resp.get("tag_found", False)),
                    resp.get("tag_id", ""),
                    resp.get("tag_family", ""),
                    resp.get("tag_size_mm", ""),
                    resp.get("camera_frame", ""),
                    resp.get("image_path", ""),
                    intr.get("fx", ""),
                    intr.get("fy", ""),
                    intr.get("cx", ""),
                    intr.get("cy", ""),
                    t_mm[0],
                    t_mm[1],
                    t_mm[2],
                    rpy[0],
                    rpy[1],
                    rpy[2],
                    resp.get("reproj_error", ""),
                    resp.get("error", ""),
                ]

                write_csv_row(csv_path, header, row)

                if args.interval_s > 0 and i < args.count - 1:
                    await asyncio.sleep(args.interval_s)

    try:
        asyncio.run(run_session())
    finally:
        try:
            arm.rm_delete_robot_arm()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
