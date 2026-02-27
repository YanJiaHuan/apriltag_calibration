#!/usr/bin/env python3
"""
Remote AprilTag capture server for eye-in-hand calibration.

This module runs on the remote machine that hosts the RealSense D435.
It captures a single RGB frame per request, detects the AprilTag with the
third_party AprilTag library (Python binding), estimates tag pose in the
OpenCV camera frame, saves the image, and appends a CSV record in /data.

Assumptions:
- Tag family is tag36h11 and tag size is provided in mm via config/vision_config.json.
- Pose returned by AprilTag is in the OpenCV camera frame (x right, y down, z forward).
- Translation is stored in millimeters, rotation in Euler XYZ (rad).
- This module must not mix robot frame conventions with OpenCV camera frame.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import json
import math
import os
import sys
from typing import Any, Dict, Optional


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


def ensure_dir(path: str) -> None:
    """
    Ensure a directory exists.

    Args:
        path (str): Directory path.

    Returns:
        None.

    Side Effects:
        Creates the directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def meters_to_mm(value_m: float) -> float:
    """
    Convert meters to millimeters.

    Args:
        value_m (float): Value in meters.

    Returns:
        float: Value in millimeters.
    """
    return value_m * 1000.0


def rotation_matrix_to_euler_xyz(r: Any) -> tuple[float, float, float]:
    """
    Convert a 3x3 rotation matrix to Euler XYZ (roll, pitch, yaw) in radians.

    This uses the convention R = Rz * Ry * Rx.

    Args:
        r (Any): 3x3 rotation matrix (list or numpy array).

    Returns:
        tuple[float, float, float]: (rx, ry, rz) in radians.
    """
    r00 = float(r[0][0])
    r10 = float(r[1][0])
    r20 = float(r[2][0])
    r21 = float(r[2][1])
    r22 = float(r[2][2])

    sy = math.sqrt(r00 * r00 + r10 * r10)
    if sy < 1e-9:
        rx = math.atan2(-float(r[1][2]), float(r[1][1]))
        ry = math.atan2(-r20, sy)
        rz = 0.0
    else:
        rx = math.atan2(r21, r22)
        ry = math.atan2(-r20, sy)
        rz = math.atan2(r10, r00)

    return rx, ry, rz


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


def init_realsense(realsense_cfg_path: str) -> tuple[Any, Dict[str, float]]:
    """
    Initialize RealSense pipeline and return intrinsics.

    Args:
        realsense_cfg_path (str): Path to realsense_D435.json.

    Returns:
        tuple[Any, dict]: (pipeline, intrinsics dict with fx, fy, cx, cy).
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
        rgb = cfg["rgb_sensor"]
        for opt_name, opt_value in rgb.items():
            try:
                opt_enum = getattr(rs.option, opt_name)
                if sensor.supports(opt_enum):
                    sensor.set_option(opt_enum, float(opt_value))
            except Exception:
                continue

    stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = stream.get_intrinsics()
    intrinsics = {
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.ppx),
        "cy": float(intr.ppy),
    }
    return pipeline, intrinsics


def build_detector(tag_family: str) -> Any:
    """
    Build an AprilTag detector for the given family.

    Args:
        tag_family (str): AprilTag family name (e.g., tag36h11).

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
    Capture one frame and run AprilTag detection + pose estimation.

    Args:
        pipeline (Any): RealSense pipeline.
        detector (Any): AprilTag detector instance.
        tag_id (int): Target tag id.
        tag_size_mm (float): Tag size in millimeters.
        intrinsics (dict): Camera intrinsics {fx, fy, cx, cy}.

    Returns:
        dict: Detection result with pose and image.
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
    if not detections:
        return {"tag_found": False, "error": "no_detections", "image": color}

    target_det = None
    for det in detections:
        if int(det.get("id", -1)) == int(tag_id):
            target_det = det
            break
    if target_det is None:
        return {"tag_found": False, "error": "tag_id_not_found", "image": color}

    tagsize_m = float(tag_size_mm) / 1000.0
    pose = detector.estimate_tag_pose(
        target_det,
        tagsize_m,
        intrinsics["fx"],
        intrinsics["fy"],
        intrinsics["cx"],
        intrinsics["cy"],
    )

    r = pose["R"]
    t = pose["t"]
    rx, ry, rz = rotation_matrix_to_euler_xyz(r)

    result = {
        "tag_found": True,
        "tag_id": int(target_det.get("id", -1)),
        "rpy_rad": [rx, ry, rz],
        "t_mm": [meters_to_mm(float(t[0][0])), meters_to_mm(float(t[1][0])), meters_to_mm(float(t[2][0]))],
        "reproj_error": float(pose.get("error", 0.0)),
        "image": color,
    }
    return result


def save_image(image: Any, image_path: str) -> None:
    """
    Save a BGR image to disk.

    Args:
        image (Any): BGR image array.
        image_path (str): Output path.

    Returns:
        None.

    Side Effects:
        Writes an image file to disk.
    """
    import cv2

    cv2.imwrite(image_path, image)


def handle_capture_request(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle a single capture request: capture, detect, save image, log CSV.

    Args:
        context (dict): Runtime context for pipeline, detector, and config.

    Returns:
        dict: Response payload to send to client.
    """
    timestamp = now_iso()
    result = capture_and_detect(
        context["pipeline"],
        context["detector"],
        context["tag_id"],
        context["tag_size_mm"],
        context["intrinsics"],
    )

    image_rel_path = ""
    if "image" in result:
        image_name = f"tag_{context['tag_id']}_{timestamp.replace(':', '-')}.png"
        image_rel_path = os.path.join(context["image_dir"], image_name)
        save_image(result["image"], image_rel_path)

    header = [
        "timestamp_iso",
        "tag_id",
        "tag_family",
        "tag_size_mm",
        "camera_frame",
        "image_path",
        "fx",
        "fy",
        "cx",
        "cy",
        "tag_found",
        "t_x_mm",
        "t_y_mm",
        "t_z_mm",
        "r_x_rad",
        "r_y_rad",
        "r_z_rad",
        "reproj_error",
        "error",
    ]

    if result.get("tag_found", False):
        t_mm = result["t_mm"]
        rpy = result["rpy_rad"]
    else:
        t_mm = ["", "", ""]
        rpy = ["", "", ""]

    row = [
        timestamp,
        context["tag_id"],
        context["tag_family"],
        context["tag_size_mm"],
        "opencv",
        image_rel_path,
        context["intrinsics"]["fx"],
        context["intrinsics"]["fy"],
        context["intrinsics"]["cx"],
        context["intrinsics"]["cy"],
        bool(result.get("tag_found", False)),
        t_mm[0],
        t_mm[1],
        t_mm[2],
        rpy[0],
        rpy[1],
        rpy[2],
        result.get("reproj_error", ""),
        result.get("error", ""),
    ]

    write_csv_row(context["csv_path"], header, row)

    response = {
        "type": "capture_result",
        "timestamp_iso": timestamp,
        "tag_found": bool(result.get("tag_found", False)),
        "tag_id": context["tag_id"],
        "tag_family": context["tag_family"],
        "tag_size_mm": context["tag_size_mm"],
        "camera_frame": "opencv",
        "image_path": image_rel_path,
        "intrinsics": context["intrinsics"],
        "t_mm": result.get("t_mm", None),
        "rpy_rad": result.get("rpy_rad", None),
        "reproj_error": result.get("reproj_error", None),
        "error": result.get("error", ""),
    }
    return response


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        None.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Remote AprilTag capture server")
    parser.add_argument("--vision-config", default="config/vision_config.json")
    parser.add_argument("--realsense-config", default="config/realsense_D435.json")
    parser.add_argument("--websocket-config", default="config/websocket.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--csv-name", default="remote_apriltag.csv")
    return parser.parse_args()


def main() -> int:
    """
    Entry point for the remote server.

    Args:
        None.

    Returns:
        int: Process exit code.
    """
    import websockets

    args = parse_args()
    vision_cfg = load_json(args.vision_config)
    tag_id = int(vision_cfg["tag"]["id"])
    tag_size_mm = float(vision_cfg["tag"]["size_mm"])
    tag_family = str(vision_cfg["tag"]["family"])

    ws_cfg = load_json(args.websocket_config)
    host = ws_cfg.get("server_host", "0.0.0.0")
    port = int(ws_cfg.get("server_port", 8766))

    data_dir = args.data_dir
    image_dir = os.path.join(data_dir, "remote_images")
    ensure_dir(data_dir)
    ensure_dir(image_dir)

    pipeline, intrinsics = init_realsense(args.realsense_config)
    detector = build_detector(tag_family)

    context = {
        "pipeline": pipeline,
        "detector": detector,
        "tag_id": tag_id,
        "tag_size_mm": tag_size_mm,
        "tag_family": tag_family,
        "intrinsics": intrinsics,
        "csv_path": os.path.join(data_dir, args.csv_name),
        "image_dir": image_dir,
        "lock": asyncio.Lock(),
    }

    async def handler(websocket, _path):
        """
        Handle websocket requests from the robot side.

        Args:
            websocket (Any): Websocket connection.
            _path (str): Request path (unused).

        Returns:
            None.
        """
        async for message in websocket:
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"type": "error", "error": "invalid_json"}))
                continue

            if payload.get("type") != "capture":
                await websocket.send(json.dumps({"type": "error", "error": "unsupported_type"}))
                continue

            async with context["lock"]:
                response = await asyncio.to_thread(handle_capture_request, context)
            if "request_id" in payload:
                response["request_id"] = payload["request_id"]
            await websocket.send(json.dumps(response))

    async def run_server() -> None:
        """
        Run the websocket server indefinitely.

        Args:
            None.

        Returns:
            None.
        """
        async with websockets.serve(handler, host, port):
            await asyncio.Future()

    try:
        asyncio.run(run_server())
    finally:
        pipeline.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
