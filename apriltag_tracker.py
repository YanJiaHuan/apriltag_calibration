#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

try:
    from dt_apriltags import Detector as DTAprilTagDetector
except ImportError:
    DTAprilTagDetector = None

from apriltag import apriltag


def _rotation_to_euler_zyx(rotation_matrix: np.ndarray) -> tuple[float, float, float]:
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


def _transform_from_rt(rotation_matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation.reshape(3)
    return transform


def _quat_from_rotation(rotation_matrix: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4, dtype=np.float64)
    trace = np.trace(rotation_matrix)
    if trace > 0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        quaternion[0] = 0.25 / scale
        quaternion[1] = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * scale
        quaternion[2] = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * scale
        quaternion[3] = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * scale
    else:
        index = int(np.argmax(np.diag(rotation_matrix)))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
            quaternion[0] = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / scale
            quaternion[1] = 0.25 * scale
            quaternion[2] = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / scale
            quaternion[3] = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / scale
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
            quaternion[0] = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / scale
            quaternion[1] = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / scale
            quaternion[2] = 0.25 * scale
            quaternion[3] = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / scale
        else:
            scale = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
            quaternion[0] = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / scale
            quaternion[1] = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / scale
            quaternion[2] = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / scale
            quaternion[3] = 0.25 * scale
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def _rotation_from_quat(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)],
    ], dtype=np.float64)


def _average_quaternions(quaternions: list[np.ndarray]) -> np.ndarray:
    accumulator = np.zeros((4, 4), dtype=np.float64)
    for quaternion in quaternions:
        aligned = quaternion if quaternion[0] >= 0 else -quaternion
        accumulator += np.outer(aligned, aligned)
    _, eigenvectors = np.linalg.eigh(accumulator)
    averaged = eigenvectors[:, -1]
    if averaged[0] < 0:
        averaged = -averaged
    return averaged / np.linalg.norm(averaged)


def _serialize_pose_result(result: dict) -> dict:
    serialized = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        else:
            serialized[key] = value
    return serialized


def _print_pose_result(result: dict, label: str) -> None:
    position_mm = result["position_mm"]
    print(f"[LocalAprilTagPoseTracker] {label}")
    print(
        f"  position_mm: x={position_mm[0]:.3f}, y={position_mm[1]:.3f}, z={position_mm[2]:.3f}"
    )
    print(
        f"  rpy_deg: roll={np.degrees(result['roll']):.3f}, pitch={np.degrees(result['pitch']):.3f}, yaw={np.degrees(result['yaw']):.3f}"
    )
    if "reproj_error" in result:
        print(f"  reproj_error: {result['reproj_error']:.6f}")
    if "reproj_error_mean" in result:
        print(f"  reproj_error_mean: {result['reproj_error_mean']:.6f}")
    if "tag_size_px" in result:
        print(f"  tag_size_px: {result['tag_size_px']:.3f}")
    if "tag_size_px_mean" in result:
        print(f"  tag_size_px_mean: {result['tag_size_px_mean']:.3f}")
    if "pnp_z_mm" in result:
        print(f"  pnp_z_mm: {result['pnp_z_mm']:.3f}")
    if "pnp_z_mm_mean" in result:
        print(f"  pnp_z_mm_mean: {result['pnp_z_mm_mean']:.3f}")
    if "frames_used" in result:
        print(f"  frames_used: {result['frames_used']}")


def _draw_detection_debug(image: np.ndarray, detections, target_tag_id: int) -> np.ndarray:
    if image.ndim == 2:
        debug_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        debug_image = image.copy()
    for detection in detections:
        corners = _detection_corners(detection).astype(np.float32).reshape(4, 2)
        corners_int = corners.astype(int)
        current_tag_id = _detection_id(detection)
        color = (0, 255, 0) if current_tag_id == target_tag_id else (0, 200, 255)
        for idx in range(4):
            p0 = tuple(corners_int[idx])
            p1 = tuple(corners_int[(idx + 1) % 4])
            cv2.line(debug_image, p0, p1, color, 2)
        center = np.mean(corners, axis=0).astype(int)
        cv2.putText(debug_image, f"id={current_tag_id}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return debug_image


def _format_preview_image(image: np.ndarray, detections, target_tag_id: int, pose: dict | None, accepted_frames: int, status_text: str) -> np.ndarray:
    preview = _draw_detection_debug(image, detections, target_tag_id)
    lines = [status_text]
    if pose is not None:
        lines.append(f"reproj={float(pose['reproj_error']):.3f}px")
        lines.append(f"tag_px={float(pose['tag_size_px']):.1f}")
    lines.append(f"accepted={accepted_frames}")
    y = 30
    for line in lines:
        cv2.putText(preview, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += 30
    return preview


def _detection_id(detection) -> int:
    if hasattr(detection, "tag_id"):
        return int(detection.tag_id)
    if isinstance(detection, dict) and "id" in detection:
        return int(detection["id"])
    raise KeyError("AprilTag detection missing tag id")


def _detection_corners(detection) -> np.ndarray:
    if hasattr(detection, "corners"):
        return np.asarray(detection.corners, dtype=np.float64).reshape(4, 2)
    if isinstance(detection, dict) and "lb-rb-rt-lt" in detection:
        raw_corners = detection["lb-rb-rt-lt"]
        return np.array([
            raw_corners[3],
            raw_corners[2],
            raw_corners[1],
            raw_corners[0],
        ], dtype=np.float64)
    raise KeyError("AprilTag detection missing corners")


def _run_detector(detector, backend: str, gray: np.ndarray):
    if backend == "dt_apriltags":
        return detector.detect(gray, estimate_tag_pose=False)
    return detector.detect(gray)


class LocalAprilTagPoseTracker:
    def __init__(
        self,
        camera_config_path: str,
        tag_size_mm: float,
        tag_family: str,
        tag_id: int,
        resolution: str | None = 'high',
        stream_name: str | None = None,
    ):
        self.camera_config_path = str(camera_config_path)
        self.tag_size_mm = float(tag_size_mm)
        self.tag_size_m = self.tag_size_mm / 1000.0
        self.tag_family = str(tag_family)
        self.tag_id = int(tag_id)
        self.camera_root_config = self._load_camera_config(self.camera_config_path)
        self.camera_config = self._camera_runtime_config(self.camera_root_config)
        self.stream_name = self._resolve_stream_name(stream_name)
        self.stream_config = self._resolve_stream_config(self.stream_name)
        self.stream_type, self.stream_index = self._resolve_stream_type(self.stream_name)
        self.pipeline = rs.pipeline()
        self.profile = None
        self.color_format = rs.format.bgr8
        self.color_to_gray_code = cv2.COLOR_BGR2GRAY
        self._gray_code_primary = cv2.COLOR_BGR2GRAY
        self._gray_code_alt = None
        self._gray_mode = "locked"
        self.frame_timeout_ms = 15000
        self.frame_retry_count = 3
        self.intrinsics_source = "unknown"
        self.runtime_intrinsics_path = str(Path(self.camera_config_path).with_name("camera_intrinsics_runtime.yaml"))
        self._start_pipeline(resolution)
        self._load_intrinsics_from_config_or_stream()
        self._save_runtime_intrinsics()
        self._configure_camera_parameters()
        if DTAprilTagDetector is not None:
            self.detector = DTAprilTagDetector(
                families=self.tag_family,
                nthreads=4,
                quad_decimate=1.0,
            )
            self.detector_backend = "dt_apriltags"
        else:
            self.detector = apriltag(self.tag_family)
            self.detector_backend = "apriltag"
        half_size = self.tag_size_m / 2.0
        self.object_points = np.array([
            [-half_size, half_size, 0.0],
            [half_size, half_size, 0.0],
            [half_size, -half_size, 0.0],
            [-half_size, -half_size, 0.0],
        ], dtype=np.float64)
        self.t_cam_to_robot = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ], dtype=np.float64)
        if self.color_format == rs.format.rgb8:
            self._gray_code_primary = cv2.COLOR_RGB2GRAY
            self._gray_code_alt = cv2.COLOR_BGR2GRAY
            self._gray_mode = "auto"

    def close(self) -> None:
        if self.profile is not None:
            self.pipeline.stop()
            self.profile = None
        try:
            cv2.destroyWindow("Eye-to-Hand Local Camera")
        except Exception:
            pass

    def _load_camera_config(self, camera_config_path: str) -> dict:
        config_path = Path(camera_config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"camera config not found: {camera_config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return config

    def _camera_runtime_config(self, camera_config: dict) -> dict:
        if "camera" in camera_config:
            return camera_config["camera"]
        if "realsense" in camera_config:
            return camera_config["realsense"]
        return camera_config

    def _available_streams(self) -> dict:
        streams = self.camera_root_config.get("streams", {})
        return streams if isinstance(streams, dict) else {}

    def _resolve_stream_name(self, stream_name: str | None) -> str:
        if stream_name is not None:
            return str(stream_name)
        configured_stream = self.camera_config.get("apriltag_stream")
        if configured_stream:
            return str(configured_stream)
        configured_stream = self.camera_config.get("stream_name")
        if configured_stream:
            return str(configured_stream)
        streams = self._available_streams()
        if "color" in streams:
            return "color"
        if "infrared_1" in streams:
            return "infrared_1"
        if "infrared" in streams:
            return "infrared"
        return "color"

    def _resolve_stream_config(self, stream_name: str) -> dict:
        streams = self._available_streams()
        if stream_name in streams and isinstance(streams[stream_name], dict):
            return streams[stream_name]
        stream_cfg = self.camera_config.get("stream", {})
        return stream_cfg if isinstance(stream_cfg, dict) else {}

    def _resolve_stream_type(self, stream_name: str):
        if stream_name == "color":
            return rs.stream.color, None
        if stream_name.startswith("infrared"):
            parts = stream_name.split("_", 1)
            stream_index = 1
            if len(parts) == 2 and parts[1].isdigit():
                stream_index = int(parts[1])
            return rs.stream.infrared, stream_index
        raise ValueError(f"unsupported stream name: {stream_name}")

    def _resolution_override(self, resolution: str | None) -> tuple[int, int] | None:
        if resolution is None:
            return None
        presets = {
            "low": (640, 480),
            "medium": (1280, 720),
            "high": (1920, 1080),
        }
        if resolution not in presets:
            raise ValueError(f"unknown resolution preset: {resolution}")
        return presets[resolution]

    def _desired_stream(self, resolution: str | None) -> tuple[int, int, int, str]:
        stream = self.stream_config
        default_width = 1920 if self.stream_type == rs.stream.color else 640
        default_height = 1080 if self.stream_type == rs.stream.color else 480
        width = int(stream.get("width", default_width))
        height = int(stream.get("height", default_height))
        fps = int(stream.get("fps", 30))
        default_format = "BGR8" if self.stream_type == rs.stream.color else "Y8"
        pixel_format = str(stream.get("format", default_format))
        override = self._resolution_override(resolution)
        if override is not None:
            width, height = override
        return width, height, fps, pixel_format

    def _desired_device(self):
        serial = self.camera_config.get("serial")
        if not serial:
            serial = self.camera_root_config.get("device", {}).get("serial_number")
        return str(serial) if serial else None

    def _find_device(self):
        target_serial = self._desired_device()
        context = rs.context()
        for device in context.query_devices():
            serial = device.get_info(rs.camera_info.serial_number)
            if target_serial is None or serial == target_serial:
                return device
        if target_serial is None:
            return None
        raise RuntimeError(f"realsense device with serial {target_serial} not found")

    def _format_candidates(self, pixel_format: str) -> list[tuple[rs.format, int]]:
        normalized = pixel_format.strip().upper()
        if normalized == "RGB8":
            return [(rs.format.rgb8, cv2.COLOR_RGB2GRAY), (rs.format.bgr8, cv2.COLOR_BGR2GRAY)]
        if normalized == "BGR8":
            return [(rs.format.bgr8, cv2.COLOR_BGR2GRAY), (rs.format.rgb8, cv2.COLOR_RGB2GRAY)]
        return [(rs.format.bgr8, cv2.COLOR_BGR2GRAY), (rs.format.rgb8, cv2.COLOR_RGB2GRAY)]

    def _list_color_profiles(self, device) -> list[tuple[int, int, int, rs.format]]:
        profiles: list[tuple[int, int, int, rs.format]] = []
        if device is None:
            return profiles
        for sensor in device.query_sensors():
            if not sensor.supports(rs.camera_info.name):
                continue
            sensor_name = sensor.get_info(rs.camera_info.name)
            if "RGB Camera" not in sensor_name:
                continue
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() != rs.stream.color:
                    continue
                video_profile = profile.as_video_stream_profile()
                profiles.append((video_profile.width(), video_profile.height(), video_profile.fps(), video_profile.format()))
        return profiles

    def _select_best_profile(self, profiles: list[tuple[int, int, int, rs.format]], width: int, height: int, fps: int) -> tuple[int, int, int, rs.format] | None:
        best_profile = None
        best_score = float("inf")
        for candidate_width, candidate_height, candidate_fps, candidate_format in profiles:
            if candidate_format not in (rs.format.bgr8, rs.format.rgb8):
                continue
            score = abs(candidate_width - width) + abs(candidate_height - height) + abs(candidate_fps - fps) * 10
            if candidate_format == rs.format.bgr8:
                score -= 1.0
            if score < best_score:
                best_score = score
                best_profile = (candidate_width, candidate_height, candidate_fps, candidate_format)
        return best_profile

    def _start_pipeline(self, resolution: str | None) -> None:
        width, height, fps, pixel_format = self._desired_stream(resolution)
        target_serial = self._desired_device()
        self.active_width = width
        self.active_height = height
        self.active_fps = fps
        if self.stream_type == rs.stream.infrared:
            config = rs.config()
            if target_serial is not None:
                config.enable_device(target_serial)
            config.enable_stream(
                rs.stream.infrared,
                self.stream_index if self.stream_index is not None else 1,
                width,
                height,
                rs.format.y8,
                fps,
            )
            try:
                self.profile = self.pipeline.start(config)
                self.color_format = rs.format.y8
                return
            except RuntimeError as exc:
                raise RuntimeError(
                    f"failed to start infrared stream {self.stream_name} at {width}x{height}@{fps}: {exc}"
                ) from exc
        for stream_format, gray_code in self._format_candidates(pixel_format):
            config = rs.config()
            if target_serial is not None:
                config.enable_device(target_serial)
            config.enable_stream(rs.stream.color, width, height, stream_format, fps)
            try:
                self.profile = self.pipeline.start(config)
                self.color_format = stream_format
                self.color_to_gray_code = gray_code
                return
            except RuntimeError:
                continue
        device = self._find_device()
        best_profile = self._select_best_profile(self._list_color_profiles(device), width, height, fps)
        if best_profile is None:
            raise RuntimeError("no supported RGB stream profile found for the selected RealSense device")
        best_width, best_height, best_fps, best_format = best_profile
        config = rs.config()
        if target_serial is not None:
            config.enable_device(target_serial)
        config.enable_stream(rs.stream.color, best_width, best_height, best_format, best_fps)
        self.profile = self.pipeline.start(config)
        self.color_format = best_format
        self.color_to_gray_code = cv2.COLOR_BGR2GRAY if best_format == rs.format.bgr8 else cv2.COLOR_RGB2GRAY
        active_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.active_width = active_stream.width()
        self.active_height = active_stream.height()
        self.active_fps = active_stream.fps()

    def _load_intrinsics_from_config_or_stream(self) -> None:
        try:
            if self.stream_type == rs.stream.color:
                stream_profile = self.profile.get_stream(rs.stream.color)
            else:
                stream_profile = self.profile.get_stream(
                    rs.stream.infrared,
                    self.stream_index if self.stream_index is not None else 1,
                )
            intrinsics = stream_profile.as_video_stream_profile().get_intrinsics()
            self.fx = float(intrinsics.fx)
            self.fy = float(intrinsics.fy)
            self.cx = float(intrinsics.ppx)
            self.cy = float(intrinsics.ppy)
            self.dist_coeffs = np.asarray(intrinsics.coeffs, dtype=np.float64)
            self.intrinsics_source = "realsense_sdk"
        except Exception:
            stream_intrinsics = self.stream_config
            stream_width = int(stream_intrinsics.get("width", -1)) if stream_intrinsics else -1
            stream_height = int(stream_intrinsics.get("height", -1)) if stream_intrinsics else -1
            if stream_intrinsics and "camera_matrix" in stream_intrinsics and \
               stream_width == self.active_width and stream_height == self.active_height:
                self.camera_matrix = np.asarray(stream_intrinsics["camera_matrix"], dtype=np.float64).reshape(3, 3)
                self.fx = float(self.camera_matrix[0, 0])
                self.fy = float(self.camera_matrix[1, 1])
                self.cx = float(self.camera_matrix[0, 2])
                self.cy = float(self.camera_matrix[1, 2])
                coeffs = stream_intrinsics.get("distortion_coefficients", [0.0, 0.0, 0.0, 0.0, 0.0])
                self.dist_coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1)
                self.intrinsics_source = "camera_config_stream"
                return
            intrinsics_cfg = self.camera_config.get("intrinsics", {})
            if intrinsics_cfg and all(key in intrinsics_cfg for key in ("fx", "fy", "cx", "cy")):
                self.fx = float(intrinsics_cfg["fx"])
                self.fy = float(intrinsics_cfg["fy"])
                self.cx = float(intrinsics_cfg["cx"])
                self.cy = float(intrinsics_cfg["cy"])
                coeffs = intrinsics_cfg.get("coeffs", [0.0, 0.0, 0.0, 0.0, 0.0])
                self.dist_coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1)
                self.intrinsics_source = "camera_config_legacy"
            else:
                raise
        self.camera_matrix = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def _wait_for_frames(self):
        last_error = None
        for _ in range(self.frame_retry_count):
            try:
                return self.pipeline.wait_for_frames(timeout_ms=self.frame_timeout_ms)
            except RuntimeError as exc:
                last_error = exc
                time.sleep(0.1)
        if last_error is not None:
            raise last_error
        raise RuntimeError("failed to acquire frames from RealSense pipeline")

    def _save_runtime_intrinsics(self) -> None:
        runtime_path = Path(self.runtime_intrinsics_path)
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        device = self.profile.get_device()
        device_info = {}
        for info_key, field_name in (
            (rs.camera_info.name, "name"),
            (rs.camera_info.serial_number, "serial_number"),
            (rs.camera_info.firmware_version, "firmware_version"),
            (rs.camera_info.product_line, "product_line"),
        ):
            try:
                if device.supports(info_key):
                    device_info[field_name] = device.get_info(info_key)
            except Exception:
                continue
        runtime_intrinsics = {
            "device": device_info,
            "runtime": {
                "stream_name": self.stream_name,
                "width": int(self.active_width),
                "height": int(self.active_height),
                "fps": int(self.active_fps),
                "intrinsics_source": self.intrinsics_source,
                "camera_matrix": self.camera_matrix.tolist(),
                "distortion_coefficients": self.dist_coeffs.reshape(-1).astype(float).tolist(),
                "fx": float(self.fx),
                "fy": float(self.fy),
                "cx": float(self.cx),
                "cy": float(self.cy),
            },
        }
        with runtime_path.open("w", encoding="utf-8") as handle:
            yaml.dump(runtime_intrinsics, handle, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"[LocalAprilTagPoseTracker] intrinsics saved to {runtime_path}")

    def _find_rgb_sensor(self):
        device = self.profile.get_device()
        for sensor in device.query_sensors():
            if sensor.supports(rs.camera_info.name):
                sensor_name = sensor.get_info(rs.camera_info.name)
                if "RGB Camera" in sensor_name:
                    return sensor
        return None

    def _configure_camera_parameters(self) -> None:
        if self.stream_type != rs.stream.color:
            return
        rgb_sensor = self._find_rgb_sensor()
        if rgb_sensor is None:
            return
        sensor_params = self.camera_config.get("rgb_sensor", {})
        if not sensor_params:
            return
        parameter_map = {
            "enable_auto_exposure": rs.option.enable_auto_exposure,
            "enable_auto_white_balance": rs.option.enable_auto_white_balance,
            "exposure": rs.option.exposure,
            "gain": rs.option.gain,
            "white_balance": rs.option.white_balance,
            "brightness": rs.option.brightness,
            "contrast": rs.option.contrast,
            "gamma": rs.option.gamma,
            "saturation": rs.option.saturation,
            "sharpness": rs.option.sharpness,
        }
        for key, option in parameter_map.items():
            if key in sensor_params and rgb_sensor.supports(option):
                try:
                    rgb_sensor.set_option(option, sensor_params[key])
                except Exception:
                    continue

    def _detect_tags(self, image: np.ndarray):
        if image.ndim == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, self._gray_code_primary)
        detections = _run_detector(self.detector, self.detector_backend, gray)
        if image.ndim == 3 and self._gray_mode == "auto" and not detections and self._gray_code_alt is not None:
            gray_alt = cv2.cvtColor(image, self._gray_code_alt)
            detections_alt = _run_detector(self.detector, self.detector_backend, gray_alt)
            if detections_alt:
                gray = gray_alt
                detections = detections_alt
                self._gray_code_primary = self._gray_code_alt
                self._gray_mode = "locked"
        return gray, detections

    def _get_detection_id(self, detection) -> int:
        return _detection_id(detection)

    def _get_corners_from_detection(self, detection) -> np.ndarray:
        return _detection_corners(detection)

    def _refine_corners_subpix(self, gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners_np = corners.astype(np.float32).reshape(-1, 1, 2)
        cv2.cornerSubPix(gray, corners_np, (5, 5), (-1, -1), criteria)
        return corners_np.reshape(-1, 2).astype(np.float64)

    def _estimate_pose(self, corners: np.ndarray) -> tuple[bool, np.ndarray, np.ndarray]:
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.object_points,
            corners,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        return bool(success), rotation_vector, translation_vector

    def _refine_pose(self, corners: np.ndarray, rotation_vector: np.ndarray, translation_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        try:
            rotation_vector_refined, translation_vector_refined = cv2.solvePnPRefineLM(
                self.object_points,
                corners,
                self.camera_matrix,
                self.dist_coeffs,
                rotation_vector,
                translation_vector,
            )
            return rotation_vector_refined, translation_vector_refined
        except Exception:
            return rotation_vector, translation_vector

    def _compute_reprojection_error(self, corners: np.ndarray, rotation_vector: np.ndarray, translation_vector: np.ndarray) -> float:
        projected, _ = cv2.projectPoints(
            self.object_points,
            rotation_vector,
            translation_vector,
            self.camera_matrix,
            self.dist_coeffs,
        )
        projected = projected.reshape(-1, 2)
        diff = corners - projected
        return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

    def _compute_tag_size_px(self, corners: np.ndarray) -> float:
        p0, p1, p2, p3 = corners
        width_top = np.linalg.norm(p1 - p0)
        width_bottom = np.linalg.norm(p2 - p3)
        height_left = np.linalg.norm(p3 - p0)
        height_right = np.linalg.norm(p2 - p1)
        return float((width_top + width_bottom + height_left + height_right) / 4.0)

    def _corners_for_ippe_square(self, corners: np.ndarray) -> np.ndarray:
        corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        return np.array([
            corners[3],
            corners[2],
            corners[1],
            corners[0],
        ], dtype=np.float64)

    def _convert_frame(self, rotation_matrix: np.ndarray, translation: np.ndarray, frame: str) -> tuple[np.ndarray, np.ndarray]:
        if frame in ("opencv", "camera"):
            return rotation_matrix, translation
        if frame == "robot":
            converted_rotation = self.t_cam_to_robot @ rotation_matrix @ self.t_cam_to_robot.T
            converted_translation = self.t_cam_to_robot @ translation
            return converted_rotation, converted_translation
        raise ValueError(f"unknown frame: {frame}")

    def capture_single_with_debug(self, frame: str = "robot"):
        frames = self._wait_for_frames()
        aligned_frames = frames
        if self.stream_type == rs.stream.color:
            image_frame = aligned_frames.get_color_frame()
        else:
            image_frame = aligned_frames.get_infrared_frame(
                self.stream_index if self.stream_index is not None else 1
            )
        if not image_frame:
            return None, None, []
        image = np.asanyarray(image_frame.get_data())
        gray, detections = self._detect_tags(image)
        target = None
        for detection in detections:
            if self._get_detection_id(detection) == self.tag_id:
                target = detection
                break
        if target is None:
            return None, image, detections
        detected_corners = self._get_corners_from_detection(target)
        detected_corners = self._refine_corners_subpix(gray, detected_corners)
        corners = detected_corners
        pnp_corners = self._corners_for_ippe_square(corners)
        success, rotation_vector, translation_vector = self._estimate_pose(pnp_corners)
        if not success:
            return None, image, detections
        rotation_vector, translation_vector = self._refine_pose(pnp_corners, rotation_vector, translation_vector)
        reprojection_error = self._compute_reprojection_error(pnp_corners, rotation_vector, translation_vector)
        tag_size_px = self._compute_tag_size_px(corners)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        rotation_matrix, translation = self._convert_frame(rotation_matrix, translation_vector, frame)
        roll, pitch, yaw = _rotation_to_euler_zyx(rotation_matrix)
        return {
            "R": rotation_matrix,
            "t_m": translation,
            "T": _transform_from_rt(rotation_matrix, translation),
            "position_mm": (translation.reshape(3) * 1000.0).tolist(),
            "position_camera_mm": (translation_vector.reshape(3) * 1000.0).tolist(),
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "reproj_error": reprojection_error,
            "tag_size_px": tag_size_px,
            "pnp_z_mm": float(translation_vector.reshape(3)[2] * 1000.0),
            "detected_corners": detected_corners.tolist(),
            "pnp_corners": pnp_corners.tolist(),
            "frame": frame,
            "stream_name": self.stream_name,
        }, image, detections

    def get_tag_pose(self, frame: str = "robot") -> dict | None:
        pose, _, _ = self.capture_single_with_debug(frame=frame)
        return pose

    def get_preview_frame(self, frame: str = "robot", reproj_error_max: float | None = None, accepted_frames: int = 0) -> np.ndarray | None:
        pose, image, detections = self.capture_single_with_debug(frame=frame)
        if image is None:
            return None
        status_text = "no target"
        if pose is not None:
            status_text = "candidate"
            if reproj_error_max is None or pose["reproj_error"] <= reproj_error_max:
                status_text = "accepted"
            else:
                status_text = "rejected"
        return _format_preview_image(image, detections, self.tag_id, pose, accepted_frames, status_text)

    def capture_pose(self, duration_s: float = 0.5, min_frames: int = 10, max_frames: int = 60, frame: str = "robot", reproj_error_max: float | None = None, include_debug_image: bool = False, show_preview: bool = False, preview_window_name: str = "Eye-to-Hand Local Camera") -> dict | None:
        start_time = time.time()
        if show_preview:
            cv2.namedWindow(preview_window_name, cv2.WINDOW_NORMAL)
        rotations: list[np.ndarray] = []
        translations: list[np.ndarray] = []
        reprojection_errors: list[float] = []
        tag_sizes_px: list[float] = []
        pnp_zs_mm: list[float] = []
        best_debug_image = None
        best_debug_error = None
        while time.time() - start_time < duration_s and len(rotations) < max_frames:
            pose, image, detections = self.capture_single_with_debug(frame=frame)
            if show_preview and image is not None:
                status_text = "no target"
                if pose is not None:
                    status_text = "candidate"
                    if reproj_error_max is None or pose["reproj_error"] <= reproj_error_max:
                        status_text = "accepted"
                    else:
                        status_text = "rejected"
                preview = _format_preview_image(image, detections, self.tag_id, pose, len(rotations), status_text)
                cv2.imshow(preview_window_name, preview)
                cv2.waitKey(1)
            if pose is None:
                continue
            if reproj_error_max is not None and pose["reproj_error"] > reproj_error_max:
                continue
            rotations.append(np.asarray(pose["R"], dtype=np.float64))
            translations.append(np.asarray(pose["t_m"], dtype=np.float64))
            reprojection_errors.append(float(pose["reproj_error"]))
            tag_sizes_px.append(float(pose["tag_size_px"]))
            if pose.get("pnp_z_mm") is not None:
                pnp_zs_mm.append(float(pose["pnp_z_mm"]))
            if include_debug_image and image is not None:
                current_error = float(pose["reproj_error"])
                if best_debug_error is None or current_error < best_debug_error:
                    best_debug_error = current_error
                    best_debug_image = _draw_detection_debug(image, detections, self.tag_id)
        if len(rotations) < min_frames:
            return None
        translation_avg = np.mean(np.hstack(translations), axis=1, keepdims=True)
        quaternions = [_quat_from_rotation(rotation) for rotation in rotations]
        rotation_avg = _rotation_from_quat(_average_quaternions(quaternions))
        roll, pitch, yaw = _rotation_to_euler_zyx(rotation_avg)
        result = {
            "R": rotation_avg,
            "t_m": translation_avg,
            "T": _transform_from_rt(rotation_avg, translation_avg),
            "position_mm": (translation_avg.reshape(3) * 1000.0).tolist(),
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "reproj_error_mean": float(np.mean(reprojection_errors)),
            "tag_size_px_mean": float(np.mean(tag_sizes_px)),
            "frames_used": len(rotations),
            "frame": frame,
            "stream_name": self.stream_name,
        }
        if pnp_zs_mm:
            result["pnp_z_mm_mean"] = float(np.mean(pnp_zs_mm))
        if include_debug_image and best_debug_image is not None:
            result["debug_image"] = best_debug_image
        if show_preview and best_debug_image is not None:
            final_preview = _format_preview_image(best_debug_image, [], self.tag_id, {"reproj_error": result["reproj_error_mean"], "tag_size_px": result["tag_size_px_mean"]}, len(rotations), "fused")
            cv2.imshow(preview_window_name, final_preview)
            cv2.waitKey(1)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Local AprilTag pose tracker")
    parser.add_argument("--camera-config", default="config/camera_intrinsics.yaml")
    parser.add_argument("--tag-size-mm", type=float, default=101.0)
    parser.add_argument("--tag-family", default="tag36h11")
    parser.add_argument("--tag-id", type=int, default=0)
    parser.add_argument("--resolution", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--stream-name", default="color")
    parser.add_argument("--mode", choices=["intrinsics", "single", "capture"], default="intrinsics")
    parser.add_argument("--frame", choices=["camera", "robot"], default="robot")
    parser.add_argument("--duration-s", type=float, default=0.5)
    parser.add_argument("--min-frames", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--reproj-error-max", type=float, default=None)
    parser.add_argument("--debug-image", default="debug_output/apriltag_single_debug.png")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    tracker = LocalAprilTagPoseTracker(
        camera_config_path=args.camera_config,
        tag_size_mm=args.tag_size_mm,
        tag_family=args.tag_family,
        tag_id=args.tag_id,
        resolution=args.resolution,
        stream_name=args.stream_name,
    )
    try:
        print(f"[LocalAprilTagPoseTracker] intrinsics source: {tracker.intrinsics_source}")
        print(f"[LocalAprilTagPoseTracker] stream: {tracker.stream_name} {tracker.active_width}x{tracker.active_height}@{tracker.active_fps}")
        print(f"[LocalAprilTagPoseTracker] tag_family={tracker.tag_family}, tag_id={tracker.tag_id}, tag_size_mm={tracker.tag_size_mm:.3f}")
        print(f"[LocalAprilTagPoseTracker] fx={tracker.fx:.6f}, fy={tracker.fy:.6f}, cx={tracker.cx:.6f}, cy={tracker.cy:.6f}")
        print(f"[LocalAprilTagPoseTracker] runtime intrinsics file: {tracker.runtime_intrinsics_path}")
        result = None
        if args.mode == "single":
            result, debug_image, detections = tracker.capture_single_with_debug(frame=args.frame)
            if debug_image is not None:
                debug_path = Path(args.debug_image)
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                debug_canvas = _draw_detection_debug(debug_image, detections, tracker.tag_id)
                cv2.imwrite(str(debug_path), debug_canvas)
                print(f"[LocalAprilTagPoseTracker] debug image saved to {debug_path}")
            if result is None:
                print("[LocalAprilTagPoseTracker] no AprilTag pose detected in single frame")
            else:
                _print_pose_result(result, "single frame pose")
        elif args.mode == "capture":
            result = tracker.capture_pose(
                duration_s=args.duration_s,
                min_frames=args.min_frames,
                max_frames=args.max_frames,
                frame=args.frame,
                reproj_error_max=args.reproj_error_max,
            )
            if result is None:
                print("[LocalAprilTagPoseTracker] failed to collect enough valid AprilTag detections")
            else:
                _print_pose_result(result, "fused pose")
        if args.output is not None and result is not None:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as handle:
                yaml.dump(_serialize_pose_result(result), handle, default_flow_style=False, allow_unicode=True, sort_keys=False)
            print(f"[LocalAprilTagPoseTracker] pose result saved to {output_path}")
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
