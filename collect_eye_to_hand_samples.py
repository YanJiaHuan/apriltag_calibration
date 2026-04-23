#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cv2
import json
import select
import sys
import time
from pathlib import Path

import numpy as np

WORKSPACE_ROOT = Path("/home/tr4/workspace")
JS_PROJECT_ROOT = WORKSPACE_ROOT / "JS_Project"
DEFAULT_OUTPUT_PATH = str(WORKSPACE_ROOT / "eye_to_hand" / "eye_to_hand_samples.json")
DEFAULT_CONFIG_PATH = str(JS_PROJECT_ROOT / "config" / "vision_config.json")
DEFAULT_LOCAL_CAMERA_CONFIG_PATH = str(WORKSPACE_ROOT / "eye_to_hand" / "head_camera.yaml")
DEFAULT_TAG_FAMILY = "tag36h11"
DEFAULT_TAG_SIZE_MM = 101.0
DEFAULT_SAVE_FRAME = "robot"
DEFAULT_DURATION_S = 1.5
DEFAULT_MIN_FRAMES = 5
DEFAULT_MAX_FRAMES = 60
DEFAULT_REPROJ_ERROR_MAX = 5.0
DEFAULT_ROBOT_STATE_RETRY_COUNT = 3
DEFAULT_ROBOT_STATE_RETRY_DELAY_S = 0.2
T_OPENCV_TO_ROBOT = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
], dtype=np.float64)

if str(JS_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(JS_PROJECT_ROOT))

from src.vision.robot_state import RobotStateManager
from src.vision.tag_pose_ws import TagPoseWebSocketClient
from src.vision.vision_config import load_vision_config

from apriltag_tracker import LocalAprilTagPoseTracker


class TagPoseSource:
    def __init__(self, config_path: str, local_camera_config_path: str, use_websocket: bool, tag_id: int | None, tag_size_mm: float | None, tag_family: str | None, show_preview: bool, preview_window_name: str):
        self.config_path = config_path
        self.local_camera_config_path = local_camera_config_path
        self.config = load_vision_config(config_path)
        self.camera_cfg = self.config.get("camera", {})
        self.capture_cfg = self.config.get("capture", {})
        self.tag_cfg = self.config.get("tag", {})
        self.tag_id = int(tag_id if tag_id is not None else self.tag_cfg.get("id", 0))
        self.tag_size_mm = float(tag_size_mm if tag_size_mm is not None else self.tag_cfg.get("size_mm", DEFAULT_TAG_SIZE_MM))
        self.tag_family = str(tag_family if tag_family is not None else self.tag_cfg.get("family", DEFAULT_TAG_FAMILY))
        self.use_websocket = bool(use_websocket)
        self.show_preview = bool(show_preview) and not self.use_websocket
        self.preview_window_name = str(preview_window_name)
        self.ws_client = None
        self.tracker = None

        if self.use_websocket:
            self.ws_client = TagPoseWebSocketClient(config=self.config, config_path=config_path)
        else:
            self.tracker = LocalAprilTagPoseTracker(
                camera_config_path=self.local_camera_config_path,
                tag_size_mm=self.tag_size_mm,
                tag_family=self.tag_family,
                tag_id=self.tag_id,
                resolution=self.camera_cfg.get("resolution"),
            )

    def detect_tag(self, duration_s: float, min_frames: int, max_frames: int, reproj_error_max: float, frame: str) -> dict | None:
        if self.ws_client is not None:
            tag_data = self.ws_client.request_pose(
                duration_s=duration_s,
                min_frames=min_frames,
                max_frames=max_frames,
                reproj_error_max=reproj_error_max,
                frame=frame,
            )
            if tag_data is None:
                return None
            t_camera_tag = np.asarray(tag_data["T_tag_cam"], dtype=float)
            t_tag_cam = np.asarray(tag_data["T_cam_tag"], dtype=float)
            normalized = dict(tag_data)
            normalized["T_camera_tag"] = t_camera_tag
            normalized["T_cam_tag"] = t_camera_tag
            normalized["T_tag_cam"] = t_tag_cam
            if "position_mm" in normalized:
                normalized["position_mm"] = (t_camera_tag[:3, 3] * 1000.0).astype(np.float64)
            return normalized

        pose = self.tracker.capture_pose(
            duration_s=duration_s,
            min_frames=min_frames,
            max_frames=max_frames,
            frame=frame,
            reproj_error_max=reproj_error_max,
            include_debug_image=True,
            show_preview=self.show_preview,
            preview_window_name=self.preview_window_name,
        )
        if pose is None:
            return None
        t_camera_tag = np.asarray(pose["T"], dtype=float)
        t_tag_cam = np.linalg.inv(t_camera_tag)
        return {
            "tag_id": self.tag_id,
            "frame": pose.get("frame", frame),
            "T_tag_cam": t_tag_cam,
            "T_cam_tag": t_camera_tag,
            "T_camera_tag": t_camera_tag,
            "position_mm": np.asarray(pose["position_mm"], dtype=np.float64),
            "rpy_rad": np.array([pose["roll"], pose["pitch"], pose["yaw"]], dtype=np.float64),
            "reproj_error": float(pose.get("reproj_error_mean", pose.get("reproj_error", 0.0))),
            "frames_used": int(pose.get("frames_used", 1)),
            "tag_size_px": float(pose.get("tag_size_px_mean", pose.get("tag_size_px", 0.0))),
            "debug_image": pose.get("debug_image"),
        }

    def close(self) -> None:
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None

    def update_preview(self, frame: str, reproj_error_max: float, accepted_frames: int = 0) -> None:
        if self.tracker is None or not self.show_preview:
            return
        cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)
        preview = self.tracker.get_preview_frame(
            frame=frame,
            reproj_error_max=reproj_error_max,
            accepted_frames=accepted_frames,
        )
        if preview is None:
            return
        cv2.imshow(self.preview_window_name, preview)
        cv2.waitKey(1)


def load_existing_samples(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []
    with output_path.open("r", encoding="utf-8") as handle:
        samples = json.load(handle)
    if not isinstance(samples, list):
        raise ValueError(f"existing samples file is invalid: {output_path}")
    return samples


def save_samples(output_path: Path, samples: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(samples, indent=2), encoding="utf-8")


def wait_for_user_command(tag_source: TagPoseSource, frame: str, reproj_error_max: float, sample_index: int) -> str:
    prompt = f"\n[{sample_index:02d}] ready> "
    if not tag_source.show_preview:
        return input(prompt).strip()

    print(prompt, end="", flush=True)
    while True:
        tag_source.update_preview(frame=frame, reproj_error_max=reproj_error_max, accepted_frames=0)
        ready, _, _ = select.select([sys.stdin], [], [], 0.03)
        if ready:
            line = sys.stdin.readline()
            if line == "":
                raise KeyboardInterrupt()
            return line.strip()


def is_escape_sequence_input(user_input: str) -> bool:
    return bool(user_input) and user_input.startswith("\x1b")


def get_robot_state_with_retry(robot: RobotStateManager, attempts: int = DEFAULT_ROBOT_STATE_RETRY_COUNT, delay_s: float = DEFAULT_ROBOT_STATE_RETRY_DELAY_S) -> dict | None:
    last_error = None
    for attempt_idx in range(attempts):
        try:
            return robot.get_current_pose()
        except Exception as exc:
            last_error = exc
            if attempt_idx + 1 < attempts:
                print(f"  [WARN] Failed to read robot pose ({exc}); retry {attempt_idx + 2}/{attempts}...")
                time.sleep(delay_s)
    print(f"  [SKIP] Failed to read robot pose after {attempts} attempts: {last_error}")
    return None


def warmup_preview(tag_source: TagPoseSource, frame: str, reproj_error_max: float, duration_s: float = 0.8) -> None:
    if not tag_source.show_preview:
        return
    start_time = time.time()
    while time.time() - start_time < duration_s:
        tag_source.update_preview(frame=frame, reproj_error_max=reproj_error_max, accepted_frames=0)
        time.sleep(0.03)


def save_debug_image(output_path: Path, timestamp_s: float, debug_image: np.ndarray | None) -> str | None:
    if debug_image is None:
        return None
    debug_dir = output_path.parent / "debug_frames"
    debug_dir.mkdir(parents=True, exist_ok=True)
    image_path = debug_dir / f"{timestamp_s:.6f}.png"
    cv2.imwrite(str(image_path), debug_image)
    return str(image_path)


def convert_tag_transform_to_robot_frame(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=float)
    converted = np.eye(4, dtype=np.float64)
    converted[:3, :3] = T_OPENCV_TO_ROBOT @ transform[:3, :3] @ T_OPENCV_TO_ROBOT.T
    converted[:3, 3] = T_OPENCV_TO_ROBOT @ transform[:3, 3]
    return converted


def normalize_tag_data_to_robot_frame(tag_data: dict) -> dict:
    current_frame = str(tag_data.get("frame", DEFAULT_SAVE_FRAME))
    if current_frame == DEFAULT_SAVE_FRAME:
        normalized = dict(tag_data)
        normalized["frame"] = DEFAULT_SAVE_FRAME
        normalized["T_camera_tag"] = np.asarray(normalized["T_camera_tag"], dtype=float)
        normalized["T_cam_tag"] = np.asarray(normalized["T_camera_tag"], dtype=float)
        normalized["T_tag_cam"] = np.asarray(normalized["T_tag_cam"], dtype=float)
        return normalized
    if current_frame not in ("opencv", "camera"):
        raise ValueError(f"unsupported tag frame for sample saving: {current_frame}")

    t_camera_tag_robot = convert_tag_transform_to_robot_frame(np.asarray(tag_data["T_camera_tag"], dtype=float))
    t_tag_cam_robot = np.linalg.inv(t_camera_tag_robot)
    normalized = dict(tag_data)
    normalized["frame"] = DEFAULT_SAVE_FRAME
    normalized["T_tag_cam"] = t_tag_cam_robot
    normalized["T_cam_tag"] = t_camera_tag_robot
    normalized["T_camera_tag"] = t_camera_tag_robot
    if "position_mm" in normalized:
        normalized["position_mm"] = (t_camera_tag_robot[:3, 3] * 1000.0).astype(np.float64)
    return normalized


def build_sample(sample_index: int, note: str | None, frame: str, robot_state: dict, tag_data: dict) -> dict:
    timestamp_s = float(time.time())
    return {
        "sample_index": sample_index,
        "mode": "eye_to_hand",
        "timestamp_s": timestamp_s,
        "frame": frame,
        "note": note,
        "T_base_flange": np.asarray(robot_state["T"], dtype=float).tolist(),
        "T_tag_cam": np.asarray(tag_data["T_tag_cam"], dtype=float).tolist(),
        "T_cam_tag": np.asarray(tag_data["T_cam_tag"], dtype=float).tolist(),
        "T_camera_tag": np.asarray(tag_data["T_camera_tag"], dtype=float).tolist(),
        "tag_frame": tag_data.get("frame", frame),
        "robot_pose_m_rad": np.asarray(robot_state["pose_m_rad"], dtype=float).tolist(),
        "xyz_mm": np.asarray(robot_state["xyz_mm"], dtype=float).tolist(),
        "rpy_rad": np.asarray(robot_state["rpy_rad"], dtype=float).tolist(),
        "joint_deg": np.asarray(robot_state["joint_deg"], dtype=float).tolist(),
        "reproj_error": float(tag_data["reproj_error"]),
        "frames_used": int(tag_data["frames_used"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect eye-to-hand calibration samples")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Path to save eye-to-hand sample JSON")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Vision config JSON path")
    parser.add_argument("--local-camera-config", default=DEFAULT_LOCAL_CAMERA_CONFIG_PATH, help="Local RealSense camera YAML used in local-camera mode")
    parser.add_argument("--robot-ip", default="10.168.1.18")
    parser.add_argument("--frame", default=None, help="Tag pose output frame, default from vision config")
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--reproj-error-max", type=float, default=DEFAULT_REPROJ_ERROR_MAX)
    parser.add_argument("--tag-id", type=int, default=None)
    parser.add_argument("--tag-size-mm", type=float, default=DEFAULT_TAG_SIZE_MM)
    parser.add_argument("--tag-family", default=DEFAULT_TAG_FAMILY)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--show-preview", dest="show_preview", action="store_true")
    parser.add_argument("--no-preview", dest="show_preview", action="store_false")
    parser.add_argument("--preview-window-name", default="Eye-to-Hand Local Camera")
    parser.add_argument("--use-websocket", dest="use_websocket", action="store_true")
    parser.add_argument("--local-camera", dest="use_websocket", action="store_false")
    parser.set_defaults(use_websocket=False)
    parser.set_defaults(show_preview=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    existing_samples = load_existing_samples(output_path) if args.append else []

    tag_source = TagPoseSource(
        config_path=args.config,
        local_camera_config_path=args.local_camera_config,
        use_websocket=args.use_websocket,
        tag_id=args.tag_id,
        tag_size_mm=args.tag_size_mm,
        tag_family=args.tag_family,
        show_preview=args.show_preview,
        preview_window_name=args.preview_window_name,
    )
    robot = RobotStateManager(robot_ip=args.robot_ip)

    frame = args.frame or DEFAULT_SAVE_FRAME
    duration_s = float(args.duration_s)
    min_frames = int(args.min_frames)
    max_frames = int(args.max_frames)
    reproj_error_max = float(args.reproj_error_max)

    samples = list(existing_samples)

    print("=" * 72)
    print("EYE-TO-HAND SAMPLE COLLECTION")
    print("=" * 72)
    print(f"output:        {output_path}")
    print(f"vision config: {args.config}")
    if not tag_source.use_websocket:
        print(f"camera yaml:   {args.local_camera_config}")
    print(f"save frame:    {DEFAULT_SAVE_FRAME}")
    print(f"tag source:    {'websocket' if tag_source.use_websocket else 'local_camera'}")
    print(f"tag family:    {tag_source.tag_family}")
    print(f"tag size mm:   {tag_source.tag_size_mm:.3f}")
    if not tag_source.use_websocket:
        print(f"preview:       {'on' if tag_source.show_preview else 'off'}")
    print(f"existing:      {len(existing_samples)} samples")
    print()
    print("Assumption:")
    print("  - camera is fixed in the environment")
    print("  - calibration board / tag is rigidly mounted on the robot flange")
    print("  - move robot to a pose, wait until stable, then capture one sample")
    print()
    print("Commands:")
    print("  - press Enter to capture current pose")
    print("  - input text to store it as note for the next captured sample")
    print("  - input 's' to save summary and quit")
    print("  - input 'q' to quit immediately (captured samples are already saved)")

    if tag_source.show_preview:
        print("Opening preview window...")
        warmup_preview(tag_source=tag_source, frame=frame, reproj_error_max=reproj_error_max)

    try:
        while True:
            user_input = wait_for_user_command(
                tag_source=tag_source,
                frame=frame,
                reproj_error_max=reproj_error_max,
                sample_index=len(samples) + 1,
            )

            if user_input.lower() == "q":
                print(f"Quit. Current samples on disk: {len(samples)}")
                return 0
            if user_input.lower() == "s":
                save_samples(output_path, samples)
                print(f"Saved {len(samples)} samples to: {output_path}")
                break
            if is_escape_sequence_input(user_input):
                print("  [INFO] Ignored terminal navigation key input.")
                tag_source.update_preview(frame=frame, reproj_error_max=reproj_error_max, accepted_frames=0)
                continue

            note = user_input if user_input else None
            robot_state = get_robot_state_with_retry(robot)
            if robot_state is None:
                tag_source.update_preview(frame=frame, reproj_error_max=reproj_error_max, accepted_frames=0)
                continue
            tag_data = tag_source.detect_tag(
                duration_s=duration_s,
                min_frames=min_frames,
                max_frames=max_frames,
                reproj_error_max=reproj_error_max,
                frame=frame,
            )

            if tag_data is None:
                print("  [SKIP] Tag not detected or not enough valid frames.")
                tag_source.update_preview(frame=frame, reproj_error_max=reproj_error_max, accepted_frames=0)
                continue

            tag_data = normalize_tag_data_to_robot_frame(tag_data)

            sample = build_sample(
                sample_index=len(samples) + 1,
                note=note,
                frame=DEFAULT_SAVE_FRAME,
                robot_state=robot_state,
                tag_data=tag_data,
            )
            debug_image_path = save_debug_image(output_path, sample["timestamp_s"], tag_data.get("debug_image"))
            if debug_image_path is not None:
                sample["debug_image_path"] = debug_image_path
            samples.append(sample)
            save_samples(output_path, samples)

            xyz_mm = np.asarray(robot_state["xyz_mm"], dtype=float)
            print("  [OK] Sample captured")
            print(f"       base->flange xyz: [{xyz_mm[0]:.1f}, {xyz_mm[1]:.1f}, {xyz_mm[2]:.1f}] mm")
            print(f"       reproj error:     {sample['reproj_error']:.3f} px")
            print(f"       frames used:      {sample['frames_used']}")
            print(f"       saved samples:    {len(samples)}")
            tag_source.update_preview(frame=frame, reproj_error_max=reproj_error_max, accepted_frames=sample['frames_used'])

        print()
        print("Next:")
        print(f"  python3 /home/tr4/workspace/eye_to_hand/solve_eye_to_hand_refined.py --samples {output_path}")
        return 0
    finally:
        tag_source.close()


if __name__ == "__main__":
    raise SystemExit(main())
