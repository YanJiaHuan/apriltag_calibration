# Hand-Eye Calibration Upgrade Design

**Date**: 2026-04-21  
**Scope**: Add eye-to-hand calibration, clean up code duplication, improve interfaces, write README

---

## Background

The existing codebase implements **eye-in-hand** calibration (camera mounted on robot end-effector) using a WebSocket-based architecture where the camera runs on a remote machine. The system works and achieves ~4mm position std dev.

New requirements:
- Add **eye-to-hand** calibration (camera fixed externally, tag on end-effector)
- Support local camera (camera and robot controller on the same machine, no WebSocket needed)
- Expose clean Python APIs so scripts can import functions directly
- Eliminate duplicated math code across files
- Add per-camera config files (D435, D405, D415)
- Write README with quick-start tutorial

---

## Coordinate Systems and Units

### Units (must be preserved at all module boundaries)
- Translation: **millimeters (mm)**
- Rotation: **radians (rad)**
- Rotation representation: **Euler XYZ**

### Coordinate frames
- **OpenCV camera frame**: X+ right, Y+ down, Z+ forward
- **RealMan robot frame** (base and end both use this): X+ right, Y+ forward (out), Z+ up
- **Conversion matrix** (opencv → robot): `R_map = [[1,0,0],[0,0,1],[0,-1,0]]`
  - X_robot = X_opencv
  - Y_robot = Z_opencv
  - Z_robot = -Y_opencv
  - det(R_map) = 1 (pure rotation, not reflection)

---

## Eye-in-Hand Math (existing, AX=XB)

```
T[end|base]_i  @  T[cam|end]  =  T[cam|end]  @  T[tag|cam]_i
```

- Known per sample: T[end|base]_i (robot pose), T[tag|cam]_i (AprilTag detection)
- Unknown (solve once): T[cam|end] = camera relative to end-effector
- Side output: T[tag|base] = estimated fixed tag pose in base frame
- Solver: `cv2.calibrateHandEye`

## Eye-to-Hand Math (new, AX=YB)

```
T[cam|base]  @  T[tag|cam]_i  =  T[end|base]_i  @  T[tag|end]
```

Rearranged as AX=YB:
- A_i = T[end|base]_i, B_i = T[tag|cam]_i
- **X = T[tag|end]** (tag relative to end, unknown — tag placement on box is imprecise)
- **Y = T[cam|base]** (camera relative to base, the primary result)
- Solver: `cv2.calibrateRobotWorldHandEye`

Both unknowns are solved simultaneously from >= 3 diverse robot poses.

---

## File Structure

### New files

| File | Purpose |
|------|---------|
| `scripts/handeye_utils.py` | Shared math: euler↔rotation, make_transform, invert_transform, opencv↔robot conversion |
| `scripts/collect_eye_to_hand.py` | Local data collection for eye-to-hand (no WebSocket) |
| `scripts/eye_to_hand_calibrate.py` | Eye-to-hand solver CLI + `solve_eye_to_hand()` API |
| `config/realsense_d405.json` | D405 config (new) |
| `config/realsense_d415.json` | D415 config (new) |
| `README.md` | Quick-start tutorial |

### Modified files (minimal changes)

| File | Change |
|------|--------|
| `scripts/handeye_calibrate.py` | Import from handeye_utils, remove duplicate math, expose `solve_eye_in_hand()` |
| `scripts/validate_handeye_calibration.py` | Import from handeye_utils, remove duplicate math |
| `scripts/compare_calibration_methods.py` | Rewrite: call `solve_eye_in_hand()` directly, no subprocess |

### Unchanged files

| File | Reason |
|------|--------|
| `scripts/remote_apriltag_server.py` | WebSocket server, complete and correct |
| `scripts/robot_eye_in_hand_client.py` | WebSocket client, complete and correct |
| `tools/reset_arm.py` | Verified, do-not-modify per project docs |
| `config/realsense_D435.json` | D435 config, kept as-is (remote_apriltag_server.py default points here) |
| `config/vision_config.json` | Tag config, unchanged |
| `config/websocket.json` | Network config, unchanged |

---

## Interface Definitions

### `scripts/handeye_utils.py` (public API)

```python
euler_xyz_to_rot(rx, ry, rz) -> np.ndarray         # Euler XYZ → 3x3
rot_to_euler_xyz(r) -> tuple[float, float, float]   # 3x3 → Euler XYZ
make_transform(r, t) -> np.ndarray                  # → 4x4 homogeneous
invert_transform(tmat) -> np.ndarray                # invert 4x4
opencv_to_robot_rotation() -> np.ndarray            # R_map 3x3
convert_opencv_pose_to_robot(r, t) -> (R, t)        # apply R_map to pose
```

Internal helpers (not intended for external use):
```python
rot_to_quat, quat_to_rot, average_rotations
```

### `scripts/handeye_calibrate.py` — new importable API

```python
def solve_eye_in_hand(
    csv_path: str,
    method: str = "tsai",
    max_reproj_error: float = 2.0,
    min_samples: int = 3,
) -> dict:
    """
    Returns:
        {
            "r_cam2end": np.ndarray,   # 3x3
            "t_cam2end": np.ndarray,   # (3,) mm
            "r_tag2base": np.ndarray,  # 3x3
            "t_tag2base": np.ndarray,  # (3,) mm
            "n_samples": int,
        }
    """
```

### `scripts/eye_to_hand_calibrate.py` — new importable API

```python
def solve_eye_to_hand(
    csv_path: str,
    method: str = "tsai",
    max_reproj_error: float = 2.0,
    min_samples: int = 3,
) -> dict:
    """
    Returns:
        {
            "r_cam2base": np.ndarray,  # 3x3
            "t_cam2base": np.ndarray,  # (3,) mm
            "r_tag2end": np.ndarray,   # 3x3
            "t_tag2end": np.ndarray,   # (3,) mm
            "n_samples": int,
        }
    """
```

### `scripts/collect_eye_to_hand.py` — CLI

```
python3 scripts/collect_eye_to_hand.py \
    --robot-ip <ip> \
    --robot-port <port> \
    --realsense-config config/realsense_d405.json \
    --vision-config config/vision_config.json \
    --count 20 \
    --manual
```

CSV format identical to existing `robot_eye_in_hand.csv` so the same solver infrastructure can be reused.

---

## Camera Config Files

Each file has the same JSON structure. Fields may differ per model (resolution, sensor options).

```json
{
  "quality": { "width": 1280, "height": 720, "fps": 30 },
  "rgb_sensor": {
    "enable_auto_exposure": 0,
    "exposure": 150,
    "gain": 10
  }
}
```

D405 note: supports closer working distances (~7cm min), same pyrealsense2 API.  
D415 note: wider FOV, standard working distance.  
All three use identical `init_realsense(config_path)` call — only the config file differs.

---

## README Structure

1. **Quick Start**
   - Eye-in-Hand (WebSocket / remote camera mode)
   - Eye-to-Hand (local mode, new)
2. **What Each Script Does and Why**
3. **Coordinate Systems and Units** (brief)
4. **How to Verify Calibration Success**
5. **How to Improve Accuracy**
6. **Camera Config Selection** (D435 / D405 / D415)

---

## Validation / Proof of Calibration Success

Eye-in-hand (existing): `validate_handeye_calibration.py`
- Move robot to N different poses, capture tag each time
- Compute T[tag|base] from each pose using calibration result
- Pass criterion: position std dev < 5mm, max deviation < 10mm

Eye-to-hand (new): same principle
- Move robot to N poses, capture tag
- Compute T[cam|base] consistency check: tag position in base frame should be constant
- Same pass criterion

---

## Tests

All new functions get unit tests in `/test`:
- `test_handeye_utils.py`: test shared math functions
- `test_eye_to_hand_calibrate.py`: test solver with synthetic data
- `test_collect_eye_to_hand.py`: test pure utility functions (no hardware)

Existing tests remain unchanged.
