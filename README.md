# AprilTag Hand-Eye Calibration

Calibration tools for Realman robotic arms using AprilTag fiducial markers.
Supports two configurations:

- **Eye-in-Hand**: camera mounted on the robot end-effector, tag fixed in world
- **Eye-to-Hand**: camera fixed in world, tag mounted on the robot end-effector

---

## Quick Start

### Prerequisites

```bash
pip install numpy opencv-python pyrealsense2 websockets
# Build and install the apriltag Python binding from third_party/apriltag
cd third_party/apriltag && mkdir build && cd build
cmake .. && make -j4
cd ../../..
```

---

### Eye-in-Hand (camera on robot, tag fixed, WebSocket mode)

This is the original setup. The camera machine and robot controller can be on
separate hosts connected by a network.

**Step 1 — Start the camera server** (on the machine with the RealSense):

```bash
python3 scripts/remote_apriltag_server.py \
    --realsense-config config/realsense_D435.json \
    --vision-config config/vision_config.json \
    --websocket-config config/websocket.json
```

**Step 2 — Collect calibration data** (on the robot controller host):

Move the robot to 15–30 diverse poses. For each pose the script captures
the tag detection and records the robot end-effector pose.

```bash
python3 scripts/robot_eye_in_hand_client.py \
    --robot-ip 10.168.1.18 \
    --robot-port 8080 \
    --websocket-config config/websocket.json \
    --count 20 \
    --manual
```

Data is saved to `data/robot_eye_in_hand.csv`.

**Step 3 — Solve calibration**:

```bash
python3 scripts/handeye_calibrate.py \
    --input-csv data/robot_eye_in_hand.csv \
    --method tsai
```

Output: `camera|end (T_end_cam)` translation in mm and rotation in rad.

**Step 4 — Compare all methods** (optional):

```bash
python3 scripts/compare_calibration_methods.py \
    --input-csv data/robot_eye_in_hand.csv
```

**Step 5 — Validate**:

```bash
python3 scripts/validate_handeye_calibration.py \
    --robot-ip 10.168.1.18 \
    --robot-port 8080 \
    --websocket-config config/websocket.json \
    --cam-in-end-x <X> --cam-in-end-y <Y> --cam-in-end-z <Z> \
    --cam-in-end-rx <RX> --cam-in-end-ry <RY> --cam-in-end-rz <RZ> \
    --num-samples 8 \
    --manual
```

Pass criterion: position std dev < 5 mm, max deviation < 10 mm.

---

### Eye-to-Hand (camera fixed, tag on robot end-effector, local mode)

Camera and robot controller are on the same machine. No WebSocket needed.

**Step 1 — Collect calibration data**:

Mount the AprilTag on the robot end-effector. Move the robot to 15–30 diverse
poses while the fixed camera observes the tag.

```bash
python3 scripts/collect_eye_to_hand.py \
    --robot-ip 10.168.1.18 \
    --robot-port 8080 \
    --realsense-config config/realsense_d405.json \
    --vision-config config/vision_config.json \
    --count 20 \
    --manual
```

Data is saved to `data/eye_to_hand.csv`.

**Step 2 — Solve calibration**:

```bash
python3 scripts/eye_to_hand_calibrate.py \
    --input-csv data/eye_to_hand.csv \
    --method shah
```

Output:
- `camera|base (T_cam2base)`: camera pose relative to robot base
- `tag|end (T_tag2end)`: tag pose relative to end-effector (solved simultaneously)

---

## What Each Script Does and Why

| Script | Role | Why |
|--------|------|-----|
| `remote_apriltag_server.py` | WebSocket server, captures frames and detects tags | Separates camera hardware from robot controller |
| `robot_eye_in_hand_client.py` | WebSocket client, records robot + tag pose to CSV | Synchronises robot state with each capture |
| `collect_eye_to_hand.py` | Local collection for eye-to-hand (no network) | Camera and robot are on the same machine |
| `handeye_calibrate.py` | Solves eye-in-hand AX=XB via cv2.calibrateHandEye | Standard closed-form hand-eye solver |
| `eye_to_hand_calibrate.py` | Solves eye-to-hand AX=ZB via cv2.calibrateRobotWorldHandEye | Simultaneously solves T[cam\|base] and T[tag\|end] |
| `compare_calibration_methods.py` | Runs all 5 eye-in-hand methods, prints comparison table | Lets you pick the most consistent result |
| `validate_handeye_calibration.py` | Moves robot, checks tag-in-base consistency | Quantitative proof that calibration is correct |

---

## Coordinate Systems and Units

**Translation**: millimeters (mm) everywhere.  
**Rotation**: radians (rad), Euler XYZ convention (R = Rz·Ry·Rx).

| Frame | X+ | Y+ | Z+ |
|-------|----|----|-----|
| OpenCV camera | right | down | forward |
| RealMan robot (base and end) | right | forward (out) | up |

Conversion applied automatically inside the solvers:
```
X_robot = X_opencv
Y_robot = Z_opencv   (forward = forward)
Z_robot = -Y_opencv  (up = -down)
```

---

## How to Verify Calibration Success

Run `validate_handeye_calibration.py` after eye-in-hand calibration.  
For eye-to-hand, manually verify by computing T[tag|base] from different robot
poses using the solved T[cam|base] and T[tag|end]; the result should be constant.

| Metric | Pass | Warning | Fail |
|--------|------|---------|------|
| Position std dev | < 5 mm | 5–10 mm | > 10 mm |
| Max position deviation | < 10 mm | 10–20 mm | > 20 mm |

---

## How to Improve Accuracy

1. **More diverse poses** (15–30 samples): vary X, Y, Z, and all three rotation axes.  
   Avoid degenerate motion (pure translation only, or small angular changes).
2. **Lower reprojection error threshold**: try `--max-reproj-error 1.0`.
3. **Larger AprilTag**: bigger tag → more stable corner detection.
4. **Disable auto-exposure**: already done in the config files.
5. **Try all methods**: `compare_calibration_methods.py` shows which is most consistent.
6. **Camera intrinsic calibration**: poor intrinsics are the biggest source of error.
   Recalibrate with a ChArUco board if accuracy is below 2 mm.

---

## Camera Config Selection

| Model | Config file | Notes |
|-------|-------------|-------|
| RealSense D435 | `config/realsense_D435.json` | Standard, used with WebSocket server |
| RealSense D405 | `config/realsense_d405.json` | Short range (~7 cm min), high precision |
| RealSense D415 | `config/realsense_d415.json` | Wide FOV, standard working distance |

Pass the config with `--realsense-config config/realsense_d405.json`.  
All three use the same `pyrealsense2` API — only exposure/gain settings differ.
