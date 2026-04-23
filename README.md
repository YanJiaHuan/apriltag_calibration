# AprilTag Hand-Eye Calibration

Calibration tools for Realman robotic arms using AprilTag fiducial markers.
Supports two independent configurations — choose **one** based on your hardware setup:

- **Eye-in-Hand**: camera mounted on the robot end-effector, AprilTag fixed in the world
- **Eye-to-Hand**: camera fixed in the world, AprilTag mounted on the robot end-effector

The two workflows use different scripts and produce different results. They are not combined.

---

## Prerequisites

```bash
pip install numpy opencv-python pyrealsense2 websockets
# Build and install the apriltag Python binding from third_party/apriltag
cd third_party/apriltag && mkdir build && cd build
cmake .. && make -j4
cd ../../..
```

---

## Workflow A: Eye-in-Hand

**Setup**: RealSense camera is mounted on the robot end-effector. An AprilTag is fixed
somewhere in the workspace (e.g. on a wall or table). The camera and robot controller
run on **separate machines** connected over a network.

**Goal**: Find `T[cam|end]` — the pose of the camera relative to the robot end-effector.

**Why do we need this?** Once we know how the camera sits on the arm, we can transform
any object detected in the camera frame into the robot base frame, which is what the
robot uses to plan motions.

---

### Step A1 — Start the camera server

Run this on the machine that has the RealSense attached:

```bash
python3 scripts/remote_apriltag_server.py \
    --realsense-config config/realsense_D435.json \
    --vision-config config/vision_config.json \
    --websocket-config config/websocket.json
```

**Why**: The camera is on a separate host from the robot controller. This script
streams AprilTag detections over WebSocket so the robot side can receive them.

---

### Step A2 — Collect calibration data

Run this on the robot controller host. Move the robot to **15–30 diverse poses**
(vary position and all three rotation axes). Press Enter at each pose to capture.

```bash
python3 scripts/robot_eye_in_hand_client.py \
    --robot-ip 10.168.1.18 \
    --robot-port 8080 \
    --websocket-config config/websocket.json \
    --count 20 \
    --manual
```

**Why**: Each capture records a pair: the robot's end-effector pose (from the robot)
and the AprilTag pose (from the camera). The solver needs many such pairs from diverse
poses to constrain the unknown `T[cam|end]`.

Data is saved to `data/robot_eye_in_hand.csv`.

---

### Step A3 — Solve the calibration

```bash
python3 scripts/handeye_calibrate.py \
    --input-csv data/robot_eye_in_hand.csv \
    --method tsai
```

**Why**: This runs the closed-form AX=XB hand-eye solver. It outputs `T[cam|end]`
(translation in mm, rotation as Euler XYZ in rad).

**Optional — compare all 5 solver methods** to pick the most consistent one:

```bash
python3 scripts/compare_calibration_methods.py \
    --input-csv data/robot_eye_in_hand.csv
```

This prints a side-by-side table of results for tsai, park, horaud, andreff, and
daniilidis. Choose the method whose translation values are most consistent, then
re-run Step A3 with `--method <chosen>`.

---

### Step A4 — Validate the result

Move the robot to several new poses while the camera sees the tag. The script
computes the tag position in the base frame from each pose and checks consistency —
if the calibration is correct, all estimates should agree.

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

Fill in `<X>` etc. with the `T[cam|end]` values from Step A3.

**Pass criterion**: position std dev < 5 mm, max deviation < 10 mm.

---

## Workflow B: Eye-to-Hand

**Setup**: RealSense camera is fixed in the world (e.g. on a tripod or mount). An
AprilTag is attached to the robot end-effector. The camera and robot controller run
on **the same machine**. No WebSocket needed.

**Goal**: Find two transforms simultaneously:
- `T[cam|base]` — pose of the camera relative to the robot base (primary result)
- `T[tag|end]` — pose of the tag relative to the end-effector (the tag placement on
  the box is imprecise, so this is also unknown and solved at the same time)

**Why do we need both?** The tag placement on the end-effector box is not known
precisely, so we cannot assume `T[tag|end]` is identity. The solver finds both
unknowns simultaneously from the set of observations.

---

### Step B1 — Collect calibration data

Mount the AprilTag on the robot end-effector. Move the robot to **15–30 diverse
poses** (vary position and all three rotation axes). Press Enter at each pose.

```bash
python3 scripts/collect_eye_to_hand.py \
    --robot-ip 10.168.1.18 \
    --robot-port 8080 \
    --realsense-config config/realsense_d405.json \
    --vision-config config/vision_config.json \
    --count 20 \
    --manual
```

**Why**: Each capture records the robot end-effector pose and the AprilTag pose seen
by the fixed camera. The solver needs pairs from diverse poses to disambiguate
`T[cam|base]` from `T[tag|end]`.

Data is saved to `data/eye_to_hand.csv`.

---

### Step B2 — Solve the calibration

```bash
python3 scripts/eye_to_hand_calibrate.py \
    --input-csv data/eye_to_hand.csv \
    --method shah
```

**Why**: This runs the AX=ZB robot-world hand-eye solver
(`cv2.calibrateRobotWorldHandEye`). It solves for both `T[cam|base]` and `T[tag|end]`
at once. Two methods are available: `shah` (default) and `li`. You can run both and
compare:

```bash
python3 scripts/eye_to_hand_calibrate.py --input-csv data/eye_to_hand.csv --method shah
python3 scripts/eye_to_hand_calibrate.py --input-csv data/eye_to_hand.csv --method li
```

If the `t_cam2base` values agree within ~2 mm, the result is reliable.

Output:
- `camera|base (T_cam2base)`: camera translation (mm) and rotation (rad) in base frame
- `tag|end (T_tag2end)`: tag translation (mm) and rotation (rad) relative to end-effector

**Note**: `compare_calibration_methods.py` and `validate_handeye_calibration.py` are
for Eye-in-Hand (Workflow A) only and do not apply here.

---

## Coordinate Systems and Units

**Translation**: millimeters (mm) everywhere.  
**Rotation**: radians (rad), Euler XYZ convention (R = Rz·Ry·Rx).

| Frame | X+ | Y+ | Z+ |
|-------|----|----|-----|
| OpenCV camera | right | down | forward |
| RealMan robot (base and end) | right | forward (out) | up |

The conversion is applied automatically inside all solvers — you do not need to
handle it manually.

---

## How to Improve Accuracy

1. **More diverse poses** (15–30 samples): vary X, Y, Z, and all three rotation axes.
   Pure translation without rotation changes gives degenerate data.
2. **Lower reprojection error threshold**: try `--max-reproj-error 1.0` to discard
   noisier detections.
3. **Larger AprilTag**: a physically larger tag gives more stable corner detection.
4. **Disable auto-exposure**: already done in the config files — manual exposure
   prevents the camera from adjusting mid-capture.
5. **Camera intrinsic calibration**: poor intrinsics are the biggest accuracy bottleneck.
   If you cannot get below ~5 mm error, recalibrate with a ChArUco board.

---

## Camera Config Selection

| Model | Config file | Notes |
|-------|-------------|-------|
| RealSense D435 | `config/realsense_D435.json` | Standard, default for WebSocket server |
| RealSense D405 | `config/realsense_d405.json` | Short range (~7 cm min), high precision |
| RealSense D415 | `config/realsense_d415.json` | Wide FOV, standard working distance |

Pass the config with `--realsense-config config/realsense_d405.json`.  
All three use the same `pyrealsense2` API — only exposure/gain settings differ.
