# Hand-Eye Calibration Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add eye-to-hand calibration, extract shared math into handeye_utils, expose clean Python APIs, add per-camera configs, and write README.

**Architecture:** Extract duplicated math from handeye_calibrate.py and validate_handeye_calibration.py into handeye_utils.py. Expose solve_eye_in_hand() and solve_eye_to_hand() as importable APIs. Add collect_eye_to_hand.py as a local (non-WebSocket) collection script. All scripts run from repo root; sys.path manipulation ensures correct imports.

**Tech Stack:** Python 3, NumPy, OpenCV (cv2.calibrateHandEye, cv2.calibrateRobotWorldHandEye), pyrealsense2, apriltag Python bindings (third_party/apriltag), RM_API2 (third_party/RM_API2/Python)

---

## File Map

| Action | File | What it does |
|--------|------|--------------|
| CREATE | `scripts/handeye_utils.py` | Shared math: euler↔rot, make_transform, invert_transform, opencv→robot conversion |
| CREATE | `test/test_handeye_utils.py` | Unit tests for handeye_utils |
| MODIFY | `scripts/handeye_calibrate.py` | Import from handeye_utils, remove duplicates, add solve_eye_in_hand() |
| MODIFY | `scripts/validate_handeye_calibration.py` | Import from handeye_utils, remove duplicates |
| REWRITE | `scripts/compare_calibration_methods.py` | Direct calls to solve_eye_in_hand(), no subprocess |
| CREATE | `scripts/eye_to_hand_calibrate.py` | solve_eye_to_hand() using calibrateRobotWorldHandEye + CLI |
| CREATE | `test/test_eye_to_hand_calibrate.py` | Unit tests with synthetic data (no hardware) |
| CREATE | `scripts/collect_eye_to_hand.py` | Local camera + robot collection, no WebSocket |
| CREATE | `test/test_collect_eye_to_hand.py` | Unit tests for pure functions only |
| CREATE | `config/realsense_d405.json` | D405 camera config |
| CREATE | `config/realsense_d415.json` | D415 camera config |
| CREATE | `README.md` | Quick-start tutorial |

---

## Task 1: Create handeye_utils.py

**Files:**
- Create: `scripts/handeye_utils.py`
- Create: `test/test_handeye_utils.py`

- [ ] **Step 1.1: Write the failing test**

```python
# test/test_handeye_utils.py
#!/usr/bin/env python3
"""Unit tests for handeye_utils shared math helpers."""

import math
import unittest

import numpy as np

from scripts import handeye_utils


class TestHandeyeUtils(unittest.TestCase):
    """Tests for coordinate and transform utilities."""

    def test_euler_roundtrip(self):
        """Euler XYZ should roundtrip through rotation matrix."""
        for angles in [
            (0.1, -0.2, 0.3),
            (0.0, 0.0, 0.0),
            (math.pi / 4, -math.pi / 6, math.pi / 3),
        ]:
            r = handeye_utils.euler_xyz_to_rot(*angles)
            rx, ry, rz = handeye_utils.rot_to_euler_xyz(r)
            self.assertAlmostEqual(rx, angles[0], places=6)
            self.assertAlmostEqual(ry, angles[1], places=6)
            self.assertAlmostEqual(rz, angles[2], places=6)

    def test_make_invert_roundtrip(self):
        """make_transform then invert_transform should yield identity."""
        r = handeye_utils.euler_xyz_to_rot(0.1, -0.2, 0.3)
        t = np.array([10.0, -5.0, 3.0])
        tmat = handeye_utils.make_transform(r, t)
        inv = handeye_utils.invert_transform(tmat)
        np.testing.assert_allclose(tmat @ inv, np.eye(4), atol=1e-10)

    def test_opencv_to_robot_is_proper_rotation(self):
        """opencv_to_robot_rotation should have det=1."""
        r = handeye_utils.opencv_to_robot_rotation()
        self.assertAlmostEqual(float(np.linalg.det(r)), 1.0, places=6)

    def test_convert_opencv_z_maps_to_robot_y(self):
        """Z+(forward) in OpenCV maps to Y+(out) in robot frame."""
        t_opencv = np.array([0.0, 0.0, 100.0])
        _, t_robot = handeye_utils.convert_opencv_pose_to_robot(np.eye(3), t_opencv)
        self.assertAlmostEqual(t_robot[0], 0.0, places=6)
        self.assertAlmostEqual(t_robot[1], 100.0, places=6)
        self.assertAlmostEqual(t_robot[2], 0.0, places=6)

    def test_convert_opencv_y_maps_to_robot_neg_z(self):
        """Y+(down) in OpenCV maps to Z-(down) in robot frame."""
        t_opencv = np.array([0.0, 100.0, 0.0])
        _, t_robot = handeye_utils.convert_opencv_pose_to_robot(np.eye(3), t_opencv)
        self.assertAlmostEqual(t_robot[0], 0.0, places=6)
        self.assertAlmostEqual(t_robot[1], 0.0, places=6)
        self.assertAlmostEqual(t_robot[2], -100.0, places=6)

    def test_quat_rot_roundtrip(self):
        """rot_to_quat then quat_to_rot should recover original matrix."""
        r = handeye_utils.euler_xyz_to_rot(0.3, -0.1, 0.5)
        q = handeye_utils.rot_to_quat(r)
        r2 = handeye_utils.quat_to_rot(q)
        np.testing.assert_allclose(r, r2, atol=1e-10)

    def test_average_rotations_single(self):
        """Average of one rotation should return that rotation unchanged."""
        r = handeye_utils.euler_xyz_to_rot(0.1, 0.2, 0.3)
        r_avg = handeye_utils.average_rotations([r])
        np.testing.assert_allclose(r_avg, r, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
python3 -m pytest test/test_handeye_utils.py -v
```
Expected: `ModuleNotFoundError: No module named 'scripts.handeye_utils'`

- [ ] **Step 1.3: Create scripts/handeye_utils.py**

```python
#!/usr/bin/env python3
"""
Shared math utilities for hand-eye calibration.

Provides coordinate conversion, rotation, and homogeneous transform helpers
used by both eye-in-hand and eye-to-hand calibration pipelines.

Assumptions:
- Translation units are millimeters (mm).
- Rotation units are radians (rad), Euler XYZ convention (R = Rz @ Ry @ Rx).
- OpenCV camera frame: X+ right, Y+ down, Z+ forward.
- RealMan robot frame: X+ right, Y+ forward (out), Z+ up.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def euler_xyz_to_rot(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Convert Euler XYZ angles to rotation matrix (R = Rz @ Ry @ Rx).

    Args:
        rx (float): Rotation about X in radians.
        ry (float): Rotation about Y in radians.
        rz (float): Rotation about Z in radians.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz_m @ ry_m @ rx_m


def rot_to_euler_xyz(r: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler XYZ angles in radians.

    Args:
        r (np.ndarray): 3x3 rotation matrix (list or numpy array).

    Returns:
        tuple[float, float, float]: (rx, ry, rz) in radians.
    """
    r = np.asarray(r, dtype=float)
    r00, r10, r20 = float(r[0, 0]), float(r[1, 0]), float(r[2, 0])
    r21, r22 = float(r[2, 1]), float(r[2, 2])
    sy = math.sqrt(r00 * r00 + r10 * r10)
    if sy < 1e-9:
        rx = math.atan2(-float(r[1, 2]), float(r[1, 1]))
        ry = math.atan2(-r20, sy)
        rz = 0.0
    else:
        rx = math.atan2(r21, r22)
        ry = math.atan2(-r20, sy)
        rz = math.atan2(r10, r00)
    return rx, ry, rz


def make_transform(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Build a 4x4 homogeneous transform from rotation matrix and translation.

    Args:
        r (np.ndarray): 3x3 rotation matrix.
        t (np.ndarray): Translation vector (3,) or (3,1) in mm.

    Returns:
        np.ndarray: 4x4 homogeneous transform matrix.
    """
    t = np.asarray(t, dtype=float).reshape(3, 1)
    tmat = np.eye(4)
    tmat[:3, :3] = np.asarray(r, dtype=float)
    tmat[:3, 3] = t[:, 0]
    return tmat


def invert_transform(tmat: np.ndarray) -> np.ndarray:
    """
    Invert a 4x4 homogeneous transform using R^T (exact, no numerical inverse).

    Args:
        tmat (np.ndarray): 4x4 homogeneous transform.

    Returns:
        np.ndarray: 4x4 inverted transform.
    """
    tmat = np.asarray(tmat, dtype=float)
    r = tmat[:3, :3]
    t = tmat[:3, 3]
    r_inv = r.T
    out = np.eye(4)
    out[:3, :3] = r_inv
    out[:3, 3] = -r_inv @ t
    return out


def opencv_to_robot_rotation() -> np.ndarray:
    """
    Return the rotation matrix that maps OpenCV camera axes to RealMan robot axes.

    OpenCV: X+ right, Y+ down, Z+ forward.
    Robot:  X+ right, Y+ forward (out), Z+ up.
    Mapping: X_robot=X_opencv, Y_robot=Z_opencv, Z_robot=-Y_opencv.
    det=1 (pure rotation, not a reflection).

    Args:
        None.

    Returns:
        np.ndarray: 3x3 rotation matrix R_map such that v_robot = R_map @ v_opencv.
    """
    return np.array(
        [[1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0],
         [0.0, -1.0, 0.0]],
        dtype=float,
    )


def convert_opencv_pose_to_robot(
    r_target2cam: np.ndarray, t_target2cam: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a target-in-camera pose from OpenCV axes to robot camera axes.

    Args:
        r_target2cam (np.ndarray): 3x3 rotation matrix in OpenCV camera axes.
        t_target2cam (np.ndarray): Translation (3,) in OpenCV camera axes (mm).

    Returns:
        tuple[np.ndarray, np.ndarray]: (R, t) expressed in robot camera axes.
    """
    r_map = opencv_to_robot_rotation()
    r_out = r_map @ np.asarray(r_target2cam, dtype=float)
    t_out = r_map @ np.asarray(t_target2cam, dtype=float).reshape(3)
    return r_out, t_out


def rot_to_quat(r: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion [w, x, y, z].

    Args:
        r (np.ndarray): 3x3 rotation matrix.

    Returns:
        np.ndarray: Quaternion [w, x, y, z].
    """
    r = np.asarray(r, dtype=float)
    m00, m01, m02 = r[0, 0], r[0, 1], r[0, 2]
    m10, m11, m12 = r[1, 0], r[1, 1], r[1, 2]
    m20, m21, m22 = r[2, 0], r[2, 1], r[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w, x = 0.25 * s, (m21 - m12) / s
        y, z = (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w, x = (m21 - m12) / s, 0.25 * s
        y, z = (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w, x = (m02 - m20) / s, (m01 + m10) / s
        y, z = 0.25 * s, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w, x = (m10 - m01) / s, (m02 + m20) / s
        y, z = (m12 + m21) / s, 0.25 * s
    return np.array([w, x, y, z], dtype=float)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to rotation matrix.

    Args:
        q (np.ndarray): Quaternion [w, x, y, z].

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ],
        dtype=float,
    )


def average_rotations(rotations: List[np.ndarray]) -> np.ndarray:
    """
    Compute mean rotation matrix via quaternion averaging.

    Args:
        rotations (list[np.ndarray]): List of 3x3 rotation matrices.

    Returns:
        np.ndarray: Averaged 3x3 rotation matrix.
    """
    if not rotations:
        raise ValueError("no rotations to average")
    quats = []
    ref = rot_to_quat(rotations[0])
    for r in rotations:
        q = rot_to_quat(r)
        if np.dot(q, ref) < 0.0:
            q = -q
        quats.append(q)
    mean_q = np.mean(np.stack(quats, axis=0), axis=0)
    return quat_to_rot(mean_q)
```

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
python3 -m pytest test/test_handeye_utils.py -v
```
Expected: 7 tests PASSED

- [ ] **Step 1.5: Commit**

```bash
git add scripts/handeye_utils.py test/test_handeye_utils.py
git commit -m "feat: add handeye_utils shared math library"
```

---

## Task 2: Update handeye_calibrate.py — import utils, add solve_eye_in_hand()

**Files:**
- Modify: `scripts/handeye_calibrate.py`

- [ ] **Step 2.1: Verify existing tests still pass before touching anything**

```bash
python3 -m pytest test/test_handeye_calibrate.py -v
```
Expected: all PASSED (baseline)

- [ ] **Step 2.2: Replace the module docstring and imports block**

Replace the import section at lines 1–32 of `scripts/handeye_calibrate.py` with:

```python
#!/usr/bin/env python3
"""
Hand-eye calibration solver for eye-in-hand using robot + AprilTag data.

This module reads a robot-side CSV containing robot end pose (end|base)
and AprilTag pose (tag|camera) in the OpenCV camera frame. It filters out
invalid detections, converts the OpenCV camera frame to the robot camera
frame, solves for the camera pose relative to the robot end (camera|end),
and estimates the fixed tag pose relative to the robot base (tag|base).

Assumptions:
- Robot pose is end in base (x,y,z in mm, rx,ry,rz in rad; Euler XYZ).
- Tag pose is tag in camera (x,y,z in mm, rx,ry,rz in rad; Euler XYZ).
- OpenCV camera frame (x right, y down, z out) is converted to
  robot frame (x right, y out, z up) before solving.
- Output translation is mm and rotation is Euler XYZ in rad.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from scripts.handeye_utils import (
    average_rotations,
    convert_opencv_pose_to_robot,
    euler_xyz_to_rot,
    invert_transform,
    make_transform,
    opencv_to_robot_rotation,
    quat_to_rot,
    rot_to_euler_xyz,
    rot_to_quat,
)
```

- [ ] **Step 2.3: Delete the nine duplicated function bodies**

Delete these complete function definitions from `scripts/handeye_calibrate.py` (they are now imported from handeye_utils):
- `opencv_to_robot_rotation` (lines ~109–132)
- `convert_opencv_pose_to_robot` (lines ~135–154)
- `euler_xyz_to_rot` (lines ~157–180)
- `rot_to_euler_xyz` (lines ~183–213)
- `make_transform` (lines ~216–234)
- `invert_transform` (lines ~237–258)
- `rot_to_quat` (lines ~261–305)
- `quat_to_rot` (lines ~308–331)
- `average_rotations` (lines ~334–357)

Also remove `import math` from the imports (it is no longer used directly in this file).

- [ ] **Step 2.4: Add solve_eye_in_hand() before main()**

Insert this function immediately before the `def main()` function:

```python
def solve_eye_in_hand(
    csv_path: str,
    method: str = "tsai",
    max_reproj_error: float = 2.0,
    min_samples: int = 3,
) -> dict:
    """
    Load CSV data and solve eye-in-hand hand-eye calibration.

    Args:
        csv_path (str): Path to robot_eye_in_hand.csv.
        method (str): Calibration method: tsai, park, horaud, andreff, daniilidis.
        max_reproj_error (float): Max reprojection error filter (pixels).
        min_samples (int): Minimum valid samples required.

    Returns:
        dict: {
            "r_cam2end": np.ndarray (3x3),   camera rotation in end frame
            "t_cam2end": np.ndarray (3,) mm, camera translation in end frame
            "r_tag2base": np.ndarray (3x3),  tag rotation in base frame
            "t_tag2base": np.ndarray (3,) mm,tag translation in base frame
            "n_samples": int,
        }

    Raises:
        ValueError: If CSV not found or not enough valid samples.
    """
    if not os.path.exists(csv_path):
        raise ValueError(f"csv not found: {csv_path}")

    r_g2b, t_g2b, r_t2c, t_t2c, warnings = load_samples(csv_path, max_reproj_error)
    for w in warnings:
        print(f"warning: {w}")

    if len(r_g2b) < min_samples:
        raise ValueError(f"not enough valid samples: {len(r_g2b)} < {min_samples}")

    r_cam2gripper, t_cam2gripper = solve_handeye(r_g2b, t_g2b, r_t2c, t_t2c, method)
    t_cam2gripper = np.array(t_cam2gripper, dtype=float).reshape(3)

    r_base_tag, t_base_tag = estimate_tag_in_base(
        r_g2b, t_g2b, r_cam2gripper, t_cam2gripper, r_t2c, t_t2c
    )

    return {
        "r_cam2end": np.array(r_cam2gripper),
        "t_cam2end": t_cam2gripper,
        "r_tag2base": np.array(r_base_tag),
        "t_tag2base": np.array(t_base_tag),
        "n_samples": len(r_g2b),
    }
```

- [ ] **Step 2.5: Run existing tests to verify no regressions**

```bash
python3 -m pytest test/test_handeye_calibrate.py test/test_handeye_utils.py -v
```
Expected: all PASSED (functions are re-exported via import, so `handeye_calibrate.euler_xyz_to_rot` still works)

- [ ] **Step 2.6: Commit**

```bash
git add scripts/handeye_calibrate.py
git commit -m "refactor: handeye_calibrate imports from handeye_utils, adds solve_eye_in_hand()"
```

---

## Task 3: Update validate_handeye_calibration.py — remove duplicated math

**Files:**
- Modify: `scripts/validate_handeye_calibration.py`

- [ ] **Step 3.1: Replace the import block at the top of validate_handeye_calibration.py**

Replace the current imports (lines 1–34) with:

```python
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
    opencv_to_robot_rotation,
    rot_to_euler_xyz,
)
```

- [ ] **Step 3.2: Delete the five duplicated function bodies**

Delete these complete function definitions from `scripts/validate_handeye_calibration.py`:
- `opencv_to_robot_rotation`
- `convert_opencv_pose_to_robot`
- `euler_xyz_to_rot`
- `rot_to_euler_xyz`
- `make_transform`

Also remove `import math` (no longer needed directly in this file).

- [ ] **Step 3.3: Run full test suite**

```bash
python3 -m pytest test/ -v
```
Expected: all PASSED

- [ ] **Step 3.4: Commit**

```bash
git add scripts/validate_handeye_calibration.py
git commit -m "refactor: validate_handeye_calibration imports from handeye_utils"
```

---

## Task 4: Rewrite compare_calibration_methods.py

**Files:**
- Rewrite: `scripts/compare_calibration_methods.py`

- [ ] **Step 4.1: Replace the entire file**

```python
#!/usr/bin/env python3
"""
Compare all eye-in-hand calibration methods on the same dataset.

Calls solve_eye_in_hand() directly for each method (no subprocess).
Prints a side-by-side table of camera|end translation and rotation.
"""

from __future__ import annotations

import argparse
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.handeye_calibrate import solve_eye_in_hand
from scripts.handeye_utils import rot_to_euler_xyz


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        None.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compare hand-eye calibration methods"
    )
    parser.add_argument("--input-csv", default="data/robot_eye_in_hand.csv")
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
    methods = ["tsai", "park", "horaud", "andreff", "daniilidis"]

    print("=" * 70)
    print("Hand-Eye Calibration Method Comparison")
    print("=" * 70)
    print(f"Input CSV: {args.input_csv}")
    print(f"Max reproj error: {args.max_reproj_error}  Min samples: {args.min_samples}")
    print("=" * 70)

    results = []
    for method in methods:
        try:
            result = solve_eye_in_hand(
                args.input_csv,
                method=method,
                max_reproj_error=args.max_reproj_error,
                min_samples=args.min_samples,
            )
            results.append({"method": method, "success": True, **result})
            print(f"  [{method.upper():>10}] OK  ({result['n_samples']} samples)")
        except Exception as exc:
            results.append({"method": method, "success": False, "error": str(exc)})
            print(f"  [{method.upper():>10}] FAILED: {exc}")

    print("\nCamera|End Translation [mm]:")
    print(f"  {'Method':<12} {'X':>10} {'Y':>10} {'Z':>10}")
    print("  " + "-" * 36)
    for r in results:
        if r["success"]:
            t = r["t_cam2end"]
            print(f"  {r['method']:<12} {t[0]:>10.2f} {t[1]:>10.2f} {t[2]:>10.2f}")
        else:
            print(f"  {r['method']:<12} {'FAILED':>10}")

    print("\nCamera|End Rotation [rad]:")
    print(f"  {'Method':<12} {'RX':>10} {'RY':>10} {'RZ':>10}")
    print("  " + "-" * 36)
    for r in results:
        if r["success"]:
            rx, ry, rz = rot_to_euler_xyz(r["r_cam2end"])
            print(f"  {r['method']:<12} {rx:>10.4f} {ry:>10.4f} {rz:>10.4f}")
        else:
            print(f"  {r['method']:<12} {'FAILED':>10}")

    print("\nRecommendation: choose the method with most consistent translation results.")
    print("Validate using: python3 scripts/validate_handeye_calibration.py ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4.2: Run full test suite**

```bash
python3 -m pytest test/ -v
```
Expected: all PASSED

- [ ] **Step 4.3: Commit**

```bash
git add scripts/compare_calibration_methods.py
git commit -m "rewrite: compare_calibration_methods uses solve_eye_in_hand() directly"
```

---

## Task 5: Create eye_to_hand_calibrate.py

**Files:**
- Create: `scripts/eye_to_hand_calibrate.py`
- Create: `test/test_eye_to_hand_calibrate.py`

- [ ] **Step 5.1: Write the failing test**

```python
# test/test_eye_to_hand_calibrate.py
#!/usr/bin/env python3
"""
Unit tests for eye_to_hand_calibrate.

Uses synthetic data with known ground-truth transforms to verify the solver
recovers T[cam|base] and T[tag|end] within acceptable tolerance.
No hardware required.
"""

import math
import unittest

import numpy as np

from scripts import handeye_utils
from scripts.eye_to_hand_calibrate import solve_eye_to_hand_raw


class TestEyeToHandCalibrate(unittest.TestCase):
    """Tests for eye-to-hand solver using synthetic data."""

    def _make_synthetic_data(self):
        """
        Generate 8 robot poses and the corresponding T[tag|cam] observations
        given known T[cam|base] and T[tag|end].

        Returns:
            tuple: (r_g2b, t_g2b, r_t2c, t_t2c, r_cb_gt, t_cb_gt, r_te_gt, t_te_gt)
        """
        # Ground truth: camera 500mm to the right of base, 300mm up, tilted 90° about X
        r_cam2base_gt = handeye_utils.euler_xyz_to_rot(math.pi / 2, 0.0, 0.0)
        t_cam2base_gt = np.array([500.0, 0.0, 300.0])
        t_cb = handeye_utils.make_transform(r_cam2base_gt, t_cam2base_gt)
        t_bc = handeye_utils.invert_transform(t_cb)

        # Ground truth: tag is 50mm along Z from end-effector, no rotation
        r_tag2end_gt = handeye_utils.euler_xyz_to_rot(0.0, 0.0, 0.0)
        t_tag2end_gt = np.array([0.0, 0.0, 50.0])
        t_te = handeye_utils.make_transform(r_tag2end_gt, t_tag2end_gt)

        # 8 diverse robot poses (x, y, z mm, rx, ry, rz rad)
        poses = [
            (100.0,    0.0, 400.0,  0.0,   0.0,  0.0),
            (200.0,    0.0, 400.0,  0.15,  0.0,  0.0),
            (100.0,  120.0, 400.0,  0.0,   0.15, 0.0),
            (100.0, -120.0, 400.0,  0.0,  -0.15, 0.0),
            (150.0,   60.0, 450.0,  0.15,  0.15, 0.0),
            (200.0,  -60.0, 350.0, -0.10,  0.15, 0.1),
            ( 80.0,   80.0, 420.0,  0.10, -0.10, 0.15),
            (220.0,  -80.0, 380.0, -0.15,  0.10, -0.1),
        ]

        r_g2b_list, t_g2b_list, r_t2c_list, t_t2c_list = [], [], [], []
        for (x, y, z, rx, ry, rz) in poses:
            r_eb = handeye_utils.euler_xyz_to_rot(rx, ry, rz)
            t_eb = np.array([x, y, z])
            t_eb_mat = handeye_utils.make_transform(r_eb, t_eb)
            # T[tag|cam] = T[base|cam] @ T[end|base] @ T[tag|end]
            t_tc = t_bc @ t_eb_mat @ t_te
            r_g2b_list.append(r_eb)
            t_g2b_list.append(t_eb)
            r_t2c_list.append(t_tc[:3, :3])
            t_t2c_list.append(t_tc[:3, 3])

        return (
            r_g2b_list, t_g2b_list, r_t2c_list, t_t2c_list,
            r_cam2base_gt, t_cam2base_gt, r_tag2end_gt, t_tag2end_gt,
        )

    def test_solve_recovers_translation_shah(self):
        """Shah method should recover T[cam|base] and T[tag|end] translations."""
        (r_g2b, t_g2b, r_t2c, t_t2c,
         _, t_cb_gt, _, t_te_gt) = self._make_synthetic_data()

        r_cb, t_cb, r_te, t_te = solve_eye_to_hand_raw(
            r_g2b, t_g2b, r_t2c, t_t2c, "shah"
        )

        np.testing.assert_allclose(t_cb.reshape(3), t_cb_gt, atol=2.0)
        np.testing.assert_allclose(t_te.reshape(3), t_te_gt, atol=2.0)

    def test_solve_recovers_translation_li(self):
        """Li method should also recover the transforms within tolerance."""
        (r_g2b, t_g2b, r_t2c, t_t2c,
         _, t_cb_gt, _, t_te_gt) = self._make_synthetic_data()

        r_cb, t_cb, r_te, t_te = solve_eye_to_hand_raw(
            r_g2b, t_g2b, r_t2c, t_t2c, "li"
        )

        np.testing.assert_allclose(t_cb.reshape(3), t_cb_gt, atol=2.0)
        np.testing.assert_allclose(t_te.reshape(3), t_te_gt, atol=2.0)

    def test_output_shapes(self):
        """Solver outputs should have correct shapes."""
        (r_g2b, t_g2b, r_t2c, t_t2c, *_) = self._make_synthetic_data()
        r_cb, t_cb, r_te, t_te = solve_eye_to_hand_raw(
            r_g2b, t_g2b, r_t2c, t_t2c, "shah"
        )
        self.assertEqual(r_cb.shape, (3, 3))
        self.assertEqual(r_te.shape, (3, 3))
        self.assertIn(t_cb.size, [3, 4])  # (3,) or (3,1)
        self.assertIn(t_te.size, [3, 4])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 5.2: Run test to verify it fails**

```bash
python3 -m pytest test/test_eye_to_hand_calibrate.py -v
```
Expected: `ModuleNotFoundError: No module named 'scripts.eye_to_hand_calibrate'`

- [ ] **Step 5.3: Create scripts/eye_to_hand_calibrate.py**

```python
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
```

- [ ] **Step 5.4: Run tests to verify they pass**

```bash
python3 -m pytest test/test_eye_to_hand_calibrate.py -v
```
Expected: 3 tests PASSED

- [ ] **Step 5.5: Run full test suite**

```bash
python3 -m pytest test/ -v
```
Expected: all PASSED

- [ ] **Step 5.6: Commit**

```bash
git add scripts/eye_to_hand_calibrate.py test/test_eye_to_hand_calibrate.py
git commit -m "feat: add eye_to_hand_calibrate with solve_eye_to_hand() API"
```

---

## Task 6: Create collect_eye_to_hand.py

**Files:**
- Create: `scripts/collect_eye_to_hand.py`
- Create: `test/test_collect_eye_to_hand.py`

- [ ] **Step 6.1: Write the failing test (pure functions only, no hardware)**

```python
# test/test_collect_eye_to_hand.py
#!/usr/bin/env python3
"""
Unit tests for collect_eye_to_hand pure utility functions.
Hardware (camera, robot) is not required.
"""

import csv
import os
import tempfile
import unittest

from scripts import collect_eye_to_hand


class TestCollectEyeToHand(unittest.TestCase):
    """Tests for pure utility functions in collect_eye_to_hand."""

    def test_meters_to_mm(self):
        """meters_to_mm should multiply by 1000."""
        self.assertAlmostEqual(collect_eye_to_hand.meters_to_mm(1.0), 1000.0)
        self.assertAlmostEqual(collect_eye_to_hand.meters_to_mm(0.123), 123.0)
        self.assertAlmostEqual(collect_eye_to_hand.meters_to_mm(0.0), 0.0)

    def test_write_csv_row_creates_header(self):
        """write_csv_row should write header on first call."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            os.unlink(path)
            header = ["a", "b", "c"]
            collect_eye_to_hand.write_csv_row(path, header, [1, 2, 3])
            with open(path, newline="") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], ["a", "b", "c"])
            self.assertEqual(rows[1], ["1", "2", "3"])
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_write_csv_row_appends(self):
        """write_csv_row should append without repeating the header."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            os.unlink(path)
            header = ["x", "y"]
            collect_eye_to_hand.write_csv_row(path, header, [1, 2])
            collect_eye_to_hand.write_csv_row(path, header, [3, 4])
            with open(path, newline="") as f:
                rows = list(csv.reader(f))
            self.assertEqual(len(rows), 3)   # 1 header + 2 data rows
            self.assertEqual(rows[0], ["x", "y"])
        finally:
            if os.path.exists(path):
                os.unlink(path)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 6.2: Run test to verify it fails**

```bash
python3 -m pytest test/test_collect_eye_to_hand.py -v
```
Expected: `ModuleNotFoundError: No module named 'scripts.collect_eye_to_hand'`

- [ ] **Step 6.3: Create scripts/collect_eye_to_hand.py**

```python
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

    import numpy as np as _np  # already imported above, just reference
    r_mat = _np.array(pose["R"])
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
```

**Note:** There is a syntax error in the `capture_and_detect` function above (`import numpy as np as _np`). Fix it by removing the duplicate import line — `numpy` is already imported at the top of that function as `np`. Replace:
```python
    import numpy as np as _np  # already imported above, just reference
    r_mat = _np.array(pose["R"])
```
with:
```python
    r_mat = np.array(pose["R"])
```

- [ ] **Step 6.4: Run tests to verify they pass**

```bash
python3 -m pytest test/test_collect_eye_to_hand.py -v
```
Expected: 3 tests PASSED

- [ ] **Step 6.5: Run full test suite**

```bash
python3 -m pytest test/ -v
```
Expected: all PASSED

- [ ] **Step 6.6: Commit**

```bash
git add scripts/collect_eye_to_hand.py test/test_collect_eye_to_hand.py
git commit -m "feat: add collect_eye_to_hand local collection script"
```

---

## Task 7: Add camera config files

**Files:**
- Create: `config/realsense_d405.json`
- Create: `config/realsense_d415.json`

- [ ] **Step 7.1: Create config/realsense_d405.json**

D405 is a short-range, high-precision depth camera. For hand-eye calibration we only use the RGB stream. Its RGB sensor has the same pyrealsense2 API as D435. Use conservative exposure settings to reduce motion blur.

```json
{
  "quality": {
    "width": 1280,
    "height": 720,
    "fps": 30
  },
  "rgb_sensor": {
    "enable_auto_exposure": 0,
    "enable_auto_white_balance": 0,
    "exposure": 120,
    "gain": 8,
    "brightness": -5,
    "contrast": 60,
    "gamma": 50,
    "saturation": 50,
    "sharpness": 50
  }
}
```

- [ ] **Step 7.2: Create config/realsense_d415.json**

D415 has a wider FOV and standard working distance. Slightly higher exposure than D405 for the wider field.

```json
{
  "quality": {
    "width": 1280,
    "height": 720,
    "fps": 30
  },
  "rgb_sensor": {
    "enable_auto_exposure": 0,
    "enable_auto_white_balance": 0,
    "exposure": 166,
    "gain": 10,
    "brightness": -5,
    "contrast": 60,
    "gamma": 50,
    "saturation": 50,
    "sharpness": 50
  }
}
```

- [ ] **Step 7.3: Commit**

```bash
git add config/realsense_d405.json config/realsense_d415.json
git commit -m "feat: add realsense config files for D405 and D415"
```

---

## Task 8: Write README.md

**Files:**
- Create: `README.md`

- [ ] **Step 8.1: Create README.md**

```markdown
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
```

- [ ] **Step 8.2: Run full test suite one final time**

```bash
python3 -m pytest test/ -v
```
Expected: all PASSED

- [ ] **Step 8.3: Commit**

```bash
git add README.md
git commit -m "docs: add README with quick-start tutorial for eye-in-hand and eye-to-hand"
```

---

## Self-Review

**Spec coverage check:**
- [x] External libs understood (apriltag RGB-only, RM_API2) — documented in README
- [x] Shared math extracted to handeye_utils (Task 1)
- [x] handeye_calibrate imports utils, exposes solve_eye_in_hand() (Task 2)
- [x] validate_handeye_calibration imports utils, no duplicates (Task 3)
- [x] compare_calibration_methods rewritten with direct calls (Task 4)
- [x] eye_to_hand_calibrate.py with solve_eye_to_hand() API (Task 5)
- [x] collect_eye_to_hand.py local collection (Task 6)
- [x] realsense_d405.json, realsense_d415.json (Task 7)
- [x] README with quick-start, explanation, validation, accuracy tips (Task 8)
- [x] remote_apriltag_server.py and robot_eye_in_hand_client.py untouched
- [x] tools/reset_arm.py untouched
- [x] realsense_D435.json untouched (default in unchanged scripts)

**Known issue fixed inline:** The `capture_and_detect` function in Task 6.3 had a duplicate `import numpy` line. The corrected version removes it (Step 6.3 note).

**Type consistency:** `solve_eye_in_hand()` returns `t_cam2end` (3,) and `r_cam2end` (3x3). `compare_calibration_methods.py` calls `rot_to_euler_xyz(r["r_cam2end"])` — matches. `solve_eye_to_hand()` returns `t_cam2base` (3,) and `r_cam2base` (3x3) — consistent with usage in main().
