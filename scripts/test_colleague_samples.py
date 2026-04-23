#!/usr/bin/env python3
"""
Quick test: solve eye-to-hand calibration from colleague's sample JSON.

Data format assumptions:
- T_base_flange: 4x4 homogeneous transform, flange in base frame (T[flange|base])
- T_tag_cam:     4x4 homogeneous transform, tag in camera frame (T[tag|cam])
- Both already in robot coordinate system (no OpenCV→robot conversion needed)
- Translation units in the JSON are meters → converted to mm here

Usage:
    python3 scripts/test_colleague_samples.py
    python3 scripts/test_colleague_samples.py --json path/to/eye_to_hand_samples.json
    python3 scripts/test_colleague_samples.py --method li
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np

from scripts.eye_to_hand_calibrate import solve_eye_to_hand_raw
from scripts.handeye_utils import invert_transform, make_transform, rot_to_euler_xyz


def load_samples(json_path: str):
    """
    Load samples from JSON, extract rotation and translation arrays.

    Returns:
        tuple: (r_g2b, t_g2b, r_t2c, t_t2c)
            r_g2b: list of 3x3 rotations  T[flange|base]
            t_g2b: list of (3,) mm         T[flange|base]
            r_t2c: list of 3x3 rotations  T[tag|cam]  (robot axes)
            t_t2c: list of (3,) mm         T[tag|cam]  (robot axes)
    """
    with open(json_path, "r") as f:
        samples = json.load(f)

    r_g2b, t_g2b, r_t2c, t_t2c = [], [], [], []
    for s in samples:
        t_bf = np.array(s["T_base_flange"], dtype=float)
        t_tc = np.array(s["T_tag_cam"], dtype=float)

        r_g2b.append(t_bf[:3, :3])
        t_g2b.append(t_bf[:3, 3] * 1000.0)   # m → mm

        r_t2c.append(t_tc[:3, :3])
        t_t2c.append(t_tc[:3, 3] * 1000.0)   # m → mm

    return r_g2b, t_g2b, r_t2c, t_t2c


def print_transform(label: str, r: np.ndarray, t: np.ndarray) -> None:
    rx, ry, rz = rot_to_euler_xyz(r)
    t = np.asarray(t).reshape(3)
    print(f"{label}:")
    print(f"  t_mm  : [{t[0]:10.3f}, {t[1]:10.3f}, {t[2]:10.3f}]")
    print(f"  rpy_rad: [{rx:10.6f}, {ry:10.6f}, {rz:10.6f}]")


def diagnose_samples(r_g2b: list, t_g2b: list) -> None:
    """Print rotational diversity of collected poses."""
    rpys = [rot_to_euler_xyz(r) for r in r_g2b]
    rpys_arr = np.array(rpys)
    rng = rpys_arr.max(axis=0) - rpys_arr.min(axis=0)
    print("Rotational diversity (Euler XYZ, rad):")
    print(f"  {'':>8}  {'rx':>8}  {'ry':>8}  {'rz':>8}")
    print(f"  range  :  {rng[0]:>8.3f}  {rng[1]:>8.3f}  {rng[2]:>8.3f}")
    print(f"  std    :  {rpys_arr.std(axis=0)[0]:>8.3f}  {rpys_arr.std(axis=0)[1]:>8.3f}  {rpys_arr.std(axis=0)[2]:>8.3f}")
    if rng[1] < 1.0:
        print(f"  WARNING: ry range is only {rng[1]:.3f} rad ({np.degrees(rng[1]):.1f} deg) — poor diversity around Y axis")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve eye-to-hand from colleague JSON")
    parser.add_argument(
        "--json",
        default=os.path.join(_repo_root, "eye_to_hand_samples.json"),
    )
    parser.add_argument("--method", default="shah", choices=["shah", "li"])
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"error: file not found: {args.json}")
        return 1

    r_g2b, t_g2b, r_t2c, t_t2c = load_samples(args.json)
    print(f"Loaded {len(r_g2b)} samples from {args.json}")
    print(f"Method: {args.method}")
    if len(r_g2b) < 15:
        print(f"WARNING: only {len(r_g2b)} samples — recommend 15–30 for reliable calibration")
    print()

    diagnose_samples(r_g2b, t_g2b)

    r_cam2base, t_cam2base, r_tag2end, t_tag2end = solve_eye_to_hand_raw(
        r_g2b, t_g2b, r_t2c, t_t2c, args.method
    )

    print_transform("T[cam|base]  (camera in base frame)", r_cam2base, t_cam2base)
    print_transform("T[tag|end]   (tag in end-effector frame)", r_tag2end, t_tag2end)

    # Sanity check: for each sample, compute T[tag|base] two ways and compare
    print()
    print("Consistency check — T[tag|base] from each sample (should be constant):")
    print(f"  {'sample':>6}  {'tx_mm':>10}  {'ty_mm':>10}  {'tz_mm':>10}")
    print("  " + "-" * 40)

    t_cb = make_transform(r_cam2base, t_cam2base)
    tag_positions = []
    for i, (r_tc, t_tc) in enumerate(zip(r_t2c, t_t2c)):
        # T[tag|base] = T[cam|base] @ T[tag|cam]
        t_tag_base = t_cb @ make_transform(r_tc, t_tc)
        tx, ty, tz = t_tag_base[:3, 3]
        tag_positions.append([tx, ty, tz])
        print(f"  {i+1:>6}  {tx:>10.2f}  {ty:>10.2f}  {tz:>10.2f}")

    positions = np.array(tag_positions)
    std = positions.std(axis=0)
    max_dev = np.abs(positions - positions.mean(axis=0)).max(axis=0)
    print()
    print(f"  std_mm   : [{std[0]:8.2f}, {std[1]:8.2f}, {std[2]:8.2f}]  (pass < 5)")
    print(f"  max_dev_mm: [{max_dev[0]:8.2f}, {max_dev[1]:8.2f}, {max_dev[2]:8.2f}]  (pass < 10)")

    pos_std = float(np.linalg.norm(std))
    if pos_std < 5.0:
        print(f"\n  PASS  position std = {pos_std:.2f} mm")
    elif pos_std < 10.0:
        print(f"\n  WARNING  position std = {pos_std:.2f} mm (borderline)")
    else:
        print(f"\n  FAIL  position std = {pos_std:.2f} mm")
        print("  → 数据不足或多样性不够，建议重新采集 20+ 个样本，确保三个旋转轴都有充分变化")

    return 0


if __name__ == "__main__":
    sys.exit(main())
