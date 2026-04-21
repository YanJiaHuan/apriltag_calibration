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
