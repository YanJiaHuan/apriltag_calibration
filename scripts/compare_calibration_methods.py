#!/usr/bin/env python3
"""
Compare different hand-eye calibration methods.

This script runs calibration with all available methods (tsai, park, horaud,
andreff, daniilidis) and compares their results.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def run_calibration(csv_path: str, method: str, max_reproj: float, min_samples: int) -> dict:
    """
    Run calibration with specified method.

    Args:
        csv_path (str): Input CSV path.
        method (str): Calibration method name.
        max_reproj (float): Max reprojection error.
        min_samples (int): Minimum samples required.

    Returns:
        dict: Calibration results or error info.
    """
    cmd = [
        "python3",
        "scripts/handeye_calibrate.py",
        "--input-csv",
        csv_path,
        "--method",
        method,
        "--max-reproj-error",
        str(max_reproj),
        "--min-samples",
        str(min_samples),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout

        # Parse output
        lines = output.strip().split("\n")
        cam_t = None
        cam_r = None
        tag_t = None
        tag_r = None
        valid_samples = None

        for line in lines:
            if "valid_samples:" in line:
                valid_samples = int(line.split(":")[1].strip())
            elif "t_mm:" in line and cam_t is None:
                # First t_mm is camera|end
                cam_t = line.split("[")[1].split("]")[0]
            elif "rpy_rad:" in line and cam_r is None:
                # First rpy_rad is camera|end
                cam_r = line.split("[")[1].split("]")[0]
            elif "t_mm:" in line and cam_t is not None and tag_t is None:
                # Second t_mm is apriltag|base
                tag_t = line.split("[")[1].split("]")[0]
            elif "rpy_rad:" in line and cam_r is not None and tag_r is None:
                # Second rpy_rad is apriltag|base
                tag_r = line.split("[")[1].split("]")[0]

        return {
            "success": True,
            "method": method,
            "valid_samples": valid_samples,
            "camera_end_t": cam_t,
            "camera_end_r": cam_r,
            "tag_base_t": tag_t,
            "tag_base_r": tag_r,
        }

    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "method": method,
            "error": e.stderr,
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
        description="Compare different hand-eye calibration methods"
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

    print("=" * 80)
    print("Hand-Eye Calibration Method Comparison")
    print("=" * 80)
    print(f"Input CSV: {args.input_csv}")
    print(f"Max reproj error: {args.max_reproj_error}")
    print(f"Min samples: {args.min_samples}")
    print("=" * 80)

    results = []

    for method in methods:
        print(f"\n[{method.upper()}] Running calibration...")
        result = run_calibration(
            args.input_csv, method, args.max_reproj_error, args.min_samples
        )
        results.append(result)

        if result["success"]:
            print(f"  ✓ Success ({result['valid_samples']} samples)")
        else:
            print(f"  ✗ Failed")

    # Print comparison table
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    print("\nCamera|End (T_end_cam) Translation [mm]:")
    print(f"{'Method':<12} {'X':>12} {'Y':>12} {'Z':>12}")
    print("-" * 50)
    for result in results:
        if result["success"]:
            t = result["camera_end_t"]
            print(f"{result['method']:<12} {t}")
        else:
            print(f"{result['method']:<12} {'FAILED':>12}")

    print("\nCamera|End (T_end_cam) Rotation [rad]:")
    print(f"{'Method':<12} {'RX':>12} {'RY':>12} {'RZ':>12}")
    print("-" * 50)
    for result in results:
        if result["success"]:
            r = result["camera_end_r"]
            print(f"{result['method']:<12} {r}")
        else:
            print(f"{result['method']:<12} {'FAILED':>12}")

    print("\nAprilTag|Base (T_base_tag) Translation [mm]:")
    print(f"{'Method':<12} {'X':>12} {'Y':>12} {'Z':>12}")
    print("-" * 50)
    for result in results:
        if result["success"]:
            t = result["tag_base_t"]
            print(f"{result['method']:<12} {t}")
        else:
            print(f"{result['method']:<12} {'FAILED':>12}")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("1. Choose the method with most consistent camera|end results")
    print("2. Validate each method using validate_handeye_calibration.py")
    print("3. If results vary significantly, collect more diverse data")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
