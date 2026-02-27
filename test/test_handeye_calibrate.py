#!/usr/bin/env python3
"""
Unit tests for hand-eye calibration math helpers.
"""

import math
import unittest

try:
    import numpy as np
except Exception:
    np = None

from scripts import handeye_calibrate


@unittest.skipIf(np is None, "numpy not available")
class TestHandeyeCalibrateMath(unittest.TestCase):
    """Unit tests for core math functions."""

    def test_euler_roundtrip_identity(self):
        """Euler XYZ identity should roundtrip through rotation matrix."""
        r = handeye_calibrate.euler_xyz_to_rot(0.0, 0.0, 0.0)
        rx, ry, rz = handeye_calibrate.rot_to_euler_xyz(r)
        self.assertAlmostEqual(rx, 0.0, places=6)
        self.assertAlmostEqual(ry, 0.0, places=6)
        self.assertAlmostEqual(rz, 0.0, places=6)

    def test_euler_roundtrip_z_90(self):
        """Euler XYZ Z-rotation should roundtrip approximately."""
        r = handeye_calibrate.euler_xyz_to_rot(0.0, 0.0, math.pi / 2.0)
        rx, ry, rz = handeye_calibrate.rot_to_euler_xyz(r)
        self.assertAlmostEqual(rx, 0.0, places=6)
        self.assertAlmostEqual(ry, 0.0, places=6)
        self.assertAlmostEqual(rz, math.pi / 2.0, places=6)

    def test_transform_inverse(self):
        """Inverting a transform should produce identity when multiplied."""
        r = handeye_calibrate.euler_xyz_to_rot(0.1, -0.2, 0.3)
        t = np.array([10.0, -5.0, 3.0])
        tmat = handeye_calibrate.make_transform(r, t)
        inv = handeye_calibrate.invert_transform(tmat)
        ident = tmat @ inv
        self.assertAlmostEqual(float(ident[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(ident[1, 1]), 1.0, places=6)
        self.assertAlmostEqual(float(ident[2, 2]), 1.0, places=6)
        self.assertAlmostEqual(float(ident[0, 3]), 0.0, places=6)
        self.assertAlmostEqual(float(ident[1, 3]), 0.0, places=6)
        self.assertAlmostEqual(float(ident[2, 3]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
