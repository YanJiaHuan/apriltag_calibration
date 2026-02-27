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

    def test_estimate_tag_in_base_no_shadow_bug(self):
        """estimate_tag_in_base should handle multiple samples without shape errors."""
        r_be = handeye_calibrate.euler_xyz_to_rot(0.0, 0.0, 0.0)
        t_be = np.array([0.0, 0.0, 0.0])
        r_end_cam = handeye_calibrate.euler_xyz_to_rot(0.0, 0.0, 0.0)
        t_end_cam = np.array([0.0, 0.0, 0.0])

        r_t2c = handeye_calibrate.euler_xyz_to_rot(0.0, 0.0, 0.0)
        t_t2c = np.array([0.0, 0.0, 0.0])

        r_base_tag, t_base_tag = handeye_calibrate.estimate_tag_in_base(
            [r_be, r_be],
            [t_be, t_be],
            r_end_cam,
            t_end_cam,
            [r_t2c, r_t2c],
            [t_t2c, t_t2c],
        )

        self.assertEqual(r_base_tag.shape, (3, 3))
        self.assertEqual(t_base_tag.shape, (3,))

    def test_estimate_tag_in_base_chain(self):
        """estimate_tag_in_base should chain base->end->cam->tag correctly."""
        r_id = handeye_calibrate.euler_xyz_to_rot(0.0, 0.0, 0.0)
        t_be = np.array([1.0, 0.0, 0.0])
        t_ec = np.array([0.0, 1.0, 0.0])
        t_ct = np.array([0.0, 0.0, 1.0])

        r_base_tag, t_base_tag = handeye_calibrate.estimate_tag_in_base(
            [r_id],
            [t_be],
            r_id,
            t_ec,
            [r_id],
            [t_ct],
        )

        self.assertAlmostEqual(float(t_base_tag[0]), 1.0, places=6)
        self.assertAlmostEqual(float(t_base_tag[1]), 1.0, places=6)
        self.assertAlmostEqual(float(t_base_tag[2]), 1.0, places=6)

    def test_opencv_to_robot_rotation(self):
        """OpenCV-to-robot rotation should be proper (det=1)."""
        r_map = handeye_calibrate.opencv_to_robot_rotation()
        det = float(np.linalg.det(r_map))
        self.assertAlmostEqual(det, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
