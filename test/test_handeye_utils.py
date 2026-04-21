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
