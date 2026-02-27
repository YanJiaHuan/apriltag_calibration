#!/usr/bin/env python3
"""
Unit tests for simple math/utility functions in the calibration scripts.

These tests avoid hardware and network dependencies by focusing on
pure functions only.
"""

import math
import unittest

from scripts import remote_apriltag_server
from scripts import robot_eye_in_hand_client


class TestHandeyeUtils(unittest.TestCase):
    """Unit tests for conversion utilities."""

    def test_meters_to_mm_server(self):
        """meters_to_mm should convert meters to millimeters (server)."""
        self.assertAlmostEqual(remote_apriltag_server.meters_to_mm(1.0), 1000.0)
        self.assertAlmostEqual(remote_apriltag_server.meters_to_mm(0.123), 123.0)

    def test_meters_to_mm_robot(self):
        """meters_to_mm should convert meters to millimeters (robot)."""
        self.assertAlmostEqual(robot_eye_in_hand_client.meters_to_mm(2.0), 2000.0)
        self.assertAlmostEqual(robot_eye_in_hand_client.meters_to_mm(0.5), 500.0)

    def test_rotation_matrix_identity(self):
        """Identity rotation should yield zero Euler angles."""
        r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        rx, ry, rz = remote_apriltag_server.rotation_matrix_to_euler_xyz(r)
        self.assertAlmostEqual(rx, 0.0)
        self.assertAlmostEqual(ry, 0.0)
        self.assertAlmostEqual(rz, 0.0)

    def test_rotation_matrix_x_90(self):
        """90-degree rotation about X should return rx=pi/2."""
        r = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        rx, ry, rz = remote_apriltag_server.rotation_matrix_to_euler_xyz(r)
        self.assertAlmostEqual(rx, math.pi / 2.0, places=6)
        self.assertAlmostEqual(ry, 0.0, places=6)
        self.assertAlmostEqual(rz, 0.0, places=6)

    def test_rotation_matrix_z_90(self):
        """90-degree rotation about Z should return rz=pi/2."""
        r = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        rx, ry, rz = remote_apriltag_server.rotation_matrix_to_euler_xyz(r)
        self.assertAlmostEqual(rx, 0.0, places=6)
        self.assertAlmostEqual(ry, 0.0, places=6)
        self.assertAlmostEqual(rz, math.pi / 2.0, places=6)

    def test_reproj_error_ok(self):
        """Reprojection error thresholding should behave as expected."""
        self.assertTrue(remote_apriltag_server.is_reproj_error_ok(0.5, 1.0))
        self.assertFalse(remote_apriltag_server.is_reproj_error_ok(2.0, 1.0))
        self.assertFalse(remote_apriltag_server.is_reproj_error_ok(None, 1.0))

    def test_reproj_error_ok_robot(self):
        """Robot-side reprojection error check should behave as expected."""
        self.assertTrue(robot_eye_in_hand_client.is_reproj_error_ok(0.5, 1.0))
        self.assertFalse(robot_eye_in_hand_client.is_reproj_error_ok(2.0, 1.0))
        self.assertFalse(robot_eye_in_hand_client.is_reproj_error_ok("", 1.0))


if __name__ == "__main__":
    unittest.main()
