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

try:
    import cv2 as _cv2  # noqa: F401
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from scripts import handeye_utils
from scripts.eye_to_hand_calibrate import solve_eye_to_hand_raw


@unittest.skipUnless(CV2_AVAILABLE, "cv2 not available")
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


class TestSolveEyeToHandErrors(unittest.TestCase):
    """Tests for solve_eye_to_hand() error paths (no cv2 needed)."""

    def test_missing_csv_raises(self):
        """solve_eye_to_hand should raise ValueError if CSV path doesn't exist."""
        from scripts.eye_to_hand_calibrate import solve_eye_to_hand

        with self.assertRaises(ValueError) as ctx:
            solve_eye_to_hand("/tmp/does_not_exist_12345.csv")
        self.assertIn("csv not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
