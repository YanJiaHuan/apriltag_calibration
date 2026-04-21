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
