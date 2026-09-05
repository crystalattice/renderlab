"""Negative tests for the approved mask and containment gates."""

import unittest

import numpy as np

from validate_preparation import mask_checks, output_metrics, validate


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.core = np.zeros((4, 4), dtype=np.uint8)
        self.core[1, 1] = 255
        self.gen = self.core.copy()
        self.gen[1, 2] = 255
        self.comp = self.core.copy()
        self.comp[1, 2] = 128

    def test_approved_package(self):
        self.assertEqual(validate()[0]["status"], "PASS")

    def test_core_cannot_be_feathered(self):
        self.comp[1, 1] = 254
        with self.assertRaises(AssertionError):
            mask_checks(self.core, self.gen, self.comp)

    def test_feather_cannot_escape_generation(self):
        self.comp[0, 0] = 1
        with self.assertRaises(AssertionError):
            mask_checks(self.core, self.gen, self.comp)

    def test_generation_must_contain_core(self):
        self.gen[1, 1] = 0
        with self.assertRaises(AssertionError):
            mask_checks(self.core, self.gen, self.comp)

    def test_generation_must_be_binary(self):
        self.gen[1, 2] = 254
        with self.assertRaises(AssertionError):
            mask_checks(self.core, self.gen, self.comp)

    def test_single_channel_outside_error_fails(self):
        source = np.zeros((896, 1160, 3), dtype=np.uint8)
        raw = np.full_like(source, 255)
        core = np.zeros((896, 1160), dtype=np.uint8)
        core[10, 10] = 255
        production = source.copy()
        production[10, 10] = 255
        production[0, 0, 2] = 1
        with self.assertRaises(AssertionError):
            output_metrics(source, raw, production, core, core)

    def test_one_core_pixel_different_from_raw_fails(self):
        source = np.zeros((896, 1160, 3), dtype=np.uint8)
        raw = np.full_like(source, 255)
        core = np.zeros((896, 1160), dtype=np.uint8)
        core[10, 10] = 255
        with self.assertRaises(AssertionError):
            output_metrics(source, raw, source, core, core)


if __name__ == "__main__":
    unittest.main()
