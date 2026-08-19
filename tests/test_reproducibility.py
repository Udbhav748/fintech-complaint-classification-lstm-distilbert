"""Tests for seed control and environment information capture."""

import random
import unittest

import numpy as np

from src.reproducibility import EnvironmentInfo, set_seed


class TestReproducibility(unittest.TestCase):
    def test_seed_control_numpy_and_python(self):
        set_seed(1234)
        val1_py = random.random()
        val1_np = np.random.rand(5)

        # Re-set seed and verify exact deterministic output
        set_seed(1234)
        val2_py = random.random()
        val2_np = np.random.rand(5)

        self.assertEqual(val1_py, val2_py)
        np.testing.assert_array_equal(val1_np, val2_np)

    def test_environment_capture(self):
        env = EnvironmentInfo.capture()
        self.assertTrue(len(env.python_version) > 0)
        self.assertIn("pandas", env.packages)
        self.assertIn("numpy", env.packages)
        self.assertIn("scikit-learn", env.packages)
        self.assertIn("device_type", env.device)


if __name__ == "__main__":
    unittest.main()
