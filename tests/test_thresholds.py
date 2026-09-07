import unittest

import numpy as np

from nast.thresholds import iqr_lower_threshold, low_alignment_threshold, multi_otsu_thresholds


class ThresholdTests(unittest.TestCase):
    def test_iqr_lower_fence(self):
        values = np.array([0.0, 10.0, 10.0, 10.0, 10.0])
        threshold = iqr_lower_threshold(values, multiplier=1.5)
        self.assertEqual(threshold, 10.0)
        self.assertTrue(values[0] < threshold)

    def test_three_cluster_multi_otsu(self):
        values = np.concatenate(
            [
                np.linspace(0.02, 0.12, 20),
                np.linspace(0.42, 0.55, 20),
                np.linspace(0.82, 0.95, 20),
            ]
        )
        first, second = multi_otsu_thresholds(values, classes=3, bins=32)
        self.assertLess(first, second)
        self.assertGreater(first, 0.1)
        self.assertLess(first, 0.5)
        self.assertGreater(second, 0.5)
        self.assertLess(second, 0.9)

    def test_constant_vector_is_defined(self):
        thresholds = multi_otsu_thresholds(np.ones(12), classes=3)
        self.assertEqual(thresholds, (1.0, 1.0))

    def test_short_vector_uses_quantile(self):
        values = np.array([0.1, 0.2, 0.7, 0.9])
        threshold = low_alignment_threshold(values, small_sample_quantile=0.25)
        self.assertAlmostEqual(threshold, float(np.quantile(values, 0.25)))

    def test_empty_iqr_is_non_filtering(self):
        self.assertEqual(iqr_lower_threshold(np.array([])), float("-inf"))


if __name__ == "__main__":
    unittest.main()
