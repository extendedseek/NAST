import unittest

import numpy as np

from nast.trainer import LossNormalizer


class LossNormalizerTests(unittest.TestCase):
    def test_outputs_are_bounded_and_ordered(self):
        normalizer = LossNormalizer()
        values = normalizer.normalize(np.array([1.0, 2.0, 3.0]))
        self.assertTrue(np.all(values >= 0.0))
        self.assertTrue(np.all(values <= 1.0))
        self.assertTrue(np.all(np.diff(values) > 0))


if __name__ == "__main__":
    unittest.main()
