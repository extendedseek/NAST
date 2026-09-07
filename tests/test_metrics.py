import unittest

import numpy as np

from nast.metrics import (
    exact_match,
    extract_final_answer,
    gradient_mass_retention,
    perplexity,
    topk_gradient_overlap,
)


class MetricTests(unittest.TestCase):
    def test_gradient_mass_retention(self):
        scores = np.array([[1.0, 2.0, 3.0, 4.0]])
        response = np.array([[False, True, True, True]])
        selected = np.array([[False, False, True, True]])
        self.assertAlmostEqual(gradient_mass_retention(scores, selected, response), 7 / 9)

    def test_topk_overlap(self):
        scores = np.array([[1.0, 2.0, 3.0, 4.0]])
        response = np.array([[True, True, True, True]])
        selected = np.array([[False, False, True, True]])
        self.assertEqual(topk_gradient_overlap(scores, selected, response, 2), 1.0)

    def test_answer_extractors(self):
        self.assertEqual(extract_final_answer("work\n#### 1,234"), "1234")
        self.assertEqual(extract_final_answer(r"Thus \\boxed{7/8}."), "7/8")
        self.assertEqual(extract_final_answer(""), "")

    def test_exact_match(self):
        self.assertEqual(exact_match(["The answer is 42."], ["#### 42"]), 1.0)

    def test_perplexity(self):
        self.assertAlmostEqual(perplexity(2.0, 2), np.e)


if __name__ == "__main__":
    unittest.main()
