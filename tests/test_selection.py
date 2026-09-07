import unittest

import numpy as np

from nast.config import SelectorConfig
from nast.selection import NASTSelector


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.shape = (2, 10)
        self.response = np.zeros(self.shape, dtype=bool)
        self.response[:, 2:] = True
        base = np.linspace(0.1, 0.9, self.shape[1])
        self.ci = np.tile(base, (2, 1))
        self.pu = np.full(self.shape, 0.2)
        self.tda = np.tile(base[::-1], (2, 1))
        self.gradient = np.tile(np.arange(self.shape[1]), (2, 1)).astype(float)

    def test_selected_tokens_are_eligible_candidates(self):
        result = NASTSelector(SelectorConfig()).select(
            self.ci, self.pu, self.tda, self.gradient, self.response
        )
        self.assertTrue(np.all(result.selected_mask <= result.candidate_mask))
        self.assertTrue(np.all(result.candidate_mask <= self.response))
        self.assertTrue(np.all(result.selected_count >= 1))

    def test_pu_gate_filters_high_confidence_tokens(self):
        pu = self.pu.copy()
        pu[:, 4] = 0.01
        config = SelectorConfig(use_ci=False, use_tda=False, pu_threshold=0.05)
        result = NASTSelector(config).select(
            self.ci, pu, self.tda, self.gradient, self.response
        )
        self.assertTrue(np.all(result.pu_noise_mask[:, 4]))
        self.assertFalse(np.any(result.candidate_mask[:, 4]))

    def test_all_filtered_recovers_minimum_candidate(self):
        pu = np.zeros(self.shape)
        config = SelectorConfig(
            use_ci=False,
            use_tda=False,
            pu_threshold=0.5,
            min_candidate_tokens=1,
        )
        result = NASTSelector(config).select(
            self.ci, pu, self.tda, self.gradient, self.response
        )
        self.assertTrue(np.all(result.candidate_count == 1))
        self.assertTrue(np.all(result.selected_count == 1))
        self.assertEqual(int(result.recovered_mask.sum()), 2)

    def test_candidate_denominator_obeys_equation(self):
        config = SelectorConfig(
            use_ci=False,
            use_pu=False,
            use_tda=False,
            r_min=0.5,
            r_max=0.5,
            entropy_weight=0.0,
            loss_weight=0.0,
            budget_denominator="candidate",
        )
        result = NASTSelector(config).select(
            self.ci, self.pu, self.tda, self.gradient, self.response
        )
        np.testing.assert_array_equal(result.selected_count, np.array([4, 4]))

    def test_response_denominator_is_capped_by_candidates(self):
        pu = self.pu.copy()
        pu[:, 2:6] = 0.0
        config = SelectorConfig(
            use_ci=False,
            use_tda=False,
            pu_threshold=0.05,
            r_min=0.5,
            r_max=0.5,
            entropy_weight=0.0,
            loss_weight=0.0,
            budget_denominator="response",
        )
        result = NASTSelector(config).select(
            self.ci, pu, self.tda, self.gradient, self.response
        )
        # Four candidates survive and the response-denominator budget requests four.
        np.testing.assert_array_equal(result.candidate_count, np.array([4, 4]))
        np.testing.assert_array_equal(result.selected_count, np.array([4, 4]))

    def test_adaptive_budget_is_bounded(self):
        config = SelectorConfig(use_ci=False, use_pu=False, use_tda=False)
        result = NASTSelector(config).select(
            self.ci,
            self.pu,
            self.tda,
            self.gradient,
            self.response,
            normalized_instance_loss=np.array([0.0, 1.0]),
        )
        self.assertTrue(np.all(result.budget_ratio >= config.r_min))
        self.assertTrue(np.all(result.budget_ratio <= config.r_max))
        self.assertGreaterEqual(result.budget_ratio[1], result.budget_ratio[0])

    def test_selection_is_deterministic(self):
        selector = NASTSelector(SelectorConfig())
        first = selector.select(self.ci, self.pu, self.tda, self.gradient, self.response)
        second = selector.select(self.ci, self.pu, self.tda, self.gradient, self.response)
        np.testing.assert_array_equal(first.selected_mask, second.selected_mask)

    def test_disabled_selector_keeps_every_response_token(self):
        result = NASTSelector(SelectorConfig(enabled=False)).select(
            self.ci, self.pu, self.tda, self.gradient, self.response
        )
        np.testing.assert_array_equal(result.selected_mask, self.response)
        np.testing.assert_array_equal(result.selected_count, np.array([8, 8]))
        self.assertFalse(np.any(result.noise_mask))

    def test_random_ranking_is_reproducible(self):
        config = SelectorConfig(
            ranking_strategy="random",
            random_seed=17,
            use_ci=False,
            use_pu=False,
            use_tda=False,
            r_min=0.25,
            r_max=0.25,
            entropy_weight=0.0,
            loss_weight=0.0,
        )
        first = NASTSelector(config).select(
            self.ci, self.pu, self.tda, self.gradient, self.response
        )
        second = NASTSelector(config).select(
            self.ci, self.pu, self.tda, self.gradient, self.response
        )
        np.testing.assert_array_equal(first.selected_mask, second.selected_mask)

    def test_tda_threshold_can_use_full_sequence_population(self):
        response = np.array([[False, False, True, True]])
        ci = np.ones((1, 4))
        pu = np.ones((1, 4))
        tda = np.array([[0.0, 0.1, 0.8, 0.9]])
        gradient = np.ones((1, 4))
        config = SelectorConfig(
            use_ci=False,
            use_pu=False,
            use_tda=True,
            tda_classes=2,
            r_min=0.5,
            r_max=0.5,
        )
        result = NASTSelector(config).select(
            ci,
            pu,
            tda,
            gradient,
            response,
            tda_population_mask=np.ones((1, 4), dtype=bool),
        )
        self.assertFalse(np.any(result.tda_noise_mask))


if __name__ == "__main__":
    unittest.main()
