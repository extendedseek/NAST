import unittest

from nast.config import DatasetSpec
from nast.data import _oasst_pairs, _weighted_round_robin, load_normalized_mixture
from nast.formatting import NormalizedExample


class DataTests(unittest.TestCase):
    def test_weighted_interleave_preserves_every_record_once(self):
        left = [NormalizedExample(str(i), str(i), source="left") for i in range(3)]
        right = [NormalizedExample(str(i), str(i), source="right") for i in range(2)]
        result = _weighted_round_robin([left, right], [0.8, 0.2], seed=7)
        self.assertEqual(len(result), 5)
        self.assertEqual(sum(item.source == "left" for item in result), 3)
        self.assertEqual(sum(item.source == "right" for item in result), 2)

    def test_mmlu_is_blocked_from_training_before_loading(self):
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            load_normalized_mixture([DatasetSpec(name="mmlu")], seed=42)

    def test_oasst_language_filter_and_parent_join(self):
        rows = [
            {"message_id": "p1", "parent_id": None, "role": "prompter", "lang": "en", "text": "Hello"},
            {"message_id": "a1", "parent_id": "p1", "role": "assistant", "lang": "en", "text": "Hi"},
            {"message_id": "p2", "parent_id": None, "role": "prompter", "lang": "de", "text": "Hallo"},
            {"message_id": "a2", "parent_id": "p2", "role": "assistant", "lang": "de", "text": "Guten Tag"},
        ]
        examples = list(_oasst_pairs(rows, max_samples=None, language="en"))
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].instruction, "Hello")
        self.assertEqual(examples[0].response, "Hi")


if __name__ == "__main__":
    unittest.main()
