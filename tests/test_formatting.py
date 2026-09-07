import unittest

from nast.formatting import build_tokenized_example, normalize_record, render_user_prompt


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        ids = [10 + (sum(map(ord, token)) % 100) for token in str(text).split()]
        if add_special_tokens:
            ids = [1, *ids]
        return {"input_ids": ids}


class FormattingTests(unittest.TestCase):
    def test_open_platypus_mapping(self):
        example = normalize_record(
            "open_platypus",
            {"instruction": "Explain", "input": "context", "output": "answer"},
        )
        self.assertEqual(example.context, "context")
        self.assertEqual(example.response, "answer")

    def test_commonsense_choices_and_answer(self):
        example = normalize_record(
            "commonsenseqa",
            {
                "question": "Where?",
                "choices": {"label": ["A", "B"], "text": ["home", "moon"]},
                "answerKey": "A",
            },
        )
        self.assertEqual(example.answer_index, 0)
        self.assertEqual(example.response, "home")
        self.assertIn("A. home", render_user_prompt(example))

    def test_prompt_tokens_are_not_targets(self):
        example = normalize_record(
            "local", {"instruction": "Do this", "response": "Final response"}
        )
        feature = build_tokenized_example(example, FakeTokenizer(), max_length=32, use_chat_template=False)
        response_start = feature["response_mask"].index(True)
        self.assertTrue(all(value == -100 for value in feature["labels"][:response_start]))
        self.assertTrue(all(value != -100 for value in feature["labels"][response_start:]))
        self.assertEqual(len(feature["input_ids"]), len(feature["labels"]))

    def test_response_is_preserved_when_truncating(self):
        example = normalize_record(
            "local",
            {"instruction": " ".join(["long"] * 100), "response": "one two three"},
        )
        feature = build_tokenized_example(example, FakeTokenizer(), max_length=8, use_chat_template=False)
        self.assertLessEqual(len(feature["input_ids"]), 8)
        self.assertGreaterEqual(sum(feature["response_mask"]), 4)  # response plus EOS


if __name__ == "__main__":
    unittest.main()
