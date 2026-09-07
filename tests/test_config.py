import tempfile
import unittest
from pathlib import Path

from nast.config import load_config


class ConfigTests(unittest.TestCase):
    def test_smoke_config_loads(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "smoke.yaml")
        self.assertEqual(config.experiment_name, "nast-smoke")
        self.assertEqual(config.selector.budget_denominator, "candidate")

    def test_relative_inheritance_deep_merges(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "ablations" / "no-ci.yaml")
        self.assertFalse(config.selector.use_ci)
        self.assertTrue(config.selector.use_pu)
        self.assertEqual(config.model.name_or_path, "Qwen/Qwen2.5-7B-Instruct")

    def test_unknown_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("experiment_name: x\nunknown_field: 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown keys"):
                load_config(path)

    def test_invalid_ratio_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(
                "experiment_name: x\nselector:\n  r_min: 0.8\n  r_max: 0.2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "r_min"):
                load_config(path)

    def test_invalid_literal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(
                "experiment_name: x\nselector:\n  gradient_proxy: imaginary\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "gradient_proxy"):
                load_config(path)

    def test_cycle_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.yaml"
            second = Path(directory) / "second.yaml"
            first.write_text("extends: second.yaml\n", encoding="utf-8")
            second.write_text("extends: first.yaml\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cyclic"):
                load_config(first)


if __name__ == "__main__":
    unittest.main()
