import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TorchTests(unittest.TestCase):
    def test_causal_nll_alignment(self):
        import torch

        from nast.signals import causal_token_nll

        logits = torch.full((1, 4, 5), -20.0)
        labels = torch.tensor([[-100, 2, 3, 4]])
        logits[0, 0, 2] = 20.0
        logits[0, 1, 3] = 20.0
        logits[0, 2, 4] = 20.0
        loss = causal_token_nll(logits, labels)
        self.assertEqual(tuple(loss.shape), (1, 4))
        self.assertLess(float(loss[:, 1:].max()), 1.0e-5)


if __name__ == "__main__":
    unittest.main()
