import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from l2d_modern import ms_surrogate_loss, ova_surrogate_loss, PostHocL2D


class L2DModernTests(unittest.TestCase):
    def test_ms_zero_logits_and_gradient_direction(self):
        outputs = torch.zeros((2, 3), requires_grad=True)
        y = torch.tensor([0, 0])
        expert = torch.tensor([0, 1])  # correct, wrong
        loss = ms_surrogate_loss(outputs, expert, y)
        self.assertTrue(np.isclose(float(loss.detach()), 1.5 * np.log(3.0), atol=1e-6))
        loss.backward()
        self.assertLess(float(outputs.grad[0, -1]), 0)  # correct expert encourages deferral
        self.assertGreater(float(outputs.grad[1, -1]), 0)  # wrong expert discourages deferral

    def test_ova_zero_logits_and_gradient_direction(self):
        outputs = torch.zeros((2, 3), requires_grad=True)
        y = torch.tensor([0, 0])
        expert = torch.tensor([0, 1])
        loss = ova_surrogate_loss(outputs, expert, y)
        self.assertTrue(np.isclose(float(loss.detach()), 3.0 * np.log(2.0), atol=1e-6))
        loss.backward()
        self.assertLess(float(outputs.grad[0, -1]), 0)
        self.assertGreater(float(outputs.grad[1, -1]), 0)

    def test_posthoc_model_outputs_finite_scores_and_probabilities(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(80, 5))
        p = np.clip(1 / (1 + np.exp(-X[:, 0])), 0.01, 0.99)
        y = (X[:, 0] + 0.2 * rng.normal(size=80) > 0).astype(int)
        expert = np.where(np.arange(80) % 5, y, 1 - y)
        model = PostHocL2D(method="ova", epochs=20, seed=3).fit(X, p, y, expert)
        scores = model.defer_score(X, p)
        prob = model.estimated_expert_correctness(X, p)
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((prob >= 0) & (prob <= 1)).all())


if __name__ == "__main__":
    unittest.main()
