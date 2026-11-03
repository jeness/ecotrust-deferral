import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from governance.decision_learning import SupportGate


class SupportGateTests(unittest.TestCase):
    def test_all_distance_variants_flag_far_shift(self):
        rng = np.random.default_rng(42)
        train = rng.normal(size=(1000, 5))
        shifted = rng.normal(loc=8.0, size=(100, 5))
        for metric in ["mahalanobis_empirical", "mahalanobis_ledoit_wolf",
                       "euclidean"]:
            gate = SupportGate(eps=0.05, metric=metric).fit(train)
            self.assertGreater(gate.out_of_support(shifted).mean(), 0.99)
            self.assertAlmostEqual(gate.out_of_support(train).mean(), 0.05,
                                   delta=0.01)

    def test_invalid_configuration_rejected(self):
        with self.assertRaises(ValueError):
            SupportGate(eps=0.0)
        with self.assertRaises(ValueError):
            SupportGate(metric="unknown")


if __name__ == "__main__":
    unittest.main()
