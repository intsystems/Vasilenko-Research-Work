import unittest

import numpy as np
import torch

from scripts.exp3_contraction import run_dynamics_on_input
from src.model import IterativeTransformerBlock


class DynamicsTests(unittest.TestCase):
    def test_batched_residuals_match_individual_trajectories(self):
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = IterativeTransformerBlock(4, 2, 8, nu=0.3).double().eval()
            inputs = torch.randn(3, 2, 4, dtype=torch.float64)
        nu_values = [0.1, 1.0]
        batched = run_dynamics_on_input(model, inputs, nu_values, steps=6)
        for sample in range(len(inputs)):
            separate = run_dynamics_on_input(model, inputs[sample:sample + 1],
                                            nu_values, steps=6)
            for nu in nu_values:
                self.assertEqual(batched[nu].shape, (3, 6))
                np.testing.assert_allclose(batched[nu][sample], separate[nu][0],
                                           rtol=1e-10, atol=1e-12)
        self.assertEqual(model.nu, 0.3)


if __name__ == "__main__":
    unittest.main()
