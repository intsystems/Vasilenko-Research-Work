import unittest

import torch

from scripts.exp2_asymptotic import limiting_map
from src.model import IterativeTransformerBlock


class AsymptoticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = IterativeTransformerBlock(4, 2, 8).double().eval()
        self.model.rms_final.eps = 0.5
        self.Y = torch.randn(1, 2, 4, dtype=torch.float64)
        self.C = torch.randn_like(self.Y) * 0.1
        with torch.no_grad():
            self.model.rms_final.gamma.copy_(torch.tensor([0.5, -1.0, 2.0, 1.5]))

    def test_large_step_jacobian_approaches_computed_limit(self):
        limit = torch.autograd.functional.jacobian(
            lambda y: limiting_map(self.model, y, self.C), self.Y
        )
        errors = []
        for nu in [1e3, 1e4]:
            self.model.nu = nu
            actual = torch.autograd.functional.jacobian(
                lambda y: self.model.forward_step(y, self.C), self.Y
            )
            errors.append(torch.linalg.vector_norm(actual - limit).item())
        self.assertLess(errors[1], 0.11 * errors[0])
        self.assertLess(errors[1], 1e-3)

    def test_zero_update_row_has_no_limiting_map(self):
        with torch.no_grad():
            C = -self.model.forward_subblocks(self.Y)
        with self.assertRaises(ValueError):
            limiting_map(self.model, self.Y, C)


if __name__ == "__main__":
    unittest.main()
