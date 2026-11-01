import math
import unittest

import torch

from src.model import IterativeTransformerBlock
from src.utils import (
    compute_component_jacobian_norms,
    compute_jacobian_spectral_norm,
    compute_theoretical_bound,
    jacobian_spectral_norm,
)


def zero_subblocks(model):
    with torch.no_grad():
        for module in (model.msa, model.ffn):
            for parameter in module.parameters():
                parameter.zero_()
    return model


class JacobianBoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_bound_matches(self, model, Y, C):
        actual = compute_jacobian_spectral_norm(model, Y, C)
        bound = compute_theoretical_bound(model, Y, C)
        self.assertAlmostEqual(bound / actual, 1.0, places=9)
        return actual

    def test_rectangular_jacobian_with_noncontiguous_input(self):
        x = torch.arange(6, dtype=torch.float64).reshape(3, 2).T
        self.assertFalse(x.is_contiguous())

        def function(value):
            return torch.stack((3 * value[0, 0], 4 * value[1, 2])).reshape(2, 1)

        self.assertAlmostEqual(jacobian_spectral_norm(function, x), 4.0)

    def test_minimum_row_prevents_underestimating_bound(self):
        model = zero_subblocks(IterativeTransformerBlock(2, 1, 2).double())
        Y = torch.tensor([[[0.001, 0.0], [1.0, 0.0]]], dtype=torch.float64)
        actual = self.assert_bound_matches(model, Y, torch.zeros_like(Y))
        self.assertGreater(actual, 1000.0)

    def test_all_three_learned_scales_contribute(self):
        model = zero_subblocks(IterativeTransformerBlock(2, 1, 2, nu=0.7).double())
        with torch.no_grad():
            model.rms1.gamma.copy_(torch.tensor([2.0, -3.0]))
            model.rms2.gamma.copy_(torch.tensor([4.0, -5.0]))
            model.rms_final.gamma.copy_(torch.tensor([0.5, -0.7]))
        Y = torch.tensor([[[0.3, 0.0]]], dtype=torch.float64)
        self.assert_bound_matches(model, Y, torch.zeros_like(Y))

    def test_each_layer_epsilon_controls_the_zero_input_jacobian(self):
        model = zero_subblocks(IterativeTransformerBlock(2, 1, 2).double())
        model.rms1.eps = 0.04
        model.rms2.eps = 0.09
        model.rms_final.eps = 0.16
        Y = torch.zeros(1, 2, 2, dtype=torch.float64)
        actual = self.assert_bound_matches(model, Y, torch.zeros_like(Y))
        self.assertAlmostEqual(actual, 2.5 * (1 + 0.5 / (0.2 * 0.3)), places=10)

    def test_component_norms_are_measured_instead_of_assumed_one(self):
        model = zero_subblocks(IterativeTransformerBlock(2, 1, 2, nu=1.0).double())
        with torch.no_grad():
            model.msa.in_proj_weight[4:].copy_(torch.diag(torch.tensor([0.0, 4.0])))
            model.msa.out_proj.weight.copy_(torch.eye(2))
        Y = torch.tensor([[[math.sqrt(2), 0.0]]], dtype=torch.float64)
        M, F = compute_component_jacobian_norms(model, Y)
        self.assertAlmostEqual(M, 4.0)
        self.assertEqual(F, 0.0)
        actual = self.assert_bound_matches(model, Y, torch.zeros_like(Y))
        self.assertAlmostEqual(actual, 3.0, places=6)
        assumed_bound = compute_theoretical_bound(model, Y, torch.zeros_like(Y), M=1, F=1)
        self.assertLess(assumed_bound, actual)

    def test_ffn_norm_is_measured_after_the_first_normalization(self):
        model = zero_subblocks(IterativeTransformerBlock(2, 1, 2).double())
        with torch.no_grad():
            model.ffn[0].weight.copy_(torch.eye(2))
            model.ffn[2].weight.copy_(torch.eye(2))
        Y = torch.tensor([[[10.0, 0.0]]], dtype=torch.float64)
        M, F = compute_component_jacobian_norms(model, Y)
        self.assertEqual(M, 0.0)
        self.assertGreater(F, 1.1)
        self.assertLess(jacobian_spectral_norm(model.ffn, Y), 1.01)

    def test_random_blocks_and_broadcast_context(self):
        for batch_size, broadcast_context in ((1, False), (2, False), (2, True)):
            with self.subTest(batch_size=batch_size, broadcast_context=broadcast_context):
                torch.manual_seed(100 + batch_size)
                model = IterativeTransformerBlock(4, 2, 6, nu=0.8).double()
                Y = torch.randn(batch_size, 2, 4, dtype=torch.float64)
                C = torch.randn(2, 4, dtype=torch.float64) * 0.1
                if not broadcast_context:
                    C = C.unsqueeze(0).expand(batch_size, -1, -1)
                actual = compute_jacobian_spectral_norm(model, Y, C)
                M, F = compute_component_jacobian_norms(model, Y)
                bound = compute_theoretical_bound(model, Y, C, M=M, F=F)
                self.assertGreaterEqual(bound + 1e-10, actual)
                self.assertAlmostEqual(bound, compute_theoretical_bound(model, Y, C), places=10)
                self.assertEqual(model.forward_step(Y, C).shape, Y.shape)

    def test_nonfinite_jacobians_are_not_silently_discarded(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            jacobian_spectral_norm(lambda x: x * float("nan"), torch.ones(2))

    def test_invalid_supplied_component_norms_raise(self):
        model = zero_subblocks(IterativeTransformerBlock(2, 1, 2).double())
        Y = torch.ones(1, 1, 2, dtype=torch.float64)
        for norm in (-1.0, float("nan"), float("inf")):
            with self.subTest(norm=norm):
                with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    compute_theoretical_bound(model, Y, torch.zeros_like(Y), M=norm, F=0)


if __name__ == "__main__":
    unittest.main()
