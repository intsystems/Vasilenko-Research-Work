import os
import math
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd.functional import jacobian
import torch


def setup_experiment(seed=42, save_dir="plots"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.figsize": (10, 6),
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
            "grid.alpha": 0.3,
        }
    )
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def jacobian_spectral_norm(function, x):
    """Compute the spectral norm of the full, flattened Jacobian via SVD.

    This forms a dense matrix and is intended for small diagnostic problems.
    Autograd and SVD errors propagate to the caller rather than dropping samples.
    """
    def flattened_function(x_flat):
        return function(x_flat.reshape(x.shape)).reshape(-1)

    J = jacobian(flattened_function, x.detach().reshape(-1))
    if not torch.isfinite(J).all():
        raise ValueError("Jacobian contains non-finite values")
    return torch.linalg.matrix_norm(J, ord=2).item()


def compute_jacobian_spectral_norm(model, Y, C):
    """Compute ||J_F(Y)||_2 with the context C held fixed."""
    return jacobian_spectral_norm(lambda y: model.forward_step(y, C), Y)


def compute_component_jacobian_norms(model, Y):
    """Return (||J_MSA(Y)||_2, ||J_FFN(Y_prime)||_2)."""
    M = jacobian_spectral_norm(lambda y: model.msa(y, y, y)[0], Y)
    with torch.no_grad():
        attention, _ = model.msa(Y, Y, Y)
        Y_prime = model.rms1(Y + attention)
    F = jacobian_spectral_norm(model.ffn, Y_prime)
    return M, F


def compute_theoretical_bound(model, Y, C, M=None, F=None):
    """Evaluate the composed bound for the RMSNorm used by the model.

    Denominators are minimum row RMS values, including each layer's epsilon.
    M and F may be supplied as component norms or valid upper bounds; missing
    values are measured at Y and Y_prime, respectively, using dense Jacobians.
    """
    with torch.no_grad():
        attention, _ = model.msa(Y, Y, Y)
        input_rms1 = Y + attention
        Y_prime = model.rms1(input_rms1)
        input_rms2 = Y_prime + model.ffn(Y_prime)
        TB_Y = model.rms2(input_rms2)
        U = Y + model.nu * (C + TB_Y)

        r_in = torch.sqrt(
            input_rms1.square().mean(dim=-1) + model.rms1.eps
        ).amin().item()
        r_out = torch.sqrt(
            input_rms2.square().mean(dim=-1) + model.rms2.eps
        ).amin().item()
        r_U = torch.sqrt(
            U.square().mean(dim=-1) + model.rms_final.eps
        ).amin().item()

    if any(not math.isfinite(r) or r <= 0 for r in (r_in, r_out, r_U)):
        raise ValueError("RMSNorm denominators must be finite and positive")

    if M is None:
        M = jacobian_spectral_norm(lambda y: model.msa(y, y, y)[0], Y)
    if F is None:
        F = jacobian_spectral_norm(model.ffn, Y_prime)
    if any(not math.isfinite(n) or n < 0 for n in (M, F)):
        raise ValueError("Component Jacobian norms must be finite and nonnegative")

    g_in = model.rms1.get_gamma_max()
    g_out = model.rms2.get_gamma_max()
    g_U = model.rms_final.get_gamma_max()
    return (g_U / r_U) * (
        1 + abs(model.nu) * (g_out * g_in / (r_out * r_in)) * (1 + F) * (1 + M)
    )
