import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.experiment import save_results
from src.model import IterativeTransformerBlock
from src.utils import (
    compute_component_jacobian_norms,
    compute_jacobian_spectral_norm,
    compute_theoretical_bound,
    setup_experiment,
)


def main():
    parser = argparse.ArgumentParser(description="Check the composed Jacobian bound.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=64)
    parser.add_argument("--nu-values", type=float, nargs="+", default=np.linspace(0.1, 2, 10).tolist())
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()
    if any(not np.isfinite(nu) or nu <= 0 for nu in args.nu_values):
        parser.error("step sizes must be finite and positive")

    torch.set_num_threads(args.threads)
    setup_experiment(args.seed, args.output_dir)
    model = IterativeTransformerBlock(args.dim, args.heads, args.ffn_dim).double().eval()
    Y = torch.randn(1, args.seq_len, args.dim, dtype=torch.float64) * 0.5
    C = torch.randn_like(Y) * 0.1
    M, F = compute_component_jacobian_norms(model, Y)

    measurements = []
    for nu in tqdm(args.nu_values, desc="Checking bound"):
        model.nu = nu
        empirical = compute_jacobian_spectral_norm(model, Y, C)
        bound = compute_theoretical_bound(model, Y, C, M=M, F=F)
        if not np.isfinite([empirical, bound]).all():
            raise RuntimeError(f"Non-finite result at nu={nu}")
        if empirical > bound + 1e-9 * max(1.0, bound):
            raise AssertionError(f"Bound violated at nu={nu}: {empirical} > {bound}")
        measurements.append({"nu": nu, "jacobian_norm": empirical, "bound": bound})

    config = vars(args).copy()
    config["output_dir"] = str(args.output_dir)
    config.update({
        "dtype": "float64", "Y_scale": 0.5, "C_scale": 0.1,
        "norm_method": "dense autograd Jacobian and SVD",
        "fixed_model_and_inputs": True, "msa_norm": M, "ffn_norm": F,
        "rms_eps": model.rms1.eps,
    })
    save_results(args.output_dir / "exp1_bounds.json", config, measurements)

    plt.figure()
    plt.plot(args.nu_values, [r["jacobian_norm"] for r in measurements], "o-", label=r"$\|J_F\|_2$")
    plt.plot(args.nu_values, [r["bound"] for r in measurements], "--", label="Composed bound")
    plt.xlabel(r"Step size $\nu$")
    plt.ylabel("Spectral norm")
    plt.title("Jacobian bound at fixed weights and inputs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.output_dir / "fig1_bounds_validation.png", dpi=300)
    plt.close()
    print(f"Bound holds at all {len(measurements)} step sizes. Results: {args.output_dir}")


if __name__ == "__main__":
    main()
