import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.experiment import save_results
from src.model import IterativeTransformerBlock
from src.utils import compute_jacobian_spectral_norm, jacobian_spectral_norm, setup_experiment


def limiting_map(model, Y, C):
    """Limit of one step as nu grows, when every row of C + TB(Y) is nonzero."""
    delta = C + model.forward_subblocks(Y)
    rms = delta.square().mean(dim=-1, keepdim=True).sqrt()
    if torch.any(rms == 0):
        raise ValueError("The limiting map is undefined for a zero update row")
    # The outer epsilon is divided by nu**2 and vanishes in this limit.
    return model.rms_final.gamma * delta / rms


def main():
    parser = argparse.ArgumentParser(description="Measure the large-step Jacobian limit.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=64)
    parser.add_argument("--nu-values", type=float, nargs="+", default=np.logspace(-1, 5, 20).tolist())
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()
    if any(not np.isfinite(nu) or nu <= 0 for nu in args.nu_values):
        parser.error("step sizes must be finite and positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")

    torch.set_num_threads(args.threads)
    setup_experiment(args.seeds[0], args.output_dir)
    runs = []
    for seed in tqdm(args.seeds, desc="Fixed-model sweeps"):
        torch.manual_seed(seed)
        model = IterativeTransformerBlock(args.dim, args.heads, args.ffn_dim).double().eval()
        Y = torch.randn(1, args.seq_len, args.dim, dtype=torch.float64) * 0.5
        C = torch.randn_like(Y) * 0.1
        with torch.no_grad():
            min_update_norm = torch.linalg.vector_norm(C + model.forward_subblocks(Y), dim=-1).min().item()
        limit = jacobian_spectral_norm(lambda y: limiting_map(model, y, C), Y)
        norms = []
        for nu in args.nu_values:
            model.nu = nu
            norms.append(compute_jacobian_spectral_norm(model, Y, C))
        if not np.isfinite(norms + [limit, min_update_norm]).all():
            raise RuntimeError(f"Non-finite result for seed={seed}")
        runs.append({"seed": seed, "norms": norms, "limit_norm": limit, "min_update_row_norm": min_update_norm})

    values = np.array([run["norms"] for run in runs])
    mean, std = values.mean(axis=0), values.std(axis=0)
    mean_limit = float(np.mean([run["limit_norm"] for run in runs]))
    config = vars(args).copy()
    config["output_dir"] = str(args.output_dir)
    config.update({
        "dtype": "float64", "Y_scale": 0.5, "C_scale": 0.1,
        "norm_method": "dense autograd Jacobian and SVD",
        "fixed_model_and_inputs_within_seed": True, "rms_eps": model.rms1.eps,
        "shading": "population standard deviation across seeds",
    })
    save_results(args.output_dir / "exp2_asymptotic.json", config, {
        "runs": runs, "mean": mean.tolist(), "std": std.tolist(), "mean_limit_norm": mean_limit,
    })

    plt.figure()
    plt.plot(args.nu_values, mean, "o-", label="Mean Jacobian norm")
    plt.fill_between(args.nu_values, mean - std, mean + std, alpha=0.2, label="One standard deviation")
    plt.axhline(mean_limit, color="tab:red", linestyle="--", label="Mean norm of limiting Jacobian")
    plt.xscale("log")
    plt.xlabel(r"Step size $\nu$")
    plt.ylabel(r"Spectral norm $\|J_F\|_2$")
    plt.title("Large-step behavior at fixed weights and inputs")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(args.output_dir / "fig2_asymptotic.png", dpi=300)
    plt.close()
    print(f"Completed {len(runs)} sweeps. Results: {args.output_dir}")


if __name__ == "__main__":
    main()
