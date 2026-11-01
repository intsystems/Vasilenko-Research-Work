"""Finite-horizon residuals of untrained recurrent blocks on selected inputs."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.data import get_synthetic_sample, load_cifar10_sample
from src.experiment import environment_metadata
from src.model import IterativeTransformerBlock
from src.utils import setup_experiment


def run_dynamics_on_input(model, x_init, nu_values, steps=1000, context_scale=0.1):
    """Return per-sample Frobenius residuals, with shape (sample, step) per nu."""
    if x_init.ndim != 3 or steps < 1:
        raise ValueError("Expected (batch, sequence, feature) inputs and positive steps")
    results = {}
    original_nu = model.nu
    try:
        with torch.no_grad():
            for nu in nu_values:
                model.nu = float(nu)
                state = x_init.clone()
                context = context_scale * x_init
                residuals = torch.empty(
                    (state.shape[0], steps), dtype=state.dtype, device=state.device
                )
                for step in range(steps):
                    next_state = model.forward_step(state, context)
                    residuals[:, step] = torch.linalg.vector_norm(
                        (next_state - state).flatten(start_dim=1), dim=1
                    )
                    state = next_state
                if not torch.isfinite(residuals).all():
                    raise RuntimeError(f"Non-finite residual encountered for nu={nu}")
                results[float(nu)] = residuals.cpu().numpy()
    finally:
        model.nu = original_nu
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--image-indices", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--nu-values", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="Absolute per-sample Frobenius residual threshold")
    parser.add_argument("--tail-steps", type=int, default=20,
                        help="All of these final steps must satisfy the threshold")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--download", action="store_true", help="Download missing CIFAR-10 data")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Skip CIFAR-10; output is explicitly marked synthetic-only")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must contain distinct nonnegative integers")
    if (len(set(args.image_indices)) != len(args.image_indices)
            or any(index < 0 or index >= 10000 for index in args.image_indices)):
        parser.error("--image-indices must contain distinct indices in [0, 10000)")
    if (len(set(args.nu_values)) != len(args.nu_values)
            or any(not np.isfinite(nu) or nu <= 0 for nu in args.nu_values)):
        parser.error("--nu-values must contain distinct, finite positive values")
    if args.steps < 1 or not 1 <= args.tail_steps <= args.steps:
        parser.error("Require 1 <= --tail-steps <= --steps")
    if not np.isfinite(args.tolerance) or args.tolerance <= 0 or args.threads < 1:
        parser.error("--tolerance and --threads must be positive")
    return args


def summarize_residuals(residuals, nu_values, tolerance, tail_steps):
    """Count trials whose final residual window meets a fixed threshold."""
    summary = []
    for nu_index, nu in enumerate(nu_values):
        trials = residuals[:, nu_index].reshape(-1, residuals.shape[-1])
        passed = np.all(trials[:, -tail_steps:] <= tolerance, axis=1)
        summary.append({
            "nu": float(nu),
            "trials": len(trials),
            "reached_threshold": int(passed.sum()),
            "fraction_reached_threshold": float(passed.mean()),
            "final_residual_min": float(trials[:, -1].min()),
            "final_residual_median": float(np.median(trials[:, -1])),
            "final_residual_max": float(trials[:, -1].max()),
            "trial_reached_threshold": passed.reshape(residuals.shape[0], -1).tolist(),
        })
    return summary


def plot_residuals(datasets, nu_values, tolerance, output_path):
    figure, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5),
                                squeeze=False, sharey=True)
    colors = [f"C{index % 10}" for index in range(len(nu_values))]
    # Zero residuals remain exact zeros in the saved data. Only the log plot clips
    # them to a documented display floor.
    display_floor = min(1e-12, tolerance * 1e-3)
    for axis, (name, residuals) in zip(axes[0], datasets.items()):
        step_axis = np.arange(residuals.shape[-1])
        for nu_index, (nu, color) in enumerate(zip(nu_values, colors)):
            trials = residuals[:, nu_index].reshape(-1, residuals.shape[-1])
            low = np.maximum(trials.min(axis=0), display_floor)
            high = np.maximum(trials.max(axis=0), display_floor)
            median = np.maximum(np.median(trials, axis=0), display_floor)
            axis.fill_between(step_axis, low, high, color=color, alpha=0.15)
            axis.plot(step_axis, median, color=color, label=fr"$\nu={nu}$")
        axis.axhline(tolerance, color="black", linestyle=":", linewidth=1,
                     label=f"Threshold {tolerance:g}")
        title = "Gaussian inputs" if name == "synthetic" else "Selected CIFAR-10 images"
        axis.set_title(f"{title}\n{len(trials)} trials; median and min–max")
        axis.set_yscale("log")
        axis.set_xlabel(r"Iteration $t$")
        axis.grid(True)
        axis.legend(fontsize=10)
    axes[0, 0].set_ylabel(r"Residual $\|Y^{(t+1)}-Y^{(t)}\|_F$")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return display_floor


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    setup_experiment(seed=0, save_dir=args.output_dir)
    sequence_length, dim, heads, ffn_hidden = 32, 64, 4, 128
    sample_count = len(args.image_indices)
    cifar_metadata = None
    cifar_inputs = None
    if not args.synthetic_only:
        cifar_inputs, cifar_metadata = load_cifar10_sample(
            data_dir=args.data_dir, indices=args.image_indices,
            download=args.download, return_metadata=True,
        )
        # Match the original image standardization, now independently per image.
        image_stds = cifar_inputs.flatten(start_dim=1).std(dim=1)
        if torch.any(image_stds <= 0):
            raise ValueError("Cannot standardize a constant CIFAR-10 image")
        cifar_inputs = cifar_inputs / image_stds[:, None, None]
        cifar_metadata["image_standard_deviations"] = image_stds.tolist()
        cifar_metadata["additional_preprocessing"] = (
            "Divide each image by its sample standard deviation (correction=1), "
            "then apply a shared random, untrained Linear(96,64) projection per seed"
        )

    dataset_names = ["synthetic"] if args.synthetic_only else ["synthetic", "cifar10"]
    residuals = {name: [] for name in dataset_names}
    initial_states = {name: [] for name in dataset_names}
    rng_streams = []
    for seed in args.seeds:
        # Disjoint streams make inputs and weights independent of execution order.
        streams = {"seed": seed, "model": 3 * seed,
                   "synthetic_inputs": 3 * seed + 1, "projection": 3 * seed + 2}
        rng_streams.append(streams)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(streams["model"])
            model = IterativeTransformerBlock(dim, heads, ffn_hidden).eval()
        inputs = {"synthetic": get_synthetic_sample(
            sample_count, sequence_length, dim, seed=streams["synthetic_inputs"]
        )}
        if cifar_inputs is not None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(streams["projection"])
                projector = nn.Linear(cifar_inputs.shape[-1], dim).eval()
            with torch.no_grad():
                inputs["cifar10"] = projector(cifar_inputs)
        # Reuse these exact block weights for both datasets and every nu.
        for name, initial in inputs.items():
            histories = run_dynamics_on_input(model, initial, args.nu_values, args.steps)
            residuals[name].append(np.stack([histories[nu] for nu in args.nu_values]))
            initial_states[name].append(initial.numpy())
        print(f"Finished seed {seed}", flush=True)

    residuals = {name: np.stack(values) for name, values in residuals.items()}
    arrays = {f"{name}_residuals": values for name, values in residuals.items()}
    arrays.update({f"{name}_initial_states": np.stack(values)
                   for name, values in initial_states.items()})
    arrays.update(seeds=np.asarray(args.seeds), nu_values=np.asarray(args.nu_values))
    np.savez_compressed(args.output_dir / "exp3_dynamics.npz", **arrays)
    summaries = {name: summarize_residuals(values, args.nu_values, args.tolerance,
                                         args.tail_steps)
                 for name, values in residuals.items()}
    display_floor = plot_residuals(residuals, args.nu_values, args.tolerance,
                                   args.output_dir / "fig3_contraction.png")
    metadata = {
        "experiment": "finite_horizon_residuals",
        "environment": environment_metadata(),
        "config": {
            "seeds": args.seeds, "rng_streams": rng_streams,
            "synthetic_samples_per_seed": sample_count,
            "nu_values": args.nu_values, "steps": args.steps,
            "S": sequence_length, "D": dim, "H": heads, "D_ff": ffn_hidden,
            "context": "C = 0.1 * initial_state; fixed along each trajectory",
            "synthetic_distribution": "N(0,I)",
            "weights": "Untrained; identical block weights across datasets for each seed",
            "dtype": str(torch.get_default_dtype()),
            "threads": args.threads, "synthetic_only": args.synthetic_only,
        },
        "cifar10": cifar_metadata,
        "residual_axes": ["seed", "nu", "sample", "step"],
        "initial_state_axes": ["seed", "sample", "token", "feature"],
        "criterion": {
            "metric": "Absolute Frobenius norm per sample: ||Y(t+1)-Y(t)||_F",
            "tolerance": args.tolerance, "tail_steps": args.tail_steps,
            "rule": "All residuals in the final tail_steps are <= tolerance",
            "scope": "Finite-horizon observations; no guarantee of convergence or contraction",
        },
        "plot": {"center": "median", "band": "minimum to maximum, not a confidence interval",
                 "log_display_floor": display_floor},
        "summary": summaries,
        "files": {"arrays": "exp3_dynamics.npz", "plot": "fig3_contraction.png"},
    }
    (args.output_dir / "exp3_dynamics.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for name, summary in summaries.items():
        for result in summary:
            print(f"{name}, nu={result['nu']:g}: "
                  f"{result['reached_threshold']}/{result['trials']} trials below "
                  f"{args.tolerance:g} for all final {args.tail_steps} steps")
    print(f"Saved residuals, metadata, and plot to {args.output_dir}")


if __name__ == "__main__":
    main()
