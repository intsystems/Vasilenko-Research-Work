# Jacobian Analysis of a Recurrent Transformer Block

Code for the paper accepted at AINL 2026. The analysis gives a bound on the
Jacobian of one recurrent update with attention, an FFN, and RMSNorm. Under a
nonzero-update assumption, that Jacobian remains bounded as the step size grows.
This does not by itself imply convergence of repeated updates or bounded
gradients through arbitrarily many iterations.

## Paper and slides

- [Accepted paper](paper/Jacobian_Analysis_of_a_Recurrent_Transformer_Block.pdf)
- [Revised paper](paper/revised.pdf), [LaTeX source](paper/revised.tex), and
  [bibliography](paper/references.bib)
- [Recovered manuscript sources](paper/archive/README.md)
- [Updated slides](slides/slides.pdf)

The accepted PDF is kept unchanged. The revised paper corrects the normalization
convention, the large-step proof, the experimental protocol, and the stability
claims. It incorporates the author and related-work updates from the recovered
manuscript sources. The archived versions are kept separately and unchanged.

The figures in `plots/` are recomputed results. The original conference figures
remain in `slides/old_plots_for_conference/`; they use the earlier experimental
protocol and should not be mixed with the revised results.

## Setup

Python 3.13 or later and [uv](https://docs.astral.sh/uv/) are required.

```bash
git clone https://github.com/Dimundel/Jacobian-Analysis-of-a-Recurrent-Transformer-Block.git
cd Jacobian-Analysis-of-a-Recurrent-Transformer-Block
uv sync --frozen
```

## Experiments

All models are randomly initialized and untrained. The scripts run on CPU with
one thread by default. Each writes its numerical results and configuration to
`plots/`, including the library versions used. Use `--output-dir` to keep a new
run separate from the checked-in figures.

```bash
uv run python -m scripts.exp1_bounds
uv run python -m scripts.exp2_asymptotic
uv run python -m scripts.exp3_contraction --download
```

The last command downloads CIFAR-10 only if it is missing. If the dataset is
already available, omit `--download` and optionally set `--data-dir PATH`.
For a run without CIFAR-10, use `--synthetic-only --output-dir /tmp/synthetic-run`.

| Experiment | Default configuration | Measurement |
| --- | --- | --- |
| Bound | S=16, D=32, H=4, FFN=64; seed 42; 10 step sizes from 0.1 to 2 | Dense Jacobian norms and the composed upper bound |
| Large-step limit | Same dimensions; seeds 0–9; 20 step sizes from 0.1 to 100000 | Norms at fixed weights and inputs within each sweep |
| Residuals | S=32, D=64, H=4, FFN=128; seeds 0–4; five inputs per dataset; 1000 iterations | Per-input Frobenius residuals at step sizes 0.1, 0.5, and 1 |

Experiments 1 and 2 use float64. Spectral norms are computed from full autograd
Jacobians with SVD. These computations are intended for small diagnostic models;
the Jacobian has `(S * D)^2` entries for one sequence.

The bound uses the minimum RMS denominator over tokens, the epsilon and learned
scale of each normalization layer, and measured component Jacobian norms. Inputs
and weights stay fixed as the step size changes.

Experiment 2 averages ten complete sweeps. The shaded band is one population
standard deviation across seeds. The horizontal line is the mean norm of the
limiting Jacobian, computed separately for each seed.

Experiment 3 uses float32. CIFAR-10 test images 0–4 become 32 tokens of 96 pixel
values, are standardized per image, and pass through an untrained linear
projection to dimension 64. Each seed uses the same block weights for synthetic
and CIFAR inputs. The fixed context is `0.1 * initial_state`. The plot shows the
median and full range across 25 trajectories per dataset and step size.
A trajectory meets the reported threshold only if all final 20 residuals are at
most `1e-4`. This is a finite-horizon criterion, not a convergence guarantee.

Raw outputs:

- `exp1_bounds.json`: component norms, step sizes, measured norms, and bounds.
- `exp2_asymptotic.json`: each seed's sweep and limiting norm, plus mean and spread.
- `exp3_dynamics.json`: data indices, RNG streams, threshold counts, and metadata.
- `exp3_dynamics.npz`: residuals and initial states for every trajectory. Array
  axes are documented in the JSON; load with `numpy.load(..., allow_pickle=False)`.

The checked-in runs used Python 3.13.7 and PyTorch 2.14.1+cpu in an existing
CPU environment. `uv.lock` retains PyTorch 2.10.0. Exact floating-point results can
differ between environments; each result file records the versions actually used.

## Checks

```bash
uv run python -m unittest discover -s tests -v
```

The tests cover the short-row counterexample, different scales and epsilons,
component norms above one, the large-step Jacobian limit, and per-sample residuals
in a batch.

## Building the documents

With a LaTeX installation that includes `pdflatex` and `bibtex`:

```bash
cd paper
pdflatex revised.tex
bibtex revised
pdflatex revised.tex
pdflatex revised.tex
cd ../slides
pdflatex slides.tex
pdflatex slides.tex
```
