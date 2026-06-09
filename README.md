# Equal-weight branching particle methods for non-conservative transport–reaction–diffusion equations

Reference implementation and reproduction scripts for the manuscript

> *Adaptive equal-weight branching particle methods for non-conservative transport–reaction–diffusion equations*,
> Liyao Lyu and Huan Lei.

**Idea in one paragraph.** Non-conservative reaction terms give particle methods exponential weights: a weighted
update is unbiased, but under localized growth the represented mass concentrates onto a few particles and the
local effective sample size collapses exactly where the solution grows. This code implements an *equal-weight
stochastic branching* reaction step that converts mass creation into integer particle count (with a Poisson
birth–death kernel and a minimum-variance / stochastic-rounding integer kernel), and compares it head-to-head
with the weighted representation under shared transport randomness.

## Repository layout

| Path | Paper section | Contents |
|---|---|---|
| `experiments/mms/` | §5.1 | Manufactured-solution verification (error vs `N`, `τ`, `K`) |
| `experiments/branch_vs_weighted/` | §5.2 | Controlled branching-vs-weighted comparison + cost-matched baseline |
| `experiments/keller_segel/mass_balance/` | §5.3 | Conservative KS case with exact mass balance |
| `experiments/keller_segel/concentration/` | §5.3 | Pre-blow-up concentration case (zero-extended Fourier basis on an adaptive window) |
| `experiments/keller_segel/logistic/` | §5.3 | Non-conservative logistic KS case |
| `experiments/highdim/` | §5.4 | 4D/6D kinetic stress test + FHT low-rank marginal diagnostics |
| `experiments/allen_cahn/` | Appendix B | Allen–Cahn sanity check |
| `reference_results/` | Tables 2–4, Fig. 3–10 | CSV outputs of the production runs used in the paper, with the exact configs |

Each experiment directory is self-contained (no cross-directory imports) and has its own `README.md` (for the
three newer experiments) and a SLURM submission script `run_me`.

## Environment

Python ≥ 3.10 with [JAX](https://github.com/jax-ml/jax) (CPU is sufficient; `x64` is enabled by the scripts):

```bash
pip install -r requirements.txt
```

The SLURM scripts assume a conda environment named `heat`; edit `run_me` for your cluster.

## Quick start (smoke tests, a few minutes on CPU)

```bash
cd experiments/branch_vs_weighted && python experiment.py --smoke && python plot.py
cd ../mms                         && python experiment.py --test    # unit tests: kernel unbiasedness + MMS residual
cd ../highdim                     && python experiment.py --smoke
```

## Reproducing the paper figures and tables

| Paper item | Command |
|---|---|
| Fig. errors-vs-N/τ/K, Table MMS (§5.1) | `cd experiments/mms && python experiment.py && python plot.py` |
| Table branch-vs-weighted, Figs. snapshots/nESS/L²/boxplot (§5.2) | `cd experiments/branch_vs_weighted && python experiment.py && python plot.py` |
| Cost-matched weighted rows of the §5.2 table | `cd experiments/branch_vs_weighted && python cost_match.py` |
| KS mass balance (§5.3) | `cd experiments/keller_segel/mass_balance && python simulation.py <N>` |
| KS concentration (§5.3) | `cd experiments/keller_segel/concentration && python simulation.py <N>` (reference: `python finite_difference.py`) |
| KS logistic (§5.3) | `cd experiments/keller_segel/logistic && python simulation.py <N>` |
| Table/Figs. high-dimensional (§5.4) | `cd experiments/highdim && python experiment.py && python experiment.py --d6 && python plot.py` |
| Allen–Cahn (App. B) | `cd experiments/allen_cahn && python simulation.py <N>` |

On SLURM: `sbatch run_me` (full config) or `sbatch run_me --smoke` inside each experiment directory.
Production runs in the paper used fixed seed lists declared in each `experiment.py` `CONFIG`; the exact
configurations of the production runs are archived under `reference_results/*/config_used.json` together with
the resulting metrics CSVs.

## Notes

- All experiments run on **fixed-size particle buffers with an active mask** so that JAX/XLA compiles each time
  step once; branching compaction is performed host-side with `numpy.repeat`.
- In the branching-vs-weighted comparisons, all methods share the **same initial particles and the same
  Brownian-increment stream** per seed; only the reaction representation differs.
- `experiments/highdim/` vendors three files of the functional-hierarchical-tensor (FHT) reconstruction from
  [Xun-Tang123/FHT_for_deans_equation](https://github.com/Xun-Tang123/FHT_for_deans_equation)
  (Tang & Ying, arXiv:2503.22816, MIT license); see `THIRD_PARTY.md`. The FHT is used only as a low-rank
  *diagnostic* — the dynamics never depend on it.

## License

MIT — see `LICENSE`. The vendored FHT files retain their upstream MIT license and attribution.
