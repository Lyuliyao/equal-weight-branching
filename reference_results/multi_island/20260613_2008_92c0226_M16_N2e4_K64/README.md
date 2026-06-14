# Separated growth-island benchmark (paper §5.2) — production run

Production reference for the local-degeneracy rebuttal experiment
(`experiments/branch_vs_weighted/multi_island.py`).

## Configuration

| | |
|---|---|
| PDE | `∂_t u = D Δu + (λ G_multi(x) − β) u` on `T² = [−π,π]²` |
| islands | `M = 16` on a 4×4 grid, `a_m = 1 + 0.25 sin(2π m/M)` |
| parameters | `σ=0.16, D=0.01, λ=12, β=0.8, u0≡1, T=0.8, τ=1e-3` |
| particles | `N0 = 2×10⁴`, seeds 0–7 |
| reconstruction | Fourier `K = 64` (figures only; `E_m` is reconstruction-free) |
| reference | Fourier Strang split-step, `512²` grid (island masses converged to 0.2% vs `1024²`) |
| diagnostic regions | `B_m = {exp(−d_T(x,c_m)²/2σ²) ≥ 0.5}` (disjoint, asserted at run time) |
| resampling trigger | global nESS `< 0.5` (systematic resampling) |
| methods | `weighted`, `weighted_ess_resample`, `minvar_branch`, `poisson_branch` |

Exact config / git hash / package versions / seeds / datetime are in
`config.json` and `manifest.json`.

## How it was run

One SLURM array task per seed (`run_multi_island.sb`, array 0–7), then merged:

```bash
RUNDIR=<this dir> sbatch experiments/branch_vs_weighted/run_multi_island.sb
python experiments/branch_vs_weighted/merge_multi_island.py --in_dir <this dir> --seeds 0 1 2 3 4 5 6 7
```

Per-seed runs go to `seed_<i>/` (not committed — large); the merge produces the
combined CSVs, `fields_*.npz`, and `clouds/` here.

## Outputs

- `metrics_summary.csv` — per-method means (± std) over seeds: global `L²`, global
  nESS, max:mean weight, `mean/median/max E_m`, `#{E_m>20%}`, min/median local
  effective count, particle-steps.
- `per_seed_metrics.csv`, `time_series.csv`, `island_masses.csv`,
  `island_local_ess.csv` — the full per-seed / per-island / per-time records.
- `fields_ref.npz`, `fields_seed0.npz` — reference + per-method final fields for
  the snapshot figures.
- `clouds/cloud_{weighted,minvar_branch}_seed0.npz` — final particle clouds used
  by the resolution-hybrid Demo 2.
- `figures/figure_*.{pdf,png}` and `plot_data/figure_*.npz`.

## Regenerate the figures (no solver rerun)

```bash
python experiments/branch_vs_weighted/plot_multi_island.py --results_dir <this dir>
```

This reads the CSVs + `fields_*.npz` and writes `figures/` and `plot_data/`:
final-field comparison, per-island `E_m` heatmap, global-vs-local degeneracy time
series, diagnostic-selected zoom inset, and `E_m`-vs-amplitude.

## One-line result

Global ESS-triggered resampling keeps a healthy global nESS yet fails the weakest
islands locally; equal-weight branching maintains the local effective count in
every island (its defining property), demonstrating that **global ESS is not a
local degeneracy diagnostic** (see `metrics_summary.csv`).
