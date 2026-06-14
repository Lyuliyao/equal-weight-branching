# Resolution-hybrid reconstruction diagnostics (paper §5.x)

A small **reconstruction diagnostic**, not a new solver. It shows that the
resolution bottleneck for a concentrating particle measure is the
*reconstruction map* (particle measure → field used in drift/reaction/diagnostics),
and that the fix is **local**, not a uniform global bandwidth increase.

> Particles carry the finite measure and reveal *where* resolution is needed.
> A global spectrum captures the smooth background; local residual particles,
> blobs, or spectral windows capture the unresolved high-frequency concentration.
> The enrichment region is inferred **from the particle cloud**, not by hand.

## Files

| file | role |
|---|---|
| `reconstructors.py` | global low/high-K Fourier reconstruction; **Option B** (global low + local spectral window, residual over the global low); **Option A** (global low + local Gaussian blob); the residual-particle acceptance machinery (`residual_particle_acceptance`, `ht_residual_reconstruction`, `positive_residual_reconstruction`) |
| `detect_windows.py` | reconstruction-free core diagnostics (centroid, `R_q`, covariance, core mass), single-core window detection, multi-core detection from a particle histogram + connected components, and the `Window` taper object |
| `core_window_demo.py` | **Demo 1**: Keller–Segel core reconstruction from a particle cloud (synthetic concentrated Gaussian with *exact* ground truth, or `--cloud <file.npz>` for a real KS snapshot) |
| `reconstruct_from_snapshot.py` | **Demo 2**: per-island reconstruction of the saved multi-island clouds |
| `plot_hybrid_reconstruction.py` | the 5-panel KS-core figure (low-K, hybrid, core profile, retained residual particles, cost vs error) |
| `plot_resolution.py` | the multi-island per-island `E_m`-vs-amplitude figure + accept-rate enrichment indicator |

## The residual-particle acceptance rate is an *enrichment* rate

`residual_particle_acceptance(..., mode="ht"|"positive")` returns the particles
**retained** for a local residual reconstruction. The "accept rate" is a
**reconstruction-enrichment rate** — a fraction of particles kept for the local
residual sketch. It is **NOT** a Metropolis acceptance probability and **NOT** a
new particle-dynamics resampling step; the PDE dynamics are untouched. It is
applied to *saved snapshots*.

- `mode="ht"` — unbiased Horvitz–Thompson sketch of the empirical part of
  `μ − P_{K_g}μ dx` (retained weights divided by the acceptance `q_i`). Use this
  for quantitative tables.
- `mode="positive"` — positive-excess thinning (retain the local positive excess
  over the global spectrum). Positivity-preserving; **labelled positive-only**,
  and the positive/negative residual mass imbalance `ΔM = ∫(r⁺ − r⁻)` is reported.

## Quick start

```bash
# Demo 1: KS-core reconstruction with exact Gaussian ground truth
python core_window_demo.py --N 400000 --Kg 8 --Kfull 40 --Kl 40 --B_target 3000 --blob \
    --out_dir results/core_demo
python plot_hybrid_reconstruction.py --results_dir results/core_demo

# Demo 2: multi-island local reconstruction (needs a saved cloud + reference masses)
python reconstruct_from_snapshot.py \
    --cloud ../branch_vs_weighted/results/multi_island_prod/clouds/cloud_minvar_branch_seed0.npz \
    --ref_island_masses ../branch_vs_weighted/results/multi_island_prod/island_masses.csv \
    --out_dir results/island_demo
python plot_resolution.py --results_dir results/island_demo
```

Use `--cloud <ks_snapshot.npz>` (keys `X, w, mass_per_particle, box`) on Demo 1
to reconstruct a **real** Keller–Segel concentration snapshot instead of the
synthetic Gaussian.

## What the demos report

Reconstruction-free (robust) vs bandwidth-sensitive diagnostics are kept
separate, as required:

- **reconstruction-free**: `R_0.5`, `R_0.8` (particle quantile radii), core mass
  in `B(center, R_0.8)`, particle-counted island mass;
- **bandwidth-sensitive** (labelled as such): reconstructed peak `‖u‖_∞`,
  reconstructed `L²`.

Demo 1 (exact Gaussian core, `mass=10π`, `a=84`): global low-`K_g=8` undershoots
the peak by ~13 %, global high-`K=40` and the hybrid window both recover it to
~1 %, while `R_0.5/R_0.8` from particles match the analytic values exactly. The
hybrid keeps the **global** bandwidth at `K_g`=8 and spends the high resolution
only on the particle-detected core window.

Demo 2 (separated growth islands): particle counting (reconstruction-free) is
accurate; reconstructing-then-integrating at low global `K` is catastrophic for
the sharp islands (mass leaks outside `B_m`); a local spectral window recovers
counting accuracy. Local reconstruction matters most for the weakest islands.

## Outputs

Each run writes `config.json`, `manifest.json` (git hash, command line, package
versions, datetime), metrics CSVs, `residual_acceptance.csv`,
`plot_data/figure_*.npz`, `residual_particles/window_*.npz`, and `figures/*.pdf`
+ `*.png`. Production copies live under
`reference_results/resolution_hybrid/<run_id>/`.

## Caveats

- Residual corrections are *signed*; the minimum field value (negativity) is
  reported. The positive-excess variant is positivity-preserving but
  positive-only, with the residual mass imbalance reported.
- This is a diagnostic of where an adaptive function-approximation / quadrature
  rule should be placed. It is **not** fed back into the time-stepping solver and
  the manuscript does not claim a full adaptive algorithm.
