# Local reconstruction diagnostics (paper §5.x) — reference run

Reference outputs for the hybrid-reconstruction diagnostics
(`experiments/resolution_hybrid/`). These show that the resolution bottleneck of a
concentrating particle measure is the *reconstruction map*, and that the fix is
**local**, not a uniform global bandwidth increase.

## Contents

| dir | demo | what it shows |
|---|---|---|
| `core_demo/` | Demo 1 (KS core) | reconstruct a concentrated Keller–Segel core (exact Gaussian, mass `10π`, `a=84`) at several bandwidths |
| `island_demo/` | Demo 2 (multi-island) | per-island mass via particle counting vs global/local reconstruction, on the saved branching cloud |

Each demo dir has `config.json`, `manifest.json`, metrics CSVs,
`residual_acceptance.csv`, `plot_data/figure_*.npz`,
`residual_particles/window_*.npz`, and `figures/*.{pdf,png}`.

## Key results

**Demo 1 (core_demo/core_demo_metrics.csv).** Against the *exact* Gaussian core:

| scheme | global modes | peak rel. err | reconstruction-free? |
|---|---|---|---|
| global low `K_g=8` | 64 | 13% | — |
| global high `K=40` | 1600 | 0.5% | — |
| hybrid (low + local window) | 64 (+local) | 1.3% | — |
| HT residual particles | 64 | 7.8% | — |
| `R_0.5`, `R_0.8` (particles) | — | exact to 3 digits | **yes** |

The hybrid recovers the high-`K` core accuracy while keeping the **global**
bandwidth at `K_g=8`. The reconstruction-free core radii match the analytic
values; the reconstructed peak/`L²` are labelled bandwidth-sensitive. The
residual-particle **accept rate is an enrichment rate**, not a Metropolis
acceptance and not a particle-dynamics resampling step.

**Demo 2 (island_demo/island_reconstruction.csv).** Particle counting
(reconstruction-free) is accurate, while integrating a low-`K` global Fourier
field over the sharp islands `B_m` is catastrophic (mass leaks outside `B_m`); a
local spectral window recovers counting accuracy.

## Regenerate

```bash
cd experiments/resolution_hybrid
python core_window_demo.py --N 400000 --Kg 8 --Kfull 40 --Kl 40 --B_target 3000 --blob --out_dir <core_demo>
python plot_hybrid_reconstruction.py --results_dir <core_demo>
python reconstruct_from_snapshot.py --cloud <multi_island>/clouds/cloud_minvar_branch_seed0.npz \
       --ref_island_masses <multi_island>/island_masses.csv --out_dir <island_demo>
python plot_resolution.py --results_dir <island_demo>
```

All plot data live under `plot_data/`; figures regenerate without rerunning any
solver.
