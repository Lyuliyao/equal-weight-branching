# `sf_blob` — physical-space blob-residual solver-field sweep (corrected dual-CFL)

**Status: diagnostic record. NOT used in the paper.** Git `84e6111`.

Tests the principled Scenario-C operator from the taper sweep: the local high part of
the chemotactic-drift v-field is an `η_h` **Gaussian blob** (KDE) instead of a high-`Kl`
local **spectrum** (whose gradient amplifies high-mode Monte-Carlo noise ~`Kl`):

    v̂ = v_lo + χ·(η_h*μ_v − η_h*(v_lo dx)),   v_lo = P_{Kg=8},   χ = radial taper.

This is the FIRST sweep with the **corrected dual-CFL diagnostics**: `drift_cfl_solver_field`
is the field that actually drives/aborts the u-particles; `drift_cfl_fourier_diag` is the
single-`K` diagnostic. (The prior `sf_taper` sweep logged only the Fourier diagnostic; its
"smoother drift 2.6 vs 4.5" claim was withdrawn — see `../sf_taper_*/README.md`.)

## Design

24 tasks = 6 configs × 4 seeds, `N=8e4`, `K=10`, `τ=2.5e-7`, 800 steps (→ `t=2e-4`),
`--dg_readout_n 80 160`. Configs: `current_fourier`, `spectral` (Kl=24, taper_hi=0.25,
the best from `sf_taper`), `blob frac_L` at `c_h = h/L = 0.04 / 0.06 / 0.09` (bracketing
and exceeding the spectral scale `L/Kl≈0.042`), and one `blob core_spacing c_h=1.2`
(the plan's spacing rule, `h~R₀.₈/√N₀.₈`, ~25× finer than `L/Kl`).

## Result

| config | abort t (mean±std) | **cflS** (real solver) | cflF (Fourier diag) | residual_E @1e-4 |
|---|---|---|---|---|
| current_fourier | 1.905e-4 ±1.4e-5 | 4.45 | 4.45 *(same field)* | — |
| spectral taper0.25 | 1.887e-4 ±1.9e-5 | **3.63** | 2.63 | — |
| blob c_h=0.04 | 1.65e-4 ±2.5e-5 | 4.38 | 1.88 | 0.032 |
| blob c_h=0.06 | **1.95e-4 ±0.8e-5** | **3.54** | 3.13 | 0.018 |
| blob c_h=0.09 | **1.99e-4 ±0.2e-5** | **3.53** | 5.22 | 0.009 |
| blob core_spacing | 1.9e-5 *(aborts immediately)* | 4.90 | 0.81 | — |

- **Q1 (stability) — corrected-metric win.** The smoother blobs (`c_h=0.06,0.09`) recover/
  exceed the global-`K` abort time **and** have a genuinely smaller *real solver* CFL (3.5)
  than both global-`K` (4.45) and the spectral hybrid (3.63). The spacing-rule blob aborts
  almost immediately — confirming the bandwidth analysis (`h` ~25× too fine → shot noise).
- **The dual-CFL fix matters.** The spectral hybrid's true solver CFL is **3.63**, not the
  2.63 the old code logged (confirming the withdrawn `sf_taper` claim). And blob `c_h=0.09`'s
  Fourier diag (5.22) ≫ its solver CFL (3.53): the blob makes a **tighter core with a smoother
  drift**; the old single-column code would have mislabeled it "spiky" and could have wrongly
  aborted it. See `figures/figure_blob_sweep.{pdf,png}` panel (b).
- **Q3 (resolution) — same honest catch.** Reconstruction-free `R_0.2` at `t=1e-4` is within
  ~4× seed noise across configs; residual energy is small (1–3%) and *shrinks* as the blob
  smooths, i.e. the most-stable setting (`c_h=0.09`) barely corrects `v_lo`. No demonstrable
  improvement in tracked concentration.

**Decision:** the blob is the best-behaved solver-level local operator (most stable, smoothest
real drift, no spectral Gibbs), and the corrected metric shows a real — if modest — stability
gain over both global-`K` and the spectral hybrid. But, like the spectral taper, it does **not**
demonstrably improve the pre-singular tracked concentration (Q3 within seed noise; the stable
setting adds <1% residual). So it remains a **reconstruction/diagnostic option, not a validated
accuracy improvement** — not in the paper. Consistent with §5.5 (core radii reconstruction-free;
reconstructed peak/L2 bandwidth-sensitive).

## Regenerate

```bash
RUNDIR=$PWD/reference_results/keller_segel_ldg_pp/sf_blob_<new_id> \
    sbatch experiments/keller_segel/ldg_comparison/run_solver_field_blob_sweep.sb   # ~25-35 min
cd experiments/keller_segel/ldg_comparison
python analyze_blob_sweep.py --sdir <this_dir> --t_match 1.0e-4   # -> blob_sweep_compare.json
python plot_blob_sweep.py    --sdir <this_dir>                    # -> figures/ + plot_data/
```

Per-run `u`-snapshots (`*/snapshots/*.npz`) and logs are git-ignored; `diag_*.csv`,
`blob_sweep_compare.json`, `abort_diagnostics.json`, `figures/`, `plot_data/` are kept.
The 12 runs that aborted before `2e-4` have an `abort_diagnostics.json` (solver-vs-Fourier
CFL, max-grad particle index/position).
