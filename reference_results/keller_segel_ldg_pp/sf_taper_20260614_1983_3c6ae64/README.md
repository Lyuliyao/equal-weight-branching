# `sf_taper` — solver-level residual-drift local-operator smoothness sweep

**Status: diagnostic / negative-leaning record. NOT used in the paper.**

This sweep tests the Scenario-C fix for the Form I solver-level residual drift
(`v̂ = v_lo + χ(v_hi − v_lo)` fed into the u-particle chemotactic drift; see
`REVISION_RESULTS.md` §3.8 and `hybrid_vfield.py`).

## Question

The sharp two-level hybrid (`v_hi = P_{Kl=24}` on the core window) aborts the
drift-CFL guard *earlier* than the plain global-`K=10` Fourier drift, because
differentiating `P_{Kl}` amplifies high-mode Monte-Carlo particle noise in `∇v̂`.
A Gaussian taper of width `h_hi` on the high-`Kl` part is exactly an `η_h` blob
residual (the blob's Fourier transform is `exp(−h²k²/2)`), i.e. a **smoother local
operator**. Smaller `h_hi` = smoother = less noise, less core bandwidth.

Does smoothing recover the drift stability of the global-`K` drift while keeping the
hybrid's inner-core reconstruction?

## Design

16 tasks = 4 configs × 4 seeds, `N=8e4`, `K=10`, `τ=2.5e-7`, 800 steps (→ `t=2e-4`),
report times `6e-5, 1.2e-4, 2e-4`. Configs:

| config | drift field |
|---|---|
| `current_fourier` | global `K=10` Fourier ∇v |
| `two_level_taper0.5` | Form I residual, `Kg=8`, `Kl=24`, `h_hi=0.50` (sharpest) |
| `two_level_taper0.35` | Form I residual, `h_hi=0.35` |
| `two_level_taper0.25` | Form I residual, `h_hi=0.25` (smoothest) |

## Result

Abort time recovers **monotonically** as the local operator smooths:

| config | abort `t` (mean ± std) | max drift_cfl | R_0.2 @ 1e-4 (recon-free) |
|---|---|---|---|
| `two_level_taper0.5` | 1.39e-4 ± 2.1e-5 | 1.4 | 0.0193 |
| `two_level_taper0.35` | 1.58e-4 ± 1.9e-5 | 1.8 | 0.0171 |
| `two_level_taper0.25` | **1.89e-4 ± 2.0e-5** | 2.6 | 0.0152 |
| `current_fourier` | 1.91e-4 ± 1.4e-5 | 4.5 | 0.0109 |

- **Q1 (drift stability):** `h_hi=0.25` recovers the global-`K` abort time (0.99×)
  with a *smoother* drift (cfl_max 2.6 vs 4.5). Confirms the early abort was a
  high-`Kl` noise artifact, removable by smoothing — not intrinsic dynamics.
- **Q3 (core resolution) — honest catch:** the `R_0.2/h_core` "gain" (1.0→3.1) is
  dominated by the `Kl=24`-vs-`K=10` grid factor, not better-tracked concentration.
  The reconstruction-free `R_0.2` per-seed spread is ~4×, so the config means are
  within seed noise; no config has a demonstrably tighter actual core.

**Decision:** Form I residual drift is drift-stable once smoothed, but a
reconstruction/diagnostic option, not a validated accuracy improvement (costs
~2.5×/step). Not in paper.

## Regenerate

```bash
# (re-run the dynamics — SLURM, ~20 min slowest task)
RUNDIR=$PWD/reference_results/keller_segel_ldg_pp/sf_taper_<new_id> \
    sbatch experiments/keller_segel/ldg_comparison/run_solver_field_sweep.sb

# analysis + figure from the saved diag CSVs only (no solver rerun):
cd experiments/keller_segel/ldg_comparison
python analyze_taper_sweep.py --sdir <this_dir> --t_match 1.0e-4   # -> taper_sweep_compare.json
python plot_taper_sweep.py   --sdir <this_dir> --seed 1            # -> figures/ + plot_data/
```

Per-run `u`-snapshots (`*/snapshots/*.npz`, regenerable) and logs are git-ignored;
the `diag_*.csv`, `taper_sweep_compare.json`, `figures/`, and `plot_data/` are kept.
