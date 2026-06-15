# q_window sensitivity of the particle core-collapse time (Experiment B)

**Status: robustness check for the `T_core` result. NOT a paper claim by itself.** Git `f4af0ae`.

Tests whether the Fourier-window quantile `q_window` (which sets the drift window scale
`L(t)`, hence `h_eff=L/K`) explains the particle-vs-LDG `T_core` offset. `current_fourier`,
`K=10`, `τ=2e-7`, 1000 steps; `q_window ∈ {0.5,0.65,0.8,0.9}` × `N ∈ {8e4,3.2e5}` × 4 seeds
= 32 runs. `T_core` via the repaired common-grid seed-mean + 1000× seed bootstrap (q-set
{0.1,0.2,0.3}); LDG reference `T_core = 1.215e-4`.

## Result (N=3.2e5)

| q_window | T_core (boot) | offset vs LDG | CI ∋ LDG? | outside_v_frac | R_0.2/h_eff |
|---|---|---|---|---|---|
| 0.5 | 1.363e-4 | +12% | yes | 0.81 | 1.89 |
| 0.65 | 1.315e-4 | +8% | yes | 0.77 | 1.47 |
| **0.8** (default) | **1.259e-4** | **+3.6%** | yes | 0.72 | 0.85 |
| 0.9 | 1.164e-4 | −4% | yes | 0.64 | 0.51 |

## Reading (plan B4)

- **Within-uncertainty agreement is ROBUST to `q_window`:** every N=3.2e5 seed-bootstrap CI
  **contains the LDG value 1.215e-4**. The result does not hinge on the window choice.
- **A more core-local window does NOT reduce the offset.** The default `q_window=0.8` is
  already closest to LDG (+3.6%); `q=0.5/0.65` overshoot (+8–12%) *and* shed most of the
  chemical field (outside_v_frac → 0.77–0.81), while `q=0.9` slightly undershoots (−4%).
  So the residual ~4% offset is **not** a window-resolution artifact — consistent with it
  being within statistical noise (Case B-negative in the sense of "no lever," which here is
  good: the default needs no tuning).
- **The expected resolve-core/shed-mass tradeoff is visible** (figure b): smaller `q_window`
  raises `R_0.2/h_eff` (finer inner-core drift) but raises `outside_v_frac` (more v outside
  the reconstruction window). The default `q=0.8` keeps `outside_v_frac` lowest among the
  resolved options without a `T_core` penalty.

**Decision:** the particle `T_core`↔LDG agreement is `q_window`-robust; keep the default
`q_window=0.8`. This supports reporting `T_core` as a stable concentration-time diagnostic
(no `q_window` cherry-picking) and does not change the "no continuum blow-up time" stance.

Files: `qwin*/diag_*.csv`, `qwindow_summary.csv/.json`, `figures/qwindow_sensitivity.{pdf,png}`.
Snapshots/logs git-ignored. Regenerate: `analyze_qwindow_sensitivity.py --sweep_dir . --ldg_T 1.215e-4`
then `plot_qwindow_sensitivity.py --sweep_dir . --ldg_T 1.215e-4`.
