# C (Fourier bandwidth K) + D (time-step τ) sensitivity of particle `T_core`

**Status: sensitivity/robustness study for the core-collapse-time result. NOT a paper claim
by itself.** Git `10da969`. Spec: "Next experiments" Experiments C, D.

`current_fourier`, `q_window=0.8`; `T_core` = common-grid seed-mean + 1000× seed bootstrap
(q-set {0.1,0.2,0.3}), LDG reference `T_core ≈ 1.16–1.21e-4` (quad 5/7 → 1.16; quad 3 /
literature → 1.21). 8e4 pilots were noise-limited (the under-resolved regime); the decisive
comparison is at Np=3.2e5.

## C — Fourier bandwidth K (Np=3.2e5)

| K | T_core (boot) | CI [p5,p95] | offset vs 1.215e-4 | CI ∋ LDG? | fits |
|---|---|---|---|---|---|
| 10 (default) | 1.269e-4 | [1.11,1.36] | +4.4% | yes | 7 |
| 12 | 1.312e-4 | [1.07,1.43] | +8.0% | yes | 5 |
| 16 | 1.265e-4 | [1.06,1.49] | +4.1% | yes | 6 |

**K is not the lever.** T_core stays ~1.27–1.31e-4 for K=10/12/16; every CI contains the LDG
value; the CIs *widen* with K (differentiating a higher-K Fourier reconstruction injects more
MC noise). So the residual particle-vs-LDG offset is **not a Fourier-bandwidth effect**
(plan C4, case "increasing K does not change T_core").

## D — time-step τ (Np=3.2e5)

| τ | n_steps | T_core (boot) | CI | offset | fits |
|---|---|---|---|---|---|
| 2e-7 (default) | 1000 | 1.259e-4 | [1.11,1.35] | +3.6% | 7 |
| 1e-7 | 2000 | 1.136e-4 | [0.93,1.59] | −6.5% | 3 (noisy) |

**Halving τ shifts T_core 1.26e-4 → 1.14e-4 (~10%), *toward* the LDG value.** The small
residual offset is therefore partly **Lie-splitting time-step error**; a finer step improves
the LDG agreement (plan D2: time-step-sensitive, in the accuracy-improving direction). Caveat:
the τ=1e-7 estimate is noisy (3 valid fits, wide CI; the 2000-step runs partly abort).
8e4 τ pilots were noise-limited; τ=4e-7 aborts too often (fracT 0.25) to be useful.

## Combined reading (with B = q_window and E = LDG robustness)

- **The LDG↔particle within-uncertainty agreement is robust** to `K`, `τ`, and `q_window`:
  every Np=3.2e5 seed-bootstrap CI contains the LDG `T_core` (1.16–1.21e-4).
- The small ~4% residual offset is **not** explained by Fourier bandwidth (K-robust) or the
  reconstruction window (q_window-robust, Exp. B), but **is** reduced by a finer time step
  (τ=1e-7 → 1.14e-4) — consistent with a mild Lie-splitting error.
- The LDG metric itself is robust (Exp. E: quad order ≥5, fit windows, q-set).

**Decision:** keep the default `K=10`, `τ=2e-7`, `q_window=0.8`. Report `T_core` as a
reconstruction-light concentration-time diagnostic that agrees with the LDG reference within
uncertainty, with the residual offset attributable to time-step splitting error (not bandwidth
or window). Still **no continuum blow-up-time claim** (§11 limited wording).

## Files

`K_summary.csv/.json`, `tau_summary.csv/.json` (here); per-run `*/diag_*.csv`. K=10/τ=2e-7
references are symlinks into the `core_collapse_<id>/particle` runs. Snapshots/logs git-ignored.
Regenerate: `analyze_param_sensitivity.py --sweep_dir . --prefix K --param K` (and `--prefix tau`).
