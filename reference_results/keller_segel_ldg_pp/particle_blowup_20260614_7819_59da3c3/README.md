# LDG-style particle blow-up proxy: reconstruction-operator sweep — results & decision

Particle analogue of the LDG `L²`-resolution-gap numerical blow-up indicator, tied
to an **explicit reconstruction operator** (not a final-time visualization) and
compared to the **fixed-flux direct LDG** reference. Run:
`particle_blowup_20260614_7819_59da3c3/`.

## Method

Per readout `R`, `S^R_{N_p,n}(t)=‖R_n μ^u_{N_p}(t)‖_{L²}`, and the gap time
`t_b = inf{t: S_high(t) ≥ θ S_low(t)}` (θ=1.05, persistence ≥5e-6, ensemble over
seeds, bootstrap CI). Readouts:

- **A — LDG-matched P¹ DG projection** (primary): same modal basis + mass-matrix
  norm as the LDG solver, with a **cross/split estimator** that removes the
  Monte-Carlo self-term. *Verified:* the cross norm converges to the LDG
  `field_L2(P u0) = 114.88` as `N_p` grows (115.5→115.0→114.9→114.88); mass exact (10π).
- **B — global/core-window Fourier** `S_L2_u` (the current solver field; sensitivity baseline).
- (C — particle-adaptive residual: covered by the separate `adaptive_recon/` audit — mixed/limited.)

Three gaps decompose particle-count vs readout-resolution effects:
`main (N_p,n)→(4N_p,2n)`, `sampling (N_p,n)→(4N_p,n)`, `recon (N_p,n)→(N_p,2n)`.

## Runs

`N_p ∈ {2×10⁴ (4 seeds), 8×10⁴ (4 seeds), 3.2×10⁵ (2 seeds)}`, fully-pp particle
solver, online DG readout at `n ∈ {40,80,160,320}`, `output_dt = 10⁻⁶`. The 8×10⁴
and 3.2×10⁵ runs hit the drift-CFL guard at `t≈1.5–1.7×10⁻⁴` (the un-hybridized
global-Fourier drift destabilizes as the core concentrates) — the gap times
(`~6–9×10⁻⁵`) are before that, so they are computable but the CIs are wide
(2–4 seeds).

## Results (θ=1.05; LDG fixed-flux `tb` = 5.95×10⁻⁵ … 8.43×10⁻⁵)

| readout | gap | pair (ppc≈12.5 unless noted) | `t_b` | bootstrap CI | on LDG scale? |
|---|---|---|---|---|---|
| **A (DG cross)** | **main** | **(8e4,80)→(3.2e5,160)** | **9.2×10⁻⁵** | [6.2, 11.9]×10⁻⁵ | ✅ |
| A (DG cross) | recon | (8e4,80)→(8e4,160) | 4.8×10⁻⁵ | [3.6, 8.1]×10⁻⁵ | ✅ |
| A (DG cross) | main | (2e4,40)→(8e4,80) | 7×10⁻⁶ | tight | ❌ shot-noise |
| A (DG cross) | sampling | (2e4,80)→(8e4,80), ppc≈3.1 | 7×10⁻⁶ | tight | ❌ shot-noise |
| B (Fourier) | main | (8e4)→(3.2e5) | 1.29×10⁻⁴ | [6.2, 13]×10⁻⁵ | ✅ (higher) |
| B (Fourier) | main | (2e4)→(8e4) | 7×10⁻⁶ | tight | ❌ shot-noise |

## Decision — Scenario 1/2

**At adequate particle counts the LDG-matched DG resolution-gap time is on the LDG
scale.** The main gap for the `(8e4,80)→(3.2e5,160)` pair (ppc≈12.5) is
`t_b = 9.2×10⁻⁵`, bracketing the LDG `160→320` value `8.43×10⁻⁵`; the same-cloud
recon gap is `4.8×10⁻⁵`, near the LDG `80→160` value `5.95×10⁻⁵`.

**The low-`N_p` pairs are shot-noise limited** (the `2×10⁴`-particle readout opens a
spurious gap at `7×10⁻⁶` from early times): the metric requires adequate
particles-per-cell, which the cross estimator alone cannot fully repair when
`ppc≲3`. This is exactly the decomposition the three-gap design was meant to expose
— a naive particle "blow-up gap" computed at low `N_p` looks ~10× too early.

**Recommendation for §5.4:**
- Use **Version A (LDG-matched DG projection)** as the quantitative LDG-comparable
  resolution-gap metric, reported at adequate `ppc` with the cross estimator, and
  **state the low-`N_p` shot-noise limitation**.
- Report **Version B (Fourier)** as a reconstruction-sensitivity diagnostic (it
  lands at `1.3×10⁻⁴`, a bit higher, and is likewise shot-noise limited at low `N_p`).
- Do **not** call this a continuum blow-up time — it is a uniform-mesh / fixed-readout
  resolution-gap indicator, on the same scale as (but not a convergence of) the LDG
  value. Pair it with the reconstruction-free core radii `R_{0.5},R_{0.8}`.

Safe manuscript language: *"Using the same P¹ DG projection and L² norm as the LDG
reference, the particle method gives a resolution-gap time on the same scale as the
fixed-flux LDG indicator at adequate particle counts; at low particle counts the
metric becomes shot-noise limited. We report this as a numerical resolution-gap
indicator, not a continuum blow-up time."*

## Files

`particle_tb_summary.csv` / `.json` (all gaps, CIs), per-run `diag_*.csv` (the
`S_dg_raw_*`, `S_dg_cross_*`, `ppc_*`, `S_L2_u`, `R_*` time series). Regenerate via
`experiments/keller_segel/ldg_comparison/analyze_particle_blowup_metric.py`.
Remaining (optional): the `3.2e5` recon gap (`n=160→320`) and a `1.28e6` pair for the
LDG `160→320` match would tighten the CIs; both are expensive and deferred.
