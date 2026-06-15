# Solver-field comparison by the SAME LDG-style particle resolution-gap time `t_b`

**Status: diagnostic record. NOT a paper accuracy claim.** Code git `411f725`;
runs submitted from `0293dd7` (job 10038983, general-long). Spec: `next_stage.md`.

## What this answers

The prior `sf_blob` sweep compared solver fields by abort/final time — a *stability*
diagnostic only. This experiment compares them by the **same LDG-matched particle
resolution-gap time** already used for the particle method and the fixed-flux LDG
reference, so a difference (if any) is a concentration/accuracy difference, not just
drift smoothness.

### Diagnostic norm `S^DG_{Np,n}(t)`

Project the u-particle measure onto the **same P1 DG space** used by the LDG reference
and take its `L2` norm with the LDG mass matrix:

    S^DG_{Np,n}(t) = || Π_n^{P1DG} μ^u_{Np}(t) ||_{L2} .

Use the **cross/split estimator** (split the cloud in half, inner-product the two
projected halves) to remove the empirical self-term bias of a projected Dirac measure.
Columns `S_dg_cross_80` (n=80) and `S_dg_cross_160` (n=160) in each `diag_*.csv`.

### Particle resolution-gap time `t_b`

For the particle analogue of LDG grid refinement `(Np,n) → (4Np,2n)` (in 2D, halving
the particle spacing needs 4× the particles), with seed-mean curves `S̄`,

    t_b(θ) = inf{ t : S̄_high(t) / S̄_low(t) ≥ θ, held for ≥ 5×10⁻⁶ },   θ = 1.05,

main pair **low=(8×10⁴, 80)**, **high=(3.2×10⁵, 160)**. Bootstrap CI: 1000× independent
resampling of the low and high seeds. This is a **numerical resolution-gap indicator,
NOT a continuum blow-up time.**

## Run matrix

3 solver fields × 2 resolutions × 4 seeds = 24 runs; `K=10`, `τ=2×10⁻⁷`,
`n_steps=1000` (`T=2×10⁻⁴`, output `Δt=10⁻⁶`), `--dg_readout_n 80 160`,
`--cfl_abort 5.0 --filter_s 0.5 --q_window 0.8`. (Optional spectral reference not run.)

## Result

Fixed-flux LDG reference: `t_b(80→160)=5.95×10⁻⁵`, `t_b(160→320)=8.43×10⁻⁵`.

| solver field | **t_b** | bootstrap CI [5%,95%] | on LDG scale | frac reach `T` (3.2e5) | max solver CFL (3.2e5) | max Fourier-diag CFL (3.2e5) |
|---|---|---|---|---|---|---|
| `current_fourier` | 8.5e-5 | [7.3e-5, 1.07e-4] | yes | 0.25 | 4.17 | 4.17 |
| `blob c_h=0.06` | 8.2e-5 | [6.9e-5, 1.07e-4] | yes | **0.75** | **3.18** | 4.6 |
| `blob c_h=0.09` | 9.6e-5 | [7.9e-5, 1.03e-4] | yes | 0.50 | 4.28 | 5.6 |

(`t_b` here uses the seed-mean ratio; bootstrap CI over seeds. `cflS`/`cflF` from
`drift_cfl_solver_field` / `drift_cfl_fourier_diag`.)

- **All three `t_b` are on the LDG scale** (~8–10×10⁻⁵, comparable to the fixed-flux LDG
  interval) and **statistically indistinguishable** — every CI spans ~[7,11]×10⁻⁵ and the
  three point estimates lie inside each other's CI.
- **Reconstruction-free core radii `R_0.2`, `R_0.5` collapse on top of each other** across all
  three fields (`figures/solver_field_core_radii`), confirming the concentration *dynamics*
  are the same — blob `c_h=0.09`'s slightly later point estimate is **not** oversmoothing.
- The blob's **only** measurable effect is **stability**: `blob c_h=0.06` survives to `T` in
  3/4 of the `3.2e5` seeds (vs 1/4 for global-K) with the smallest real solver CFL (3.18).
  Note its Fourier-diagnostic CFL is *higher* (4.6) than its solver CFL — the blob makes a
  tighter core with a smoother drift (`figures/solver_field_dual_cfl`).

## Abort/final time is stability only

`t_end` (= `T` or the guard's abort time) and the solver/Fourier CFLs are reported only as
**secondary stability** diagnostics. A smoother field can abort later simply by reducing
`max|∇v̂|`; that is not evidence of better dynamics. The primary metric is `t_b`.

## Decision (next_stage.md §10 Scenario A / §11)

> **The blob residual improves numerical stability/smoothness of the drift, but does not change
> the LDG-style concentration proxy `t_b` (CIs overlap) nor the reconstruction-free core radii.
> It is not an accuracy improvement on this benchmark.** Keep record-only — not promoted to §5.4.

The **paper-safe result stands**: the current particle method + LDG-matched DG readout gives a
particle resolution-gap time on the **same scale as fixed-flux LDG** at adequate particle count.
We do **not** claim the blob (or any solver field here) resolves near-blow-up dynamics or a
continuum blow-up time.

## Regenerate

```bash
REPO=$PWD OUT=$PWD/reference_results/keller_segel_ldg_pp/solver_field_tb_<id> \
  sbatch experiments/keller_segel/ldg_comparison/run_solver_field_tb_sweep.sb     # 24 runs
cd experiments/keller_segel/ldg_comparison
python analyze_solver_field_tb.py --sdir <this_dir>   # -> solver_field_tb_summary.csv/.json + plot_data
python plot_solver_field_tb.py    --sdir <this_dir>   # -> figures/
```

`diag_*.csv`, `solver_field_tb_summary.{csv,json}`, `plot_data/`, `figures/`, and the 6
`abort_diagnostics.json` are kept; per-run snapshots/logs are git-ignored.
