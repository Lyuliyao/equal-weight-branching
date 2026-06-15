# Core-collapse time `T_core` from mass-quantile radii (LDG vs particle)

**Status: limited-positive diagnostic. Candidate for §5.5 as a supporting concentration-time
indicator; NOT a continuum blow-up-time claim.** Code git `ae4973d`; spec `next_stage.md`.

## Definitions

**Mass-quantile radius** (method-independent, reconstruction-light):

    R_q(t) = inf{ r : mu(B(x_c(t), r)) >= q M },   x_c = mass centroid.

For LDG: from sub-cell `quad_order=3` Gauss quadrature samples of the P1-DG field
(`core_radii.ldg_quad_samples`, mass-exact, raw + clipped). For particles: the
`ceil(qN)`-th ordered distance from the cloud centroid (`core_radii.particle_radii`).

**Core-collapse time**: fit `R_q(t)^2 = alpha_q - beta_q t` on late windows;
`T_q = alpha_q / beta_q`. Aggregate over `q in {0.1,0.2,0.3}` and 4 windows
{[4,9],[5,10],[6,11],[7,12]}×1e-5:  `T_core = median`, spread `[p10,p90]`.
Secondary: `T_L2` from `S^-2`, `T_peak` from `peak^-1`. A fit is valid only if
`beta>0`, `R^2>=0.9`, `T>window_end`; quotable only if `rel_spread=(p90-p10)/median<=0.25`.

## Result (repaired per "Next experiments" Exp. A: common-grid seed handling + seed bootstrap + q-set sensitivity)

`T_core` median (q-set `{0.1,0.2,0.3}`); LDG uncertainty = q/window spread, **particle
uncertainty = SEED BOOTSTRAP** (1000×, resample 4 seeds):

| method | N | **T_core** | uncertainty | valid R_q² fits | secondary |
|---|---|---|---|---|---|
| LDG | 80 | 2.46e-4 | [2.29,2.62] (q/win) | 2/12 (grid-floor) | — |
| LDG | 160 | 1.234e-4 | [1.15,1.32] (q/win) | 2/12 | — |
| **LDG** | **320** | **1.215e-4** | [1.13,1.22] (q/win) | **4/12** | **T_L2=1.219e-4, T_peak=1.211e-4** |
| particle | 8e4 | 1.505e-4 | **[0.96,1.96]** boot | 3/12 | T_L2=1.25e-4 |
| **particle** | **3.2e5** | **1.29e-4** (med 1.259 boot) | **[1.11,1.36]** boot | 7/12 | — |

(LDG `quad_order=3`, raw mass = clipped since `u≥0`. Particle = `current_fourier`, 4 seeds.
Literature LDG numerical blow-up ~1.21e-4.)

### Reading (next_stage.md §8 / §12)

- **LDG gate PASSES, with the honest fit count:** the proxy is **resolution-convergent**
  (N=80→160→320: 2.46→1.234→1.215) and at N=320 the accepted fits have small spread
  (`rel_spread 0.073`). **Only 4 of the 12** `q×window` `R_q²` fits pass the strict linear
  gate (the rest fail on curvature / grid floor — mainly `q=0.1` and the early windows);
  the 4 accepted fits cluster tightly, and the independent `T_L2` (1.219e-4) and `T_peak`
  (1.211e-4) extrapolations agree, all matching the literature LDG blow-up.
- **Particle is stable and convergent** (8e4→3.2e5: 1.505→1.29) and — with the **corrected
  seed bootstrap** — its CI **[1.11,1.36]e-4 overlaps the LDG estimate and contains the LDG
  value 1.215e-4**. The point estimates agree to ~4%. (The earlier "disjoint bands, ~8%
  offset" was an artifact of a seed-averaging bug that dropped seeds whose CSVs ended at
  different abort times, plus quoting the artificially-tight q/window spread instead of the
  seed bootstrap. Fixed: common-grid interpolation + `min_seed_coverage` window rule + seed
  bootstrap.)
- **q-sensitivity:** `q={0.2,0.3}` is the reliable particle set (7/8 valid fits); **`q=0.1`
  is too noisy / not linearly resolved** at the present particle counts. Across q-sets the
  particle `T_core` spans 1.23–1.33e-4 (~7%, under the 10% red-line) — see
  `core_T_qset_sensitivity` and `core_fit_summary_by_qset` rows.

## Decision

A **genuine improvement over the LDG-style `t_b`** and now a **within-uncertainty cross-method
agreement**: the proxy is reconstruction-light, stable + resolution-convergent on **both**
methods, and at the finest resolutions the particle seed-bootstrap CI overlaps the LDG estimate
and the literature blow-up (~1.21e-4), point estimates agreeing to ~4%. Caveats kept honest:
only 4/12 LDG fits pass the strict gate (but tightly + secondary-confirmed); `q=0.1` is the weak
particle quantile; the particle CI is wide ([1.11,1.36]). We therefore report `T_core` as a
reconstruction-light concentration-time diagnostic consistent between the methods and with the
LDG blow-up scale, and still **do not quote a single continuum blow-up time** (§11 limited
wording), pending the `q_window`/`K`/`τ` sensitivity (Experiments B–D) to confirm the residual
offset is reconstruction-controlled.

> We compute a reconstruction-light core-collapse proxy `T_core` from the extrapolated
> mass-quantile radii `R_q² ≈ a_q(T_q − t)`, identically on the LDG reference (quadrature
> masses) and on particle clouds. On LDG it is resolution-convergent and, at the finest
> resolution, stable across the accepted inner-quantile/window fits (`T_core = 1.215e-4`),
> in agreement with the independent inverse-`L2` and inverse-peak extrapolations and the LDG
> numerical blow-up. The particle method gives a stable, convergent `T_core = 1.26e-4` at
> `Np = 3.2e5` whose seed-bootstrap interval contains the LDG value. We report `T_core` as a
> concentration-time diagnostic and do not quote a continuum blow-up time.

## Files / regenerate

```text
ldg/N{80,160,320}/ldg_core_radii_N<N>.csv   # R_q(t) raw+clip, S_L2, peak, u_min
particle/current_fourier_N{80000,320000}_seed{0..3}/diag_*.csv   # R_0.05..R_0.8(t)
core_fit_all.csv, core_fit_summary.csv/.json
figures/{core_Rq2_fits,core_Tq_scatter,core_T_summary,core_halo_sep}.{pdf,png}
```

```bash
OUT=$PWD/reference_results/keller_segel_ldg_pp/core_collapse_<id>
OUT=$OUT sbatch experiments/keller_segel/core_collapse_time/run_ldg_core_radii.sb       # N=80,160,320
OUT=$OUT sbatch experiments/keller_segel/core_collapse_time/run_particle_core_radii.sb  # current_fourier 8e4,3.2e5 x4
cd experiments/keller_segel/core_collapse_time
python fit_core_collapse.py  --ldg_dir $OUT/ldg --particle_root $OUT/particle --mass raw --outdir $OUT
python plot_core_collapse.py --ldg_dir $OUT/ldg --fitdir $OUT --mass raw
```

LDG field snapshots, particle snapshots, and logs are git-ignored; the `R_q`/`diag` CSVs,
`core_fit_*`, and figures are kept.
