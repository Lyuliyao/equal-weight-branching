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

## Result

| method | N | **T_core** | [p10,p90] | rel_spread | valid fits | T_L2 | T_peak |
|---|---|---|---|---|---|---|---|
| LDG | 80 | 2.46e-4 | [2.29,2.62] | 0.14 | 2/12 | — | — |
| LDG | 160 | 1.234e-4 | [1.15,1.32] | 0.14 | 2/12 | — | — |
| **LDG** | **320** | **1.215e-4** | [1.13,1.22] | **0.073** | **12/12** | **1.219e-4** | **1.211e-4** |
| particle | 8e4 | 1.505e-4 | [1.39,1.62] | 0.15 | 3 | 1.249e-4 | 1.450e-4 |
| **particle** | **3.2e5** | **1.318e-4** | [1.26,1.33] | **0.055** | 3 | — | — |

(LDG `quad_order=3`, raw mass = clipped to machine precision since `u>=0`. Particle =
`current_fourier`, seed-mean over 4 seeds. Literature LDG numerical blow-up ~1.21e-4.)

### Reading (next_stage.md §8 / §12)

- **LDG gate PASSES.** The proxy is **resolution-convergent** (N=80→160→320: 2.46→1.234→1.215)
  and at the finest resolution **stable** (all 12 q×window fits valid, rel_spread 0.07),
  with `T_core ≈ T_L2 ≈ T_peak ≈ 1.21e-4` (three independent extrapolations agree) and
  matching the literature LDG blow-up. Coarse N is grid-floor-limited (`R_0.1 < dx` by
  `t~1.2e-4`), as the plan §3.1 anticipates.
- **Particle is stable and convergent** (8e4→3.2e5: 1.505→1.318), on the **same scale**
  and **within ~8%** of LDG, near the literature blow-up.
- **But the strict within-uncertainty agreement is NOT met:** the LDG [1.13,1.22] and
  particle [1.26,1.33] bands are disjoint; the particle `T_core` is systematically ~8%
  higher. This is consistent with the `K=10` Fourier solver drift slightly
  under-resolving the inner core (cf. §3.8: the single-K drift under-resolves `R_0.2`),
  so the particle core collapses marginally slower. The particle bands are also narrow
  but under-sampled (2/4 of the `3.2e5` seeds abort at `~1.2e-4`, trimming late windows).

## Decision

The metric is a **genuine improvement over the LDG-style `t_b`**: it is reconstruction-light,
**stable and resolution-convergent on BOTH methods**, and the two agree to **~8%** with each
other and with the LDG numerical blow-up. But the strict §12 gate (LDG↔particle agreement
*within uncertainty*) is not met (disjoint bands, ~8% offset). Per §11 we therefore use the
**limited wording** and do **not** quote a single continuum blow-up time:

> We compute a reconstruction-light core-collapse proxy `T_core` from the extrapolated
> mass-quantile radii `R_q^2 ≈ a_q(T_q − t)`. On the fixed-flux LDG reference the proxy is
> resolution-convergent and, at the finest resolution, stable across inner quantiles and
> fitting windows (`T_core = 1.215e-4`, rel. spread 0.07), in agreement with the independent
> inverse-`L2` and inverse-peak extrapolations and with the LDG numerical blow-up scale. The
> particle method on the same equation gives a stable, convergent `T_core = 1.32e-4` at
> `Np = 3.2e5`, on the same scale and within ~8% of the LDG value, with the residual offset
> consistent with the Fourier drift slightly under-resolving the inner core. We report
> `T_core` as a concentration-time diagnostic and do not quote a continuum blow-up time.

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
