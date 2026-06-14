# Core-local & reconstruction-free blow-up-proxy diagnostics (§5.5)

Pure post-processing of the §5.4 saved diagnostics (no solver run). Consumes the
FVM baseline `S_curves.csv` at grids 128/256/512 (`../ldg_pp_baseline/`) and the
particle pp diagnostics (`ldg_comparison` base/refined). Answers the §5.5 question:
**do core-local / reconstruction-free diagnostics give a more reliable numerical
blow-up proxy than the global L2-gap or the reconstructed peak?**

## What it computes (`analyze_core_proxy.py`)

- **Global vs core-local resolution-gap** `t_b^{glob}` / `t_b^{core}` from grid pairs,
  with `S_core` the L2 norm restricted to `B(x_c, 3R_0.8)`.
- **Reconstruction-free candidate concentration time** from a linear fit
  `R_q(t)² ≈ C_q(T_* - t)` over several fit windows (`q∈{0.5,0.8}`, baseline + particle).

## Honest findings (the §5.5 *limited* result)

1. **Reconstruction-free core radii `R_0.5, R_0.8` collapse robustly** in both the grid
   baseline and the particle method and agree at the reporting times — this is the
   reliable, method-agnostic concentration signal.
2. **A radius-fit candidate `T_*` is NOT stable.** Across fit windows it spreads
   1.4×–4.5× (FVM ≈ 1.2–1.9e-4, particle ≈ 1.4–6.1e-4); the radii also saturate at the
   grid/bandwidth floor, contaminating the fit. `T_*` is O(1e-4), the scale of the LDG
   numerical blow-up (~1.21e-4), but **we do not quote a continuum blow-up time**.
3. **Core-localization does not sharpen the grid proxy.** `S_core ≡ S_L2` for the
   resolving baseline (the concentrating field carries essentially all of its L2 mass
   in the core), so `t_b^{core} ≡ t_b^{global}`. The particle reconstruction already
   uses a core-adaptive window, so its global S is itself core-focused.

Conclusion: reconstruction-free radii give robust evidence of pre-singular
concentration and identify the reliable window; the reconstructed peak, the global
L2-gap `t_b`, and the radius-fit `T_*` are all resolution / bandwidth / window-
sensitive. We therefore report numerical resolution-gap indicators and
reconstruction-free radii, not a continuum blow-up time (consistent with the
project finding that the blow-up *time* is not defensibly computable while the
collapse is).

## Run

```bash
python analyze_core_proxy.py --baseline_dir <baseline_run> \
    --particle_base <particle_base_diag.csv> --particle_refined <particle_refined_diag.csv> \
    --out_dir <core_proxy_run>
python plot_core_proxy.py --core_dir <core_proxy_run> --baseline_dir <baseline_run>
```

Outputs (`reference_results/keller_segel_ldg_pp/core_proxy_<run_id>/`):
`global_core_tb.csv`, `radius_fit.csv`, `sensitivity.csv` (the full window spread),
`plot_data/radii.npz`, `figures/ks_core_proxy.{pdf,png}`.
