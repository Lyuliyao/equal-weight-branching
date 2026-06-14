# pp_injection reference run — 20260613_1929_92c0226_Nu30000_T2

Parabolic–parabolic Keller–Segel chemical equation `v_t = Δv + u − v` on the
torus `[0, 2π]²`, solved with the **cross-species injection** reaction substep
`v^{n+1} = e^{-τ} v* + (1 − e^{-τ}) u*` (two-species birth–death, no `(u−v)/v`
multiplicative rate). See
`experiments/keller_segel/pp_injection/README.md` for the full model and method
description.

## Configuration

- `n_samples = 30000` (cell mass `M_u = 1`), `M_v0 = 0.3`, `chi = 1.0`
- `T = 2.0`, `dt = 2e-3` (1000 steps), `n_freq = 5`, `seed = 0`
- injection kernel (default); population control NOT active
- git commit `92c0226` (full hash in `manifest.json`)
- reference: the exact analytic mass law (no grid/Fourier/FD/LDG solver)

Note on sizing: the eager per-step vmap-grad makes this CPU run expensive
(~minutes/100 steps). The requested 40000/T=2 default would take several hours on
CPU, so this reference uses `n_samples=30000`, `dt=2e-3` to keep wall time modest
while pushing the Monte Carlo birth–death error below the smoke level.

## Key results (validates the exact mass law)

- `max |M_u(t) − M_u(0)| = 0.0` (cells exactly conserved — pure transport, no
  reaction on `u`).
- `max rel. err of M_v(t)` vs `M_u + (M_v0 − M_u) e^{-t}` over **every** step:
  `1.386e-2`.
- final `M_v = 0.8972` vs exact `0.9053` (abs err `8.0e-3`) at `T = 2`.

The injection step is unbiased: each `v` decays with prob `1 − e^{-τ}` and each
`u` spawns a `v` with prob `1 − e^{-τ}` (since `ω_u = ω_v`), giving conditional
mean exactly `e^{-τ} v* + (1 − e^{-τ}) u*`. The residual ~1% is Monte Carlo
birth–death noise and shrinks as `1/√N`.

## Files

- `config.json` — model / reaction substep / resolved args / `p = 1−e^{-τ}`.
- `manifest.json` — git commit, command line, Python + package versions, seed,
  output dir, ISO datetime, population-control flag.
- `mass_balance.csv` — `step,t,M_u,M_v,M_v_exact,rel_err,n_birth,n_death,N_u,N_v`.
- `metrics_summary.csv` — scalar summary.
- `plot_data/figure_mass_balance.npz` — arrays for the figure.
- `figures/figure_mass_balance.{pdf,png}` — chemical mass-balance figure.
- `run_stderr.log` — stderr (harmless CUDA→CPU fallback warnings).

## Regenerate the figure WITHOUT rerunning the solver

```bash
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
$PY experiments/keller_segel/pp_injection/plot.py \
    --results_dir reference_results/pp_injection/20260613_1929_92c0226_Nu30000_T2
```

`plot.py` reads only `plot_data/figure_mass_balance.npz` (CSV fallback) and never
calls the solver.
