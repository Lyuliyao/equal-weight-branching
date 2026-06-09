# Experiment 2 — Manufactured-Solution (MMS) Verification

Convergence verification of the particle method against an **exact** solution on
the 2-D torus `T^2 = [-pi,pi]^2`:

```
d_t u = -div(b u) + D Δu + r(t,x) u ,   constant b = (b1,b2),
u_ex(t,x) = M(t)·( 1 + a cos x1 + b_ sin x2 + c cos(x1+x2) ),  M(t)=exp(gamma t).
```

With small `a,b_,c` the exact solution stays strictly positive. The reaction
coefficient that makes `u_ex` exact is computed **analytically** (closed form):
with `P = 1 + a cos x1 + b_ sin x2 + c cos(x1+x2)`,

```
r(t,x) = gamma + ( b1·∂_x1 P + b2·∂_x2 P - D·ΔP ) / P
∂_x1 P = -a sin x1 - c sin(x1+x2)
∂_x2 P =  b_ cos x2 - c sin(x1+x2)
ΔP     = -a cos x1 - b_ sin x2 - 2 c cos(x1+x2)
```

(constant `b` ⇒ `div(b u) = b·∇u`). A built-in unit test confirms `r` by
finite differences: the FD residual of the full PDE is `~1e-4` relative
(truncation floor), i.e. `r` is correct.

The particle method uses the **Poisson branching** kernel (unbiased). Transport
drift is `-b` (so `d_t u = -div(b u) + ...`), diffusion is `sqrt(2 D tau)·dW`.
The reconstructed field is `P_K × measure_mass`, `measure_mass = (N_active/N0)·M0`,
`M0 = ∫ u_ex(0,·) = L^2` (the cos/sin terms integrate to zero on the torus).
Same `P_K` Fourier estimator as the rest of the codebase (`common_particle.py`).

## Studies (each writes its own CSV)
- **(a) `errors_vs_N.csv`** — fix `tau, K`; vary `N ∈ {2e3,4e3,8e3,1.6e4,3.2e4,6.4e4}`,
  multiple seeds; report mean ± std L2-rel. Expected slope ≈ **-1/2** in N
  (Monte-Carlo rate).
- **(b) `errors_vs_tau.csv`** — fix large `N, K`; vary `tau`.
- **(c) `errors_vs_K.csv`** — fix large `N`, small `tau`; vary `K`.

A fitted-slope summary (N and tau) is printed at the end.

## Files
- `common_particle.py` — shared module (identical copy of the one in
  `branch_vs_weighted/`).
- `experiment.py` — unit tests + the three convergence studies.
- `plot.py` — log-log error-vs-N (with fitted slope), error-vs-tau, error-vs-K.
- `run_me` — SLURM submission script.
- `README.md` — this file.

## How to run

Unit tests only (kernel means vs `exp(r tau)`, and MMS `r` FD check):
```
python experiment.py --test
```

Smoke (verified; runs unit tests + tiny sweeps, ~110 s on CPU):
```
python experiment.py --smoke
```

Full run (foreground):
```
python experiment.py
```

Full run via SLURM:
```
sbatch run_me            # full config
sbatch run_me --smoke    # tiny config on the cluster
```

Plots:
```
python plot.py
```

## Configuration
`CONFIG` at top of `experiment.py`. Defaults: `D=0.05, b=(0.5,0.3), gamma=0.3,
a=0.2, b_=0.15, c=0.1, T=0.5, grid=256, seeds=[0,1,2,3]`, with per-study
`N_list / tau_list / K_list` and the fixed values for each study. Override via
`--config my.json`; `--smoke` shrinks every list and uses 2 seeds.

## Outputs (in `results/mms/`)
- `errors_vs_N.csv`, `errors_vs_tau.csv`, `errors_vs_K.csv` — columns
  `var, value, N, tau, K, mean_L2_rel, std_L2_rel, n_seeds`.
- `config_used.json`.
- PDFs: `errors_vs_N.pdf`, `errors_vs_tau.pdf`, `errors_vs_K.pdf`.

## Cost
Each particle run scales with `N × steps`. The full study (a) at `N=6.4e4`,
`tau=2.5e-3` (200 steps) × 4 seeds is the dominant term. CPU smoke (`N≤4e3`,
10 steps) is ~110 s total. Extrapolating, the full sweep is on the order of a
**few hours on CPU**, well within the 12 h SLURM wall; faster on GPU. The
branching buffer is `8·N`; with `gamma=0.3, T=0.5` the population grows only by
`exp(0.15) ≈ 1.16×`, so overflow is not a concern here.
