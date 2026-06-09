# Experiment 1 — Branching vs Weighted Particles in Localized Growth

The decisive experiment. A **linear** reaction–diffusion PDE on the 2-D torus
`T^2 = [-pi,pi]^2`,

```
d_t u = D Δu + r(x) u ,   r(x) = lambda·G(x) - beta ,
G(x) = exp( -|x - x0|_T^2 / (2 sigma^2) )    (periodic distance)
```

is solved three ways with a particle method, all sharing **identical initial
particles** and **identical Brownian increments** per seed:

1. **weighted** — positions diffuse only; weights `w_i *= exp(r(X_i) tau)`.
2. **poisson** — equal-weight integer branching, unbiased: if `r>=0`,
   `nu = 1 + Poisson(m-1)`; if `r<0`, `nu = Bernoulli(m)`, `m = exp(r tau)`.
3. **minvar** — equal-weight minimum-variance integer branching:
   `nu = floor(m) + Bernoulli(m - floor(m))`.

The ground truth `u_ref(t,x)` is a deterministic **Fourier split-step** solve of
the *same* linear PDE on a 256×256 grid (diffusion exact in Fourier space,
reaction `exp(r tau)` in physical space, Strang-split).

The density of a particle cloud is reconstructed with the project's standard
Fourier estimator `P_K` (see `common_particle.py`, same conventions as
`Keller_Segel/case2_test3/density.py`): coefficients = (weighted) mean of the
cos/sin basis over active particles, so `P_K` is a unit-mass probability
density; the physical field is `P_K × measure_mass`, where the measure mass is
`(sum_w / N0) · M0` (weighted) or `(N_active / N0) · M0` (branching), and
`M0 = ∫ u0`.

## Files
- `common_particle.py` — shared module: Fourier `P_K`, EM transport with shared
  Brownian increments, the three reaction kernels, nESS.
- `experiment.py` — runs all three methods + reference, writes metrics & fields.
- `plot.py` — builds the four publication PDFs.
- `run_me` — SLURM submission script (`-A Multiscaleml`, `-C amr`, heat env).
- `README.md` — this file.

## Configuration
All knobs live in the `CONFIG` dict at the top of `experiment.py`. Defaults:
`D=0.05, lambda=8.0, beta=1.0, sigma=0.5, x0=(0,0), T=1.0, tau=2e-3 (500 steps),
N0=20000, K=16, grid=256, eta=0.5, 20 snapshots, seeds=[0..7]`.
The growth region is `B = {x : G(x) >= eta}` with `eta=0.5` fixed in advance.
Override via `--config my.json` (JSON keys overwrite CONFIG). `--smoke` uses
`N0=2000, tau=0.05 (20 steps), seeds=[0,1]`.

## How to run

Smoke (already verified, ~70 s/seed on CPU):
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

Plots (run after the experiment):
```
python plot.py
```

## Outputs (in `results/branch_vs_weighted/`)
- `metrics.csv` — per seed × method × snapshot time: `total_mass, L1_err,
  L2_rel_err, peak_height, peak_height_err, peak_loc_err, local_mass_B,
  local_mass_B_err, global_nESS, local_nESS_B, max_w_over_mean_w, N_active,
  N_local_B, runtime_s`. nESS / max_w columns are `nan` for branching.
- `fields_seed{seed}.npz` — final-time 256-grid fields (`reference, weighted,
  poisson, minvar`) plus `XX, YY, xs`.
- `config_used.json` — resolved config.
- PDFs: `snapshots_final.pdf`, `l2_vs_t.pdf`, `ness_vs_t.pdf`,
  `boxplot_final_l2.pdf`.

## Cost
CPU smoke: ~70 s/seed (N0=2000, 20 steps). A scaling probe at the full
`N0=20000` with 100 steps took ~430 s/seed, peak RAM ≈ 6 GB. Extrapolating to
the full 500-step config gives roughly **0.5–1 h per seed on CPU**
(≈ 4–8 h for 8 seeds), well within the 12 h SLURM wall. A GPU node will be much
faster. The branching buffer is `8·N0`; smoke and the 100-step probe never
overflowed (poisson peaked well under the buffer). If `overflow` is reported,
increase `buffer_mult` or reduce `lambda`/`T`.
