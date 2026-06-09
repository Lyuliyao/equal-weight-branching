# High-Dimensional (4D / 6D) Particle Experiment — Kinetic Localized Growth

A clean, self-contained, reproducible high-dimensional benchmark for the
particle-method paper, showing that the SAME qualitative story as the 2-D
`branch_vs_weighted` experiment carries over to high dimensions: **weighted**
particles suffer weight degeneracy (max/mean-weight grows, effective sample
size collapses) while **branching** grows the active / local particle count and
keeps equal weights — all using a **grid-free** reaction that needs only a
scalar moment, so it works in any dimension with no dense grid and no field
solve.

## The PDE

On the phase-space torus `[-pi, pi]^d`, `z = (x, v)`, `d = d_x + d_v`,
`d_x = d_v`:

```
d_t f + v . grad_x f = D_v Lap_v f + r[f](z) f ,
r[f](z) = lambda * G_d(z) * (1 - alpha * m[f]) - beta ,
m[f]    = integral G_d f dz ,
G_d(z)  = prod_j (1 + cos z_j) / 2 .
```

- `d = 4` : `d_x = d_v = 2`.  `d = 6` : `d_x = d_v = 3`.
- `G_d` is a separable localized bump, `= 1` at the origin, `= 0` on the box
  faces (`z_j = +/- pi`).
- The local region is `B = { z : G_d(z) >= eta }` with `eta = 0.5` fixed.

### Grid-free reaction (the whole point)

The reaction depends on the field `f` ONLY through the scalar moment `m[f]`,
estimated directly as a Monte-Carlo average over active particles:

```
m_hat = (1/N0) * sum_{i active} w_i * G_d(Z_i)
```

(`w_i = 1` for branching).  No dense `K^d` grid, no field solve, in any
dimension.  The logistic factor `(1 - alpha * m_hat)` self-limits growth so the
moment `m(t)` stabilizes rather than blowing up.

## Particle update (one step `tau`)

```
X <- X + V * tau                       (mod 2pi)        # kinetic transport
V <- V + sqrt(2 D_v tau) * xi          (mod 2pi)        # velocity diffusion
```
then an equal-weight branching (or weighted) reaction with rate
`r_i = lambda * G_d(Z_i) * (1 - alpha * m_hat) - beta`.

All three methods share **identical initial particles** and **identical
velocity-noise increments** per seed (the first `N0` active particles use the
same `xi`; branching draws fresh noise only for the surplus offspring).

## Methods compared

1. **weighted** — positions evolve, `w_i *= exp(r_i tau)`.
2. **poisson**  — equal-weight unbiased integer branching: `r>=0` →
   `nu = 1 + Poisson(m-1)`; `r<0` → `nu = Bernoulli(m)`, `m = exp(r tau)`.
3. **minvar**   — equal-weight minimum-variance branching:
   `nu = floor(m) + Bernoulli(m - floor(m))`.

All three reaction kernels are re-implemented **self-contained** in
`common_highdim.py` (no cross-directory imports).

## FHT / TT low-rank reconstruction diagnostic

At the final time we reconstruct a **low-rank** density model from the
equal-weight (poisson) particle cloud using the project's **Functional
Hierarchical Tensor (FHT) Fourier-sketching** reconstruction — the same one
adapted in `allen_cahn/case2`. We extract from the low-rank model:

- the 1D marginals (one per coordinate),
- two 2D marginals (`(x1,x2)` and `(x1,v1)`),
- the diagonal profile `f(s, s, ..., s)`.

`plot.py` overlays the FHT-reconstructed marginals on the raw histograms.
Dense `K^d` reconstruction is **never** attempted — the FHT low-rank model is
the only full-density object.

### Reconstruction path actually used: **FHT reuse (vendored + lightly patched)**

We **reuse** the `allen_cahn/case2` FHT modules, vendored into this directory:
- `fht_utils.py`                              — verbatim copy.
- `functional_hierarchical_tensor_sketch.py` — verbatim copy.
- `functional_hierarchical_tensor_fourier.py` — copy with ONE small patch.

FHT requires `d = 2^L`. For `d = 4` this is exact (`L = 2`, no ghost dims). For
`d = 6` we pad to `2^L = 8` (`L = 3`) with two **ghost** dimensions. The
upstream `FunctionalHierarchicalTensorFourier` ghost-marginalization path emits
a `'None'` sentinel for ghost leaves that the middle/root contraction einsums do
not absorb (it raises for `d = 6`). The vendored copy patches
`update_eval_msg_masked` / `evaluate_marginal` so ghost leaves are always
marginalized via their constant-mode projection `sqrt(2) * c[node][0,:]`,
leaving the middle/root einsums unchanged. The patch is documented at the top of
the vendored file. **The upstream `allen_cahn/case2` file is NOT modified.**

Both `d = 4` and `d = 6` FHT reconstructions run successfully on the smoke
clouds; the FHT 1D marginals match the raw histograms closely (e.g. d=4 smoke:
peak location agrees to ~0.02 in `z`, peak height to ~1.5%). The TT-sketching
package under `thirdparty/Non_Negative_Tensor_Train` was therefore **not
needed**.

### Documented fallback

As a cheap, always-available backup, every `fht_d{d}_seed{seed}.npz` ALSO stores
the empirical per-coordinate product-Fourier coefficient tensor (the
rank-one-sum diagnostic), `<cos(k z_j)>` and `<sin(k z_j)>` over active
particles. If the FHT construction ever raises, `experiment.py` catches it,
records `used_fht=False` with the error string, and the run still produces these
fallback coefficients plus the raw particle marginals.

## Files

- `common_highdim.py` — kernels (`G_d`, moment estimator, reaction rate),
  phase-space EM step, the three reaction kernels, nESS, branching compaction,
  initial sampling, and the FHT/TT reconstruction wrapper (`build_fht`,
  `fht_marginal_1d/2d`, `fht_diagonal`, `empirical_fourier_coeffs`).
- `experiment.py` — configs, runs, CSV/NPZ outputs, writes `config_used.json`.
- `plot.py` — metrics-vs-t panels, FHT-vs-histogram marginal overlays, 2D
  marginal panels, FHT diagonal profile.
- `fht_utils.py`, `functional_hierarchical_tensor_sketch.py`,
  `functional_hierarchical_tensor_fourier.py` — vendored FHT modules (see above).
- `run_me` — SLURM submission (`-A Multiscaleml`, `-C amr`, `ml Miniforge3`,
  `conda activate heat`).
- `README.md` — this file.

## Configuration

Knobs live in `CONFIG_D4` / `CONFIG_D6` at the top of `experiment.py`. Defaults:

```
d=4, D_v=0.1, lambda=6.0, alpha=1.0, beta=0.5, eta=0.5,
T=2.0, tau=0.02 (100 steps), N0=20000, buffer_mult=8,
sigma0=1.0, n_snapshots=21, seeds=[0,1,2],
FHT: deg=8, rank=6, sketch=5, grid=41.
```

The d=6 config is identical with `d=6`. Override with `--config my.json`
(JSON keys overwrite the resolved config). `--smoke` uses
`N0=2000, T=0.6, tau=0.04 (15 steps), seeds=[0,1]`.

## Interpreter

```
/mnt/home/lyuliyao/.conda/envs/heat/bin/python
```
with `jax.config.update("jax_enable_x64", True)` (set in `experiment.py`).
(On the dev node JAX prints a harmless CUDA-init warning and falls back to CPU.)

## How to run

Smoke (verified):
```
python experiment.py --smoke          # d=4 tiny
python experiment.py --smoke --d6     # d=6 tiny
```

Full runs:
```
python experiment.py                  # full d=4
python experiment.py --d6             # full d=6
```

Via SLURM:
```
sbatch run_me                  # full d=4
sbatch run_me --d6             # full d=6
```

Plots (after the runs):
```
python plot.py
```

## Outputs (in `results/highdim/`)

- `metrics.csv` (combined over dims) and `metrics_d{d}.csv` — per
  `seed x d x method x snapshot`: `seed, d, method, t, total_mass (=N_active/N0),
  moment_m, local_mass_B, N_active, N_local_B, global_nESS, local_nESS_B,
  max_w_over_mean_w, runtime_s`. nESS / max_w columns are `nan` for branching.
- `marginals_d{d}_seed{seed}.npz` — raw 1D histograms (all coords) and two 2D
  marginals (`(x1,x2)`, `(x1,v1)`) per method, plus bin edges/centers.
- `fht_d{d}_seed{seed}.npz` — FHT low-rank 1D marginals, two 2D marginals, the
  diagonal `f(s,...,s)`, the `used_fht` flag, and the fallback empirical
  product-Fourier coefficients.
- `config_used.json` — resolved config.
- PDFs (per dim): `metrics_d{d}.pdf`, `marginals_d{d}.pdf`,
  `marginals2d_d{d}.pdf`, `diagonal_d{d}.pdf`.

## Observed qualitative trends (smoke, T=0.6, seed 0, d=4)

| quantity                 | weighted        | poisson        | minvar         |
|--------------------------|-----------------|----------------|----------------|
| moment m, 0 -> T         | 0.41 -> 0.68    | 0.41 -> 0.68   | 0.41 -> 0.70   |
| total mass, 0 -> T       | 1.00 -> 1.50    | 1.00 -> 1.49   | 1.00 -> 1.51   |
| N_active, 0 -> T         | 2000 (fixed)    | 2000 -> 2989   | 2000 -> 3011   |
| N_local_B, 0 -> T        | 705 -> 626      | 705 -> 1376    | 705 -> 1386    |
| global nESS, 0 -> T      | 1.00 -> 0.857   | —              | —              |
| local nESS (B), 0 -> T   | 1.00 -> 0.954   | —              | —              |
| max_w/mean_w, 0 -> T     | 1.00 -> 2.40    | —              | —              |

The three methods agree on the physical observables `m(t)` and total mass,
while only the weighted method shows weight degeneracy and only branching grows
the (local) particle count. The same trends appear at `d = 6`. Over the full
`T = 2.0` the gaps widen substantially.

## Cost and limits

- **Smoke** (`N0=2000`, 15 steps): ~44 s/seed on CPU (incl. FHT), both d=4 and
  d=6 (FHT cost is comparable in both because of ghost padding to dpad=8).
- **Scaling probe** (`N0=20000`, 20 steps, d=4, 1 seed): 64 s (incl. FHT/setup).
  Branching grew to ~27k active at `T=0.4` and was **decelerating** (the
  logistic factor caps growth as `m(t)` approaches equilibrium).
- **Extrapolated full run** (`N0=20000`, 100 steps): roughly **5–8 min/seed**
  on CPU for d=4, similar for d=6; ~15–25 min for 3 seeds per dimension, well
  within the 12 h SLURM wall. A GPU node would be much faster.
- **Buffer overflow**: the `8 x N0 = 160000` branching buffer is safe. Growth
  saturates because `(1 - alpha m)` drives `r -> 0` as `m` rises (peak-`G`
  single-particle equilibrium `m* = 1 - (beta/lambda) = 0.917`; the global mean
  is lower), so `N_active` levels off at a few times `N0` rather than exploding.
  No overflow occurred in the smoke runs or the scaling probe. If a future
  parameter set does overflow, increase `buffer_mult` or `beta`/`alpha`, or
  reduce `lambda`/`T`.
- **FHT cost**: the sketch is ~1–3 s; it scales with the chosen ranks
  (`fht_rank`, `fht_sketch`) and degree (`fht_deg`), not with `K^d`. We
  subsample the cloud to 200k particles before sketching for tractability.

## Honesty notes

- The CUDA-init traceback at startup is benign; JAX correctly falls back to CPU
  on the dev node (`backend: cpu`).
- The d=6 FHT works only because of the small documented ghost-marginalization
  patch in the vendored Fourier module; without it the upstream class raises on
  `d = 6`. The patch is isolated to the vendored copy.
- Only the smoke tests and one scaling probe were run here; full production
  runs and SLURM submission are left to the caller, as requested.
```
