# 3D Keller--Segel focusing / self-convergence stress test (`case_3d_focusing`)

A **parabolic--elliptic** 3D Keller--Segel focusing and self-convergence study.
A single conservative particle cloud carries `rho` under chemotaxis + diffusion
(no branching, no reaction); the chemical field `c` is obtained by a **3D
screened-Poisson spectral solve** each step.

## Model

On the periodic box `[-L/2, L/2]^3` with `L = 12`:

```
d_t rho = Delta rho - chi div(rho grad c),
-Delta c + kappa^2 c = rho - rho_bar,        chi = 1, kappa = 0.1.
```

Particle SDE (Euler--Maruyama), `chi = 1`:

```
X_{n+1} = wrap( X_n + chi * grad c(X_n) * tau + sqrt(2 tau) * xi_n ),  xi ~ N(0, I_3).
```

The drift is **`+chi grad c`** (inward / aggregating: `grad c` points toward the
chemical peak, which sits over the density peak). `rho` is a single conservative
cloud, so the particle **count is constant** and total mass is conserved exactly
(diagnostic 5.5(a)).

### 3D screened-Poisson solve (`field3d_screened.py`)

Generalizes the 2D symbol `lam = KX^2 + KY^2 + kappa^2` of
`../blowup_time/adaptive_window.py` to 3D on a **fixed periodic box of side `L`**:

- Wavenumbers `k_n = (2*pi/L) * n`, modes `n = 0..H-1` per axis (`H` = bandwidth).
- Real cos/sin coefficient tensors of the empirical **probability** density
  (integrates to 1), built by an einsum over particles (same normalization as
  `density3d.py`, with `k_n` set by the box side `L`).
- Physical density `rho = M * p`: the **mass `M` enters only as the field scale**
  (it does *not* change the particle count). Same role as `mass` in the 2D
  `chem_force`.
- Screened solve per mode: `c_hat_k = rho_hat_k / (|k|^2 + kappa^2)`.
- **Gauge / `rho_bar` subtraction: `c_hat_0 = 0`** (drop the DC mode; lives only
  in the `ccc` block at `n=(0,0,0)`).
- `grad c` assembled **analytically** (`d/dx cos(k.x) = -k sin`, etc.), a fresh
  `i*k` spectral assembly (not the autodiff path of `density3d.py`).

`selftest_field3d()` (written, **not run**) checks: (1) a single-mode source
`rho - rho_bar = cos(2*pi x1/L)` gives `c = cos(2*pi x1/L)/((2*pi/L)^2 + kappa^2)`;
(2) `grad c` vs finite difference; (3) the **drift sign** (`+grad c` is inward).

## Initial conditions (`ic_focusing.py`)

- **5.5(b) radial Gaussian mass family:** `rho_0^M = M (2 pi sigma^2)^{-3/2}
  exp(-|x|^2/(2 sigma^2))`, `sigma = 0.45`, `M in {20,40,60,80,100}`. Positions
  `x ~ N(0, sigma^2 I_3)`; `M` is carried as the field scale.
- **5.5(c) nonradial 4-cluster (tetrahedral):** centers `(1,1,1),(1,-1,-1),
  (-1,1,-1),(-1,-1,1)`, `M = 80`, `sigma_c = 0.25`, particles split equally.

## Diagnostics (`diagnostics_focusing.py`, grid-free where possible)

Torus-aware on the box. Per snapshot:

- `x_c(t)` torus-aware centroid; core radii `R_0.5(t)`, `R_0.9(t)` (quantile
  radii about `x_c`).
- `rho_core(t) = 0.5 M / ((4/3) pi R_0.5^3)` (grid-free peak-density surrogate).
- `P_H(t) = ||P_H rho||_inf` at bandwidth `H` (max over a coarse eval grid and
  particle positions).
- `C_H(t) = ||c_H||_inf` (chemical peak).
- `Q_c(t) = ||c_{H_hi}||_inf / ||c_{H_lo}||_inf` from the **same cloud**
  (default `H_lo=12`, `H_hi=24`; `16/32` optional) -- a self-convergence ratio
  that exposes under-resolution.
- `mass_drift = |N_t/N_0 - 1|` (sanity; ~0 for the conservative cloud).
- **Tetra extras:** per-cluster centroids and `R_0.5/R_0.9`, and the minimum
  inter-cluster center distance (merging indicator).

## Framing caution

3D classical Keller--Segel has **no finite universal critical mass**. Describe
results as a **family-dependent focusing transition** / a **numerical focusing
threshold within this one-parameter mass family**, supported by **self-convergence**
(`Q_c`, `(N,H)` grid). This is a self-convergence + focusing study. It is **not**
a comparison to any post-arXiv SIPF work and makes **no** universal-critical-mass
or blow-up-time claim.

## File map

| File | Purpose |
|------|---------|
| `field3d_screened.py` | 3D screened-Poisson spectral solver; `density_coeffs`, `screened_solve`, `eval_c`, `grad_c`, `eval_density`, `selftest_field3d()`. |
| `ic_focusing.py` | `gaussian_ic`, `tetra_clusters_ic`. |
| `diagnostics_focusing.py` | core radii, `rho_core`, `P_H`, `C_H`, `Q_c`, mass drift, cluster diagnostics. |
| `simulation_focusing.py` | transport loop; per-snapshot CSV; optional slice npz; config + README into out_dir. |
| `config_radial.json` / `config_tetra.json` | experiment-grid descriptions. |
| `plot_focusing.py` | figures (Agg, PDF). |
| `run_focusing.sb` | SLURM template + SMOKE + production grid commands. |

## Commands

**Codex cold-verify every code-executing command first (CLAUDE.md protocol).**

Self-test the field solver (written, not yet run):
```
python field3d_screened.py
```

SMOKE (tiny, fast end-to-end check):
```
python simulation_focusing.py --ic_type radial --N 20000 --M 60 --sigma 0.45 \
    --H 12 --tau 1e-4 --T 0.005 --L 12 --kappa 0.1 --chi 1.0 \
    --seed 0 --n_report 6 --out_dir scratch_smoke
```

Production (one run; see `run_focusing.sb` for the full grid):
```
python simulation_focusing.py --ic_type radial --N 800000 --M 80 --sigma 0.45 \
    --H 24 --tau 1e-4 --T 1.0 --L 12 --kappa 0.1 --chi 1.0 \
    --seed 0 --n_report 11 --out_dir results/radial_M80_N8e5_H24_s0
```

Tetra:
```
python simulation_focusing.py --ic_type tetra --N 800000 --M 80 --sigma_c 0.25 \
    --H 24 --tau 1e-4 --T 1.0 --L 12 --kappa 0.1 --chi 1.0 \
    --seed 0 --n_report 11 --out_dir results/tetra_M80_N8e5_H24_s0
```

Figures:
```
python plot_focusing.py --runs results/radial_M20_* results/radial_M40_* ... \
    --out_dir figures
```

## Resolution grid

`N in {2e5, 8e5, 3.2e6}`, `H in {12,18,24,32}`, `tau in {1e-4, 5e-5}`, `T = 1`.

## Outputs

- `diagnostics.csv` columns: `t, n_active, x_c_0, x_c_1, x_c_2, R_0.5, R_0.9,
  rho_core, P_H, C_H, Qc, C_Hlo, C_Hhi, mass_drift` (+ tetra: `min_intercluster_dist`,
  `cl{m}_c0..c2`, `cl{m}_R0.5`, `cl{m}_R0.9`).
- `density_slices.npz` (if `--save_slices`): keys `t{step}_grid`, `t{step}_slice`
  (z=centroid-plane density), `t{step}_xc`.
- `config_used.json`, `README.txt` per run.
