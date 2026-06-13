# KS 2D — LDG-style blow-up / concentration diagnostics

LDG-style concentration diagnostics for the 2D parabolic–elliptic Keller–Segel
particle method, layered on top of the **validated** core-adaptive
reconstruction in `../blowup_time/`. This directory adds new diagnostics only; it
**does not modify or copy** any existing file — the validated modules are reused
by importing them (`sys.path` is extended to `../blowup_time` at import time).

## What is new here

Beyond the particle-scale core-collapse diagnostics already in `../blowup_time`,
this directory computes the two LDG concentration diagnostics:

- **`S_{K,N}(t) = ||P_K μ||_{L2}`** — the reconstructed L2 norm of the physical
  `u` field on the adaptive window (a focusing indicator that rises as mass
  concentrates).
- **`M_core(r,t) = μ(B(x_c,r))`** — core mass inside small balls
  `r ∈ {0.01,0.02,0.04}` about the cluster centre (each particle carries mass
  `mass/N`).

and a **resolution-gap focusing time** `t_gap` from paired `(N,K)`+`(4N,2K)`
runs: `t_gap = inf{ t : S_{2K,4N}(t)/S_{K,N}(t) ≥ thresh }`, `thresh ∈ {1.05,
1.10}`.

## Model (unchanged from `../blowup_time`)

`u_t = Δu − χ∇·(u∇v)`, `−Δv+v=u`, `χ=1`, on the plane via the core-adaptive
window. IC `u0 = 840 exp(−84|x|²)`, mass `10π > 8π` (super-critical). Particle
update `dX = +χ∇v dt + √(2 dt) ξ` (inward, blow-up driving; drift-sign rationale
is documented in `../blowup_time/simulation_blowup.py` / `README.md`). The window
geometry, density reconstruction and screened-Poisson force come from
`../blowup_time/adaptive_window.py`; the physical-density rescaling
`u_phys = mass·(π/L)²·ρ_y` is used identically to `peak_density`.

## File map

| file | purpose |
|------|---------|
| `ldg_diagnostics.py` | pure functions `recon_L2_norm`, `core_mass` (+ `_selftest`). Imports `eval_density_y` from `../blowup_time/adaptive_window.py`. |
| `simulation_ldg.py` | thin variant of `../blowup_time/simulation_blowup.py`: adds `S_L2`, `Mcore_*` CSV columns and saves reconstructed-field `.npz` snapshots at report times. Reuses `sample_u0`, `quantile_radii`, `core_counts`, `compute_window`, `density_coeffs_y`, `chem_force`, `peak_density`. |
| `tgap.py` | post-processing: `t_gap(N,K)` from paired base/refined CSVs (no simulation). |
| `plot_ldg.py` | snapshot heatmaps, `S_L2`/peak/`R_0.5²` curves, `t_gap` table → PDFs. |
| `run_ldg.sh` | SLURM **templates** (not auto-submitting) + the SMOKE command. |
| `LDG_benchmark_notes.md` | benchmark cross-check skeleton (TODOs; verify `li2017local` via Codex before quoting times). |
| `results/` | output dir (CSV, config.json, README, `snapshots/`, PDFs, `tgap_table.*`). |

## Reuse of `../blowup_time`

`ldg_diagnostics.py` and `simulation_ldg.py` prepend `../blowup_time` to
`sys.path` and import: `compute_window`, `density_coeffs_y`, `chem_force`,
`peak_density`, `eval_density_y`, `sample_u0`, `quantile_radii`, `core_counts`.
Nothing in `../blowup_time` is edited. The `S_L2` rescaling mirrors
`peak_density` (`u_phys = mass·(π/L)²·ρ_y`, window side `2L`); the L2 integral
uses composite-trapezoid quadrature on the endpoint mesh (physical node spacing
`2L/(n_grid-1)`), Codex-verified.

## Environment

`/mnt/home/lyuliyao/.conda/envs/heat/bin/python` with
`MPLBACKEND=Agg JAX_PLATFORM_NAME=cpu`. The CUDA-load warning on CPU nodes is
harmless (jax falls back to CPU). SLURM: `-A Multiscaleml -C amr`,
`--time=16:00:00 --mem=48G --cpus-per-task=16`.

## Smoke command (run AFTER Codex approval, BEFORE the grid)

```bash
cd /mnt/gs21/scratch/lyuliyao/SDE_PDE/Numerical_experiment/Keller_Segel/ks2d_ldg
MPLBACKEND=Agg JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=16 \
  /mnt/home/lyuliyao/.conda/envs/heat/bin/python simulation_ldg.py \
  --N 20000 --K 5 --dt 1e-7 --n_steps 200 --diag_every 10 --seed 0 \
  --q_window 0.8 --verbose --outdir results_smoke
```

Optional self-test of the diagnostics on a known Gaussian:

```bash
MPLBACKEND=Agg JAX_PLATFORM_NAME=cpu \
  /mnt/home/lyuliyao/.conda/envs/heat/bin/python ldg_diagnostics.py
```

## Production grid

`N ∈ {4e4, 1.6e5, 6.4e5, 2.56e6}`, `K ∈ {5, 7, 9, 11}`, `τ ∈ {1e-7, 5e-8}`,
plus paired `(N,K)`+`(4N,2K)` runs for `t_gap`. All commands are SLURM
**templates** in `run_ldg.sh` (do not auto-submit). Report/snapshot times include
`5e-5, 1e-4, 1.5e-4`. After the grid finishes:

```bash
# t_gap table
python tgap.py --pairs base.csv:refined.csv ... --out results/tgap_table
# plots
python plot_ldg.py --out_dir results
```

## Outputs

- `results/diag_<tag>.csv` — columns:
  `step, t, N, K, L, h_eff, xc_x, xc_y, S_u, R2, R_0.5, R_0.8, R_0.9, R_0.99,`
  `N_0.5, N_0.8, N_0.9, N_0.99, R50_over_heff, peak_PK_u, S_L2,`
  `Mcore_0.01, Mcore_0.02, Mcore_0.04`.
- `results/snapshots/snap_<tag>_t<rt>.npz` — keys:
  `X, Y, U` (physical-`u` field on the side-`2L` window mesh), `x_c, L, t,`
  `report_t, mass, K, N, peak, S_L2`.
- `results/config.json`, `results/README` — run metadata.
- `results/tgap_table.{json,csv}` — `t_gap(N,K)` table.
- `results/*.pdf` — `curves_<tag>.pdf`, `snapshots_<tag>.pdf`, `tgap_table.pdf`.
```
