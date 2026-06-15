# LDG-style parabolic–parabolic Keller–Segel concentration comparison

Particle-field, **LDG-style** comparison of the equal-weight branching / injection
particle method on the 2D parabolic–parabolic Keller–Segel system, aligned with
the positivity-preserving LDG study of **Li, Shu, Yang** (*Local discontinuous
Galerkin method for the Keller–Segel chemotaxis model*).

## Model

On the plane (via a core-adaptive Fourier window):

```
u_t = Δu − χ div(u ∇v)        (cells,   χ = 1)
v_t = Δv + u − v              (chemical, DYNAMIC — its own particle cloud)
```

This is the **parabolic–parabolic** model: `v` is a genuine dynamic field carried
by its own equal-weight particle cloud, *not* the elliptic reduction
`−Δv + v = u`. The cell cloud `u` is conservative (no reaction); it transports by
chemotaxis `+χ ∇v` (gradient reconstructed directly from the v-cloud, `lam=1`, no
elliptic solve) plus diffusion.

### Cross-species injection kernel (the chemical reaction)

The chemical source `u − v` is **not** a multiplicative rate `(u−v)/v` on existing
v-particles. It is the exact reaction substep of `v_t = u − v` over a step `τ`:

```
v^{n+1} = e^{−τ} v* + (1 − e^{−τ}) u*
```

realized as two unbiased components on the *transported* clouds, with **equal**
per-particle mass for both species (`ω_u = ω_v`, so `q = (1−e^{−τ}) ω_u/ω_v`
reduces to a Bernoulli probability `p = 1−e^{−τ}`):

- **decay**: each transported v-particle survives with probability `e^{−τ}`
  (dies with prob `p = 1−e^{−τ}`);
- **injection**: each transported u-particle spawns a new equal-weight v-particle
  at its own post-transport location with probability `p = 1−e^{−τ}`.

This is unbiased for `(1−e^{−τ}) μ_u*` and avoids division by small `v`.

### Initial condition (Li–Shu–Yang)

Super-critical concentrated Gaussians, mass `10π` each:

```
u0(x) = 840 exp(−84 |x|²)     (∫u0 = 840π/84 = 10π > 8π)
v0(x) = 420 exp(−42 |x|²)     (∫v0 = 420π/42 = 10π)
```

Equal per-particle mass `10π / N` for both clouds.

## LDG alignment and reporting language

- We use the **same initial condition and reporting times** as the LDG study:
  snapshots of `u` (and `v`) at `t = 6e-5, 1.2e-4, 2.0e-4`.
- The particle-field method **tracks the pre-singular concentration up to the
  resolution gap**; **we do not claim the singular blow-up time**.
- The resolution-gap indicator
  `t_gap(N,K;θ) = inf{ t : S_{2K,4N}(t)/S_{K,N}(t) ≥ θ }`, `θ ∈ {1.05, 1.10}`,
  with `S_{K,N}(t) = ‖P_K μ_t^N‖_{L²}`, is a **resolution-gap indicator, not a
  blow-up time** (computed by `tgap.py` from a base `(N,K)` and a refined `(4N,2K)`
  run on a common time span).
- The reconstructed **peak is bandwidth-sensitive**; the **core radii
  `R_0.5(t)`, `R_0.8(t)` are reconstruction-free** (particle quantiles).

## Boundary-condition disclosure

The LDG paper uses **homogeneous Neumann** boundary conditions on a square domain.
This implementation uses a **periodic / core-adaptive Fourier window**. This is
therefore an **LDG-style** comparison, **not a strict Neumann method-to-method
benchmark**. Boundary effects are negligible over the very short reporting times
because the Gaussian stays far from the window edge: the half-width
`L(t) = max(L_min, γ R_q, γ_diff √(2Dτ))` tracks the collapsing core, and the
concentrated mass never reaches `|y| ≈ π`. (The `outside_v_frac` column logs the
fraction of v-particles that alias out of the window; it stays small through the
reporting window.) This disclosure is also recorded in `config.json` /
`manifest.json` of every run.

## Files

| file | role |
|---|---|
| `simulation.py` | particle solver; writes `diag_*.csv`, `snapshots/*.npz`, `config.json`, `manifest.json` |
| `adaptive_window.py` | vendored — core-adaptive window geometry + Fourier density coeffs |
| `field_pp.py` | vendored — `grad_v` from the v-cloud (`lam=1`, low-pass taper) + recon peak/L2/grid |
| `tgap.py` | vendored — post-processing `t_gap` resolution-gap indicator from a paired (base, refined) run |
| `plot_ldg_style.py` | publication figures from saved snapshots + `diag_*.csv` (does **not** rerun the solver) |

`diag_*.csv` carries `N`, `K`, and an `S_L2` column (alias of `S_L2_u`) so
`tgap.py` reads its default column and `(N,K)→(4N,2K)` pairing check unchanged.

## How to run

Smoke (tiny, ~30 s on CPU; still reaches the first report time):

```bash
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
$PY simulation.py --smoke --seed 0 --outdir results/smoke
$PY plot_ldg_style.py --results_dir results/smoke
```

Production pair (base `(N,K)` and refined `(4N,2K)` from the same IC over a common
time span), e.g.:

```bash
# BASE  (N, K)
$PY simulation.py --N 20000 --K 5 --tau 1e-6 --n_steps 200 --diag_every 4 \
    --seed 0 --report_times 6e-5 1.2e-4 2.0e-4 --outdir <run>/base
# REFINED (4N, 2K)
$PY simulation.py --N 80000 --K 10 --tau 1e-6 --n_steps 200 --diag_every 4 \
    --seed 0 --report_times 6e-5 1.2e-4 2.0e-4 --outdir <run>/refined
# resolution-gap indicator
$PY tgap.py --pairs <run>/base/diag_*.csv:<run>/refined/diag_*.csv \
    --out <run>/tgap_table
# figures (run on the REFINED dir; snapshots + diag there)
$PY plot_ldg_style.py --results_dir <run>/refined --out_dir <run>/figures
```

## Regenerating figures from saved data

The figures depend **only** on the saved `snapshots/*.npz` and `diag_*.csv`; no
solver rerun is needed:

```bash
$PY plot_ldg_style.py --results_dir <reference_results_dir>
```

Each figure also ships a small `plot_data/figure_<name>.npz` with the exact arrays
plotted, so a downstream user can re-style without any of the above.

## Key diagnostics (columns in `diag_*.csv`)

- `M_u`, `M_v` — particle masses (`M_u` conserved at `10π`; `M_v` follows
  `M_u + (M_v0 − M_u) e^{−t}`, here constant since `M_u0 = M_v0 = 10π`);
- `S_L2` (= `S_L2_u`) — reconstructed `L²` norm `S_{K,N}(t)` (**bandwidth-sensitive**);
- `R_0.5`, `R_0.8`, `R_0.9` — core radii (**reconstruction-free** particle quantiles);
- `peak_PK_u` — reconstructed peak (**bandwidth-sensitive**);
- `outside_v_frac` — v-particle aliasing fraction (boundary/window sanity);
- `drift_cfl` — chemotactic-drift CFL guard.
