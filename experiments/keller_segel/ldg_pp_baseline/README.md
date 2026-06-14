# LDG-aligned fully parabolic–parabolic Keller–Segel: grid baseline (§5.4)

Deterministic **grid baseline** for the fully parabolic–parabolic Keller–Segel
concentration benchmark of Li–Shu–Yang (LDG), against which the particle method is
compared. This is the grid reference; the particle solver is
`experiments/keller_segel/ldg_comparison/` (see `../pp_particle_ldg/README.md`).

## System

On `Ω = [-0.5, 0.5]²` with homogeneous **Neumann** boundary conditions:

```
u_t = div( grad u - u grad v )     (cell density, conservative)
v_t = Δv + u - v                   (chemoattractant)
u0  = 840 exp(-84 r²),  v0 = 420 exp(-42 r²)    (M_u = 10π > 8π, supercritical)
```

The Gaussian has std `1/√(2·84) ≈ 0.077`, so its value at `|x|=0.5` is `e^-21 ≈ 7e-10`:
the Neumann / periodic / whole-plane distinction is numerically negligible over the
reported times. This makes the periodic core-adaptive **particle** run and this
Neumann **grid** baseline comparable on the same near-whole-plane dynamics.

## Scheme (`fvm_baseline.py`)

Cell-centered finite volume, `n×n` cells:
- **u** flux `F = -grad u + u grad v`, `u_t = -div F`: central diffusion + **first-order
  upwind** advection (positivity-preserving), zero boundary flux ⇒ exact discrete mass
  conservation;
- **v**: central 5-point Laplacian + reaction `u - v`;
- explicit Euler, adaptive `dt = cfl·min(dx²/4, dx/max|∇v|)`, `cfl=0.25`.

Verified (independent adversarial audit): mass conserved to ~1e-16, positivity
`u_min ≥ 0` (~1e-16), correct chemotaxis sign, provably positivity-preserving at
`cfl=0.25`.

## Diagnostics

Per save time: `S_L2 = ||u||_L2` (global), `S_core` (core-local, `B(x_c, 3R_0.8)`),
peak, `u_min` (positivity), masses, reconstruction-free radii `R_0.5, R_0.8`.

`tb_from_pair.py` forms the **LDG-style resolution-gap proxy**

```
t_b(n; θ) = inf{ t : S_{2n}(t) ≥ θ S_n(t) }
```

from grid pairs `(n)→(2n)`. This is a **numerical resolution-gap indicator, not a
continuum blow-up time**: it is itself resolution-dependent (`t_b(1.05)` = 3.5e-5 at
128→256, 5.0e-5 at 256→512) and a few × 1e-5, the same order as the LDG numerical
blow-up window (~1.21e-4) and preceding it, as a resolution-gap onset should.

## Run

```bash
# heat conda env. From this directory:
python fvm_baseline.py --smoke                                  # n=64, T=6e-5
for n in 128 256 512; do python fvm_baseline.py --n $n --T 2e-4 --cfl 0.25 --out_dir results/n$n; done
python tb_from_pair.py --pairs results/n128/S_curves.csv:results/n256/S_curves.csv \
                               results/n256/S_curves.csv:results/n512/S_curves.csv --out results/tb_global
python plot_baseline.py --baseline_dir <baseline_run> \
    --particle_base <ldg_comparison base diag.csv> \
    --particle_refined <ldg_comparison refined diag.csv> --out_dir <baseline_run>/figures
```

Production results are archived under
`reference_results/keller_segel_ldg_pp/baseline_<run_id>/` (`S_curves.csv`,
`snapshots.npz`, `config_used.json`, `tb_global.csv`, figures, plot_data).

## Result (honest scope)

The baseline reproduces LDG-style supercritical concentration on the same equation
and initial data, with positivity and exact cell-mass conservation, and yields a
finite resolution-gap proxy `t_b` of order 1e-5–1e-4. We report it as a numerical
resolution-gap indicator and **do not** quote a continuum blow-up time (see
`../core_local_proxy/` for the reconstruction-free radius diagnostics and the
window-sensitivity of any radius-fit candidate concentration time).
