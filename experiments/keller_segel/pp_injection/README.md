# Parabolic–parabolic Keller–Segel: cross-species injection chemical reaction

This experiment ports the working multispecies-injection prototype into the
revision repository. It demonstrates the **correct** multi-species representation
of the parabolic–parabolic Keller–Segel chemical equation on the torus
`[0, 2π]²`.

## Model

Two equal-weight particle clouds with the **same per-particle mass**
`ω_u = ω_v = M_u / n_samples`:

- **cells** `u`: `u_t = ∇·(∇u − χ u ∇v) = 0` — conservative chemotaxis +
  diffusion, **no reaction**, so the cell mass `M_u` is exactly conserved;
- **chemical** `v`: `v_t = Δv + u − v` — diffusion plus the **cross-species
  source** `u − v`.

## The injection kernel (this is the point)

The chemical reaction half-step `v_t = u − v` has the exact integrator over a
step `τ`:

```
v^{n+1} = e^{-τ} v* + (1 − e^{-τ}) u*
```

We realize this as a genuine **two-species birth–death process** — *not* as a
multiplicative `(u−v)/v` rate on existing `v` particles:

- **DECAY**: each transported `v`-particle dies with probability `p = 1 − e^{-τ}`
  (survives with `e^{-τ}`);
- **BIRTH (injection)**: each transported `u`-particle spawns a new `v`-particle
  **at its own location** with probability `p = 1 − e^{-τ}`. The general birth
  mean is `(1 − e^{-τ}) ω_u/ω_v`, which reduces to `1 − e^{-τ}` here because
  `ω_u = ω_v`.

In conditional mean this gives
`E[μ_v^{n+1}] = e^{-τ} μ_v* + (1 − e^{-τ}) μ_u*`, i.e. exactly the substep above.
There is no division by small `v`, no positivity floor, and `v` can be created
where `v` is currently absent. The cloud mass then follows the **exact analytic
law**

```
M_u(t) = M_u(0),     M_v(t) = M_u + (M_v(0) − M_u) e^{-t}.
```

The simulation validates this law step by step.

### Legacy multiplicative form (deprecated)

`--legacy_multiplicative_v_source` raises `NotImplementedError` on purpose. The
multiplicative `(u−v)/v` rate on existing `v` particles is fragile near small
`v`, conceptually wrong for a cross-species source, and **is not used in the
paper**. The default path is the injection kernel.

## Running

Interpreter (CPU; the CUDA/jax_plugin warnings on stderr are harmless):

```
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
```

Smoke test (tiny/fast — `n_samples=8000`, `T=1.0`, `dt=2e-3`):

```bash
$PY simulation.py --smoke --seed 0
$PY plot.py --results_dir results_smoke
```

Production (modest):

```bash
$PY simulation.py --n_samples 40000 --Mv0 0.3 --chi 1.0 --T 2.0 --dt 1e-3 \
    --seed 0 --out_dir <out_dir>
$PY plot.py --results_dir <out_dir>
```

Key flags: `--n_samples`, `--Mv0`, `--chi`, `--T`, `--dt`, `--n_freq`, `--seed`,
`--out_dir`, `--smoke`, `--legacy_multiplicative_v_source`.

## Output files

Written into `--out_dir`:

- `config.json` — model description, reaction substep, resolved args, `p = 1−e^{-τ}`,
  initial masses, `population_control_active=false`, reference-solver note.
- `manifest.json` — reproducibility record: git commit hash, exact command line
  (`sys.argv`), Python + numpy/jax/matplotlib versions, resolved args, seed,
  output dir, ISO datetime, population-control flag.
- `mass_balance.csv` — time series:
  `step,t,M_u,M_v,M_v_exact,rel_err,n_birth,n_death,N_u,N_v`.
- `metrics_summary.csv` — key scalars: `max_abs_Mu_drift`,
  `max_relerr_Mv_law`, `M_v_final`, `M_v_exact_final`, etc.
- `plot_data/figure_mass_balance.npz` — arrays for the figure (so the plot can be
  regenerated without rerunning the solver).
- `figures/figure_mass_balance.{pdf,png}` — produced by `plot.py`.

## Regenerating the figure WITHOUT rerunning the solver

`plot.py` reads only `plot_data/figure_mass_balance.npz` (with a
`mass_balance.csv` fallback) — it never calls the solver:

```bash
$PY plot.py --results_dir <dir-with-plot_data-or-csv>
```

This works on any saved results directory, including those under
`reference_results/pp_injection/<run_id>/`.
