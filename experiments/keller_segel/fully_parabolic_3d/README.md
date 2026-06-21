# 3D fully parabolic–parabolic Keller–Segel particle solver

Equal-weight branching / injection particle method for the **fully parabolic–parabolic**
Keller–Segel system on the periodic box `T_L^3 = [-L/2,L/2]^3` (L=12):

```
u_t = D_u Δu − χ ∇·(u ∇v)          (cell density, conservative particles)
v_t = D_v Δv + α u − β v            (chemical, created from u; v0 = 0)
```

This is NOT the old parabolic–elliptic / screened-Poisson model. The chemical field `v`
is carried by its own particle cloud and is created from scratch (`v0=0`) by a cross-species
**injection** step — never an `(u−v)/v` multiplicative rate.

## Algorithm (first-order Lie split, step τ)

1. reconstruct `∇v` at the cell particles from the current v-cloud (periodic Fourier, bandwidth `K_dyn`);
2. transport u: `X += χ ∇v τ + sqrt(2 D_u τ) ξ`, wrap to torus;
3. transport v: `Y += sqrt(2 D_v τ) ζ`, wrap;
4. exact decay–injection: `μ_v^{n+1} = e^{−βτ} μ_v* + (α/β)(1−e^{−βτ}) μ_u*`
   — existing v-particles survive w.p. `e^{−βτ}`; transported u-particles inject new v-particles
   with the **minimum-variance integer kernel** of mean `(α/β)(1−e^{−βτ}) ω_u/ω_v`.

Equal particle masses `ω_u = ω_v`. Concentration diagnostics are the **reconstruction-free**
particle quantile radii `R_q` (no field reconstruction). Performance: a JITTED fixed-capacity
grad-v buffer (`field3d_fourier.grad_v_buffer`, used via `simulate(..., fast=True)`), verified
equivalent to the eager dynamic-cloud path in `test_buffer_equiv.py`.

## Code

| file | role |
|------|------|
| `field3d_fourier.py` | periodic Fourier reconstruction of v and ∇v; jitted `grad_v_buffer` |
| `injection_kernel.py` | exact decay–injection (survival + min-variance integer injection) |
| `exact_linear_modes.py` | analytic û_k, v̂_k, mass laws for the linear verification |
| `initial_conditions.py` | radial / tetra IC samplers, torus wrap |
| `diagnostics_pp3d.py` | torus centroid, core radii R_q, cluster diagnostics |
| `simulation_pp3d.py` | the coupled solver `simulate(cfg, seed, fast=)` |
| `run_linear_verification.py` | Experiment A driver (exact-linear) |
| `run_radial_pilot.py` | Experiment B pilot: sweep M (`--Ms`) or χ (`--chis`), `--fast` |
| `run_radial_production.py` | Experiment B production driver (one config/seed) |
| `run_tetra_control.py` | Experiment C driver (active χ>0 / control χ=0) |
| `plot_radial_response.py` | Figure B (reads diagnostics only) |
| `plot_tetra_control.py` | Figure C (reads diagnostics only) |
| `test_*.py` | unit tests (field, injection, exact-linear, buffer equivalence) |

Tests: `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python test_buffer_equiv.py` (and the other `test_*.py`).
Env: `/mnt/home/lyuliyao/.conda/envs/heat/bin/python`. GPU submits use `submit_*.sb`
(`-A Multiscaleml`, `--constraint="a100|h200|l40s"` to avoid 16GB OOM at K=12/large N).

## Experiments and results (under `reference_results/keller_segel_pp3d/`)

### A. Exact-linear verification — `linear_*`
χ=0, D_u=D_v, v0=0. Mass law `M_v(t)=M_u(0)(1−e^{−t})` and Fourier-mode errors decrease at the
Monte-Carlo rate `N^{−1/2}`. All acceptance gates pass (validates the coupled injection algorithm).

### B. Radial delayed response — `radial_*_M88_M96_K12_8seed` → **Figure B**
Normalized regime D=α=β=χ=1, single radial Gaussian (σ=0.45), the effective coupling is F=M.
Sharp transition between diffusion (R_0.5 grows) and **delayed focusing** (cloud expands as the
chemical builds from 0, then turns over and concentrates). Delayed config **M=96**: core ratio
R_0.5(T)/R_0.5(0) ≈ **0.21**, N-converged (0.219/0.211/0.206 at N=2e4/1e5/3.2e5), genuine delay
(t_turn≈0.20, peak expansion ×1.26). Weak config **M=72**: diffuses (ratio 3.75). See the dir
README for the full table and the bandwidth-dependence of the transition mass (M\*∈(72,80] at K=12
vs (88,92] at K=8) and the R_0.8 seed/N-sensitivity caveat. Pilot record: `msweep_*`, `ksens_*`,
`PILOT_FINDINGS_msweep.md`.

Regenerate: `python plot_radial_response.py --run_dir <radial dir> --baseN 100000 --K 12`

### C. Tetra multi-cluster — `tetra_*_a1_M240_K12` → **Figure C**
Four selected-cluster-mass clusters (M=240, σ_c=0.25; each focuses at this fixed bandwidth) on a tetrahedron, active χ=1 vs diffusion control
χ=0 (common seed). **Mutual chemotactic attraction + individual collapse**: active per-cluster
R_0.5 → 0.16 (collapse) vs control 3.78 (spread), a ~23× contrast; active d_min ↓ vs flat control.
Pilots: `tetra_pilot_*`, `tetra_pilot2_*`. See the dir README for the table and caveats.

Regenerate: `python plot_tetra_control.py --run_dir <tetra dir>`

## Manuscript text

Draft §5.6 text (radial + tetra + algorithm/setup + language guardrails) is in
`DRAFT_TEXT_pp3d_section.md` — markdown for hand-paste into Overleaf (paper/ is not edited here).
Guardrails honored: no continuum blow-up / no universal critical-mass claim (transition is
numerical/bandwidth-dependent); cross-species source is injection, not `(u−v)/v`; reconstruction-free
core radii separated from bandwidth-sensitive peaks/R_0.8.

---

## Validation closure (2026-06-20)

A focused closure pass on the remaining 3D fully parabolic–parabolic validation items.
All new code is committed; all new data, figures, and a neutral analysis report live under
`reference_results/keller_segel_pp3d/validation_closure_<UTC>_<sha>/`. **No paper file is
edited by this pass.**

### Important readout caveat

`R_0.5` (and the other quantile radii `R_q`) are **reconstruction-free as a readout** —
they are computed directly from particle positions with no field reconstruction. However,
their **dynamics still depend on `K_dyn`** through the reconstructed chemotactic drift `∇v`
(bandwidth `K_dyn`). The validation-closure resolution audit quantifies this via a
same-cloud `∇v` discrepancy between `K=8/12/16`. The radial/tetra "transitions" are
**fixed-bandwidth numerical transitions**, not continuum critical masses.

### New code

| file | role |
|------|------|
| `repro.py` | shared reproducibility metadata (git, argv, command.txt, host, versions, device, buffer capacity, population_control) |
| `test_validation_extra.py` | axis-permutation symmetry, periodic wrapping, fixed-seed reproducibility, no-hidden-population-control, row-wise injection location |
| `bench_performance.py` | fast/slow timing benchmark (Task A.2) |
| `vc_load.py` | config-safe diagnostics loader (never pools differing M/N/K/tau) |
| `analyze_radial_validation.py` | B.1 tau + B.2 K metrics → `figure_radial_tau_K_validation` |
| `analyze_resolution_audit.py` | B.3 `Q_0.2` gate audit + B.4 same-cloud drift → `figure_radial_resolution_audit` |
| `plot_radial_state.py` | Task C u/v xy mass-marginal state-evolution figure from saved clouds |
| `analyze_tetra_validation.py` | Task D one-factor refinement + centroid reliability → `figure_tetra_resolution_validation` |

`simulation_pp3d.simulate` gained an optional `drift_probe_K` (same-cloud `∇v` discrepancy
at diagnostic times; draws no RNG and does not enter transport → trajectory unchanged) and
the tetra diagnostics gained the per-cluster circular-resultant reliability `A_{m,j}` plus
the six pairwise centroid distances. `run_radial_production.py` gained `--drift_probe` and
`--save_times` (cloud snapshots). `plot_radial_response.py` / `plot_tetra_control.py` now
take explicit baseline selectors and group seed means by the FULL config tuple (they never
auto-pick the smallest `tau` or pool different bandwidths).

### Reproduce

```bash
# tests (CPU)
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python test_validation_extra.py   # + the other test_*.py

# GPU runs (SLURM); set OUT to a validation_closure_* subdir
REPO=$PWD OUT=<vc>/radial_K          sbatch submit_radial_K.sb
REPO=$PWD OUT=<vc>/radial_tau        sbatch submit_radial_tau.sb
REPO=$PWD OUT=<vc>/radial_state_figure sbatch submit_radial_state.sb
REPO=$PWD OUT=<vc>/tetra_refinement  sbatch submit_tetra_refine.sb
REPO=$PWD OUT=<vc>/performance       sbatch submit_bench.sb

# figures (solver-free, from saved CSV/NPZ)
python analyze_radial_validation.py --K_dir <vc>/radial_K --tau_fine_dir <vc>/radial_tau \
    --tau_base_dir <radial production dir> --out_root <vc>
python analyze_resolution_audit.py  --K_dir <vc>/radial_K --out_root <vc>
python plot_radial_state.py --clouds <vc>/radial_state_figure/state_*/snapshots/clouds_seed0.npz --out_root <vc>
python analyze_tetra_validation.py  --refine_dir <vc>/tetra_refinement --out_root <vc>
```

See `validation_closure_*/ANALYSIS_REPORT.md` for the validation matrix and findings.
