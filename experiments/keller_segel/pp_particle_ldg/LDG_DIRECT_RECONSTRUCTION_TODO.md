# BLOCKER for §5.4: direct LDG comparison + particle-adaptive reconstruction

This note supersedes the current §5.4 workflow that compares the particle method mainly against the finite-volume baseline.  The present results are not sufficient for the paper until the two items below are done.

## 0. Why the current §5.4 is not enough

The current repository already has a fully parabolic--parabolic particle run and a deterministic grid reference, but the grid reference is a finite-volume baseline, not LDG.  Therefore it should not be presented as a direct comparison with the LDG literature.  The current particle table also changes both particle number and Fourier bandwidth when it refines `(N,K) -> (4N,2K)`, so its `t_gap` is partly a reconstruction-bandwidth artifact rather than the same LDG-style grid-resolution indicator.

The revised §5.4 must therefore answer two separate questions:

1. Can we reproduce the LDG benchmark directly with an LDG discretization, or at least compare against published LDG curves/numbers without replacing LDG by FVM?
2. Once the particle cloud has concentrated, can we use the particle distribution itself to refine the reconstruction, instead of relying on a uniform global Fourier bandwidth?

Until both questions are addressed, do not claim that §5.4 is a direct LDG comparison and do not use the current finite-volume baseline as the main comparator.

---

## 1. Direct LDG comparison, not finite volume

### Required change

Create a new experiment directory:

```text
experiments/keller_segel/ldg_direct/
    ldg_solver.py              # or a wrapper around a verified LDG implementation
    run_ldg.py
    compare_published_ldg.py   # optional digitized/published LDG curve comparison
    plot_ldg_direct.py
    README.md
```

Archive production results in:

```text
reference_results/keller_segel_ldg_direct/ldg_<run_id>/
    config_used.json
    S_curves.csv
    snapshots.npz
    tb_table.csv
    figures/*.pdf
    figures/*.png
    README.md
```

### Benchmark to match

Use the fully parabolic--parabolic Keller--Segel system

```math
u_t - \nabla\cdot(\nabla u-u\nabla v)=0,\qquad
v_t - \Delta v = u-v,
```

with the same concentrated Gaussian data and reporting times currently used in §5.4:

```math
u_0(x,y)=840\exp[-84(x^2+y^2)],\qquad
v_0(x,y)=420\exp[-42(x^2+y^2)],
```

```text
t = 6e-5, 1.2e-4, 2e-4.
```

Use the LDG domain and boundary condition from the Li--Shu--Yang benchmark after checking the paper directly.  Do not silently replace the LDG boundary condition by periodic or FVM boundary conditions.

### Minimum acceptable LDG evidence

The main paper must include at least one of these two pieces of evidence:

1. **Our LDG reproduction:** LDG snapshots and `S(t)=||u(t)||_L2` curves at two or more grid refinements, plus the LDG-style resolution-gap time

   ```math
   t_b(h;\theta)=\inf\{t:S_{h/2}(t)\ge \theta S_h(t)\},\qquad \theta=1.05.
   ```

2. **Published LDG comparison:** a clearly sourced comparison to the LDG paper's reported snapshots/curves/`t_b`, with digitization uncertainty stated if curves are digitized.

FVM can remain as an appendix sanity check, but it must not be the main LDG comparison.

### Do not write

```text
"direct LDG comparison" if the comparator is only fvm_baseline.py.
"LDG numerical blow-up time reproduced" unless the LDG reproduction/digitized comparison supports it.
"particle t_gap equals LDG t_b" if the particle refinement changes both N and reconstruction bandwidth.
```

---

## 2. Particle-adaptive reconstruction from the particle cloud

### Motivation

The particle method carries the finite measure and naturally tells us where the solution has concentrated.  The current global Fourier reconstruction is too blunt: low global `K` under-resolves the core, while high global `K` injects Monte-Carlo noise everywhere.  The right reconstruction for §5.4 should use particles to place local resolution in the core.

This is not optional appendix material for §5.4.  It is the likely reason the current estimates are not accurate enough.

### Required reconstruction modes

Add a post-processing script first:

```text
experiments/keller_segel/pp_particle_ldg/adaptive_reconstruct.py
```

It must read saved particle snapshots from the fully pp particle run and produce:

```text
reference_results/keller_segel_ldg_direct/particle_adaptive_<run_id>/
    config_used.json
    adaptive_S_curves.csv
    adaptive_peak_curves.csv
    adaptive_tb_table.csv
    adaptive_snapshots.npz
    plot_data/*.npz
    figures/*.pdf
    figures/*.png
    README.md
```

Implement at least the following two reconstructions for comparison:

### A. Global Fourier baseline

This is the current output:

```math
u^{global}_{K_g}=P_{K_g}\mu^N.
```

Keep it as the baseline, but label it as bandwidth-sensitive.

### B. Particle-detected local residual reconstruction

Use the particle cloud to detect the core and refine only there:

1. Compute the particle center `x_c(t)` from the cell particles.
2. Compute reconstruction-free radii `R_0.5(t)` and `R_0.8(t)`.
3. Define the local window automatically, e.g.

   ```math
   W(t)=B(x_c(t),\alpha R_{0.8}(t)),\qquad \alpha\in[2,4].
   ```

4. Reconstruct a coarse global background `P_{K_g} μ`.
5. On `W(t)`, reconstruct the local residual using either a local Fourier window, a local KDE/blob, or a local finite-volume histogram with a mass-conserving kernel:

   ```math
   u^{adapt}(x)=P_{K_g}\mu^N(x)
   + \chi_W(x)\,\mathcal R_{loc}\{\mu^N-P_{K_g}\mu^N dx\}(x).
   ```

6. Choose local resolution from the local particle spacing, not by hand:

   ```math
   h_{loc}(t) \approx \sqrt{|W(t)|/N_W(t)},
   \qquad
   K_{loc}(t) \approx c\,R_W(t)/h_{loc}(t),
   ```

   with `c` recorded in `config_used.json`.

7. Report mass conservation, non-negativity/minimum value if the residual reconstruction is signed, and sensitivity to `(K_g, alpha, c)`.

For multi-core or non-radial tests later, replace the single ball by particle-histogram connected components.  For the one-bulge LDG benchmark, the quantile-radius window is enough.

### Diagnostics to recompute with adaptive reconstruction

For each particle run, compute all of these from both the global and adaptive reconstructions:

```text
S_L2(t) = ||u(t)||_L2
peak(t) = ||u(t)||_infty
t_gap(theta=1.05) using the same reconstruction rule for both resolutions
core-local S_core(t)
R_0.5(t), R_0.8(t) directly from particles, no reconstruction
```

The final §5.4 comparison should have columns like:

```text
method | resolution | reconstruction | S(6e-5) | S(1.2e-4) | S(2e-4) | peak(2e-4) | R_0.8(2e-4) | gap time
LDG    | h           | DG polynomial  | ...     | ...       | ...     | ...         | ...          | ...
particle | N         | global Fourier | ...     | ...       | ...     | ...         | ...          | ...
particle | N         | particle-adaptive local residual | ... | ... | ... | ... | ... | ...
```

### Acceptance criteria

The adaptive reconstruction is useful only if at least one of the following happens:

1. `S_L2(t)` and peak move closer to the direct LDG result at the LDG reporting times.
2. The particle `t_gap(1.05)` becomes less sensitive to the arbitrary global bandwidth `K`.
3. The adaptive reconstruction agrees with reconstruction-free radius collapse in the resolved window, while global Fourier does not.

If none of these hold, report the negative result honestly and do not claim that adaptive reconstruction fixes §5.4.

---

## 3. Manuscript rewrite after the new runs

Replace the current §5.4 structure by:

```text
5.4 Fully parabolic--parabolic Keller--Segel: direct LDG comparison
    5.4.1 LDG reproduction / published LDG comparison
    5.4.2 Particle method on the same equation
    5.4.3 Global Fourier versus particle-adaptive reconstruction
```

The finite-volume baseline should move to an appendix as an implementation sanity check.

A safe paragraph if the adaptive result is positive:

```latex
The initial particle method output based on a uniform Fourier reconstruction underestimates the concentrating core unless the global bandwidth is increased, but increasing the global bandwidth also amplifies Monte-Carlo coefficient noise.  We therefore add a particle-adaptive reconstruction: the particle cloud determines the core center and the quantile radius \(R_{0.8}\), and a local residual reconstruction is applied only inside this window.  This refinement is not an externally prescribed mesh; it is induced by the empirical measure.  With this reconstruction the particle diagnostics at the LDG reporting times move closer to the direct LDG calculation, while the reconstruction-free radii remain unchanged.
```

A safe paragraph if the adaptive result is mixed:

```latex
The particle cloud gives a natural location for local reconstruction refinement, but in the present implementation the improvement is mainly diagnostic rather than a fully resolved blow-up-time estimate.  The adaptive reconstruction reduces the bandwidth dependence of the reconstructed peak and \(L^2\) norm in the pre-singular window, while the late-time gap time remains sensitive to local bandwidth and fit window.  We therefore compare reporting-time concentration and reconstruction-free core radii to the LDG benchmark, but do not quote a continuum blow-up time.
```

---

## 4. Immediate next actions

1. Stop using `ldg_pp_baseline/fvm_baseline.py` as the main §5.4 comparator.
2. Implement or import a direct LDG solver, or digitize/compare to the published LDG result with uncertainty.
3. Add `adaptive_reconstruct.py` and run it on saved particle snapshots before rerunning particle dynamics.
4. Only after this post-processing audit, decide whether the dynamics also need online adaptive reconstruction for `∇v`.
5. Rewrite §5.4 so that the reader sees LDG vs particle global Fourier vs particle-adaptive reconstruction, not FVM vs particle.
