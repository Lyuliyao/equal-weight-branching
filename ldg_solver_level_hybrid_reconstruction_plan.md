# Solver-level hybrid residual reconstruction for the LDG Keller--Segel benchmark

This note replaces the previous post-processing-only reconstruction plan for §5.4.
The key change is:

> The hybrid residual reconstruction must be used **inside the solver** to evaluate the fields that drive particle motion and reaction. It is not enough to apply it only to the final particle cloud for plotting or diagnostics.

The second key change is:

> §5.4 must compare directly with the LDG Keller--Segel benchmark. A finite-volume baseline may remain as an internal sanity check or appendix, but it must not be the main reference for the paper's LDG comparison.

---

## 1. Paper-level purpose

The revised §5.4 should answer the following concrete question:

> On the fully parabolic--parabolic Keller--Segel benchmark used in the LDG literature, can the particle solver reproduce the LDG concentration regime when the reconstruction used **inside the time step** is locally enriched by the particle cloud?

The intended message is not simply that a particle cloud can be post-processed better. The intended message is:

1. A low global spectrum is enough for the smooth background.
2. The particle cloud detects where this global spectrum fails.
3. Local signed residual reconstruction in those windows improves the **field used by the solver**.
4. The resulting particle trajectory should be compared against the LDG refinement trend.
5. We still avoid claiming a continuum blow-up time; all blow-up times reported here are numerical resolution-gap indicators.

---

## 2. LDG benchmark that must be used in §5.4

### 2.1 Equation

Use the fully parabolic--parabolic Keller--Segel system

\[
 u_t - \nabla\cdot(\nabla u-u\nabla v)=0,
 \qquad
 v_t-\Delta v=u-v .
\]

Equivalently,

\[
 u_t=\Delta u-\nabla\cdot(u\nabla v),
 \qquad
 v_t=\Delta v+u-v .
\]

### 2.2 Domain and boundary condition

The LDG benchmark should be treated as a Neumann-boundary computation on

\[
 \Omega=[-1/2,1/2]^2 .
\]

Use homogeneous Neumann boundary conditions for the LDG reference.

For the particle solver, the existing adaptive periodic/Fourier window may be used only if the manuscript explicitly states that the particle run is LDG-aligned rather than a strict Neumann particle implementation. Because the initial Gaussian core is far from the boundary and the reported times are very short, the boundary mismatch is expected to be small, but it must not be hidden.

### 2.3 Initial data

Use the LDG benchmark initial data

\[
 u_0(x,y)=840\exp[-84(x^2+y^2)],
 \qquad
 v_0(x,y)=420\exp[-42(x^2+y^2)] .
\]

The cell mass is

\[
 M_u=\int_{\mathbb R^2}840e^{-84r^2}\,dxdy=10\pi>8\pi,
\]

so this is the supercritical two-dimensional concentration test.

### 2.4 Reporting times

Use the LDG reporting times

\[
 t=6\times10^{-5},\qquad 1.2\times10^{-4},\qquad 2.0\times10^{-4}.
\]

Do not substitute a different set of times unless it is explicitly labeled as an auxiliary diagnostic.

### 2.5 LDG reference requirement

For the main text, do **not** replace LDG by FVM. Use one of the following:

1. Implement a direct LDG solver for this benchmark; or
2. Digitize / extract the published LDG reporting-time data and resolution curves, with a clear statement that these are published LDG reference data.

The FVM code can remain for debugging mass conservation, positivity, and sign conventions. In the manuscript, it should be appendix/sanity-check material only.

### 2.6 LDG diagnostics

For each LDG resolution, save

\[
 S_h(t)=\|u_h(t)\|_{L^2(\Omega)},
 \qquad
 P_h(t)=\|u_h(t)\|_{L^\infty(\Omega)},
\]

mass, positivity / minimum value, and reconstruction-free or cell-mass radii

\[
 R_{0.5}(t),\qquad R_{0.8}(t).
\]

For refinement pairs, compute the LDG-style resolution-gap proxy

\[
 t_b(h;\theta)=\inf\{t:S_{h/2}(t)\ge \theta S_h(t)\},
 \qquad \theta=1.05.
\]

This is a numerical resolution-gap indicator. It is not a continuum blow-up time.

---

## 3. Why solver-level reconstruction is necessary

The current particle algorithm reconstructs a field from the empirical measure to evaluate the coefficients used in the time step. Therefore, if the reconstruction is under-resolved, the particle dynamics themselves are affected.

For the Keller--Segel solver, the critical coefficient is the chemotactic drift

\[
 dX_u=\nabla v(X_u)\,dt+\sqrt{2}\,dW .
\]

Thus the reconstruction of the chemical field \(v\) is not merely an output operation. It determines the motion of the cell particles.

The new §5.4 solver comparison must therefore include at least two particle solvers:

1. **global-spectrum solver:** uses a single global Fourier reconstruction for \(v\);
2. **hybrid-residual solver:** uses a low global spectrum plus local residual reconstruction for \(v\), and uses this hybrid field inside the transport step.

A third solver, using a Horvitz--Thompson residual sketch, can be added after the deterministic hybrid solver works.

---

## 4. Hybrid residual reconstruction operator

At each time step, let the relevant empirical measure be

\[
 \mu=\sum_i\omega_i\delta_{X_i}.
\]

For Keller--Segel drift evaluation, this measure is usually the chemical particle measure \(\mu_v\). For diagnostics of the cell density, the same construction can be applied to \(\mu_u\).

### 4.1 Low global spectrum

First compute a low-order global background

\[
 u_{\rm lo}(x)=P_{K_g}\mu(x),
 \qquad K_g\ll K_{\rm full}.
\]

This is meant to capture the smooth background and long-range field.

### 4.2 Particle-detected windows

Detect local windows from the particle cloud, not from the reconstructed image.

For the single-core LDG benchmark, start with a one-window rule:

\[
 x_c=\frac{\sum_i\omega_iX_i}{\sum_i\omega_i},
\]

\[
 R_q=\inf\left\{r:\frac{\sum_{|X_i-x_c|\le r}\omega_i}{\sum_i\omega_i}\ge q\right\}.
\]

Use

\[
 W=B(x_c,\alpha R_{0.8}),
 \qquad
 W^{\rm pad}=B(x_c,\alpha_{\rm pad}R_{0.8}),
\]

with typical values

```text
alpha      = 2.5 or 3.0
alpha_pad  = 1.5 * alpha or 2.0 * alpha
```

For the LDG benchmark, the first production version should use the single core. Multi-window detection can be added later for separated islands / nonradial tests.

### 4.3 Local reconstruction operator

Use one of two local operators:

**Option A: local blob / KDE**

\[
 T_j\nu=\eta_{h_j}*\nu .
\]

**Option B: local spectral window**

\[
 T_j\nu=P_{K_\ell}^{W_j}\nu,
 \qquad K_\ell>K_g .
\]

The local spectral option is closer to the existing Fourier code and gives analytic gradients. Use it first if implementation time permits. The blob option is cheaper and easier, but gradients must be handled carefully.

### 4.4 Correct residual form

Do not add the local blob or local spectrum directly to the global spectrum. That double-counts low-frequency mass.

The solver-level hybrid field should be

\[
 \mathcal R_{\rm hyb}(\mu)(x)
 =
 u_{\rm lo}(x)
 +
 \sum_j\chi_j(x)
 \left[
 T_j(\mu|_{W_j^{\rm pad}})(x)
 -
 T_j(u_{\rm lo}\mathbf 1_{W_j^{\rm pad}}dx)(x)
 \right].
\]

For the local blob version,

\[
 \mathcal R_{\rm hyb}(\mu)(x)
 =
 u_{\rm lo}(x)
 +
 \sum_j\chi_j(x)
 \left[
 \eta_{h_j}*(\mu|_{W_j^{\rm pad}})(x)
 -
 \int_{W_j^{\rm pad}}\eta_{h_j}(x-y)u_{\rm lo}(y)\,dy
 \right].
\]

For the local spectral version,

\[
 \mathcal R_{\rm hyb}(\mu)(x)
 =
 u_{\rm lo}(x)
 +
 \sum_j\chi_j(x)
 \left[
 P_{K_\ell}^{W_j}(\mu|_{W_j^{\rm pad}})(x)
 -
 P_{K_\ell}^{W_j}(u_{\rm lo}\mathbf 1_{W_j^{\rm pad}}dx)(x)
 \right].
\]

The second term is signed. This is expected and mathematically necessary: the residual is an approximation to

\[
 \mu-u_{\rm lo}dx,
\]

which is a signed measure.

### 4.5 Gradient formula

When the field enters the drift, compute

\[
 \nabla\mathcal R_{\rm hyb}(\mu)
 =
 \nabla u_{\rm lo}
 +
 \sum_j\nabla\left(\chi_j r_j\right),
\]

where

\[
 r_j=T_j(\mu|_{W_j^{\rm pad}})-T_j(u_{\rm lo}\mathbf 1_{W_j^{\rm pad}}dx).
\]

Do not drop the taper-gradient term:

\[
 \nabla(\chi_j r_j)=\chi_j\nabla r_j+r_j\nabla\chi_j .
\]

Use a smooth raised-cosine taper so that \(\nabla\chi_j\) is bounded and the correction turns off smoothly at the padded-window boundary.

---

## 5. Solver-level use in parabolic--parabolic Keller--Segel

The time step should use the hybrid reconstruction as follows.

### 5.1 Field construction before transport

At time \(t_n\), build the chemical field

\[
 v_n^{\rm hyb}=\mathcal R_{\rm hyb}(\mu_{v,n}).
\]

The windows may be detected from the cell cloud \(\mu_{u,n}\), from the chemical cloud \(\mu_{v,n}\), or from their union. For the first LDG-core implementation, use the cell cloud \(\mu_u\) to locate the concentrating core, then reconstruct \(v\) in that same core window. This aligns the enriched chemical gradient with the cell concentration region.

### 5.2 Transport step

Update cell particles with

\[
 X_{u,i}^{*}
 =
 X_{u,i}^n
 +\tau\nabla v_n^{\rm hyb}(X_{u,i}^n)
 +\sqrt{2\tau}\,\xi_i^n .
\]

Update chemical particles by diffusion:

\[
 X_{v,i}^{*}=X_{v,i}^n+\sqrt{2\tau}\,\zeta_i^n .
\]

Freeze the reconstruction during this Lie step. Do not update \(v_n^{\rm hyb}\) inside the particle loop.

### 5.3 Chemical decay / injection step

Use the exact cross-species injection step for

\[
 v_t=u-v.
\]

In measure form,

\[
 \mu_v^{n+1}=e^{-\tau}\mu_v^*+(1-e^{-\tau})\mu_u^* .
\]

Implementation:

- existing \(v\)-particles survive with probability \(e^{-\tau}\);
- transported \(u\)-particles inject \(v\)-particles with mean \((1-e^{-\tau})\omega_u/\omega_v\);
- use the minimum-variance integer injection kernel;
- do not use quotient branching such as \((u-v)/v\).

### 5.4 Output reconstruction

For fair reporting, save both:

1. the **solver field** used during the time step;
2. a **common diagnostic reconstruction** applied to all particle trajectories after the step.

This is important because otherwise the hybrid solver may look better simply because it reports its own enriched field. The reconstruction-free radii and common diagnostic reconstruction are needed to determine whether the particle trajectory itself improved.

---

## 6. Horvitz--Thompson residual sketch

The first solver-level implementation should use the deterministic full residual. After that works, add a cost-reduced HT residual sketch.

### 6.1 Residual score

For each window \(W_j\), build a cheap pilot reconstruction \(u_{{\rm pilot},j}\), and define particle scores

\[
 s_i^{(j)}
 =
 \chi_j(X_i)
 \frac{
 |u_{{\rm pilot},j}(X_i)-u_{\rm lo}(X_i)|
 }{
 u_{{\rm pilot},j}(X_i)+|u_{\rm lo}(X_i)|+\varepsilon
 }.
\]

This score estimates how much local density near particle \(X_i\) is not explained by the global low spectrum.

### 6.2 HT accept probabilities

For the quantitative signed residual sketch, use

\[
 q_i^{(j)}=\min\{1,\max(q_{\min},\lambda_j s_i^{(j)})\},
\]

where \(\lambda_j\) is chosen by bisection so that

\[
 \sum_i q_i^{(j)}\approx B_j .
\]

Then draw

\[
 A_i^{(j)}\sim {\rm Bernoulli}(q_i^{(j)}).
\]

### 6.3 HT residual field

The HT residual field is

\[
 \widehat u_{{\rm res},j}^{\rm HT}(x)
 =
 \chi_j(x)
 \left[
 \sum_{i:X_i\in W_j^{\rm pad}}
 \frac{A_i^{(j)}}{q_i^{(j)}}\omega_i\eta_{h_j}(x-X_i)
 -
 \int_{W_j^{\rm pad}}\eta_{h_j}(x-y)u_{\rm lo}(y)\,dy
 \right].
\]

It is essential that the retained empirical weights are divided by \(q_i^{(j)}\). Do **not** use \(q_iA_i\omega_i\).

Conditioned on the current particle cloud,

\[
 \mathbb E_A\widehat u_{{\rm res},j}^{\rm HT}(x)
 =
 \chi_j(x)
 \left[
 \eta_{h_j}*(\mu|_{W_j^{\rm pad}})(x)
 -
 \eta_{h_j}*(u_{\rm lo}\mathbf 1_{W_j^{\rm pad}}dx)(x)
 \right].
\]

### 6.4 Diagnostic accept rate versus budgeted accept rate

The budgeted HT accept rate is not by itself a clean unresolved-residual diagnostic, because if no clipping occurs then \(\sum_i q_i\approx B_j\), so

\[
 \bar q_j\approx B_j/N(W_j^{\rm pad}).
\]

Therefore save both:

\[
 \bar s_j=\frac1{N(W_j^{\rm pad})}\sum_{X_i\in W_j^{\rm pad}}s_i^{(j)},
\]

and a separate diagnostic accept rate

\[
 q_{i,{\rm diag}}^{(j)}=\min\{1,\lambda_{\rm diag}s_i^{(j)}\},
\]

with \(\lambda_{\rm diag}\) fixed across windows and times. Then

\[
 \bar q_{j,{\rm diag}}
 =
 \frac1{N(W_j^{\rm pad})}\sum_iq_{i,{\rm diag}}^{(j)}
\]

can be interpreted as:

```text
low diagnostic accept rate     : global spectrum already explains this window
high diagnostic accept rate    : unresolved local residual structure
growing diagnostic accept rate : reconstruction resolution should concentrate here
```

### 6.5 Positive-excess particles

The positive-excess version is useful for visualization only:

\[
 u_{\rm lo}^{+}(x)=\max\{u_{\rm lo}(x),0\},
\]

\[
 r_j^{+}(x)=(u_j(x)-u_{\rm lo}^{+}(x))_+,
\]

\[
 \alpha_j(x)=\min\left\{1,\frac{r_j^{+}(x)}{u_j(x)+\varepsilon_{\rm dens}}\right\}.
\]

This can show which particles correspond to positive residual structure, but it is not an unbiased signed residual estimator. In figures, label it explicitly as

```text
positive-residual-only diagnostic
```

Do not use it as the quantitative field inside the main solver comparison.

---

## 7. Implementation tasks

### 7.1 New files

Add a solver-level hybrid reconstruction module:

```text
experiments/keller_segel/ldg_comparison/hybrid_field_pp.py
```

It should provide a field object with:

```python
field.value(points)
field.grad(points)
field.diagnostics()
```

Add or update a solver driver:

```text
experiments/keller_segel/ldg_comparison/simulation_hybrid.py
```

or add flags to the existing `simulation.py`:

```text
--field_recon fourier|hybrid_full|hybrid_ht
--Kg <int>
--Kl <int>
--local_type spectrum|blob
--window_source u|v|union
--window_alpha <float>
--window_pad <float>
--B_target <int>
--q_min <float>
--hybrid_update_every <int>
--save_cloud_snapshots
```

### 7.2 Required behavior

The hybrid field must be used in the transport step. In the current parabolic--parabolic KS code path, replace the global Fourier `grad_v_from_cloud` call by

```python
hyb_v = build_hybrid_field_from_cloud(...)
gradv = hyb_v.grad(X_u)
```

and use `gradv` in the actual Euler--Maruyama update of the \(u\)-particles.

This is the central requirement. A script that only reconstructs final snapshots is not sufficient.

### 7.3 Save raw clouds

At every LDG reporting time, save

```text
X_u, X_v
weights_u, weights_v if applicable
mass_per_particle_u, mass_per_particle_v
t, report_time, seed
x_c, R_0.5, R_0.8, R_0.9
window definitions
solver reconstruction parameters
```

Do not save only reconstructed fields. Raw clouds are needed to audit the solver-level reconstruction and to evaluate common diagnostics.

### 7.4 Stability diagnostics

At every diagnostic time, save

```text
mass_u, mass_v
R_0.5, R_0.8
peak_solver_field
S_L2_solver_field
peak_common_diagnostic
S_L2_common_diagnostic
min_v_hyb
mass_v_hyb
mass_correction_v_hyb
max_grad_v_hyb
drift_cfl
residual_L2 / low_field_L2
mean_score_j
mean_q_diag_j
expected_HT_count_j
actual_HT_count_j
HT_effective_sample_size_j
positive_residual_mass_j
negative_residual_mass_j
```

For the deterministic full-residual solver, HT columns may be empty but the residual energy and score diagnostics should still be saved.

### 7.5 Safeguards

Use these safeguards in the first implementation:

1. Freeze windows and reconstruction during one Lie step.
2. Use smooth tapers; include \(r\nabla\chi\) in gradients.
3. Abort or reduce time step if drift CFL exceeds a set threshold.
4. Report minimum reconstructed field value.
5. If a field is used inside a reaction rate requiring nonnegative input, apply a small coefficient floor and record it. For the parabolic--parabolic KS drift, signed values of \(v\) are less dangerous than signed values in a multiplicative reaction coefficient, but they should still be reported.
6. If the residual correction changes total mass, either report the mass defect or apply a constant mass correction. For a drift field, a constant correction does not affect the gradient; for reaction rates it matters.

---

## 8. Numerical experiment matrix

### 8.1 LDG reference

Run or extract LDG data at multiple resolutions:

```text
LDG h-levels: at least 128, 256, 512 equivalent grid/cell resolutions if feasible
report times: 6e-5, 1.2e-4, 2.0e-4
save: S_L2, peak, mass, positivity, R_0.5, R_0.8, snapshots
compute: t_b(theta=1.05)
```

### 8.2 Particle solvers

Use the same fully parabolic--parabolic equation and the same initial data.

Run at least:

```text
A. fourier_global:
   field_recon = fourier
   K = current stable K, e.g. 8 or 10

B. hybrid_full:
   field_recon = hybrid_full
   Kg = 5 or 8
   Kl = 24, 32, or 40 for local spectrum
   local_type = spectrum first, blob optional

C. hybrid_HT, optional second stage:
   field_recon = hybrid_ht
   same Kg/Kl or blob h
   B_target = 1000, 3000, 8000 sensitivity
```

Particle levels:

```text
base:    N_u=N_v=2e4 or 4e4
refined: N_u=N_v=8e4 or 1.6e5
high:    optional if runtime permits
```

Time step:

```text
Use the current stable parabolic--parabolic time step first.
Then repeat one key run with tau/2 to check that the hybrid field is not creating a time-step artifact.
```

### 8.3 Comparison metrics

Compare LDG and particle solvers using:

```text
S_L2(t)
peak(t)
R_0.5(t), R_0.8(t)
resolution-gap proxy t_b(theta=1.05)
mass conservation / mass law
drift CFL and min reconstructed field
```

For particle solvers, report both:

```text
solver-field S_L2 and peak
common-diagnostic S_L2 and peak
```

The common diagnostic is needed to show that the trajectory improves, not merely that the solver reports a different field.

### 8.4 Success criteria

Strong result:

```text
Hybrid-full solver follows the LDG S_L2 and radius trends better than the global-spectrum solver at comparable particle number.
Hybrid-full gives a resolution-gap proxy closer to the LDG refinement trend.
Reconstruction-free radii remain stable and consistent with LDG.
```

Moderate result:

```text
Hybrid solver does not fully match LDG t_b, but improves core drift, radii, and reporting-time concentration over the global-spectrum solver.
This supports the claim that reconstruction, not the particle measure alone, was the bottleneck.
```

Negative but useful result:

```text
Hybrid field introduces instability or large signed residual artifacts.
Then §5.4 should not claim solver improvement; report it as a limitation and keep reconstruction-free radii as the reliable concentration diagnostic.
```

---

## 9. Figures and tables for the manuscript

### Figure: direct LDG versus particle solvers

Columns:

```text
LDG reference
particle global-spectrum solver
particle hybrid-full solver
optional particle hybrid-HT solver
```

Rows:

```text
t = 6e-5
t = 1.2e-4
t = 2.0e-4
```

Use the same physical zoom around the core. Annotate peak and \(R_{0.8}\).

### Figure: curves

Plot:

```text
S_L2(t) for LDG refinements and particle solvers
peak(t) for LDG refinements and particle solvers
R_0.5(t), R_0.8(t)
residual diagnostics: mean score, q_diag, residual energy fraction
```

### Table: LDG comparison

Suggested columns:

```text
method
resolution / particle count
field reconstruction used inside solver
S_L2(2e-4)
peak(2e-4)
R_0.8(2e-4)
gap time theta=1.05
mass error
notes
```

### Figure: residual refinement mechanism

For the hybrid solver at one reporting time, show:

```text
low global spectrum u_lo
local pilot or local spectrum
signed residual correction
positive-residual-only retained particles, clearly labeled as visualization only
```

This figure explains where the reconstruction resolution is spent.

---

## 10. Manuscript language target

Use language like this:

```latex
The reconstruction used inside the particle time step is no longer a single global Fourier projector.  We compute a low-order global spectrum for the smooth background and enrich it in particle-detected windows by adding a signed local residual correction.  In a window \(W_j\), the correction has the form
\[
\chi_j\{T_j(\mu|_{W_j^{\rm pad}})-T_j(u_{\rm lo}{\bf 1}_{W_j^{\rm pad}}dx)\},
\]
so that the local reconstruction does not double count the part of the measure already represented by the global spectrum.  For the parabolic--parabolic Keller--Segel benchmark, this hybrid reconstruction is used to evaluate \(\nabla v\) in the cell-particle transport step, not only to postprocess the final cloud.
```

For the LDG comparison:

```latex
We compare with the LDG Keller--Segel benchmark on the same fully parabolic--parabolic equation, initial data, and reporting times.  The finite-volume computation is used only as a reproducibility and sign-convention check; the main comparison is made against the LDG reference curves and reporting-time concentration diagnostics.  The reported gap times are numerical resolution-gap indicators and are not interpreted as continuum blow-up times.
```

---

## 11. What not to claim

Do not claim:

```text
We compute the continuum blow-up time.
The particle method avoids all adaptive reconstruction.
The hybrid residual particles are a positive residual measure.
The HT residual makes nonlinear coefficient evaluation unbiased.
FVM is an LDG replacement in the main benchmark.
```

Allowed claim if results support it:

```text
Using the particle cloud to enrich the field reconstruction inside the solver improves the LDG-aligned concentration benchmark relative to a single global spectrum, and gives a more appropriate particle analogue of local mesh refinement.
```

---

## 12. Claude/Codex workflow

Before coding:

1. Inspect the current Keller--Segel parabolic--parabolic solver.
2. Identify the exact code path where \(\nabla v\) is reconstructed and used in the \(u\)-particle transport.
3. Ask Codex for a cold review of the proposed insertion point and gradient formula.

During implementation:

1. Implement `hybrid_field_pp.py` with deterministic full residual first.
2. Add solver flags but keep the default behavior unchanged.
3. Add raw cloud snapshots at reporting times.
4. Add diagnostics for residual score, mass, minimum field, and drift CFL.
5. Run a tiny smoke test for 5--10 steps.
6. Ask Codex to verify that the hybrid field is used in the actual transport update, not only in diagnostics.

After implementation:

1. Run the LDG benchmark particle experiments.
2. Compare against direct LDG data, not FVM, in the main plots.
3. Only after deterministic hybrid is stable, add the HT residual solver variant.
4. Ask Codex to verify the manuscript claims against the actual plotted data.

