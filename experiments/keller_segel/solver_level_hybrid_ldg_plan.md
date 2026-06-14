# Solver-level hybrid residual reconstruction and direct LDG comparison plan

Status: planning note for the Keller--Segel redesign of Sections 5.4--5.5.

This note records a **hard change in direction**:

1. The hybrid reconstruction must be used **inside the solver** to compute coefficients, drift, reaction, and diagnostics. It is not only a final-time post-processing reconstruction.
2. The Keller--Segel benchmark in Section 5.4 must compare directly with the **LDG benchmark** of Li--Shu--Yang on the same fully parabolic--parabolic equation. A finite-volume run can remain as an appendix sanity check, but not as the main reference comparison.

---

## 1. Why this note exists

The old numerical story was too weak in two separate ways.

First, the Keller--Segel comparison was only **LDG-aligned**. The main deterministic reference in the repository is currently a finite-volume baseline. That is useful for debugging signs, mass conservation, positivity, and reporting-time concentration, but it does not satisfy the paper-level requirement: Section 5.4 should compare against the LDG benchmark itself, not a surrogate finite-volume method.

Second, the existing `experiments/resolution_hybrid/` direction treated the hybrid reconstruction mainly as a **diagnostic/post-processing** tool. That is not enough. In the particle method, reconstruction is part of the solver: it is used to evaluate the nonlinear coefficients. Therefore, if the reconstruction is the bottleneck near a concentrating Keller--Segel core, the reconstruction must be improved **during time stepping**, not only after the trajectory has already been generated.

The revised Section 5.4 should therefore test:

> On the fully parabolic--parabolic LDG benchmark, does a particle-field solver whose in-step field reconstruction is a low global spectrum plus particle-detected local signed residual correction better track the LDG concentration diagnostics than a pure global Fourier reconstruction?

---

## 2. Non-negotiable decisions

### 2.1 No FVM main comparison in Section 5.4

For the main Section 5.4 text:

```text
FVM is not an acceptable replacement for LDG.
```

Allowed use of FVM:

```text
Appendix / internal sanity check only:
    mass conservation,
    positivity,
    chemotaxis sign,
    approximate reporting-time concentration.
```

Not allowed:

```text
Do not write that the method is compared to LDG if the only deterministic reference is FVM.
Do not use an FVM resolution-gap time as the main LDG numerical blow-up proxy.
Do not hide boundary-condition or discretization differences.
```

### 2.2 Hybrid reconstruction must enter the solver

The hybrid reconstruction must replace the field reconstruction used to compute drift or reaction rates in the time step. It is not enough to reconstruct the final snapshots differently.

For Keller--Segel, this means the u-particle transport must use

\[
    dX_u = \nabla v_{\rm hyb}(X_u)\,dt + \sqrt{2}\,dW,
\]

not

\[
    dX_u = \nabla P_K\mu_v(X_u)\,dt + \sqrt{2}\,dW.
\]

The particle trajectory must change when `--field_recon hybrid_full` or `--field_recon hybrid_ht` is selected.

### 2.3 Fully parabolic--parabolic Keller--Segel only

The main benchmark must stay on

\[
    u_t-\nabla\cdot(\nabla u-u\nabla v)=0,
    \qquad
    v_t-\Delta v=u-v.
\]

Do not use parabolic--elliptic Keller--Segel as the main LDG comparison.

---

## 3. Solver-level hybrid residual reconstruction

Let the current empirical measure be

\[
    \mu_n = \sum_i \omega_i\delta_{X_i^n}.
\]

The solver should no longer be restricted to a single global Fourier projector. Define a reconstruction operator

\[
    \mathcal R_n\mu_n
    = u_{{\rm lo},n} + \sum_j u_{{\rm res},j,n}.
\]

### 3.1 Low-order global spectrum

Use a low-order global spectrum for the smooth background:

\[
    u_{{\rm lo},n}(x)=P_{K_g}\mu_n(x),
    \qquad K_g\ll K_{\rm full}.
\]

This should be cheap and stable. It is not expected to resolve the Keller--Segel core.

Suggested first values:

```text
Kg = 5 or 8
```

### 3.2 Particle-detected local windows

Detect windows from the particle cloud, not from a reconstructed image.

For the single-core LDG benchmark, start with the weighted center and quantile radii of the u-particles:

\[
    x_c^n=\frac{1}{M_n}\sum_i \omega_iX_i^n,
    \qquad
    M_n=\sum_i\omega_i,
\]

\[
    R_q^n=\inf\left\{r:
    \sum_{|X_i^n-x_c^n|\le r}\omega_i\ge qM_n
    \right\}.
\]

Define

\[
    W_n=B(x_c^n,\alpha R_{0.8}^n),
    \qquad
    W_n^{\rm pad}=B(x_c^n,\alpha_{\rm pad}R_{0.8}^n).
\]

Suggested first values:

```text
alpha      = 2.5 or 3.0
alpha_pad  = 4.0 or 4.5
```

For multi-core or island tests later, replace this by histogram connected components or kNN clustering. Do not do that first for the LDG single-core benchmark.

### 3.3 Local reconstruction operator

In each detected window, use a local reconstruction operator \(T_{j,n}\). Two acceptable first implementations:

**Option A: local blob / KDE**

\[
    T_{j,n}\nu = \eta_{h_j}*\nu.
\]

The bandwidth should be tied to the particle spacing in the detected window, for example

\[
    h_j = \eta_h\frac{R_{0.8}^n}{\sqrt{N_{0.8}^n}}.
\]

**Option B: local spectral window**

\[
    T_{j,n}\nu = P_{K_\ell}^{W_j}\nu,
    \qquad K_\ell>K_g.
\]

This is closer to the existing Fourier code. The local window is small, so a moderate \(K_\ell\) gives much higher physical resolution than a global spectrum with the same number of modes.

Suggested first values:

```text
local_type = spectrum first, blob second
Kl = 24, 32, or 40
```

### 3.4 Signed residual correction: no double counting

Do not add the local blob/spectrum directly to the global spectrum. That double-counts low-frequency mass.

The correct local correction is a signed residual:

\[
    u_{{\rm res},j,n}(x)
    =
    \chi_{j,n}(x)
    \left[
        T_{j,n}\bigl(\mu_n|_{W_{j,n}^{\rm pad}}\bigr)(x)
        -
        T_{j,n}\bigl(u_{{\rm lo},n}\mathbf 1_{W_{j,n}^{\rm pad}}dx\bigr)(x)
    \right].
\]

Thus

\[
    \boxed{
    \mathcal R_n\mu_n(x)
    =
    P_{K_g}\mu_n(x)
    +
    \sum_j\chi_{j,n}(x)
    \left[
        T_{j,n}\bigl(\mu_n|_{W_{j,n}^{\rm pad}}\bigr)(x)
        -
        T_{j,n}\bigl(P_{K_g}\mu_n\,\mathbf 1_{W_{j,n}^{\rm pad}}dx\bigr)(x)
    \right].
    }
\]

This is the central formula. It says:

```text
The global spectrum explains the background.
The local window only adds what the global spectrum did not explain.
The local correction is signed.
```

### 3.5 Gradient evaluation

For Keller--Segel, the solver needs gradients. If

\[
    u_{{\rm res},j}=\chi_j r_j,
\]

then

\[
    \nabla u_{{\rm res},j}
    =\chi_j\nabla r_j + r_j\nabla\chi_j.
\]

The taper-gradient term must not be dropped. Use a smooth raised-cosine taper:

```text
chi_j = 1 inside the core window,
chi_j decreases smoothly in the padded annulus,
chi_j = 0 at the padded boundary.
```

The field object must provide both

```python
field.eval(points)
field.grad(points)
field.diagnostics()
```

---

## 4. How the hybrid field enters the parabolic--parabolic KS solver

The LDG benchmark equation is

\[
    u_t-\nabla\cdot(\nabla u-u\nabla v)=0,
    \qquad
    v_t-\Delta v=u-v.
\]

The particle algorithm should use conservative u-particles and a v-particle cloud.

### 4.1 Transport step

At time step \(n\), build a hybrid reconstruction of the chemical field from the v-cloud, using u-particle-detected core windows:

\[
    v_{{\rm hyb},n} = \mathcal R_n\mu_{v,n}.
\]

Then transport the u-particles with

\[
    X_{u,i}^{*}
    = X_{u,i}^n
    +\tau\nabla v_{{\rm hyb},n}(X_{u,i}^n)
    +\sqrt{2\tau}\,\xi_{u,i}^n.
\]

Transport v-particles by diffusion:

\[
    X_{v,i}^{*}
    = X_{v,i}^n+\sqrt{2\tau}\,\xi_{v,i}^n.
\]

### 4.2 v decay / u-to-v injection step

Use the exact reaction update for

\[
    v_t=u-v.
\]

At the measure level:

\[
    \mu_v^{n+1}=e^{-\tau}\mu_v^*+(1-e^{-\tau})\mu_u^*.
\]

Particle implementation:

```text
existing v-particles survive with probability exp(-tau);
transported u-particles inject v-particles with mean (1-exp(-tau))*omega_u/omega_v;
use the minimum-variance integer kernel;
do not use quotient branching of the form (u-v)/v.
```

For the LDG benchmark, the u-equation is conservative. There is no u birth/death.

### 4.3 Refresh fields

At the end of the step, rebuild diagnostic fields from the new clouds. Solver fields and diagnostic fields must be clearly separated:

```text
solver field:
    the reconstruction actually used inside the time step.

common diagnostic field:
    the same high-quality diagnostic reconstruction applied to all solver trajectories,
    used only for fair comparison of S(t), peak, and plots.

reconstruction-free diagnostics:
    radii and particle counts computed directly from particles.
```

---

## 5. Horvitz--Thompson residual sketch

The deterministic full residual should be implemented first. The HT residual is a cost-reduced solver variant and should be tested second.

### 5.1 Residual score

For a cheap pilot reconstruction \(u_{{\rm pilot},j}\), define for particles in \(W_j^{\rm pad}\):

\[
    s_i^{(j)}
    =
    \chi_j(X_i)
    \frac{|u_{{\rm pilot},j}(X_i)-u_{\rm lo}(X_i)|}
    {u_{{\rm pilot},j}(X_i)+|u_{\rm lo}(X_i)|+\varepsilon}.
\]

Interpretation:

```text
s_i is large where the local density near the particle is not explained by the low global spectrum.
```

### 5.2 HT accept probabilities

For a target retained count \(B_j\), define

\[
    q_i^{(j)}=\min\{1,\max(q_{\min},\lambda_j s_i^{(j)})\},
\]

where \(\lambda_j\) is chosen by bisection so that

\[
    \sum_i q_i^{(j)}\approx B_j.
\]

Avoid the normalized formula as the only diagnostic. If \(q_i\) is forced to sum to \(B_j\), then the mean accept rate is mostly \(B_j/N(W_j^{\rm pad})\). For unresolved-structure diagnostics, also save

\[
    \bar s_j=\frac1{N(W_j^{\rm pad})}\sum_i s_i^{(j)},
\]

and a separate fixed-scale diagnostic accept rate

\[
    q_{i,{\rm diag}}^{(j)}=\min\{1,\lambda_{\rm diag}s_i^{(j)}\},
    \qquad
    \bar q_{j,{\rm diag}}=\frac1{N(W_j^{\rm pad})}\sum_i q_{i,{\rm diag}}^{(j)}.
\]

Then the diagnostic interpretation is valid:

```text
low diagnostic accept rate     : global spectrum already explains the window;
high diagnostic accept rate    : core/island has unresolved residual structure;
growing diagnostic accept rate : reconstruction resolution should be concentrated here.
```

### 5.3 HT residual field

The signed HT residual field is

\[
    \widehat u_{{\rm res},j}^{\rm HT}(x)
    =
    \chi_j(x)
    \left[
    \sum_{i:X_i\in W_j^{\rm pad}}
    \frac{A_i^{(j)}}{q_i^{(j)}}\omega_i\eta_{h_j}(x-X_i)
    -
    \int_{W_j^{\rm pad}}\eta_{h_j}(x-y)u_{\rm lo}(y)dy
    \right].
\]

The factor must be \(A_i/q_i\), not \(q_iA_i\). Conditionally on the current particle cloud,

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

### 5.4 HT inside the solver: caution

If the HT field is used inside the solver, the coefficient becomes random. In general,

\[
    \mathbb E_A b(\mathcal R_{\rm HT}\mu)
    \ne
    b(\mathbb E_A\mathcal R_{\rm HT}\mu).
\]

Therefore:

```text
Main solver variant first: hybrid_full deterministic residual.
Second variant: hybrid_ht cost-reduced residual.
For hybrid_ht, use q_min, drift-CFL guards, and possibly refresh the residual sketch every m steps rather than every step.
```

### 5.5 Positive-excess particles are visualization only

The positive-excess version

\[
    u_{\rm lo}^+=\max\{u_{\rm lo},0\},
    \qquad
    r_j^+=(u_j-u_{\rm lo}^+)_+,
\]

is useful for visualizing particles not explained by the global spectrum. It is not an unbiased signed residual estimator and should not be used for the quantitative LDG comparison unless explicitly labelled:

```text
positive-residual-only diagnostic, not a signed residual estimator.
```

---

## 6. Direct LDG comparison: benchmark contract

### 6.1 Equation and data

Use the fully parabolic--parabolic benchmark:

\[
    u_t-\nabla\cdot(\nabla u-u\nabla v)=0,
    \qquad
    v_t-\Delta v=u-v.
\]

Domain and boundary condition for LDG:

```text
Omega = [-0.5, 0.5]^2
homogeneous Neumann boundary conditions
```

Initial data:

\[
    u_0(x,y)=840\exp[-84(x^2+y^2)],
    \qquad
    v_0(x,y)=420\exp[-42(x^2+y^2)].
\]

The cell mass is

\[
    M_u=10\pi>8\pi.
\]

Reporting times:

\[
    t=6\times10^{-5},\quad 1.2\times10^{-4},\quad 2.0\times10^{-4}.
\]

### 6.2 What counts as direct LDG reference

Acceptable:

```text
1. Implement the Li--Shu--Yang LDG benchmark directly and save LDG curves/snapshots; or
2. Use published/digitized LDG reporting-time curves and blow-up-window data, explicitly labelled as published LDG reference.
```

Not acceptable for main text:

```text
FVM as the only grid reference.
FD as the only grid reference.
A parabolic--elliptic reference.
A different initial condition.
A different reporting-time set without explanation.
```

### 6.3 LDG data products

Create a directory:

```text
experiments/keller_segel/ldg_reference/
    README.md
    ldg_solver.py or scripts_to_import_published_ldg_data.py
    run_ldg_sweep.py
    plot_ldg_particle_compare.py
```

Archive results under:

```text
reference_results/keller_segel_ldg_pp/ldg_<run_id>/
    config_used.json
    S_curves.csv
    tb_ldg.csv
    snapshots.npz
    figures/*.pdf
    figures/*.png
    README.md
```

Minimum `S_curves.csv` columns:

```text
t
resolution_id
h_or_n
poly_degree
S_L2
S_core
peak
mass_u
mass_v
umin
R_0.5
R_0.8
xc_x
xc_y
```

Minimum snapshot contents:

```text
u at each reporting time
v at each reporting time
grid coordinates
resolution metadata
```

### 6.4 LDG-style numerical blow-up proxy

For LDG refinements \(h\to h/2\), define

\[
    S_h(t)=\|u_h(t)\|_{L^2},
\]

\[
    t_b^{\rm LDG}(h;\theta)
    =
    \inf\{t:S_{h/2}(t)\ge \theta S_h(t)\},
    \qquad \theta=1.05.
\]

This is a numerical resolution-gap indicator, not a continuum blow-up time.

### 6.5 Particle comparison to LDG

Particle solvers to run:

```text
A. fourier_global_current:
   current global Fourier reconstruction used inside the solver.

B. global_low_only:
   P_Kg reconstruction inside the solver, no local residual.
   This is expected to under-resolve the core and is a control.

C. hybrid_full:
   P_Kg + deterministic local signed residual inside the solver.
   This is the main proposed solver-level reconstruction.

D. hybrid_ht:
   P_Kg + HT-sketched signed residual inside the solver.
   This is the cost-reduced variant, not the first success criterion.
```

Suggested initial sweep:

```text
N_u=N_v=2e4, 8e4, optionally 3.2e5
tau=1e-6 or smaller after drift-CFL check
T=2e-4
report_times=6e-5,1.2e-4,2e-4
Kg=5 or 8
Kl=24,32,40
local_type=spectrum first; blob second
window_alpha=2.5 or 3.0
window_pad=4.0 or 4.5
```

The particle adaptive-window/periodic implementation is not the same boundary condition as the LDG Neumann reference. This must be disclosed. Because the Gaussian is extremely small at \(|x|=0.5\), the short-time core dynamics should still be comparable, but this is an LDG benchmark comparison, not a theorem that the boundary problems are identical.

---

## 7. Diagnostics and plots

### 7.1 Reconstruction-free diagnostics

Always report:

```text
R_0.5(t)
R_0.8(t)
core particle count
mass_u(t)
mass_v(t)
center x_c(t)
```

These come directly from particles or LDG cell masses, not from the reconstruction.

### 7.2 Solver-field diagnostics

For each solver, report diagnostics using the field that the solver actually used:

```text
S_solver(t) = ||u_solver_field(t)||_L2
peak_solver(t)
min field value
mass of reconstructed field
drift_CFL
```

### 7.3 Common diagnostic reconstruction

For fair comparison of particle trajectories, also apply the same diagnostic reconstruction to every saved trajectory:

```text
common high-quality diagnostic field:
    same grid,
    same local windows or same high-K diagnostic rule,
    not used to advance particles.
```

Use this to compare:

```text
S_common(t)
peak_common(t)
LDG reporting-time profiles
core profile cuts
```

Without this, a hybrid solver can look better merely because its own diagnostic field has more local modes.

### 7.4 Residual diagnostics

For every hybrid window save:

```text
window center
R_0.5, R_0.8
window radius and padded radius
n_window
mean residual score sbar_j
fixed-scale diagnostic accept rate qbar_diag_j
expected retained count sum q_i
actual retained count sum A_i
HT effective sample size
residual L2 / low-field L2
positive residual mass
negative residual mass
residual mass imbalance
field minimum after signed correction
```

### 7.5 Required figures

Minimum figures for Section 5.4:

```text
Fig. 5.4a:
    LDG snapshots at the reporting times.

Fig. 5.4b:
    Particle solver snapshots for fourier_global_current and hybrid_full.

Fig. 5.4c:
    S_L2(t) curves:
        LDG refinements,
        particle global-Fourier solver,
        particle hybrid-full solver.

Fig. 5.4d:
    R_0.5(t), R_0.8(t) curves from LDG cell masses and particles.

Fig. 5.4e:
    residual score / diagnostic accept-rate trace showing where the solver adds local reconstruction resolution.
```

Optional figure:

```text
Positive-excess residual particles overlaid on the core.
Label clearly: positive-residual-only diagnostic, not signed estimator.
```

---

## 8. Implementation tasks

### 8.1 New solver module

Add:

```text
experiments/keller_segel/ldg_comparison/hybrid_field_pp.py
```

Required API:

```python
class HybridField:
    def eval(self, points):
        ...

    def grad(self, points):
        ...

    def diagnostics(self):
        ...


def build_hybrid_field_from_cloud(
    X,
    weights,
    mass,
    global_box,
    Kg,
    windows,
    local_type="spectrum",      # "spectrum" or "blob"
    Kl=None,
    h=None,
    residual_mode="full",       # "full" or "ht"
    B_target=None,
    q_min=0.0,
    rng=None,
):
    ...
```

The implementation must include both value and gradient. For local spectrum, use analytic gradients. For blob, use differentiable Gaussian convolution or grid gradient with a recorded discretization.

### 8.2 Modify the KS solver

Modify:

```text
experiments/keller_segel/ldg_comparison/simulation.py
```

Add flags:

```text
--field_recon fourier_global|global_low|hybrid_full|hybrid_ht
--Kg
--Kl
--local_type spectrum|blob
--window_alpha
--window_pad
--B_target
--q_min
--hybrid_refresh_every
--save_cloud_snapshots
```

In the transport loop, replace the direct global Fourier gradient path by:

```python
if args.field_recon == "fourier_global":
    gradv = grad_v_from_cloud(...)
elif args.field_recon in {"global_low", "hybrid_full", "hybrid_ht"}:
    windows = detect_windows_from_u_cloud(...)
    hyb_v = build_hybrid_field_from_cloud(X2, ..., windows=windows, ...)
    gradv = hyb_v.grad(X1)
```

The key requirement:

```text
hybrid field must be used before X1 is advanced.
```

### 8.3 Save raw cloud snapshots

At every reporting time save:

```text
X_u
X_v
weights_u or all-ones marker
weights_v or all-ones marker
mass_per_particle_u
mass_per_particle_v
box/window metadata
R_0.5, R_0.8
N_0.5, N_0.8
solver field configuration
hybrid diagnostics if used
```

Existing reconstructed-field-only snapshots are insufficient for residual-particle analysis.

### 8.4 Plotting and analysis scripts

Add or update:

```text
experiments/keller_segel/ldg_comparison/run_hybrid_solver_sweep.py
experiments/keller_segel/ldg_comparison/analyze_hybrid_vs_ldg.py
experiments/keller_segel/ldg_comparison/plot_hybrid_ldg_compare.py
```

The analysis must be idempotent: it reads saved CSV/NPZ and does not rerun the solver.

---

## 9. Acceptance criteria

### Strong success

The revised Section 5.4 can claim:

```text
Direct LDG reference reproduces the LDG concentration benchmark.
The particle solver with solver-level hybrid residual reconstruction tracks the LDG reporting-time concentration better than the global-Fourier particle solver at comparable N and tau.
Reconstruction-free core radii agree across LDG and particle trajectories in the pre-singular window.
The residual-score / diagnostic-accept-rate trace shows that local reconstruction resolution is automatically concentrated in the particle-detected core.
No continuum blow-up time is claimed.
```

### Limited success

Acceptable but weaker:

```text
The hybrid solver improves the field used inside the time step and reduces bandwidth sensitivity, but the LDG gap time remains resolution-sensitive.
The paper reports LDG-aligned pre-singular concentration, core radius collapse, and solver-level reconstruction diagnostics, without quoting a continuum blow-up time.
```

### Failure condition

If direct LDG reproduction fails or the hybrid solver is unstable:

```text
Do not write a blow-up-time or LDG-match story.
Move this to an implementation/diagnostic appendix.
Keep the main paper focused on branching vs weighted-particle evidence and report Keller--Segel limitations honestly.
```

---

## 10. Manuscript language target

A possible paragraph for Section 5.4:

```latex
The reconstruction is part of the particle solver, not only a visualization step.  In the concentrating Keller--Segel benchmark, a single global Fourier bandwidth either under-resolves the core or injects high-mode particle noise into the drift.  We therefore replace the in-step field reconstruction by a hybrid operator: a low-order global spectrum represents the smooth background, while particle-detected core windows add only signed local residual corrections.  In a window \(W_j\), the correction is
\[
\chi_j\{T_j(\mu|_{W_j^{\rm pad}})-T_j(P_{K_g}\mu\,\mathbf 1_{W_j^{\rm pad}}dx)\},
\]
so the local reconstruction does not double-count the mass already represented by the global spectrum.  This hybrid field is used to compute the chemotactic drift during the time step.
```

A possible paragraph for the LDG comparison:

```latex
We compare with the fully parabolic--parabolic LDG benchmark of Li--Shu--Yang using the same initial data and reporting times.  The LDG computation provides the grid-based concentration and resolution-gap reference.  The finite-volume calculation is retained only as a sanity check for mass conservation and sign conventions.  The comparison below therefore tests whether the particle solver, with reconstruction resolution concentrated by the particle cloud itself, reproduces the LDG pre-singular concentration diagnostics without imposing a global mesh refinement.
```

---

## 11. Verification protocol

Before any production run:

```text
1. Run a small smoke test with field_recon=fourier_global and confirm it reproduces the old trajectory.
2. Run field_recon=global_low and confirm it changes only the field bandwidth, not the injection logic.
3. Run field_recon=hybrid_full for a short time and verify:
       finite gradients,
       reasonable drift_CFL,
       mass conservation for u,
       v mass balance,
       no missing taper-gradient term.
4. Run hybrid_ht only after hybrid_full is stable.
5. Independently verify the formulas and code path before launching large jobs.
```

Do not start expensive LDG or particle sweeps until these checks pass.
