# Fully parabolic three-dimensional Keller--Segel experiment plan

## Status and decision

This note defines the replacement for the current three-dimensional
`keller_segel/focusing_3d/` experiment.

The old experiment solves a **parabolic--elliptic** screened-Poisson model. It may remain
in the repository as a historical/diagnostic record, but it must not be used as the
three-dimensional result in the revised paper.

The new three-dimensional experiment must solve the **fully parabolic--parabolic** system

\[
\partial_t u
= D_u\Delta u-\chi\nabla\!\cdot(u\nabla v),
\qquad
\partial_t v
= D_v\Delta v+\alpha u-\beta v,
\qquad x\in\mathbb T_L^3,
\]

with the default coefficients

\[
D_u=D_v=\alpha=\beta=\chi=1.
\]

The scientific purpose is not to estimate a universal critical mass or a continuum
blow-up time. The purpose is to test the complete three-dimensional coupled particle
algorithm:

1. conservative particle transport for `u`;
2. three-dimensional chemical diffusion for `v`;
3. cross-species creation of `v` particles from `u` particles;
4. decay of existing `v` particles;
5. reconstruction of the three-dimensional chemical drift `grad v`;
6. delayed chemotactic aggregation and, if supported, nonradial cluster interaction.

The main paper must continue to use the localized-growth experiments for the claim that
branching outperforms weighted particles. This 3D example has a different role: it checks
that the coupled non-conservative particle mechanism works in three dimensions.

---

## 1. Hard constraints

### 1.1 Do not use the old parabolic--elliptic model

The new solver must not use

```text
-Delta v + kappa^2 v = u - u_bar
```

at any point in the dynamics. In particular:

- there is no `kappa` parameter;
- there is no instantaneous screened-Poisson solve;
- do not reuse the old mass-sweep conclusion `M=40--60 transition`;
- do not reuse the old tetrahedral result as evidence of cluster merging;
- do not quote the old parabolic--elliptic peak, core radius, or focusing threshold in
  the revised manuscript.

Low-level real Fourier basis/evaluation routines may be ported from the old code after
verification, but the screened solve itself must not be used.

### 1.2 Use the cross-species injection kernel

The source `alpha u` in the `v` equation is additive. It must not be rewritten as the
multiplicative quotient

\[
r_v=(\alpha u-\beta v)/v.
\]

That quotient is undefined where `v=0` and cannot create chemical particles in a region
where no `v` particle is present. The correct split reaction update is an injection and
decay step.

### 1.3 No use of `H` in the new code or paper

The old symbol `H` was a Fourier bandwidth. It is easy to confuse with a Sobolev space
and is inconsistent with the rest of the paper. Use the following notation:

```text
K_dyn   maximum Fourier mode used inside the time step to construct grad v
K_out   optional Fourier/KDE resolution used only for output figures
K_test  low-mode set used in quantitative verification
```

Define `K_dyn` mathematically by

\[
P_{K_{\rm dyn}}\mu_v(x)
=
\sum_{|k_j|\le K_{\rm dyn}}
\widehat\mu_v(k)e^{2\pi i k\cdot x/L}.
\]

If the implementation uses real sine/cosine arrays, the nonnegative array index runs
from `0` to `K_dyn`; it does not change this definition. Record both `K_dyn` and the
actual number of stored modes in every config file.

### 1.4 No silent population control or clipping

The production result must use the raw injection/decay process unless population control
is explicitly introduced as a separate, labeled diagnostic. Do not silently:

- cap `N_v`;
- renormalize the `v` mass;
- add a positivity floor to a quotient rate;
- clip the chemical drift;
- replace a failed run by a tuned parameter set without recording the failure.

### 1.5 No direct comparison with post-arXiv 3D SIPF work

This experiment is a self-contained verification/stress test of the present method. Do
not position it as a head-to-head comparison with later 3D SIPF papers and do not claim
priority or superiority from this test.

### 1.6 Execution protocol

Follow the repository `CLAUDE.md` protocol:

1. Before every code-executing command, ask Codex to cold-check the command, parameters,
   expected cost, and output path.
2. Run a smoke/unit test before any production job.
3. After each result is produced, ask Codex to inspect the code path and result files
   before reporting a conclusion.
4. Do not edit the manuscript until the numerical gates in this note pass.

---

## 2. Particle representation and time step

### 2.1 Common base mass

Use a fixed base particle mass `omega` and represent

\[
\mu_u^n=\omega\sum_{i=1}^{N_u}\delta_{X_i^n},
\qquad
\mu_v^n=\omega\sum_{j=1}^{N_v^n}\delta_{Y_j^n}.
\]

For the default setup,

\[
\omega=\frac{M_u(0)}{N_u},
\]

and `N_u` is fixed. The primary experiment uses `v_0=0`, so `N_v^0=0` is valid and
must be supported by the implementation.

The same per-particle mass for both species makes the population interpretation
transparent. If an auxiliary run uses a different chemical particle mass, the ratio
`omega_u/omega_v` must be stated explicitly and treated as a cost/resolution parameter,
not hidden as an implementation detail.

### 2.2 Transport substep

Given `mu_u^n` and `mu_v^n`, reconstruct only the field needed by the dynamics,

\[
v_K^n=P_{K_{\rm dyn}}\mu_v^n.
\]

Advance the cell particles by

\[
X_i^*
=
\operatorname{wrap}\left(
X_i^n+\chi\nabla v_K^n(X_i^n)\tau
+\sqrt{2D_u\tau}\,\xi_i^n
\right),
\]

and the existing chemical particles by

\[
Y_j^*
=
\operatorname{wrap}\left(
Y_j^n+\sqrt{2D_v\tau}\,\eta_j^n
\right).
\]

Here `xi` and `eta` are independent standard three-dimensional Gaussian vectors.

No reconstruction of `u` is required for the source step: the transported `u` particles
themselves are the source locations.

### 2.3 Exact reaction/injection substep

During the reaction substep, freeze `u=u*` and solve

\[
\partial_t v=\alpha u^*-\beta v.
\]

The exact update is

\[
\mu_v^{n+1}
=
e^{-\beta\tau}\mu_v^*
+
\frac{\alpha}{\beta}(1-e^{-\beta\tau})\mu_u^*.
\]

Implement it particlewise:

- **Decay:** each transported `v` particle survives with probability
  `exp(-beta*tau)`.
- **Injection:** each transported `u` particle creates an integer number of new `v`
  particles at `X_i^*` with mean

  \[
  q_{\rm inj}
  =
  \frac{\alpha}{\beta}(1-e^{-\beta\tau})
  \frac{\omega_u}{\omega_v}.
  \]

For the default `alpha=beta=1` and `omega_u=omega_v`,

\[
q_{\rm inj}=1-e^{-\tau}<1,
\]

so the minimum-variance integer kernel is a Bernoulli draw. The Poisson injection kernel
may be retained as a variance comparator, but the default production method is the
minimum-variance kernel.

The conditional mean must satisfy

\[
\mathbb E_n\mu_v^{n+1}
=
e^{-\beta\tau}\mu_v^*
+
\frac{\alpha}{\beta}(1-e^{-\beta\tau})\mu_u^*.
\]

### 2.4 Exact mass and expected-population laws

The continuum mass laws are

\[
M_u(t)=M_u(0),
\]

\[
M_v(t)
=e^{-\beta t}M_v(0)
+
\frac{\alpha}{\beta}(1-e^{-\beta t})M_u(0).
\]

For `alpha=beta=1` and `v_0=0`,

\[
M_v(t)=M_u(0)(1-e^{-t}).
\]

With equal particle masses,

\[
\mathbb E N_v(t)=N_u(1-e^{-t}),
\]

so the expected chemical population is bounded by `N_u`. This identity is one of the
primary quantitative checks.

Do not use relative error divided by `M_v(t)` near `t=0`, where the exact chemical mass
vanishes. Use

\[
E_{M_v}(t)=\frac{|M_v^N(t)-M_v(t)|}{M_u(0)}.
\]

---

## 3. Experiment A: exact three-dimensional mode verification

This is the mandatory implementation gate. It provides an exact reference without a
three-dimensional grid solver.

### 3.1 Setup

Use the periodic box `[-L/2,L/2]^3`, with default `L=12`, and turn off chemotactic
feedback only for this verification:

\[
\chi=0.
\]

Choose a positive low-mode initial density

\[
u_0(x)
=
\bar u\left[
1+\varepsilon\left(
\cos\frac{2\pi x_1}{L}
+\cos\frac{2\pi x_2}{L}
+\cos\frac{2\pi x_3}{L}
\right)
\right],
\]

with `epsilon=0.1`, and set

\[
v_0=0.
\]

The density stays nonnegative because `1-3 epsilon>0`.

### 3.2 Exact solution in Fourier space

For a Fourier mode `k`, let

\[
\lambda_k=\left(\frac{2\pi}{L}\right)^2|k|^2.
\]

Then

\[
\widehat u_k(t)
=e^{-D_u\lambda_k t}\widehat u_k(0),
\]

and, for `v_0=0`,

\[
\widehat v_k(t)
=
\alpha\widehat u_k(0)
\frac{
 e^{-D_u\lambda_k t}
 -e^{-(D_v\lambda_k+\beta)t}
}{D_v\lambda_k+\beta-D_u\lambda_k}.
\]

With the default coefficients this simplifies to

\[
\widehat v_k(t)
=
\widehat u_k(0)e^{-\lambda_k t}(1-e^{-t}).
\]

The zero mode gives the exact mass law.

### 3.3 Required diagnostics

For the modes

```text
k = (0,0,0), (1,0,0), (0,1,0), (0,0,1)
```

save the real and imaginary parts of `u_hat` and `v_hat`, together with the exact values.
Also report

\[
E_{\rm mode}^2(t)
=
\frac{1}{M_u(0)^2}
\sum_{k\in K_{\rm test}}
|\widehat v_k^N(t)-\widehat v_k(t)|^2,
\]

and the low-mode gradient error

\[
E_{\nabla v}^2(t)
=
\frac{1}{M_u(0)^2}
\sum_{0<|k|\le K_{\rm test}}
\lambda_k
|\widehat v_k^N(t)-\widehat v_k(t)|^2.
\]

Use absolute/`M_u`-normalized errors at early times; do not divide by a nearly zero
chemical mode.

### 3.4 Comparison to run

Compare minimum-variance and Poisson injection with shared initial particles and shared
transport randomness whenever possible.

Use a staged refinement:

```text
smoke:
    N_u = 2e4, K_dyn = 4, tau = 2e-3, T = 0.02, seed = 0

verification grid:
    N_u in {2e4, 8e4, 3.2e5}
    tau in {2e-3, 1e-3, 5e-4}
    K_dyn >= 2 so all tested modes are retained
    T = 1
    seeds = 0,...,7 for the N-sweep
```

Do not run the full tensor product of all parameters. Isolate one error source at a time:

- `N` sweep at fixed small `tau`;
- `tau` sweep at the largest affordable `N`;
- one `K_dyn` sanity check because the exact solution contains only low modes.

### 3.5 Acceptance gates

The implementation passes Experiment A only if:

1. `u` mass is conserved to machine precision.
2. The mean `v` mass follows the exact law, and `E_Mv` decreases with `N`.
3. The RMS low-mode error is consistent with `N^{-1/2}` at fixed `tau` before the time
   error dominates.
4. The weak/mode error decreases approximately first order under `tau -> tau/2`.
5. Minimum-variance injection has no larger seed variance than Poisson injection for
   mass and low-mode errors.
6. The solver works with `N_v(0)=0`; no dummy `v` particle or positivity floor is used.
7. The three first modes remain statistically symmetric.

If these gates fail, stop and fix the implementation before any nonlinear aggregation
run.

---

## 4. Experiment B: radial delayed chemotactic focusing

This is the main candidate for the paper.

### 4.1 Initial data

Use

\[
u_0^M(x)
=
M(2\pi\sigma^2)^{-3/2}
\exp\left(-\frac{|x|^2}{2\sigma^2}\right),
\qquad
\sigma=0.45,
\]

sampled on the periodic box `[-6,6]^3`, and set

\[
v_0=0.
\]

The primary point of `v_0=0` is that the chemical field must be generated entirely by
the cross-species injection mechanism. Initially `grad v=0`, so the cells first diffuse.
The chemical mass and spatial gradients then build up, after which chemotactic
contraction may begin. This finite response delay is the fully parabolic phenomenon that
the old instantaneous-field experiment could not test.

### 4.2 Pilot family

Use one predetermined mass family only:

```text
M in {20, 40, 80, 160}
chi = 1
L = 12
sigma = 0.45
D_u = D_v = alpha = beta = 1
v0 = 0
```

Pilot settings:

```text
N_u = 2e4
K_dyn = 8
tau = 1e-3
T = 1.5
seed = 0
```

The pilot is used only to identify:

- a clearly diffusion-dominated case;
- at most one case with delayed, persistent core contraction;
- whether the startup `v`-particle noise is acceptable with equal particle masses.

Do not call the change across `M` a critical mass. It is only a family- and
regularization-dependent numerical behavior.

If none of the four predefined masses shows delayed contraction, one final extension
`M=320` is allowed. Do not continue tuning `M`, `chi`, `sigma`, and `L` simultaneously.
If the extension also fails or becomes immediately under-resolved, retain only the exact
verification result and report the radial test as negative/limited.

### 4.3 Primary dynamics diagnostics

Compute directly from the raw `u` particles:

\[
R_q^u(t),
\qquad q\in\{0.2,0.5,0.8\},
\]

using torus-aware distances about the torus centroid. These radii are reconstruction-free
as readouts, although their dynamics still depend on `K_dyn` through `grad v`.

Measure the chemical drift actually seen by the cells:

\[
G_v(t)
=
\left[
\frac1{N_u}
\sum_{i=1}^{N_u}
|\nabla v_K(X_i(t))|^2
\right]^{1/2},
\]

and also save

```text
max_i |grad v_K(X_i)|
mean_i grad v_K(X_i)
drift displacement tau*chi*max|grad v_K|
```

Define the effective Fourier length

\[
h_K=\frac{L}{2K_{\rm dyn}+1}
\]

and log

\[
C_{\rm drift}(t)
=
\frac{\tau\chi\max_i|\nabla v_K(X_i)|}{h_K},
\qquad
Q_q(t)=\frac{R_q^u(t)}{h_K}.
\]

The simulation must not be interpreted beyond the time at which the inner core is at the
Fourier floor. Use `Q_0.2 >= 3` as the default resolved-window requirement; show the
sensitivity of this cutoff rather than hiding it.

Also save:

- exact and numerical `M_v(t)`;
- `N_v(t)`, births, and deaths per step;
- low Fourier coefficients of `u` and `v`;
- `u` covariance eigenvalues;
- fixed-resolution orthogonal slices for visualization only.

### 4.4 Delayed-focusing diagnostic

Do not define a blow-up time. Instead, test whether the radius exhibits a robust
turnaround:

1. an initial interval in which `R_0.5` increases or remains nearly flat while `G_v`
   grows from zero;
2. a later interval in which `R_0.5` decreases persistently;
3. the contraction begins after the chemical gradient has built up.

Define, for analysis only,

\[
t_{\rm turn}
=
\operatorname*{arg\,max}_{t\in[0,T_{\rm resolved}]}
R_{0.5}^u(t),
\]

provided that after this maximum the radius decreases by at least 10% and remains below
`0.95 R_0.5(t_turn)` for at least five consecutive output times. If this persistence gate
does not hold, report no resolved delayed focusing.

Report

```text
t_turn
R_0.5(t_turn) / R_0.5(0)
R_0.5(T_resolved) / R_0.5(t_turn)
G_v(t_turn)
M_v(t_turn) / M_u
```

with seed variability. These are numerical response diagnostics, not continuum singular
times.

### 4.5 Optional initial-chemical control

After selecting one radial case, run one labeled control with the same `u_0` but an
independent `v` cloud sampled from `v_0=u_0`. This remains a fully parabolic simulation;
no elliptic solve is used.

The control asks whether a pre-existing aligned chemical gradient removes the initial
response delay. Compare `G_v(t)` and `R_q^u(t)` with the primary `v_0=0` run. Do not
mix the two initial conditions in a convergence table.

### 4.6 Production refinement

Only after a pilot case passes the delayed-focusing gate, run one-at-a-time refinements.
Suggested base configuration:

```text
N_u = 1e5
K_dyn = 12
tau = 5e-4
T = 1.5
seeds = 0,1,2,3
```

One-factor checks:

```text
particle number:
    N_u in {1e5, 4e5}, fixed K_dyn=12, tau=5e-4

field bandwidth:
    K_dyn in {8, 12, 16}, fixed N_u=1e5, tau=5e-4

time step:
    tau in {5e-4, 2.5e-4}, fixed N_u=1e5, K_dyn=12
```

A reduced grid is acceptable after a measured wall-time estimate. Do not launch all
large combinations. At least the base run must use four seeds; expensive refinement
runs may use seed `0` first and be expanded only if they change the conclusion.

### 4.7 Acceptance gates for the paper

The radial result is paper-ready only if:

1. the mass and expected-population laws remain correct in the coupled run;
2. the delayed turnaround is present for most seeds;
3. `R_0.2`, `R_0.5`, and `G_v` agree under `N` refinement in a nontrivial resolved time
   interval;
4. halving `tau` does not materially change `t_turn` or the radius curve in that interval;
5. increasing `K_dyn` extends the resolved interval rather than simply producing an
   unrelated trajectory from early times;
6. all quoted conclusions stop before `R_0.2/h_K < 3` or a documented drift-stability
   guard is reached.

A bandwidth-dependent final peak or final radius is not a failure if the pre-floor
response dynamics agree. It does mean that the paper must not claim a converged singular
core.

---

## 5. Experiment C: tetrahedral nonradial aggregation (optional)

Do this only after Experiment B has produced a stable radial case. It is optional and
must not delay the main revision.

### 5.1 Initial data

Place four Gaussian `u` clusters at the tetrahedral vertices

\[
(a,a,a),\quad(a,-a,-a),\quad(-a,a,-a),\quad(-a,-a,a),
\]

with `a=1`, width `sigma_c=0.25`, and `v_0=0`.

Choose the mass per cluster from the radial pilot:

```text
M_cluster = the smallest radial mass that shows resolved delayed contraction
M_total   = 4*M_cluster
```

This choice avoids the failure mode of the old test, where each cluster was locally
diffusive and only the centroids moved inward.

### 5.2 Diagnostics

Track:

- four torus-aware cluster centroids;
- all six pairwise centroid distances and their coefficient of variation;
- minimum inter-cluster distance;
- per-cluster `R_0.5` and `R_0.8` while labels remain meaningful;
- central mass `mu_u(B(0,r_c))` for fixed `r_c` values;
- global covariance eigenvalues and anisotropy;
- chemical mass and `N_v(t)`;
- orthogonal `u` and `v` slices.

### 5.3 Gate for the word “merging”

Do not call the result cluster merging based only on decreasing centroid distance.
Use that term only if all of the following occur before loss of resolution:

1. the minimum centroid distance decreases substantially;
2. the individual clusters remain localized long enough to approach each other, rather
   than simply diffusing across the domain;
3. the mass in a fixed central ball increases;
4. the density slices show overlap into a common central aggregate;
5. tetrahedral symmetry is preserved within seed fluctuations.

If the centroids attract but each cluster radius grows comparably to the separation,
describe the result only as **mutual attraction of dispersing clouds** and keep it out of
the main paper.

---

## 6. Variance comparison: minimum-variance versus Poisson injection

The valid comparator for the additive source is another injection kernel, not a
multiplicative weighted update based on `(u-v)/v`.

For injection mean `q`, use

\[
K_{\rm minvar}
=
\lfloor q\rfloor+
\operatorname{Bernoulli}(q-\lfloor q\rfloor),
\]

with variance

\[
\operatorname{Var}(K_{\rm minvar})
=
\{q\}(1-\{q\}),
\]

and compare with

\[
K_{\rm Pois}\sim\operatorname{Poisson}(q),
\qquad
\operatorname{Var}(K_{\rm Pois})=q.
\]

The primary comparison belongs in Experiment A. For the nonlinear radial run, repeat the
base configuration with Poisson injection only if the extra cost is modest. Compare:

- seed variance of `M_v(t)`;
- seed variance of low `v` modes;
- seed variance of `G_v(t)` and `R_0.5(t)`;
- total particle-step work.

Do not turn this into a new branching-versus-weighted headline. It is a kernel-variance
check inside the coupled extension.

---

## 7. Early-time chemical-particle noise

With `v_0=0` and equal particle masses,

\[
\mathbb E N_v(t)\approx N_u t
\qquad (t\ll1),
\]

so the relative error of the chemical field can be large at very early times even though
the absolute chemical field is small.

The implementation must therefore save both absolute and relative low-mode errors, and
must not interpret relative errors at `t approximately 0` as a physical instability.

Start with `omega_v=omega_u`. If the exact mode test shows that early chemical shot
noise creates a measurable spurious cell drift in the nonlinear run, one predefined
chemical oversampling test is allowed:

```text
omega_v = omega_u / 4
```

Then the injection mean is multiplied by four and the expected equilibrium `v`
population is `4 N_u`. Report the fourfold particle cost explicitly. Do not tune this
ratio continuously. The same-weight result remains the primary method unless it fails a
stated accuracy gate.

---

## 8. Files and reproducibility

Create a new self-contained directory:

```text
experiments/keller_segel/fully_parabolic_3d/
```

Suggested files:

```text
README.md
field3d_fourier.py
injection_kernel.py
initial_conditions.py
simulation_pp3d.py
diagnostics_pp3d.py
exact_mode_reference.py
analyze_mode_convergence.py
analyze_radial_delay.py
plot_mode_verification.py
plot_radial_delay.py
plot_tetra.py
test_pp3d.py
config_mode.json
config_radial_pilot.json
config_tetra.json
run_pp3d.sb
```

The directory must be self-contained according to the repository convention. Reuse or
port validated routines, but do not import fragile code through relative paths across
experiment directories.

Archive production outputs under

```text
reference_results/keller_segel_pp3d/<run_id>/
```

Every run directory must include:

```text
config_used.json
manifest.json
command.txt or exact argv in manifest
stdout/stderr log or SLURM job id
mass_population.csv
modes.csv
radii_drift.csv
cluster_metrics.csv        # tetra only
abort_diagnostics.json     # when applicable
plot_data/*.npz
figures/*.pdf
figures/*.png
```

`manifest.json` must record at least:

```text
git commit
UTC timestamp
Python/JAX/numpy versions
hostname or cluster
seed
all PDE parameters
N_u and initial N_v
omega_u and omega_v
K_dyn, K_out, K_test
Fourier mode convention
tau, T, output cadence
injection kernel
population-control status
wall time and peak population
```

Plot scripts must read saved CSV/NPZ data only and must never rerun the solver.

---

## 9. Unit tests required before production

`test_pp3d.py` must include:

1. **Fourier constant mode:** a uniform particle/quadrature cloud gives the correct mass
   coefficient and zero gradient.
2. **Single-mode gradient:** analytic gradient versus finite difference.
3. **Axis permutation symmetry:** swapping `x`, `y`, and `z` permutes coefficients and
   gradients correctly.
4. **Injection conditional mean:** Monte Carlo mean agrees with the exact split update.
5. **Injection variance:** minimum-variance offspring variance is no larger than Poisson.
6. **Zero chemical initialization:** `N_v=0` proceeds through the first step without a
   dummy particle or division by zero.
7. **Mass law:** a no-space test reproduces the exact discrete mass recurrence.
8. **Reproducibility:** fixed seed gives byte-identical diagnostic CSVs for a tiny run.
9. **No hidden population control:** config and outputs report it as false.
10. **Periodic wrapping:** particles remain in the declared torus.

All tests must pass before the radial pilot.

---

## 10. Paper-facing output

Keep the main-paper presentation compact. The preferred final structure is:

### One small verification table

Report:

- maximum normalized `v`-mass error;
- low-mode RMS error and observed `N` slope;
- time-step slope;
- minimum-variance versus Poisson seed variance;
- maximum/mean `N_v/N_u`.

### One main 3D figure

For the selected radial case:

- top row: fixed, identically normalized orthogonal/slice views of `u` at several times;
- second row: corresponding `v` views;
- side or bottom panel: `M_v/M_u`, `G_v`, and `R_0.2`, `R_0.5`, `R_0.8` versus time;
- mark the resolved-window endpoint and do not plot an unqualified singular peak.

The tetrahedral result should be included only if it passes the strict merging gate. If
included, place it in a separate compact figure or appendix.

### Safe manuscript claim

A defensible target statement is:

> In three dimensions, the fully parabolic particle system creates an initially absent
> chemical component through a cross-species minimum-variance injection kernel, follows
> the exact bounded chemical-mass law, and reproduces the expected low-mode evolution.
> In the nonlinear radial test, the finite buildup of the chemical field produces a
> resolved delay between initial cell diffusion and subsequent chemotactic contraction.

Add, when supported:

> A tetrahedral initial condition further demonstrates genuinely three-dimensional mutual
> attraction and aggregation of separated clusters.

### Claims that are not allowed

Do not claim:

```text
universal critical mass in 3D
continuum blow-up time
converged singular peak
resolution-independent final core radius
branching superiority over weighted particles from this example
cluster merging when only centroid attraction is observed
```

---

## 11. Stop/go checklist

Proceed in this order:

```text
[ ] Create the new fully_parabolic_3d directory.
[ ] Port and verify the 3D Fourier field evaluator; remove the screened solve.
[ ] Implement zero-initial-v cross-species injection and decay.
[ ] Pass all unit tests.
[ ] Pass the exact Fourier-mode verification.
[ ] Run the four-point radial pilot only.
[ ] Select at most one delayed-focusing case.
[ ] Run one-factor N, K_dyn, and tau checks.
[ ] Decide whether the radial result is paper-ready.
[ ] Run tetrahedral aggregation only if the radial result passes.
[ ] Generate figures from saved data only.
[ ] Ask Codex to audit the final numbers and framing.
[ ] Only then revise the manuscript subsection.
```

A negative or limited result must be recorded honestly. Do not expand the parameter
search merely to force a desired three-dimensional picture.