# Core-collapse time metric for the Keller--Segel benchmark

**Purpose.**  
Define a method-independent blow-up / concentration-time diagnostic that can be computed
on both the fixed-flux LDG reference and the particle method.  This is different from the
LDG-style resolution-gap time
\[
t_b=\inf\{t:S_{\rm high}(t)/S_{\rm low}(t)\ge 1.05\},
\]
which measures when two numerical resolutions separate.  The goal here is to estimate a
geometric **core-collapse time** from the shrinking length scale of the concentrating mass.

This note is intended as the implementation plan for the next numerical diagnostic.  The
metric should be tried first on the fixed-flux LDG reference.  Only if it is stable on LDG
should it be used as a main comparison with particles.

---

## 1. Why introduce a new metric?

The LDG-style \(t_b\) is useful and fair for comparing resolution-gap behavior, but it is
not a direct blow-up time.  It answers:

\[
\text{When does the high-resolution solution start seeing more }L^2\text{ growth than the low-resolution solution?}
\]

That time depends on the resolution pair, the threshold \(1.05\), and the chosen \(S(t)\)
readout.  It often occurs before the physical or numerical collapse time.

For a concentrating Keller--Segel core, the more geometric quantity is the core length scale.
A method-independent way to measure this length scale is through **mass-quantile radii**
\[
R_q(t)=\inf\{r:\mu(B(x_c(t),r))\ge qM\}.
\]

For particles, \(R_q\) is obtained directly from sorted particle distances.  
For LDG, \(R_q\) is obtained from cell or quadrature mass samples.  
Thus the metric is almost reconstruction-free and can be applied to both methods.

The intended improvement over \(L^2\)-gap metrics is:

- It tracks the **core length scale** directly.
- It is less sensitive to Fourier/KDE/DG readout choices.
- It exposes core--halo separation:
  \[
  R_{0.8}(t)\ \text{may stop shrinking while } R_{0.1}(t),R_{0.2}(t)\ \text{continue collapsing}.
  \]
- It can be checked for stability across \(q\), fitting windows, and resolutions.

---

## 2. Mathematical definition

Let \(\mu(t)\) be the finite measure corresponding to the cell density \(u(t)\), with total mass
\[
M=\mu(t)(\Omega).
\]

For the fully parabolic--parabolic Keller--Segel benchmark considered here, \(M=10\pi\)
for the \(u\)-density and is conserved.

### 2.1 Center of the core

For the single-core radial benchmark, the simplest center is the first moment:
\[
x_c(t)=\frac{1}{M}\int_\Omega x\,d\mu(t).
\]

This is the default center.

A more robust option is an inner-mass center.  First compute \(R_{q_0}\) using the first-moment
center, with \(q_0=0.2\) or \(0.3\).  Then define
\[
x_c^{(q_0)}(t)
=
\frac{1}{\mu(B(x_c,R_{q_0}))}
\int_{B(x_c,R_{q_0})}x\,d\mu(t).
\]

In practice:

1. Start with the global center \(x_c^{(0)}\).
2. Compute \(R_{q_0}^{(0)}\).
3. Recompute the center using only the mass inside \(B(x_c^{(0)},R_{q_0}^{(0)})\).
4. Optionally repeat once.
5. Use the final center to compute all \(R_q\).

For the current single-core example, the global center should be sufficient.  The inner-center
version is a robustness check.

### 2.2 Mass-quantile radii

For each
\[
q\in Q=\{0.05,0.1,0.2,0.3,0.5,0.8\},
\]
define
\[
R_q(t)
=
\inf\left\{r:\mu(B(x_c(t),r))\ge qM\right\}.
\]

Use \(q=0.1,0.2,0.3\) as the **primary core set**.  
Use \(q=0.5\) as a secondary check.  
Use \(q=0.8\) only as a halo/core-separation diagnostic, not as the main blow-up-time estimator.

The main reason not to use \(q=0.8\) as the primary metric is that \(R_{0.8}\) can be controlled
by the outer halo while the inner core continues collapsing.

---

## 3. Core-collapse time from radius extrapolation

In a resolved pre-collapse window, assume the core length scale obeys approximately
\[
R_q(t)^2 \approx a_q(T_q-t).
\]

Equivalently, write a linear regression model
\[
Y_q(t):=R_q(t)^2 \approx \alpha_q-\beta_q t,
\qquad
\beta_q>0,
\]
so that
\[
T_q = \frac{\alpha_q}{\beta_q}.
\]

This is the **radius-extrapolated collapse time** for quantile \(q\).

An instantaneous derivative version is also possible:
\[
T_q(t)
=
t+\frac{R_q(t)^2}{-dR_q(t)^2/dt},
\]
but the regression version is less noisy and should be the primary implementation.

### 3.1 Fitting windows

Use several late-time windows, for example
\[
\mathcal I =
\{
[4\times10^{-5},9\times10^{-5}],
[5\times10^{-5},1.0\times10^{-4}],
[6\times10^{-5},1.1\times10^{-4}],
[7\times10^{-5},1.2\times10^{-4}]
\}.
\]

The exact windows should be adjusted after inspecting LDG data.  The key rule is that the
window must be:

1. late enough that the core collapse is visible;
2. early enough that the radius is not at the grid / particle / bandwidth floor;
3. before any solver abort / incomplete-data time for the particle runs;
4. common across the methods being compared when possible.

### 3.2 Aggregated definition

For each resolution and method, compute
\[
T_q(I)
\]
for all \(q\in Q_{\rm core}=\{0.1,0.2,0.3\}\) and all \(I\in\mathcal I\).

Define
\[
T_{\rm core}
=
\operatorname{median}_{q\in Q_{\rm core},\,I\in\mathcal I} T_q(I).
\]

Also report the spread:
\[
\Delta T_{\rm core}
=
\left[
\operatorname{percentile}_{10}(T_q(I)),
\operatorname{percentile}_{90}(T_q(I))
\right],
\]
or a min--max range if the number of windows is small.

Do **not** quote a single blow-up time unless the spread across \(q\), windows, and resolutions is controlled.

---

## 4. How to compute \(R_q\) on LDG

For an LDG solution \(u_h(x,t)\), do not rely only on cell averages.  Use quadrature.

### 4.1 Quadrature samples

For each cell \(C\), take quadrature points \(x_{C,m}\) and quadrature weights \(w_{C,m}\).
For \(P^1\) LDG, a \(3\times 3\) Gauss rule is sufficient for robust diagnostics; a \(4\times4\)
rule is safer and cheap.

Evaluate
\[
u_{C,m}=u_h(x_{C,m},t).
\]

Construct mass samples in two versions:

Raw:
\[
m_{C,m}^{\rm raw}=w_{C,m}u_{C,m}.
\]

Clipped:
\[
m_{C,m}^{+}=w_{C,m}\max(u_{C,m},0).
\]

The positivity limiter should make \(u_h\ge0\), but saving both raw and clipped diagnostics is useful.
The primary result should use the clipped version only if raw and clipped agree to high accuracy.

### 4.2 LDG center and radii

For the selected mass samples \(m_a\) at locations \(x_a\), compute
\[
M_h=\sum_a m_a,
\qquad
x_c=\frac{\sum_a m_a x_a}{M_h}.
\]

Then compute distances
\[
d_a=|x_a-x_c|.
\]

Sort the pairs \((d_a,m_a)\) by \(d_a\), form cumulative mass, and define
\[
R_q=\min\left\{d_a:\sum_{b:d_b\le d_a}m_b\ge qM_h\right\}.
\]

Optional interpolation: If cumulative mass jumps from below \(qM_h\) to above \(qM_h\)
between two adjacent sorted samples, linearly interpolate in \(d\).  This is not required
but makes \(R_q(t)\) smoother.

### 4.3 LDG output file

For each LDG run, write a CSV:

```text
ldg_core_radii_N<N>.csv
```

with columns:

```text
t
N
M_raw
M_clip
xc_x_raw
xc_y_raw
xc_x_clip
xc_y_clip
R_0.05_raw
R_0.1_raw
R_0.2_raw
R_0.3_raw
R_0.5_raw
R_0.8_raw
R_0.05_clip
R_0.1_clip
R_0.2_clip
R_0.3_clip
R_0.5_clip
R_0.8_clip
S_L2
peak
u_min
```

The current fixed-flux LDG runs should be re-run or postprocessed so that this time series exists at
\(\Delta t_{\rm out}\approx10^{-6}\) over \(0\le t\le2\times10^{-4}\).

---

## 5. How to compute \(R_q\) on particles

For an equal-weight particle cloud
\[
\mu^N(t)=\omega\sum_{i=1}^{N(t)}\delta_{X_i(t)},
\]
compute
\[
x_c=\frac{1}{N(t)}\sum_i X_i(t)
\]
for the default center.

Then compute distances
\[
d_i=|X_i-x_c|.
\]

For equal weights,
\[
R_q^N(t)=d_{(\lceil qN(t)\rceil)},
\]
where \(d_{(k)}\) is the \(k\)-th ordered distance.

For non-equal weights, sort \((d_i,\omega_i)\) and use the cumulative weighted mass.

### 5.1 Particle output

The current particle diagnostics already include some inner radii such as \(R_{0.1}\), \(R_{0.2}\), \(R_{0.5}\), and \(R_{0.8}\) in recent runs.  Add \(R_{0.3}\) explicitly.

Required columns:

```text
t
seed
N
M_u
xc_x
xc_y
R_0.05
R_0.1
R_0.2
R_0.3
R_0.5
R_0.8
S_dg_cross_80
S_dg_cross_160
S_L2_u
peak_PK_u
drift_cfl_solver_field
drift_cfl_fourier_diag
solver_field_mode
```

For the main comparison, use `current_fourier` particle dynamics unless there is a strong reason
to include solver-field variants.  The latest solver-field comparison shows that blob residual
improves stability but does not change the LDG-style \(t_b\) or core radii in a statistically
distinguishable way.  Therefore it is not the main accuracy result.

---

## 6. Secondary collapse-time diagnostics

These are useful cross-checks but should not be the main metric.

### 6.1 Inverse-\(L^2\) time

If the core is approximately self-similar,
\[
u(x,t)\approx R(t)^{-2}F\left(\frac{x-x_c}{R(t)}\right),
\]
then
\[
\|u(t)\|_{L^2}\sim R(t)^{-1}.
\]

Therefore
\[
S(t)^{-2}\sim R(t)^2\sim T-t.
\]

Fit
\[
S(t)^{-2}\approx \alpha_S-\beta_S t,
\qquad
T_{L^2}=\alpha_S/\beta_S.
\]

For LDG, use the LDG \(L^2\) norm.  
For particles, use the LDG-matched \(P^1\) DG readout \(S^{DG}\), not the Fourier \(S\), as the primary \(S\).  
Fourier \(S\) can be reported as sensitivity only.

### 6.2 Inverse peak time

Similarly, for a concentrated two-dimensional core,
\[
\|u(t)\|_\infty^{-1}\sim R(t)^2.
\]

Fit
\[
P(t)^{-1}\approx \alpha_P-\beta_P t,
\qquad
T_{\rm peak}=\alpha_P/\beta_P.
\]

Peak is highly reconstruction-dependent and should only be a secondary diagnostic.

---

## 7. Stability criteria before quoting a time

A core-collapse time estimate is only meaningful if the following checks pass.

### 7.1 Quantile consistency

The estimates
\[
T_{0.1},\quad T_{0.2},\quad T_{0.3}
\]
should be close.

Suggested acceptance criterion:

\[
\frac{\max T_q-\min T_q}{\operatorname{median}T_q}\le 0.25
\]
within a given resolution and fitting-window family.

### 7.2 Window consistency

The same \(q\) fitted over multiple windows should give similar \(T_q(I)\).

Suggested acceptance criterion:

\[
\frac{P_{90}(T_q(I))-P_{10}(T_q(I))}{\operatorname{median}_{I}T_q(I)}\le 0.25.
\]

### 7.3 Resolution consistency

LDG \(N=160\) and \(N=320\) should give the same order and preferably overlapping
ranges.  Particle \(N_p=8\times10^4\) and \(3.2\times10^5\) should also be consistent
within seed uncertainty.

### 7.4 Agreement with secondary diagnostics

\(T_{L^2}\) should be of the same order as \(T_{\rm core}\).  
\(T_{\rm peak}\) should not strongly contradict it, but peak is not required to agree precisely.

### 7.5 Fit quality

For the linear fit
\[
R_q(t)^2=\alpha_q-\beta_q t,
\]
require:

```text
beta_q > 0
R^2 >= 0.9    # or at least report it
T_q > max(window)
T_q not absurdly far beyond the data window
```

If the fit is curved or the estimate changes strongly with the fitting window, do not quote it as a blow-up time.

---

## 8. Interpretation scenarios

### Scenario A: strong result

LDG \(T_{\rm core}\) is stable across \(q\), windows, and resolution, and particle \(T_{\rm core}\)
agrees within uncertainty.

Paper language:

> We define a reconstruction-light core-collapse time from the extrapolated mass-quantile
> radii.  The estimate is stable across inner quantiles and fitting windows and agrees
> between the fixed-flux LDG reference and the particle method.

This would significantly strengthen the Keller--Segel benchmark.

### Scenario B: LDG stable, particle unstable

This means the metric is valid but the particle resolution / ensemble is not enough.

Action:

- increase particle count or seed count;
- check small-\(q\) quantile noise;
- consider \(q=0.2,0.3\) only;
- do not quote particle \(T_{\rm core}\) yet.

### Scenario C: LDG itself unstable

Then the metric is not robust enough for this benchmark.

Paper language:

> The core-collapse proxy is of order \(10^{-4}\), but remains window-sensitive even
> for the LDG reference.  We therefore do not quote a continuum blow-up time.

### Scenario D: \(T_{\rm core}\) stable but far from literature time

Then check:

- center definition;
- raw vs clipped LDG mass;
- fitting windows;
- whether \(R_q^2\) is actually linear;
- whether the literature value refers to a different blow-up-time definition.

Do not force the metric to match a target number.

---

## 9. Proposed repository implementation

Create:

```text
experiments/keller_segel/core_collapse_time/
    README.md
    compute_ldg_core_radii.py
    collect_particle_core_radii.py
    fit_core_collapse.py
    plot_core_collapse.py
```

### 9.1 `compute_ldg_core_radii.py`

Responsibilities:

- read LDG snapshots or LDG coefficient time series;
- evaluate \(u_h\) at quadrature points;
- compute raw and clipped \(R_q\);
- write `ldg_core_radii_N<N>.csv`.

Command:

```bash
python compute_ldg_core_radii.py \
  --ldg_dir reference_results/keller_segel_ldg_pp/ldg_20260614_2074_b41f6d4_ldg_fixed_flux \
  --N 80 160 320 \
  --q 0.05 0.1 0.2 0.3 0.5 0.8 \
  --quad_order 4 \
  --outdir reference_results/keller_segel_ldg_pp/core_collapse_<run_id>/ldg
```

If the current LDG archive does not contain enough snapshots / coefficients to postprocess, rerun LDG with online radius output.

### 9.2 `collect_particle_core_radii.py`

Responsibilities:

- read particle `diag_*.csv`;
- extract \(R_q(t)\), \(S^{DG}(t)\), Fourier \(S(t)\), peak;
- group by method, \(N_p\), seed;
- write clean ensemble CSV.

Command:

```bash
python collect_particle_core_radii.py \
  --particle_dir reference_results/keller_segel_ldg_pp/particle_blowup_<run_id> \
  --q 0.05 0.1 0.2 0.3 0.5 0.8 \
  --outdir reference_results/keller_segel_ldg_pp/core_collapse_<run_id>/particle
```

If \(R_{0.3}\) is missing from existing particle runs, either rerun or compute from saved clouds if available.
For main results, prefer rerun with \(R_{0.3}\) written online.

### 9.3 `fit_core_collapse.py`

Responsibilities:

- read LDG and particle radii;
- fit \(R_q^2=\alpha-\beta t\);
- compute \(T_q=\alpha/\beta\);
- compute \(T_{\rm core}\) median and spread;
- compute \(T_{L^2}\) from \(S^{-2}\);
- compute \(T_{\rm peak}\) from peak\(^{-1}\);
- bootstrap particle seeds.

Output files:

```text
core_fit_all.csv
core_fit_summary.csv
core_fit_summary.json
```

Suggested output columns for `core_fit_all.csv`:

```text
method
resolution
seed_group
q
window_start
window_end
quantity          # Rq2, Sminus2, peakminus1
alpha
beta
T_est
R2_fit
n_points
valid_fit
invalid_reason
```

Suggested output columns for `core_fit_summary.csv`:

```text
method
resolution
quantity
q_set
window_set
T_median
T_p10
T_p90
T_min
T_max
relative_spread
valid_quote
decision
```

### 9.4 `plot_core_collapse.py`

Figures:

1. \(R_q(t)^2\) vs \(t\) for \(q=0.1,0.2,0.3\), with fit lines.
2. \(T_q(I)\) scatter over \(q\) and fit windows.
3. LDG vs particle \(T_{\rm core}\) summary with uncertainty bars.
4. \(S(t)^{-2}\) and peak\(^{-1}\) as secondary checks.
5. Core--halo separation plot:
   \[
   R_{0.8}(t)/R_{0.2}(t)
   \]
   to show whether the halo decouples from the collapsing inner core.

---

## 10. Suggested experiment order

### Step 1: LDG-only validation

Use fixed-flux LDG \(N=80,160,320\).

Compute:

```text
R_0.05, R_0.1, R_0.2, R_0.3, R_0.5, R_0.8
T_core from R_0.1, R_0.2, R_0.3
T_L2 from S_L2^{-2}
T_peak from peak^{-1}
```

Decision:

- If LDG \(T_{\rm core}\) is unstable, stop and do not promote this metric.
- If LDG \(T_{\rm core}\) is stable, proceed to particle.

### Step 2: Particle current solver

Use current particle runs or rerun:

```text
N_p = 8e4, 3.2e5
seeds = 0,1,2,3
solver_field = current_fourier
dg_readout_n = 80,160
output_dt = 1e-6
```

Add \(R_{0.3}\) to diagnostics if missing.

### Step 3: Compare LDG and particle

Compare:

```text
T_core(LDG N=160,320)
T_core(particle Np=8e4,3.2e5)
T_L2(LDG)
T_L2(particle DG readout)
LDG-style t_b
```

### Step 4: Decide paper usage

Use in main §5.4 only if LDG and particle are both stable.

Otherwise use as a limited diagnostic and say no continuum blow-up time is quoted.

---

## 11. Suggested manuscript wording

### If stable

```latex
In addition to the LDG-style resolution-gap time, we compute a reconstruction-light
core-collapse proxy from the mass-quantile radii.  For \(q=0.1,0.2,0.3\), let
\(R_q(t)\) be the smallest radius around the cell-density center containing a fraction
\(q\) of the total cell mass.  In a resolved pre-collapse window we fit
\(R_q(t)^2\approx a_q(T_q-t)\) and define \(T_{\rm core}\) as the median over quantiles
and fitting windows.  The same definition is applied to LDG quadrature masses and to
particle clouds.  The resulting \(T_{\rm core}\) is stable across inner quantiles and
agrees between the LDG reference and the particle method, while the outer radius
\(R_{0.8}\) exposes the surrounding halo.
```

### If not stable

```latex
We also tested a reconstruction-light core-collapse proxy based on the extrapolation
of mass-quantile radii \(R_q(t)^2\).  Although the proxy gives a time scale of order
\(10^{-4}\), it remains sensitive to the fitting window and inner quantile, even on the
LDG reference.  We therefore report it only as a diagnostic and do not quote a continuum
blow-up time.
```

---

## 12. Decision rule for final paper

The new metric should improve the paper only if it passes this gate:

```text
LDG stable first.
Particle stable second.
LDG and particle agree within uncertainty.
Secondary S^{-2} check same order.
Peak not contradictory.
```

If this gate fails, the paper should keep the safer story:

```text
direct LDG comparison;
LDG-style resolution-gap time on the same scale;
reconstruction-free core radii showing concentration;
no continuum blow-up-time claim.
```

