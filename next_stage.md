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
# Next experiments for the Keller--Segel LDG comparison and branching advantage

**Purpose.**  
The recent `T_core` result is promising but not yet paper-ready. We need to separate three questions that have been mixed together:

1. **Metric question:** Is the new core-collapse time `T_core` a stable, method-independent concentration-time diagnostic?
2. **Particle-dynamics question:** Is the current particle dynamics delayed relative to LDG because of the Fourier drift reconstruction, especially the window quantile `q_window` and bandwidth `K`?
3. **Branching-novelty question:** Where does the stochastic branching mechanism actually improve over weighted particles? In the fully parabolic--parabolic Keller--Segel blow-up benchmark, `u` is conservative and does not branch; only the `v` cloud uses decay/injection to realize `v_t = Δv + u - v`. Therefore this benchmark cannot by itself prove that branching is better than LDG or better than a weighted particle method.

The goal of the next experiments is **not** to tune a blow-up time to match LDG, but to obtain a defensible result that can be used in the paper.

---

## Current facts to respect

### Current Keller--Segel particle benchmark

The fully parabolic--parabolic benchmark is

\[
u_t=\Delta u-\nabla\cdot(u\nabla v),\qquad
v_t=\Delta v+u-v .
\]

The current particle runs use:

```text
solver_field = current_fourier
K = 10
q_window = 0.8
tau = 2e-7
dg_readout_n = 80 160
N_p = 8e4, 3.2e5
seeds = 0,1,2,3
```

The particle dynamics uses a core-adaptive Fourier reconstruction of the **v-field** to compute the drift

\[
dX_i^u = \nabla \widehat v(X_i^u)\,dt + \sqrt{2}\,dW_i .
\]

The diagnostic `T_core` is different: it is computed from **raw particle mass quantile radii**, not from Fourier reconstruction.

### Current `T_core` result

The new metric is

\[
R_q(t)=\inf\{r:\mu(B(x_c(t),r))\ge qM\},
\]

and

\[
R_q(t)^2 \approx \alpha_q-\beta_q t,\qquad T_q=\alpha_q/\beta_q.
\]

The current reported values are approximately:

```text
LDG N=320:          T_core = 1.215e-4, [p10,p90] = [1.13,1.22]e-4
particle Np=3.2e5: T_core = 1.318e-4, [p10,p90] = [1.26,1.33]e-4
```

This is a strong **limited-positive** result: same scale, near the LDG blow-up scale, particle about 8% later. It is **not** yet a continuum blow-up-time claim.

### Current problems to fix before using this in the paper

1. The README/commit text says LDG `N=320` has `12/12` valid fits, but the summary file shows only `4` valid `R_q^2` fits. This must be corrected.
2. Particle seed averaging currently risks silently dropping seeds whose CSV lengths differ from seed 0. This must be fixed using a common time grid and seed coverage rules.
3. Particle `T_core` uncertainty currently reflects quantile/window spread, not true seed uncertainty. Add seed bootstrap.
4. Particle valid fits are mainly from `q=0.2,0.3`; `q=0.1` is not stable. The paper language must reflect that.

---

# Experiment A. Repair and stress-test the `T_core` metric implementation

## A1. Fix particle seed averaging

Modify:

```text
experiments/keller_segel/core_collapse_time/fit_core_collapse.py
```

Current problematic logic:

```python
t0 = seeds[0]["t"]
arrs = [s["R"][q] for s in seeds if q in s["R"] and len(s["R"][q]) == len(t0)]
Rdict[q] = np.nanmean(np.vstack(arrs), axis=0)
```

Replace this with common-grid logic:

```python
def build_common_grid(seeds, dt=1e-6, t_end=None):
    # Use the largest interval covered by enough seeds.
    # Prefer t_grid = np.arange(0, t_common + 0.5*dt, dt)
    # where t_common is the largest time such that n_seed_eff >= min_seed_coverage.
    pass

def interpolate_seed_to_grid(seed, t_grid, q_list):
    # For each q, interpolate R_q(t) to the common grid.
    # Use np.interp only within the seed's available time interval.
    # Outside coverage, fill NaN.
    pass

def seed_mean_on_common_grid(seeds, q_list, dt=1e-6, min_seed_coverage=3):
    # Return:
    #   t_grid
    #   R_mean[q]
    #   R_median[q]
    #   n_seed_eff[q](t)
    #   R_seed_arrays[q] for bootstrap
    pass
```

Window validity rule:

```text
A fitting window is valid for particle only if n_seed_eff(t) >= min_seed_coverage
for every sampled time in that fitting window.
```

Default:

```text
min_seed_coverage = 3
dt = 1e-6
```

Keep a stricter option:

```text
--min_seed_coverage 4
```

so we can see whether the result changes when all four seeds are required.

## A2. Add seed bootstrap for particle `T_core`

Add:

```text
--bootstrap_seeds 1000
```

For each bootstrap sample:

1. Sample seeds with replacement.
2. Build seed-mean `R_q(t)` on the common grid.
3. Fit `T_core` over the same q-set and fitting windows.
4. Store bootstrap `T_core`.

Report:

```text
T_core_median_qwindow
T_core_p10_qwindow
T_core_p90_qwindow
T_core_bootstrap_median
T_core_bootstrap_p10
T_core_bootstrap_p90
n_seed_eff_min_by_fit
```

The paper should cite the seed bootstrap interval, not only the q/window spread.

## A3. Add q-set sensitivity without rerunning dynamics

Use existing LDG and particle `R_q(t)` CSVs. Recompute `T_core` for:

```text
q_set = {0.1,0.2,0.3}     # current default
q_set = {0.2,0.3}         # likely more reliable for particles
q_set = {0.2}
q_set = {0.3}
q_set = {0.05,0.1,0.2}    # stress test, not for main text unless stable
```

Output:

```text
core_fit_summary_by_qset.csv
core_fit_summary_by_qset.json
figures/core_T_qset_sensitivity.pdf
```

Decision:

```text
If particle T_core changes by >10% across reasonable q_set choices, do not quote a single
particle T_core. Report only that the core-collapse proxy is on the LDG scale but q-sensitive.

If LDG is stable and particle {0.2,0.3} is stable, use {0.2,0.3} for particle and
explain that q=0.1 is below the particle resolution/noise floor in the finest run.
```

## A4. Correct current result text

Update README and `REVISION_RESULTS.md`:

```text
Wrong:
  LDG N=320 has 12/12 q-window fits valid.

Correct:
  LDG N=320 has 4/12 R_q^2 fits passing the strict linear-fit gate;
  the accepted fits are concentrated in the late resolved windows and have small spread.
  The independent S^{-2} and peak^{-1} extrapolations agree near 1.21e-4.
```

Also replace:

```text
particle q={0.1,0.2,0.3} stable
```

with:

```text
particle stable fits currently come mainly from q=0.2 and q=0.3; q=0.1 is too noisy
or not linearly resolved at the present particle counts.
```

---

# Experiment B. Sensitivity of particle dynamics to `q_window`

## Why this is necessary

The current particle dynamics chooses the Fourier reconstruction window using `q_window=0.8`. This controls the window scale \(L(t)\) and therefore the effective drift resolution

\[
h_{\rm eff}(t)=L(t)/K.
\]

If `q_window=0.8` is controlled by the outer halo, then the inner core may keep collapsing while the Fourier drift remains too coarse. This could explain why particle `T_core` is about 8% later than LDG.

## B1. Production matrix

Run current Fourier solver only:

```text
solver_field = current_fourier
K = 10
tau = 2e-7
n_steps = 1000
T = 2e-4
diag_every = 5        # output dt = 1e-6
dg_readout_n = 80 160
cfl_abort = 5.0
filter_s = 0.5
```

Sweep:

```text
q_window in {0.5, 0.65, 0.8, 0.9}
N_p in {8e4, 3.2e5}
seeds = {0,1,2,3}
```

Total:

```text
4 q_window × 2 N × 4 seeds = 32 runs
```

Output directory:

```text
reference_results/keller_segel_ldg_pp/core_collapse_qwindow_<id>/
```

## B2. Exact command template

For each run:

```bash
python experiments/keller_segel/ldg_comparison/simulation.py \
  --N <N> \
  --K 10 \
  --tau 2e-7 \
  --n_steps 1000 \
  --diag_every 5 \
  --seed <seed> \
  --solver_field current_fourier \
  --dg_readout_n 80 160 \
  --cfl_abort 5.0 \
  --filter_s 0.5 \
  --q_window <q_window> \
  --report_times 6e-5 1.2e-4 2e-4 \
  --outdir <OUT>/qwin<q_window>_N<N>_seed<seed>
```

## B3. Diagnostics to analyze

For each q-window:

```text
T_core with repaired seed averaging + seed bootstrap
LDG-style t_b using S_dg_cross
R_0.1, R_0.2, R_0.3, R_0.5, R_0.8 curves
R_0.2 / (L/K), R_0.1 / (L/K)
L(t)
outside_v_frac
drift_cfl_solver_field
abort/final time
valid fit count
seed coverage by window
```

## B4. Decision rule

### Case B-success: q_window explains the 8% offset

If some principled mid-range window such as `q_window=0.65` or `0.5` gives:

```text
particle T_core closer to LDG N=320,
particle T_core stable under seed bootstrap,
outside_v_frac remains small,
no strong increase in aborts/CFL,
and q-set sensitivity is acceptable,
```

then write:

```text
The residual delay of the particle T_core under the default q_window=0.8 is attributable
to a conservative outer-mass window that slightly under-resolves the inner drift scale.
Using a more core-local but still stable window reduces this offset.
```

Do **not** cherry-pick. If choosing a new default, justify it before looking at `T_core`, e.g.:

```text
q_window = 0.65 is selected because it keeps outside_v_frac below a tolerance while
maximizing R_0.2/(L/K), i.e. resolving the inner core without losing the chemical field.
```

### Case B-negative: q_window sensitivity is large or unstable

If `T_core` changes strongly with `q_window`, or a closer result comes with large `outside_v_frac` or instability, then do not use a particle blow-up-time claim.

Paper wording:

```text
The particle core-collapse proxy remains on the LDG scale, but its exact extrapolated
time is sensitive to the Fourier-window choice. We therefore report it as a concentration-time
diagnostic, not as a continuum blow-up time.
```

---

# Experiment C. Sensitivity to Fourier bandwidth `K`

## Why this is necessary

The 8% particle delay may come from `K=10`, not only from `q_window=0.8`. If increasing `K` moves particle `T_core` toward LDG, then the current offset is a drift-reconstruction resolution effect.

## C1. Pilot at Np = 8e4

Run:

```text
K in {8,10,12,16}
q_window = 0.8
N_p = 8e4
seeds = 0,1,2,3
tau = 2e-7
T = 2e-4
```

If `K=16` is too noisy or aborts, try stronger filter:

```text
filter_s in {0.5,0.35,0.25}
```

## C2. Main high-resolution test

Only for promising K values:

```text
K in {10,12} or {10,12,16 if stable}
N_p = 3.2e5
seeds = 0,1,2,3
```

Optional combined sweep:

```text
q_window in {0.65,0.8}
K in {10,12}
```

## C3. Metrics

Same as Experiment B:

```text
T_core with fixed seed handling
seed bootstrap CI
LDG-style t_b
core radii
R_0.2/(L/K)
drift_cfl_solver_field
abort/final time
outside_v_frac
```

## C4. Decision rule

If increasing K moves \(T_{\rm core}\) toward LDG while preserving stability:

```text
The current particle delay is reconstruction-resolution controlled.
```

If increasing K does not change \(T_{\rm core}\):

```text
The 8% offset is not primarily due to Fourier bandwidth at this resolution; it may be
particle noise, splitting error, time step, or the boundary/window model.
```

If increasing K causes instability/noise:

```text
The Fourier drift is resolution/noise limited; report only the stable default result.
```

---

# Experiment D. Time-step sensitivity

## Why this is necessary

The particle method uses Lie splitting + Euler transport with `tau=2e-7`. Before claiming an 8% physical/numerical offset from LDG, check whether `T_core` changes with `tau`.

## D1. Runs

Use the most relevant configuration from Experiments B/C, plus the current default:

```text
tau in {4e-7, 2e-7, 1e-7}
N_p = 8e4 first
seeds = 0,1,2,3
K = 10 or selected K
q_window = 0.8 or selected q_window
```

If the trend is strong, repeat at \(N_p=3.2e5\) for:

```text
tau in {2e-7, 1e-7}
```

## D2. Decision

If `T_core` changes by less than 3--5% under halving tau:

```text
time discretization is not the main source of the 8% offset.
```

If it changes significantly:

```text
the particle core-collapse time is still time-step sensitive; do not promote as final.
```

---

# Experiment E. LDG metric robustness

## E1. Quadrature order sensitivity

For LDG core radii, compute with:

```text
quad_order = 3, 5, 7
```

Check:

```text
T_core changes < 1--2%
R_q curves visually identical
raw vs clipped mass identical within roundoff
```

## E2. q-set sensitivity for LDG

Same as A3:

```text
{0.1,0.2,0.3}
{0.2,0.3}
{0.2}
{0.3}
```

If LDG is stable across q-sets but particle is not, the limitation is particle resolution/noise.

## E3. Fit-window sensitivity

Add later/narrower windows if data allows:

```text
[7e-5, 1.15e-4]
[8e-5, 1.18e-4]
[9e-5, 1.20e-4]
```

Do not use windows that include post-grid-floor artifacts.

---

# Experiment F. Actual branching advantage: weighted vs branching on non-conservative reaction

## Why this is separate from the KS blow-up benchmark

In the fully parabolic--parabolic KS blow-up benchmark, \(u\) is conservative. The `u` particles do not branch. The birth/death mechanism is only the `v` decay/injection kernel. Therefore this benchmark is not the clean place to claim "branching beats LDG" or "branching beats weighted particles."

The methodological novelty claim should be supported by a **non-conservative reaction** benchmark where particle weights would concentrate but branching keeps an equal-weight representation.

The editor/revision motivation explicitly says that the planned revision should demonstrate reduced sampling variance versus weighted-particle strategies: branching keeps an empirical measure as an unweighted sum and spawns particles where mass concentrates, whereas weighted strategies can suffer effective sample-size collapse. This is the right target for the branching-advantage experiment.

## F1. Candidate benchmark 1: localized growth islands

Use an ADR equation:

\[
\partial_t u = D\Delta u + r(x)u,
\]

with

\[
r(x)=\lambda G_{\rm multi}(x)-\beta,
\]

where \(G_{\rm multi}\) is a grid of separated Gaussian growth islands.

Suggested parameters:

```text
M = 16 islands
sigma = 0.16
D = 0.01
lambda = 12
beta = 0.8
T = 0.8
tau = 1e-3
N0 = 2e4
K = 64
```

Run both methods with identical transport and reconstruction:

```text
branching equal-weight
weighted fixed-N with weights w_i <- w_i exp(tau r_i)
```

Metrics:

```text
global ESS/N
local ESS per island
local mass per island
local variance across seeds
max weight / mean weight
MMD or W2 to a high-sample/reference run
runtime
particle population trajectory for branching
```

Expected useful result:

```text
Global ESS can look acceptable while local island ESS collapses.
Branching keeps local particle counts proportional to mass growth.
```

This directly supports the novelty claim.

## F2. Candidate benchmark 2: nonlinear logistic KS / growth KS

Use the non-conservative KS-like system from the manuscript:

\[
u_t-\nabla\cdot\left(\nabla u-\frac{4u}{1+u^2}\nabla v\right)=u(1-u),
\qquad
v_t-\Delta v=u-v.
\]

This is closer to the original paper's KS section, but it is harder to isolate branching from drift/reconstruction. Use it only after F1 succeeds.

Compare:

```text
branching birth/death for u reaction
weighted particle reaction
same transport
same reconstruction
same seeds
```

Metrics:

```text
L2 / mass error vs deterministic reference or high-N particle reference
local ESS
local variance in high-density region
particle population N(t)
weight degeneracy
```

## F3. Decision rule for paper

A paper-worthy branching result requires:

```text
branching and weighted use the same transport and same reconstruction;
branching has lower seed variance or lower local mass error at the same computational budget;
weighted has local ESS collapse or high max/mean weight;
branching population remains controlled and tracks L1 mass.
```

This directly addresses the reviewer concern and the editor letter.

---

# Experiment G. What to put in the paper depending on outcomes

## G1. Strong outcome

Use if the following all hold:

```text
Repaired T_core confirms LDG T_core ~1.21e-4.
Particle T_core matches LDG within uncertainty after seed bootstrap and q/window sensitivity.
Weighted-vs-branching shows a clear local ESS / variance advantage in a non-conservative reaction benchmark.
```

Paper message:

```text
The particle method reproduces LDG core-collapse dynamics using a reconstruction-light metric,
and the branching mechanism provides a separate variance-resolution advantage for non-conservative reactions.
```

## G2. Limited-positive outcome

Use if:

```text
LDG T_core is stable.
Particle T_core is on the same scale but offset by ~5--10%.
q/K/window sensitivity explains but does not eliminate the offset.
Weighted-vs-branching shows clear advantage.
```

Paper message:

```text
The KS benchmark demonstrates LDG-scale concentration and a stable core-collapse diagnostic,
while the branching mechanism's main advantage is shown in non-conservative reaction benchmarks.
```

## G3. Negative outcome

Use if:

```text
Particle T_core is q/K/window sensitive.
Weighted-vs-branching advantage is weak.
```

Paper message:

```text
Do not emphasize blow-up time. Report reconstruction-free radii and LDG-scale concentration only.
Refocus novelty on equal-weight reaction representation, mass tracking, high-dimensional scalability,
and weighted-particle degeneracy where it is demonstrable.
```

---

# Immediate task list for Claude/Codex

## Priority 1: repair `T_core` analysis

Implement:

```text
fit_core_collapse.py:
  --common_grid_dt
  --min_seed_coverage
  --bootstrap_seeds
  --q_sets
  seed interpolation to common grid
  seed-bootstrap CI
  n_seed_eff reporting
```

Regenerate:

```text
core_fit_all.csv
core_fit_summary.csv/json
core_fit_bootstrap.csv/json
README.md
figures/
```

## Priority 2: q-set sensitivity

Run without new dynamics:

```bash
python fit_core_collapse.py \
  --ldg_dir <OUT>/ldg \
  --particle_root <OUT>/particle \
  --mass raw \
  --q_sets "0.1,0.2,0.3" "0.2,0.3" "0.2" "0.3" \
  --bootstrap_seeds 1000 \
  --min_seed_coverage 3 \
  --outdir <OUT>/qset_sensitivity
```

## Priority 3: q_window dynamics sweep

Create:

```text
experiments/keller_segel/core_collapse_time/run_qwindow_sensitivity.sb
experiments/keller_segel/core_collapse_time/analyze_qwindow_sensitivity.py
experiments/keller_segel/core_collapse_time/plot_qwindow_sensitivity.py
```

Run the 32-run matrix.

## Priority 4: weighted vs branching non-conservative benchmark

Create or update:

```text
experiments/branching_vs_weighted/local_growth_islands/
```

Required scripts:

```text
simulate_branching.py
simulate_weighted.py
analyze_local_ess.py
plot_local_ess.py
README.md
```

This is the clean experiment for the paper's branching novelty.

---

# Recommended paper positioning after these experiments

For the Keller--Segel LDG section:

```text
Do not claim "branching is better than LDG" in the fully parabolic--parabolic blow-up benchmark.
Instead: the particle method reproduces LDG-scale concentration under a direct LDG reference,
and the core-collapse proxy is a reconstruction-light supporting diagnostic.
```

For the branching novelty section:

```text
Use the non-conservative reaction benchmark to show branching reduces sampling variance relative
to weighted particles, especially in local ESS / local mass diagnostics.
```

This division is scientifically cleaner and safer for reviewers.

