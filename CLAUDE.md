# CLAUDE.md — revision priorities for the branching-particle paper

This repository is being revised for the paper:

> **Adaptive equal-weight branching particle methods for non-conservative transport–reaction–diffusion equations**

Every code change, numerical experiment, table, figure, and paragraph must serve the paper. Do not add experiments merely because they are interesting. The paper’s central contribution is:

> Equal-weight branching is a controlled particle-level representation of non-conservative mass creation. Compared with weighted particles, it keeps particles equal-weight and makes local particle resolution follow localized reaction-driven growth.

The revision must remain focused. The main paper should not become a broad report of every diagnostic we tried.

---

## 0. Current paper-level story

The paper should make a measured, defensible claim:

1. The weighted update is unbiased for the reaction multiplier but can suffer severe weight degeneracy under localized growth.
2. Equal-weight branching stores reaction-induced mass creation as integer particle count instead of scalar weights.
3. In controlled localized-growth benchmarks, branching gives better reconstructed-field accuracy than raw weighted particles and the tested global resampling baselines at matched or comparable particle-step cost.
4. Switching-growth tests show a structural failure of global resampling: it can collapse lineage diversity onto an early growth region and leave later regions under-represented.
5. Keller–Segel, 3D focusing, and kinetic tests are stress tests and extensions of the particle-field machinery. They should not be overused to claim new branching advantages unless the comparison is controlled.

The main evidence for “branching wins” is:

```text
§5.2 stationary localized growth:
    branching beats raw weighted and tested ESS-resampling baseline in reconstructed L2 / peak metrics.

§5.3 switching growth:
    branching beats tested global resampling baselines in the second-stage region and preserves lineage diversity.
```

The Keller–Segel / 3D / kinetic sections should support robustness, concentration handling, and dimensional transfer, not carry the main branching-versus-weighted proof.

---

## 1. Hard decision: multi-island is NOT a main-paper branching-wins experiment

### Static multi-island

The static separated-island benchmark showed a useful diagnostic fact:

```text
global ESS can look healthy while local island effective sample size collapses.
```

However, it did **not** show that branching has smaller per-island mass error than ESS resampling. In the current archived static result, branching keeps much larger local particle count, but the per-island mass error `E_m` is comparable to or worse than the resampled baseline.

Therefore:

```text
Do NOT use the static multi-island benchmark as a main-paper branching-wins result.
```

It may remain in `reference_results/` as a diagnostic record.

### Staged multi-island

The staged multi-island benchmark was designed to rescue the idea by activating separated islands at different times. It used:

```text
M=16 islands, 4 time-staggered activation groups,
stratified initial particles,
weighted / ESS-resampled / cost-matched ESS / minvar branching,
late-group metrics.
```

The production 8-seed result did **not** satisfy the success criterion. Branching achieved the lowest global `L2` and the highest late-island local particle count, but cost-matched weighted+ESS won the per-island mass metrics:

```text
cost-matched ESS late mean E_m < branching late mean E_m
cost-matched ESS late local L2 <= branching late local L2
```

Conclusion:

```text
The staged multi-island benchmark is a negative / diagnostic record.
It is NOT a main-paper benchmark.
Do NOT keep tuning it with the same per-island mass metrics.
Do NOT claim that staged multi-island shows branching is better.
```

The correct interpretation is:

```text
Per-island mass E_m is a forgiving integral observable that favors weighted particles.
Branching creates more local particles, but island total mass still carries genealogical
and reproduction variance from the finite ancestor set.
Final local count high does not imply island-mass variance low.
```

Update or keep any staged multi-island README accordingly. It must say:

```text
This benchmark was designed as a candidate main-paper result, but the 8-seed
production result did not satisfy the success criterion. It is kept as a diagnostic /
negative-result record and is not used in the main paper.
```

Remove or rewrite any sentence saying:

```text
"staged multi-island replaces §5.2"
"branching is quantitatively better in staged multi-island"
"branching has the smallest max E_m"
```

These statements are false or misleading for the current data.

---

## 2. What not to do next

Do not spend more time trying to make the same staged multi-island `E_m` table work.

Do not continue tuning `lambda`, `beta`, windows, or amplitudes if the primary metric remains:

```text
E_m = island total mass error over B_m.
```

That metric is not the right battlefield for branching. Weighted particles can estimate a low-dimensional mass integral well at matched particle-step cost because they carry continuous weights. Branching is designed to improve local representation / shape / peak / reconstruction under localized growth, not to dominate every zeroth-moment integral.

Do not insert a multi-island section into the main manuscript unless the table clearly supports the claim:

```text
branching beats cost-matched ESS on the primary local shape / peak / local-L2 metrics.
```

If the result is only:

```text
branching has more local particles but worse or comparable local error,
```

then it belongs only in notes / reference_results, not the paper.

---

## 3. Optional replacement idea: compressive-readout multi-island

This is optional and lower priority than manuscript cleanup. Only run this if there is time and the main paper is already stable.

The reason to try it would be:

> Convert “branching has higher local particle count” into a visible accuracy advantage by making the local field shape sharper after growth, so insufficient local particles hurt local reconstruction.

This is not the same as the failed `E_m` experiment.

### 3.1 PDE

Use a staged multi-island reaction plus a deterministic compressive drift:

\[
\partial_t u
=
-\nabla\cdot(b(t,x)u)
+
D\Delta u
+
r(t,x)u .
\]

Reaction:

\[
r(t,x)
=
\lambda\sum_{g=1}^G s_g(t)
\sum_{m\in\mathcal G_g}
a_m
\exp\left(
-\frac{d_{\mathbb T}(x,c_m)^2}{2\sigma^2}
\right)
-\beta .
\]

Compressive drift:

\[
b(t,x)
=
-\kappa
\sum_{g=1}^G h_g(t)
\sum_{m\in\mathcal G_g}
\chi_m(x)(x-c_m)_{\mathbb T}.
\]

Here:

- \((x-c_m)_{\mathbb T}\) is the torus displacement from center \(c_m\);
- \(\chi_m(x)\) is a smooth cutoff around island \(m\);
- \(h_g(t)\) turns on after group \(g\) has grown;
- the late group is the main diagnostic group.

Suggested smooth cutoff:

\[
\chi_m(x)
=
\exp\left(
-\frac{d_{\mathbb T}(x,c_m)^2}{2\sigma_b^2}
\right),
\qquad
\sigma_b \approx 0.20.
\]

Suggested late activation for the compressive readout:

\[
h_{\rm late}(t)
=
\frac12
\left[
1+\tanh\left(\frac{t-t_{\rm comp}}{\delta_{\rm comp}}\right)
\right],
\qquad
t_{\rm comp}\approx 0.95,\quad \delta_{\rm comp}\approx 0.03.
\]

Start with constant diffusion. Do **not** introduce spatially variable diffusion first.

### 3.2 Why use drift before variable diffusion

A deterministic drift \(b(t,x)\) is cleaner than spatially variable diffusion.

Avoid variable \(\sigma(x)\) / \(D(x)\) unless the Fokker–Planck form is written carefully. For example,

\[
\partial_t u=\nabla\cdot(D(x)\nabla u)
\]

is not the same as the Fokker–Planck equation generated by

\[
dX=\sqrt{2D(X)}\,dW
\]

without an Itô correction. This will distract the paper and invite reviewer questions.

Use constant \(D\) plus compressive drift unless there is a strong reason otherwise.

---

## 4. Compressive-readout pilot design

This is a pilot only. Do not run large production until the single-seed and two-seed pilots clearly pass.

Base parameters:

```text
M = 16
G = 4
domain = [-pi, pi]^2
sigma = 0.16
D = 0.005 or 0.01
T = 1.2
tau = 1e-3
N0 = 2e4
K = 64
grid = 512
amp_var = 0.0 first
windows = [0.00,0.35], [0.25,0.60], [0.50,0.90], [0.80,1.20]
delta = 0.03
lambda = 13
beta = 0.1
```

Compression parameters to pilot:

```text
kappa = 4, 8, 12
sigma_b = 0.20
t_comp = 0.95
delta_comp = 0.03
compress only late group first
```

Methods:

```text
weighted
weighted_ess_resample
weighted_ess_resample_costmatched
minvar_branch
```

Use the same initial particles and Brownian increments across same-budget methods. For the cost-matched weighted+ESS baseline, choose

\[
N_0^{cm}
=
\left\lfloor
\frac{C_{\rm branch}}{T/\tau}
\right\rceil ,
\qquad
C_{\rm branch}
=
\sum_n N_{\rm act}^{\rm branch}(t_n).
\]

---

## 5. Primary metrics for compressive-readout multi-island

Do **not** use per-island mass `E_m` as the primary metric. It can be reported only as a sanity check.

Primary late-island metrics should be local field / shape metrics:

### 5.1 Local window L2

For each late island \(m\), define a small window \(W_m\), e.g.

\[
W_m=\{x:d_{\mathbb T}(x,c_m)\le R_W\},
\qquad
R_W\approx 0.25 .
\]

Report:

\[
E^{\rm loc}_m
=
\frac{
\|P_K\mu_T-u_{\rm ref}(T)\|_{L^2(W_m)}
}{
\|u_{\rm ref}(T)\|_{L^2(W_m)}
}.
\]

Primary columns:

```text
mean_late local L2
max_late local L2
# late islands with local L2 > threshold
```

### 5.2 Local peak error

Report:

\[
E^{\rm peak}_m
=
\frac{
\left|
\max_{x\in W_m} P_K\mu_T(x)
-
\max_{x\in W_m} u_{\rm ref}(T,x)
\right|
}{
\max_{x\in W_m} u_{\rm ref}(T,x)
}.
\]

Primary columns:

```text
mean_late peak error
max_late peak error
```

### 5.3 Narrow Gaussian observable

Use a narrow test function centered on each island:

\[
\psi_m(x)
=
\exp\left(
-\frac{d_{\mathbb T}(x,c_m)^2}{2\sigma_{\rm obs}^2}
\right),
\qquad
\sigma_{\rm obs}=0.04 \text{ or } 0.05 .
\]

Report:

\[
E^{\rm obs}_m
=
\frac{
\left|
\langle\psi_m,\mu_T\rangle
-
\int \psi_m(x)u_{\rm ref}(T,x)\,dx
\right|
}{
\int \psi_m(x)u_{\rm ref}(T,x)\,dx
}.
\]

This is more sensitive to the compressed core than the half-height disk mass.

### 5.4 Local count / local ESS

For weighted methods:

\[
N_{\rm eff}(W_m)
=
\frac{(\sum_{i:X_i\in W_m}w_i)^2}
{\sum_{i:X_i\in W_m}w_i^2}.
\]

For branching:

```text
local count in W_m
```

Report:

```text
min late local effective count
median late local effective count
```

### 5.5 Sanity only: mass E_m

Keep:

\[
E_m=
\frac{
\left|
\mu_T(B_m)-\int_{B_m}u_{\rm ref}(T,x)dx
\right|
}{
\int_{B_m}u_{\rm ref}(T,x)dx
}.
\]

But label it as:

```text
mass sanity check, not the primary metric
```

Do not use it to decide whether branching wins.

---

## 6. Success criteria for compressive-readout pilot

A compressive-readout multi-island experiment is worth promoting only if it satisfies all of:

```text
1. minvar_branch beats raw weighted, weighted+ESS, and cost-matched weighted+ESS
   on late mean local L2.

2. minvar_branch beats raw weighted, weighted+ESS, and cost-matched weighted+ESS
   on late max local L2 or late peak error.

3. minvar_branch has much larger late local count than weighted local ESS.

4. particle-steps are reported, and branching does not require absurd growth.
   Aim for final Nact <= 8 N0 unless there is a strong reason.

5. The result is stable over at least 4 seeds before considering 8-seed production.

6. The improvement is visible in one clean figure with a metric-selected inset.
```

If these fail, stop. Do not tune indefinitely.

A failed compressive-readout pilot should be recorded in `parameter_log.md` but should not enter the manuscript.

---

## 7. Figure design for any successful compressive-readout result

Main figure should be compact:

```text
reference
weighted+ESS
cost-matched weighted+ESS
minvar branching
```

Use a shared color scale.

Add one magnifier/inset around the worst late island for the cost-matched ESS baseline:

\[
m_\ast
=
\arg\max_{m\in\mathcal G_{\rm late}}
E^{\rm loc}_{m,\;{\rm costmatched}} .
\]

Do not choose the inset by eye.

The table should be short:

```text
method
particle-steps
global L2
mean_late local L2
max_late local L2
mean_late peak error
max_late peak error
min_late local eff/count
Nact final
mass E_m sanity
```

Do not include a large report-style table in the main text.

---

## 8. Manuscript policy

### 8.1 Main paper should not mention failed multi-island unless necessary

Do not add a paragraph explaining the staged multi-island failure in the main paper. It is not needed and weakens the story.

The repository can keep the negative record. The manuscript should only report experiments that support the paper claims.

### 8.2 Narrow the “outperforms weighted particles” claim

Avoid universal wording such as:

```text
Branching outperforms raw and resampled weighted particles at equal cost.
```

Prefer:

```text
On controlled localized-growth benchmarks, branching outperforms raw weighted
particles and the tested global resampling baselines in reconstructed-field accuracy
at matched particle-step work.
```

This is more accurate and safer.

### 8.3 Current main numerical story

Keep the numerical section organized as:

```text
5.1 MMS verification
5.2 stationary localized growth: main branch-vs-weighted comparison
5.3 switching growth: lineage diversity under moving source
5.4 2D Keller-Segel: coupled solver + pre-singular concentration
5.5 3D Keller-Segel focusing
5.6 6D kinetic Keller-Segel
Appendix: Allen-Cahn, logistic KS, high-dimensional dense reconstruction, negative multi-island records
```

Do not insert multi-island into the main story unless the compressive-readout version clearly passes the success criteria.

### 8.4 Keller–Segel language

Do not claim a blow-up time from reconstructed peaks or reconstructed \(L^2\).

Allowed language:

```text
pre-singular concentration
focusing indicator
resolution-gap time
bandwidth-sensitive reconstructed peak
reconstruction-free core radius
```

Not allowed:

```text
we compute the blow-up time
we resolve the singularity
the peak convergence determines blow-up
```

### 8.5 Cross-species injection

For parabolic–parabolic Keller–Segel with

\[
v_t=\Delta v+u-v,
\]

use the injection kernel:

\[
\mu_v^{n+1}
=
e^{-\tau}\mu_v^\ast
+
(1-e^{-\tau})\mu_u^\ast .
\]

Do not implement or describe this as a multiplicative \((u-v)/v\) branching rate on existing \(v\)-particles.

State clearly:

```text
existing v-particles decay; transported u-particles inject new v-particles
at their own locations.
```

---

## 9. Repository hygiene tasks

### 9.1 CLAUDE.md

Keep this file specific to `Lyuliyao/equal-weight-branching`.

Do not overwrite it with instructions from unrelated repositories.

### 9.2 staged multi-island files

Ensure the staged multi-island README and parameter log say:

```text
diagnostic / negative-result record only
not used in the main paper
```

Remove any claim that it replaces §5.2 or proves branching is better.

### 9.3 Root README

Update the root README so that:

- paper reproduction commands match the current manuscript;
- staged and static multi-island are listed as `not in paper / diagnostic record`;
- `pp_injection`, `ldg_comparison`, and any `resolution_hybrid` directory are correctly categorized;
- commands do not point to obsolete `concentration_ldg` if the manuscript uses the newer LDG-aligned parabolic–parabolic comparison.

### 9.4 Data traceability

Every number in a main-text table must have a corresponding committed or documented CSV in `reference_results/`.

Pay special attention to the localized-growth cost-matched row:

```text
weighted + resample (ESS), N0 = 3.8e4, particle-steps = 1.9e7
```

If the raw CSV / config for this row is not archived, either archive it or remove the row.

### 9.5 Figures

Every main figure must be regenerable from saved data only:

```text
plot_data/*.npz
metrics_used.csv
config.json
manifest.json
figure.pdf
figure.png
```

Plot scripts must not rerun the solver.

---

## 10. Working protocol for Claude Code

Before running any expensive job:

1. Write down the purpose of the experiment in one paragraph.
2. State which manuscript claim it supports.
3. State the success criteria before seeing the result.
4. Run a smoke test.
5. Run a 1-seed pilot.
6. Only then run 4-seed or 8-seed production.

After every run:

1. Update `parameter_log.md`.
2. Record failed pilots honestly.
3. Do not hide negative results.
4. Do not move a result into the manuscript unless it passes the stated success criteria.
5. Report particle-steps and active counts whenever comparing to resampling.

---

## 11. Guardrails

- Do not add experiments that do not strengthen the paper.
- Do not claim branching wins a metric unless the table says so.
- Do not use per-island mass `E_m` as the main argument for branching.
- Do not claim blow-up time from reconstructed peak or \(L^2\).
- Do not introduce variable diffusion without writing the correct Fokker–Planck / Itô form.
- Do not hide cost; always report particle-steps and active counts.
- Do not hide failed pilots; keep them in logs.
- Do not let the numerical section become a report.
- Keep the main paper concise and claim only what the data supports.
