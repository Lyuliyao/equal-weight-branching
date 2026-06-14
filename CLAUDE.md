# CLAUDE.md — revision priorities for the branching-particle paper

This repository is being revised for the paper. Every code change, experiment, figure, table, and paragraph should serve the paper's main contribution:

> Equal-weight branching is a better reaction representation than weighted particles when localized reaction creates mass, because resolution follows local mass creation rather than a global fixed-weight budget.

Do not add experiments merely because they are interesting. If a result does not strengthen this story, demote it to appendix/supporting notes or remove it from the main manuscript.

## Current highest-priority task: replace the static multi-island benchmark by a staged multi-island benchmark that shows branching is better

The current static separated-island result shows a useful diagnostic fact: global ESS can look healthy while some local islands are under-resolved. However, it does **not** show that branching has smaller per-island mass error than ESS resampling. Therefore it should not be used as a main-paper benchmark in its current form.

The main-paper multi-island experiment must be redesigned to prove the paper-level claim:

> When separated growth regions become important at different times, global ESS-triggered resampling can keep a healthy global ESS but lose local lineages in later regions. Equal-weight branching creates particles where the source turns on and gives smaller late-island mass/local-field errors at comparable particle-step cost.

### Do not optimize for a diagnostic-only story

Avoid a main-text claim of the form:

```text
global ESS is not a sufficient local diagnostic
```

unless it is paired with a quantitative accuracy win for branching. The paper needs:

```text
branching beats weighted and ESS-resampled weighted particles on the quantities that matter.
```

A diagnostic-only static island example can remain in `reference_results/` or appendix notes, but it should not occupy main-paper space unless branching is also better on the key error columns.

## Staged multi-island PDE design

Use a time-dependent separated-growth-island reaction:

\[
\partial_t u = D\Delta u + r(t,x)u,
\]

\[
r(t,x)=\lambda\sum_{g=1}^G s_g(t)\sum_{m\in\mathcal G_g} a_m
\exp\!\left(-\frac{d_{\mathbb T}(x,c_m)^2}{2\sigma^2}\right)-\beta.
\]

Use `M=16` centers on a `4 x 4` grid. Split the islands into `G=4` activation groups. Prefer groups that are spatially separated, e.g. checkerboard-like or column/row groups, but ensure each stage activates separated regions rather than one contiguous block.

Use smooth window functions:

\[
s_g(t)=\frac12\left[\tanh\frac{t-t_{g,on}}{\delta}-\tanh\frac{t-t_{g,off}}{\delta}\right].
\]

A good initial schedule to test:

```text
T = 1.2
windows = [0.00,0.35], [0.25,0.60], [0.50,0.90], [0.80,1.20]
delta = 0.03
sigma = 0.16
D = 0.01
tau = 1e-3
N0 = 2e4
K = 64
lambda in [8,10]
beta in [0.5,1.0]
```

The schedule can be tuned, but the tuning must be logged in `parameter_log.md` with short reasons. Do not hide failed pilots.

## Initial condition: use stratified uniform particles

The previous static experiment was dominated by initial local-ancestor noise. That is not the paper's target. The comparison should isolate the reaction representation.

Use `u0 == 1` on the torus but sample it using a stratified uniform initial cloud:

1. Divide the torus according to the `4 x 4` island cells.
2. Within each island cell, stratify the diagnostic disk
   \[
   B_m = \{x: \exp(-d_{\mathbb T}(x,c_m)^2/(2\sigma^2)) \ge 1/2\}
   \]
   and its complement.
3. Assign deterministic quotas proportional to area, so every island starts with the same number of particles inside `B_m` up to rounding.
4. Fill the background uniformly by stratified quotas.
5. Use exactly the same stratified initial particles for all methods.

This is not biasing the solution. It is a lower-variance discretization of the same uniform initial measure and removes an irrelevant source of random imbalance.

The manuscript should say:

```text
We use a stratified uniform initial cloud to remove irrelevant initial-island sampling imbalance and isolate the effect of the reaction representation. All methods use the same stratified initial positions and Brownian increments.
```

## Methods to compare

Keep the main benchmark small and clean. Use only:

1. `weighted`: raw weighted particles.
2. `weighted_ess_resample`: systematic resampling triggered by global normalized ESS below 0.5.
3. `weighted_ess_resample_costmatched`: same resampling baseline with initial particle count chosen so that integrated particle-steps match branching.
4. `minvar_branch`: equal-weight minimum-variance branching.

Do **not** put Poisson in the main table unless it adds clarity. It can remain in appendix or `reference_results`.

### Cost matching

Report integrated particle-steps:

\[
C = \sum_n N_{act}(t_n).
\]

If branching has cost `C_branch`, choose the cost-matched weighted+ESS particle count as

\[
N_0^{cm}=\left\lfloor \frac{C_{branch}}{T/\tau}\right\rceil.
\]

The staged benchmark is only paper-strong if `minvar_branch` beats both ESS baselines, including the cost-matched one, on the late-island metrics.

## Primary metrics

Do not let this experiment become another large report. The main-paper table should contain at most the following columns:

```text
method
particle-steps
global L2
mean_all E_m
max_all E_m
# all islands with E_m > 20%
mean_late E_m
max_late E_m
# late islands with E_m > 20%
max_late local L2(B_m)
min_m local effective count
final / mean active count
```

Definitions:

\[
E_m = \frac{|\mu_T(B_m)-\int_{B_m}u_{ref}(T,x)dx|}
           {\int_{B_m}u_{ref}(T,x)dx}.
\]

For weighted methods, local effective count in `B_m` is

\[
N_{eff}(B_m)=\frac{(\sum_{i:X_i\in B_m}w_i)^2}{\sum_{i:X_i\in B_m}w_i^2}.
\]

For branching, local effective count is the equal-weight particle count in `B_m`.

The late group is the islands whose activation window includes the final stage, e.g. group 4. These are the discriminating islands.

## Success criteria

Before running production, run pilots and tune `lambda`, `beta`, and activation windows so the following target behavior is achieved:

```text
1. weighted_ess_resample final global nESS >= 0.5
   The global diagnostic looks healthy.

2. weighted_ess_resample late-group min local ESS <= 100--300
   Local degeneracy remains in the late islands.

3. minvar_branch late-group min local count >= 2000
   Branching keeps local resolution where the late source turns on.

4. minvar_branch beats raw weighted, weighted_ess_resample, and cost-matched weighted_ess_resample
   in late-group mean E_m, late-group max E_m, and late-group local L2(B_m).

5. minvar_branch does not require an extreme particle explosion.
   Aim for final N_act about 4--8 times N0, not 50--100 times N0.

6. All methods share initial particles and Brownian increments.
```

If these criteria fail after reasonable tuning, do **not** force the result into the main paper. Keep the existing single-peak and switching-growth results as the main evidence, and demote the multi-island experiment to appendix or a negative diagnostic note.

## Figure design

The main text should have one multi-island figure and one compact table.

Preferred figure panels:

```text
reference
weighted+ESS
cost-matched weighted+ESS
minvar branching
```

Use one diagnostic-selected inset / magnifier around the worst late island of the cost-matched ESS baseline:

\[
m_* = \arg\max_{m \in \mathcal G_{late}} E_m^{weighted+ESS,costmatched}.
\]

Do not choose the inset by eye. The caption should say it is selected by the metric.

Use shared color scale. Mark late-stage islands. Do not include many auxiliary panels in the main text.

## Manuscript text target

The staged multi-island section should support this paragraph:

```text
The single-peak benchmark shows that branching is more accurate than weighted particles and ESS resampling at matched work. The staged multi-island benchmark then shows that this advantage is not limited to one peak. When growth appears in separated regions at different times, global ESS-triggered resampling can keep a healthy global ESS while losing local lineages in the later regions. Equal-weight branching creates particles directly where the source turns on and gives smaller late-island mass and local-field errors at comparable particle-step work.
```

Remove or rewrite any sentence claiming that the existing static multi-island branching run has the smallest `max E_m`. That statement is false for the current archived static result.

## Repository / reproducibility requirements

When implementing the staged benchmark, write outputs under a new directory, do not overwrite the existing static run:

```text
experiments/branch_vs_weighted/staged_multi_island.py
experiments/branch_vs_weighted/plot_staged_multi_island.py
experiments/branch_vs_weighted/run_staged_multi_island.sb
reference_results/staged_multi_island/<run_id>/
```

The run directory must include:

```text
config.json
manifest.json
parameter_log.md
metrics_summary.csv
per_seed_metrics.csv
time_series.csv
island_masses.csv
island_local_ess.csv
late_group_metrics.csv
fields_ref.npz
fields_seed0.npz
plot_data/*.npz
figures/*.pdf
figures/*.png
README.md
```

Figures must regenerate from saved CSV/NPZ only. Plot scripts must not rerun the solver.

## Root README and paper updates

After a successful staged run:

1. Update the root `README.md` experiment map and reproduction commands.
2. Update `paper/cmame-main.tex` and rebuild `paper/cmame-main.pdf`.
3. The numerical-section introduction must mention `sec:multi_island` if the section stays in the main paper.
4. The main text must report particle-steps whenever comparing against resampling.
5. The conclusion should emphasize the paper-level point: branching is a controlled equal-weight representation of non-conservative mass creation and a practical alternative to weighted particles when resolution must follow growth.

## Guardrails

- Do not claim a blow-up time from reconstructed peaks or reconstructed L2 norms.
- Do not over-emphasize diagnostic-only details that do not strengthen the main story.
- Do not hide cost: always report particle-steps and active counts.
- Do not hide failed pilots: log them briefly in `parameter_log.md`.
- Do not use a result in the main text if the table does not support the claim.
- Keep the main paper concise. Appendix/supporting folders can hold diagnostic details.
