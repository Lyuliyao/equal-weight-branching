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

## 1. Hard stop: do NOT continue multi-island

The static, staged, and compressive-readout multi-island lines have all been tested enough for this revision.

### Static multi-island

The static separated-island benchmark showed a useful diagnostic fact:

```text
global ESS can look healthy while local island effective sample size collapses.
```

However, it did **not** show that branching has smaller per-island mass error than ESS resampling. It is a diagnostic record only.

### Staged multi-island

The staged multi-island benchmark was designed to activate separated islands at different times. The 8-seed production result did **not** satisfy the success criterion. Branching achieved the lowest global `L2` and the highest late-island local particle count, but cost-matched weighted+ESS won the per-island mass metrics:

```text
cost-matched ESS late mean E_m < branching late mean E_m
cost-matched ESS late local L2 <= branching late local L2
```

### Compressive-readout multi-island

The compressive-readout variant added a deterministic compressive drift and changed the primary metrics to local field `L2`, local peak error, and narrow-Gaussian observables. It also failed as a main-paper benchmark:

```text
κ = 8, D = 0.01:
    branching has much larger late local count but does not win local field metrics.

κ = 12, D = 0.005:
    the deterministic reference / band-limited reconstruction under-resolves the core.
```

Conclusion:

```text
Do NOT continue multi-island experiments for this revision.
Do NOT tune λ, β, windows, amplitudes, κ, or diffusion to rescue multi-island.
Do NOT insert multi-island into the main paper.
Keep static/staged/compressive multi-island only as diagnostic or negative records in the repo.
```

The correct interpretation is:

```text
Local particle count advantage is not the same as lower per-island mass or local-field error.
Weighted particles can estimate some low-dimensional local integrals well using continuous weights.
Branching’s main advantage is demonstrated more cleanly by the single-peak and switching-growth benchmarks.
```

Remove or rewrite any sentence saying:

```text
"multi-island replaces §5.2"
"staged multi-island proves branching is better"
"compressive-readout multi-island should be promoted to the paper"
"branching has the smallest max E_m"
```

---

## 2. Current highest-priority task: improve §5.3 switching-growth Figure 6

Do **not** run new solver jobs for this task. The switching experiment already saves the needed data:

```text
reference_results/switch/metrics.csv
reference_results/switch/ancestors_at_switch.csv
reference_results/switch/fields_seed*.npz
```

The problem is visualization: the current full-domain Figure 6 is too small to show what happens in the old growth region \(B_A\) and the new growth region \(B_B\). The figure must visually explain Table 6.

### 2.1 Purpose of the switching-growth example

The switching-growth example is not a multi-island diagnostic. It has one clear paper purpose:

> Show a structural failure mode of global resampling. A fixed particle budget can be resampled into the first-stage region \(B_A\), leaving insufficient lineage diversity for the later second-stage region \(B_B\). Branching does not globally discard background lineages and can reconstruct the new growth region more accurately.

The key table result is:

```text
ESS resampling is best in B_A, the old region.
Branching is better in B_B, the new region.
Branching retains about 95% distinct ancestors at the switch, while global resampling retains about 63--68%.
```

The figure should make this mechanism visible.

---

## 3. Required Figure 6 redesign: full domain + A/B microscope zooms

Create or update:

```text
experiments/branch_vs_weighted/plot_switch.py
```

The plot script must read saved data only. It must not rerun the solver.

### 3.1 Figure layout

Replace the current four-panel full-domain figure with a 4-column × 3-row figure:

```text
columns:
    reference | weighted | weighted+ESS | min.-variance branching

row 1:
    full domain

row 2:
    microscope zoom around old growth region A

row 3:
    microscope zoom around new growth region B
```

Do not include Poisson in the main figure unless there is a very strong layout reason. Table 6 already reports Poisson. The figure should be visually clean.

### 3.2 Regions and zoom windows

Use the switching benchmark parameters:

```text
x_A = (-1.2, 0)
x_B = ( 1.2, 0)
sigma = 0.25
eta = 0.5
```

The diagnostic regions are:

\[
B_A = \{G_\sigma(x;x_A) \ge 0.5\},
\qquad
B_B = \{G_\sigma(x;x_B) \ge 0.5\}.
\]

The half-height radius is:

\[
R_B = \sigma\sqrt{2\log 2} \approx 0.294.
\]

Use microscope windows with

```text
R_zoom = 0.55 or 0.60
A zoom: x in [-1.2-R_zoom, -1.2+R_zoom], y in [-R_zoom, R_zoom]
B zoom: x in [ 1.2-R_zoom,  1.2+R_zoom], y in [-R_zoom, R_zoom]
```

Draw the diagnostic set boundaries \(B_A\) and \(B_B\) in the full-domain row and in the corresponding zoom rows.

Use distinct but non-distracting markers:

```text
B_A: circle boundary
B_B: square or circle boundary with different linestyle
```

The reader must immediately understand that Table 6 local errors are computed in these marked regions.

### 3.3 Color scales

Use row-wise shared color scales:

```text
row 1: one shared color scale for all full-domain panels
row 2: one shared color scale for all A-zoom panels
row 3: one shared color scale for all B-zoom panels
```

Do not use a separate color scale for each method; that would hide accuracy differences.

Do not force the zoom rows to use the full-domain color scale if it makes local structure invisible. The zoom rows are microscopes and may use their own row-wise scale.

### 3.4 Local error labels

Add small unobtrusive annotations in the A and B zoom rows:

```text
A: local L2 = ...
B: local L2 = ...
```

Use the final-time mean values from `metrics.csv` over all seeds, matching Table 6:

```text
weighted+ESS:
    B_A L2 ≈ 0.162
    B_B L2 ≈ 0.297

min.-variance branching:
    B_A L2 ≈ 0.170
    B_B L2 ≈ 0.174
```

The exact values should be computed from `metrics.csv`, not typed by hand.

For the reference column, leave the local-error label blank or write `reference`.

### 3.5 Output requirements

The plotting script must save:

```text
reference_results/switch/switch_snapshots_zoom.pdf
reference_results/switch/switch_snapshots_zoom.png
reference_results/switch/plot_data/switch_snapshots_zoom.npz
paper/figure/switch_snapshots.pdf
```

If the current paper uses `paper/figure/switch_snapshots.pdf`, overwrite it with the improved version so the manuscript picks it up without a LaTeX filename change.

The saved `plot_data` must contain at least:

```text
xs or grid axes
reference field
weighted field
weighted_ess field
minvar field
cA, cB, sigma, eta, R_B, R_zoom
row-wise vmin/vmax
final-time local errors used for annotations
```

The plot script must be idempotent: rerunning it from saved CSV/NPZ should reproduce the same figure without running the solver.

---

## 4. Figure 6 caption target

Replace the current caption with a mechanism-focused caption like:

```latex
Switching-growth benchmark at \(T=1.2\). Top row: full-domain reconstructed fields after the source has moved from \(x_A\) to \(x_B\). Middle and bottom rows: magnified views of the old growth region \(B_A\) and the new growth region \(B_B\), with the local diagnostic sets marked. Global ESS resampling gives the smallest error in \(B_A\), where it has concentrated the fixed particle budget before the switch, but it is less accurate in the second-stage region \(B_B\). Equal-weight branching preserves more surviving lineages and reconstructs the new growth region more accurately, consistent with Table~\ref{tab:switch}.
```

Do not write only that “the fields are smoother.” The key message is old-region versus new-region behavior and lineage diversity.

---

## 5. §5.3 text target

The text immediately before or after Table 6 should emphasize:

```text
The second-stage region B_B is the discriminating diagnostic.
ESS-triggered resampling is best in B_A because B_A was the first growth region.
Branching is better in B_B because it preserves lineage diversity and does not globally discard future-growth lineages.
```

A good paragraph is:

```text
The fixed-source benchmark is favorable to global resampling because the important region never moves. The switching-source test exposes a different failure mode. Resampling based on a global ESS can equalize weights and improve the old region \(B_A\), but it does so by committing a fixed budget of lineages to the region that was important before the switch. The second-stage region \(B_B\) is therefore the discriminating diagnostic. Table~\ref{tab:switch} and Fig.~\ref{fig:switch} show that ESS-triggered resampling has the smallest \(B_A\) error but substantially larger \(B_B\) error, while equal-weight branching retains more lineages at the switch and reconstructs \(B_B\) more accurately.
```

Keep the table. The new figure should visually support it.

---

## 6. L2-error definition to keep in the paper

The reported `L2` errors are relative errors of the reconstructed physical field, not direct distances between raw empirical measures.

Use this definition in the numerical section:

\[
E_{L^2}(t)
=
\frac{
\left(\Delta x^d\sum_j
|u_h^{N,K}(t,x_j)-u_{\rm ref}(t,x_j)|^2\right)^{1/2}
}{
\left(\Delta x^d\sum_j
|u_{\rm ref}(t,x_j)|^2\right)^{1/2}
},
\qquad
u_h^{N,K}=P_K\mu_t^N.
\]

For local regions \(B\), use the same formula with the sum restricted to grid cells in \(B\):

\[
E_{L^2(B)}(t)
=
\frac{
\left(\Delta x^d\sum_{x_j\in B}
|u_h^{N,K}(t,x_j)-u_{\rm ref}(t,x_j)|^2\right)^{1/2}
}{
\left(\Delta x^d\sum_{x_j\in B}
|u_{\rm ref}(t,x_j)|^2\right)^{1/2}
}.
\]

This definition should be stated once, near the start of the numerical section. Do not leave `L2 error` ambiguous.

---

## 7. Manuscript policy

### 7.1 Main paper should not mention failed multi-island unless necessary

Do not add a paragraph explaining the static/staged/compressive multi-island failures in the main paper. It is not needed and weakens the story.

The repository can keep the negative records. The manuscript should only report experiments that support the paper claims.

### 7.2 Narrow the “outperforms weighted particles” claim

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

### 7.3 Current main numerical story

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

Do not insert multi-island into the main story.

### 7.4 Keller–Segel language

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

### 7.5 Cross-species injection

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

## 8. Repository hygiene tasks

### 8.1 CLAUDE.md

Keep this file specific to `Lyuliyao/equal-weight-branching`.

Do not overwrite it with instructions from unrelated repositories.

### 8.2 Multi-island files

Ensure static, staged, and compressive multi-island README / parameter logs say:

```text
diagnostic or negative-result record only
not used in the main paper
```

Remove any claim that they replace §5.2 or prove branching is better.

### 8.3 Root README

Update the root README so that:

- paper reproduction commands match the current manuscript;
- static/staged/compressive multi-island are listed as `not in paper / diagnostic record`;
- `pp_injection`, `ldg_comparison`, and any `resolution_hybrid` directory are correctly categorized;
- section numbers match the final manuscript.

### 8.4 Data traceability

Every number in a main-text table must have a corresponding committed or documented CSV in `reference_results/`.

The localized-growth cost-matched row must remain traceable:

```text
weighted + resample (ESS), N0 = 3.8e4, particle-steps = 1.9e7
```

### 8.5 Figures

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

## 9. Working protocol for Claude Code

Before running any expensive job:

1. Write down the purpose of the experiment in one paragraph.
2. State which manuscript claim it supports.
3. State the success criteria before seeing the result.
4. Run a smoke test.
5. Run a 1-seed pilot.
6. Only then run 4-seed or 8-seed production.

For the current Figure 6 task, no expensive job is needed; use saved switching-growth data only.

After every run or figure-generation change:

1. Update the relevant README or notes.
2. Record failed pilots honestly.
3. Do not hide negative results.
4. Do not move a result into the manuscript unless it passes the stated success criteria.
5. Report particle-steps and active counts whenever comparing to resampling.

---

## 10. Guardrails

- Do not continue multi-island.
- Do not add experiments that do not strengthen the paper.
- Do not claim branching wins a metric unless the table says so.
- Do not use per-island mass `E_m` as the main argument for branching.
- Do not claim blow-up time from reconstructed peak or \(L^2\).
- Do not introduce variable diffusion without writing the correct Fokker–Planck / Itô form.
- Do not hide cost; always report particle-steps and active counts.
- Do not hide failed pilots; keep them in logs.
- Do not let the numerical section become a report.
- Keep the main paper concise and claim only what the data supports.
