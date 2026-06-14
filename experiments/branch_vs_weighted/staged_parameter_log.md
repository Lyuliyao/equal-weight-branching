# Staged multi-island — parameter / pilot log

Tuning log for `staged_multi_island.py` (paper §5.2). The success criteria are in
the repo `CLAUDE.md`. This log records every pilot, including failed ones.

## Design decisions

- **Groups (G=4):** 2×2 checkerboard sub-lattices, `group = (row%2)*2 + (col%2)`,
  so each activation stage turns on four islands spread across the domain
  (separated regions, not a contiguous block). Late group = group 3 (islands at
  rows/cols both odd: m = 5, 7, 13, 15), window the latest.
- **Uniform amplitudes (`amp_var=0`, a_m=1).** The static benchmark was confounded
  by amplitude-driven branching-process variance (weak islands had few ancestors
  and large per-island mass error regardless of method). The *staged* mechanism is
  about activation **timing**, so uniform amplitudes are the correct, cleaner
  design: every late island grows the same amount, isolating the lineage effect.
  (`amp_var` is a CLI/JSON knob; the variation a_m=1+0.25 sin can be re-enabled.)
- **Stratified initial cloud:** deterministic per-island quota inside `B_m`
  (equal across islands) + stratified background, shared across all same-budget
  methods. Removes initial-ancestor imbalance; same uniform measure u0≡1.
- **Reaction is time-dependent** with smooth tanh windows; reference and particles
  both evaluate the rate at the step midpoint `(s-0.5)τ`.

## Reference growth probe (grid 256², T=1.2, a_m=1+0.25 sin)

| λ, β | total growth | late-island M (mean / min) |
|---|---|---|
| 8, 0.8 | 0.6× | 0.40 / 0.25 |
| 9, 0.8 | 0.7× | 0.54 / 0.32 |
| 10, 0.8 | 0.7× | 0.74 / 0.41 |
| 9, 0.6 | 0.8× | 0.69 / 0.41 |
| 9, 1.0 | 0.5× | 0.43 / 0.25 |

Observation: the constant `−β` decays the large background, so **total mass
decays (<1×)** — branching will not explode (good for criterion 5). The late
islands grow ~4–7× locally from their `B_m` start mass ≈ 0.11. This bounds the
achievable late-island particle count; pilots below measure the actual count and
whether branching beats the cost-matched ESS baseline on the late islands.

## Pilots

### Pilot 1 — λ=9, β=0.8, N0=1e4, T=1.2 (grid 256, K=32, seed 0) — FAILED

Late-group metrics (late mean E_m / max / local L2 / min late local eff):
- weighted: 0.088 / 0.113 / 0.472 / 18 (nESS 0.305, Nact 1.0×)
- weighted_ess_resample: 0.271 / 0.510 / 0.706 / 28 (nESS 0.821)
- cost-matched ESS: 0.250 / 0.444 / 0.687 / 50 (Nact 0.8×)
- minvar_branch: 0.287 / 0.605 / 0.834 / 113 (nESS 1.0, Nact 0.7×)

FAIL: with β=0.8 the total mass DECAYS (0.7×). Branching sheds particles and
loses on every late metric; raw weighted keeps all particles and is *most*
accurate on late mass (0.088). The decaying regime is wrong — branching needs a
GROWING late region (so it creates local particles and weighted develops
degeneracy). Reason: constant `−β` decay dominates the localized island growth.
Fix: drop β to the switching-benchmark regime (β≈0.1) so islands accumulate.

### Reference growth probe, low β (grid 256², T=1.2, a_m=1)

| λ, β | total growth | late M (uniform) | late local growth |
|---|---|---|---|
| 10, 0.1 | 1.6× | 1.59 | 14.3× |
| 10, 0.3 | 1.3× | 1.25 | 11.3× |
| 9, 0.2 | 1.3× | 1.05 | 9.5× |
| 8, 0.1 | 1.3× | 0.89 | 8.0× |

Low β gives moderate total growth (no explosion) and ~14× late-island local
growth (uniform across late islands). Pilot 2 tests λ=10, β=0.1.

### Pilot 2 — λ=10, β=0.1, N0=1e4, T=1.2 (grid 256, K=32, seed 0) — partial

Late metrics (late mean E_m / max / local L2 / min late local eff / nESS / Nact):
- weighted: 0.095 / 0.122 / 0.481 / 17 / 0.223 / 1.0×
- weighted_ess_resample: 0.093 / 0.228 / 0.726 / 147 / 0.992 / 1.0×
- cost-matched ESS: 0.220 / 0.386 / 0.559 / 195 / 0.991 / 1.28×
- minvar_branch: 0.221 / 0.333 / **0.434** / 269 / 1.0 / 1.6×

Branching WINS late local L2 (field resolution: 0.434, best) and has the highest
late local count, but raw weighted/resample still win late MASS E_m (the forgiving
per-island integral). The growth is too mild for weighted to degenerate WITHIN the
late islands. Push λ.

### Pilot 3 — λ=13, β=0.1, N0=1e4, T=1.2 (grid 256, K=32, seed 0) — STRONG

- weighted: 0.114 / 0.147 / 0.510 / 14 / nESS 0.091 / 1.0×
- weighted_ess_resample: 0.150 / 0.336 / 0.512 / 98 / 0.855 / 1.0×
- cost-matched ESS: 0.130 / 0.392 / 0.430 / 141 / 0.860 / 1.67×
- minvar_branch: **0.109** / 0.249 / **0.283** / 750 / 1.0 / 2.6×

Branching WINS late mean E_m (0.109, best of all four) and late local L2 (0.283,
best by a wide margin), and beats BOTH ESS baselines on every late metric.
Criteria 1 (resample nESS 0.855 ≥ 0.5) and 2 (resample late min ESS 98 ≤ 100–300)
hold; Nact 2.6× (moderate). Remaining gap: raw weighted late MAX E_m (0.147) <
branching (0.249) — raw weighted keeps all particles so its per-island MASS is
low-variance, but it is globally degenerate (nESS 0.09) and has the worst late
field reconstruction (local L2 0.510). Push λ=15 to sharpen the late islands so
weighted degenerates within them and the late count rises toward 2000.

### Pilot 4 — λ=15, β=0.1, N0=1e4 (grid 256, seed 0) — OVERSHOOT

- weighted: 0.124 / 0.164 / 0.529 / nESS 0.055
- weighted_ess_resample: 0.129 / 0.321 / 0.544 / 0.900
- cost-matched ESS: 0.509 / 0.693 / 0.978 / 0.781 (degenerates badly at high λ)
- minvar_branch: 0.178 / 0.389 / 0.433 / minLateEff 1135 / Nact 3.9×

λ=15 is WORSE for branching mean E_m (0.178 vs 0.109 at λ=13): sharper late islands
mean more branching generations and more reproduction variance. Reference is still
converged (256-vs-512 ≤1% at both λ=13,15; late mass uniform across late islands,
u_min 0.89 > 0 — no negativity).

## Decision: production at λ=13, β=0.1

λ=13 is the sweet spot. Single-seed (pilot 3) and the reference-convergence check
support it. Branching wins late mean E_m and late local L2, and beats BOTH ESS
baselines (incl. cost-matched) on every late metric, at moderate growth (2.6×).
Raw weighted keeps a slightly lower late-MAX per-island mass error only by never
resampling, at the cost of a catastrophic global nESS (~0.09) and the worst late
field reconstruction (local L2 0.51) — which is exactly the weighted dilemma the
paper highlights. Production: N0=2e4, K=64, grid 512, 8 seeds.

## Production result (8 seeds, N0=2e4, K=64, grid 512) — CRITERION 4 FAILS

| method | global L2 | all mean E_m | late mean E_m | late max E_m | late local L2 | min late count | Nact |
|---|---|---|---|---|---|---|---|
| weighted | 1.213 | 0.148 | 0.152 | 0.258 | 1.126 | 28 | 1.0× |
| weighted+ESS | 0.494 | 0.133 | 0.169 | 0.261 | 0.589 | 213 | 1.0× |
| **cost-matched ESS** | 0.384 | **0.100** | **0.137** | **0.234** | **0.465** | 368 | 1.0× |
| minvar_branch | **0.343** | 0.145 | 0.202±0.089 | 0.381 | 0.511 | **1517** | 2.6× |

**The single-seed pilots were misleading.** With proper 8-seed statistics,
branching has the WORST late mean E_m (0.202) and the highest seed variance
(±0.089). The **cost-matched weighted+ESS baseline beats branching on every
per-island metric** (all/late mean & max E_m, late local L2). Branching wins only
the GLOBAL L2 (0.343, consistent with the single-peak benchmark) and keeps the
highest late-island local count (1517 vs 368) — but the extra particles do not
lower the per-island MASS error, because branching mass carries reproduction
variance set by the ancestor count, not the final count.

**Conclusion (honest, per CLAUDE.md guardrails).** Success criterion 4 (branching
beats cost-matched ESS on late mean/max E_m and late local L2) FAILS after
reasonable tuning (λ∈{9,10,13,15}, β∈{0.1,0.8}). The deep reason is fundamental:
per-island *mass* is a forgiving integral that favors weighted particles, and at
matched particle-step cost the larger weighted budget wins. Branching's genuine
advantage is GLOBAL L2 / peak resolution, already shown by the single-peak (§5.2)
and switching (§5.3) benchmarks. Therefore the multi-island benchmark is DEMOTED:
it stays in `reference_results/` as a diagnostic, not as a main-paper "branching
wins" benchmark.
