# Compressive-readout multi-island — parameter / pilot log (OPTIONAL, CLAUDE.md §3-7)

Tuning log for `compressive_multi_island.py`. This is the *optional* idea: add a
deterministic compressive drift that sharpens the late-island cores after growth,
so the metric becomes **local field / shape / peak / narrow-Gaussian** (CLAUDE.md
§5) rather than the forgiving per-island mass integral that favored weighted
particles in the static and staged benchmarks.

## Purpose & claim it would support

> Convert "branching keeps a higher local particle count" into a *visible local
> reconstruction-accuracy advantage*: once the late core is compressed, a method
> with too few local particles cannot reconstruct the sharp core.

It would support the §5.2/§5.3 branching-wins story (reconstructed-field accuracy)
only if it passes the §6 success criteria. **Success criterion (stated before
running):** branching beats raw weighted, weighted+ESS, and cost-matched
weighted+ESS on late mean local L2 AND on late max local L2 or late peak error,
with a much larger late local count and `Nact ≤ 8 N0`, stable over ≥4 seeds.

## Design

- Reaction = the staged 4-group schedule (λ=13, β=0.1, uniform amplitudes).
- Drift `b(t,x) = -κ h_late(t) Σ_{m∈late} χ_m(x)(x-c_m)`, `χ_m=exp(-d²/2σ_b²)`,
  σ_b=0.20, compressing only the late group from `t_comp=0.95` (δ_comp=0.03).
- Reference = pseudo-spectral split-step (exact diffusion + RK2 advection-reaction);
  `u_min` is monitored for advection under-resolution.
- Primary metrics: late local L2 over `W_m` (R_W=0.25), late local peak error,
  narrow-Gaussian observable (σ_obs=0.04, reconstruction-free). Mass `E_m` = sanity.

## Pilots

### Pilot κ=8, D=0.01, N0=1e4, grid 256, K=32, T=1.2 (seed 0) — FAILS criterion 1

Late metrics (mean local L2 / max local L2 / mean peak / narrow-obs / min late local count):
- weighted: 0.374 / 0.395 / 0.392 / **0.049** / 37 (nESS 0.09)
- weighted+ESS: 0.408 / 0.524 / 0.395 / 0.145 / 198 (nESS 0.69)
- cost-matched ESS: 0.403 / 0.437 / **0.251** / 0.164 / 396 (nESS 0.60)
- minvar_branch: 0.378 / 0.405 / 0.383 / 0.060 / **2023** (nESS 1.0, Nact 3.0×)

FAIL: branching has by far the largest late local count (2023) but does NOT win
the field metrics. Raw weighted wins late local L2 (0.374) and the narrow-Gaussian
observable (0.049); cost-matched ESS wins the late peak error (0.251). Even with
compression, 37 weighted particles + weights reconstruct the κ=8 core well enough,
and branching's reproduction variance keeps its field reconstruction from winning.
Next: κ=12, D=0.005 (sharper core) to give branching the best chance.

### Pilot κ=12, D=0.005, N0=1e4, grid 256, K=32, T=1.2 (seed 0) — REFERENCE UNDER-RESOLVED

Reference `u_min = -18.4` (strong negative Gibbs oscillation): at κ=12, D=0.005 the
compressed late core is sharper than the 256² pseudo-spectral reference (and the
explicit RK2 advection) can represent, so the reference is unreliable and the local
L2 metrics (all ≈0.71) are meaningless. The narrow-obs at this κ: cost-matched
0.057, branching 0.056, weighted 0.075 — branching ties cost-matched, does not win.

## Conclusion — compressive-readout is a NEGATIVE result (not in paper)

Two pilots span the usable range:
- κ=8 (core resolvable): branching has 5–50× the late local count but does **not**
  win the field metrics — raw weighted wins late local L2 and the narrow observable,
  cost-matched ESS wins the late peak. The criterion fails.
- κ=12 (core sharp enough that few weighted particles should fail): the reference
  and the K-bandwidth reconstruction both under-resolve the core, so there is no
  clean comparison.

This is the fundamental obstruction: making the core sharp enough that a small
weighted ensemble fails also makes it unresolvable by the deterministic reference
and the band-limited reconstruction. Branching's higher local particle count does
not convert into a local-field accuracy win here, because (a) weighted particles
reconstruct a resolvable core well from their continuous weights, and (b) once the
core is unresolvable, all methods are limited by reconstruction bandwidth, not
particle count.

Per CLAUDE.md §6 ("if these fail, stop") and §8.1, the compressive-readout
experiment is kept as a **negative / diagnostic record only and is not used in the
main paper**. The branching advantage remains the global / peak-resolution result
of the single-peak (§5.2) and switching (§5.3) benchmarks.
