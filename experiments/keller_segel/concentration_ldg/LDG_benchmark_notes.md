# LDG benchmark notes — 2D Keller–Segel concentration / pre-blow-up

Cross-check of the particle method's concentration diagnostics against the
grid-based Keller–Segel blow-up literature (LDG / Chertock–Kurganov /
Epshteyn–Kurganov). **Verified via Codex (GPT-5.2) factual review, 2026-06-12**
(thread 019ebdcf). Confidence levels are recorded per item; numbers marked
low-confidence are NOT quoted in the manuscript.

## Summary (Codex-verified)

> We use the classical one-bulge supercritical Keller–Segel Gaussian benchmark
> u0 = 840 exp(−84 r²), v0 = 420 exp(−42 r²), whose cell mass is 10π > 8π. This
> is a CK/EK/LDG-style parabolic–parabolic blow-up benchmark; we compare
> **morphology and concentration diagnostics**, not exact reproduction of any
> published blow-up time.

## Initial data
- u0 = 840 exp(−84|x|²), v0 = 420 exp(−42|x|²); ∫u0 = 840π/84 = **10π ≈ 31.42 > 8π**.
  Mass statement: **HIGH confidence**.
- Whether the *original* Li–Shu–Yang LDG 2017 paper (JSC 73:943–967) uses exactly
  these 840/420 coefficients: **UNCERTAIN** (could not verify from accessible
  text). It is confirmed to be a standard one-bulge supercritical CK/EK/LDG-style
  test. Close variants exist (e.g. 500·exp(−100((x−.25)²+(y−.25)²)) on [−.5,.5]²).
  → We therefore say "CK/EK/LDG-style benchmark", NOT "the LDG benchmark exactly".

## Model form
- Standard benchmark is **parabolic–parabolic**: u_t = Δu − χ∇·(u∇v),
  v_t = Δv − v + u, χ=1 (**medium-high confidence**).
- **Our LDG-diagnostics runs use the parabolic–ELLIPTIC reduction** −Δv+v=u
  (v slaved to u), on the core-adaptive Fourier window, because the adaptive
  window + elliptic solve is what resolves the collapsing core for the
  reconstruction-free core radii and the resolution-gap indicator. This is the
  same reduced testbed as the manuscript's "On the blow-up time" analysis. The
  full parabolic–parabolic morphology is covered by the main concentration test
  (two particle clouds). State this reduction explicitly in §5.4.

## Domain / boundary conditions
- Grid literature: bounded box, **homogeneous Neumann/no-flux** (HIGH confidence);
  one-bulge CK test domain likely [−1/2,1/2]² (medium confidence).
- Our method: effectively-unbounded plane via the core-adaptive window
  (half-width L(t), centered at x_c(t)); screened-Poisson solved spectrally on it.
  Valid pre-concentration since the bulge mass is localized far inside the window.

## Reporting times
- Exact 840/420 LDG/CK report times: **UNCERTAIN — not quoted.**
- Related benchmarks (Codex): parabolic–parabolic two-species reports t=5e-4, 1e-3;
  parabolic–elliptic two-species reports t≈3e-3, 3.3e-3 with a blow-up estimate
  ≈2.94e-3 — these are RELATED, not identical, so treated as order-of-magnitude
  guidance only (**low confidence as literature-aligned for 840/420**).
- Our runs: τ=1e-6, T=4e-3 (4000 steps), snapshots at {5e-5,1e-4,1.5e-4} (early,
  matching the manuscript's existing contour times) AND {5e-4,1e-3,2e-3,4e-3}
  (the collapse regime). We report at these times and frame as exploratory /
  morphological, not as reproducing specific LDG figure times.

## Blow-up / concentration diagnosis (ours)
- S_{K,N}(t) = ||P_K μ||_{L2} (reconstructed L2 norm) — `S_L2`.
- ||P_K u||_∞ reconstructed peak — `peak_PK_u`.
- core mass M_core(r,t)=μ(B(x_c,r)), r∈{0.01,0.02,0.04}.
- reconstruction-free core radii R_q(t); resolution ratio R_0.5/h_eff.
- resolution-gap focusing time t_gap from paired (N,K)+(4N,2K) runs.
- Grid literature diagnoses via max-density growth curves, morphology, positivity,
  mesh-refinement dependence; later DG papers note the discrete scheme cannot
  literally realize continuum blow-up (finite mass + positivity) — peaks just grow
  very large. Our framing matches: "pre-blow-up concentration", "resolution-gap
  focusing indicator". **No claim of robust finite-time blow-up capture; no claim
  of superiority over LDG.**

## Checklist
- [x] model form: par–par standard; we use par–ell reduction (stated)
- [x] IC mass 10π > 8π (high confidence); exact LDG coefficients unverified (stated)
- [x] domain/BC: Neumann grid vs our adaptive window (stated)
- [~] report times: order-of-magnitude only; not literature-exact (stated, not quoted)
- [x] diagnosis method: morphology + max-growth + resolution dependence (consistent)
