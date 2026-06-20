# Experiment B (radial) pilot findings — mass sweep + resolution check

Model: 3D fully parabolic–parabolic Keller–Segel on the periodic box `T_L^3`, L=12,
`u_t = D Δu − χ∇·(u∇v)`, `v_t = D Δv + αu − βv`, with **v0 = 0** (chemical created
from scratch by injection). Normalized regime `D_u=D_v=α=β=χ=1`, so the effective
coupling is `F = χαM/β = M` (the initial cell mass is the single knob). Radial
wrapped-Gaussian `u0`, σ=0.45, τ=1e-3, T=2.0, N_u=20000, minvar injection kernel,
JITTED fixed-capacity grad-v buffer (`--fast`, verified in test_buffer_equiv.py).
Diagnostics are reconstruction-free particle radii `R_q` (q=0.2/0.5/0.8).

## Mass sweep (one seed) — diffusion → focusing transition

| M (=F) | R_0.5(T)/R_0.5(0) | t_turn | t_focus10 | regime |
|--------|-------------------|--------|-----------|--------|
| 16     | 4.44 | 2.0 (=T) | —    | diffusion (monotone expansion) |
| 32     | 4.30 | 2.0      | —    | diffusion |
| 64     | 3.89 | 2.0      | —    | diffusion (slower) |
| 72     | 3.71 | 2.0      | —    | diffusion |
| 80     | 3.45 | 2.0      | —    | diffusion |
| 88     | 2.73 | 2.0      | —    | **marginal**: chemotaxis arrests spreading (R_0.5 plateaus ~0.93, t∈[0.12,0.72]) then diffusion wins again |
| 92     | 0.675| 0.20     | 1.36 | critically-delayed (focuses, very late onset) |
| **96** | **0.42** | **0.20** | **0.60** | **delayed focusing** (clean intermediate regime) |
| 104    | 0.35 | 0.12     | 0.40 | focusing (earlier) |
| 112    | 0.32 | 0.12     | 0.32 | focusing |
| 120    | 0.30 | 0.08     | 0.28 | near-immediate focusing |
| 128    | 0.28 | 0.08     | 0.24 | immediate hard collapse |

Critical mass (χ=α=β=D=1): **M\* ∈ (88, 92]**, sharp transition.

### M=96 delayed shape (the resolved intermediate regime)
Diffuse to t≈0.24 (R_0.5: 0.69→0.85) → **turnover** at t≈0.16–0.20 → core focuses
(R_0.5: 0.85→0.29 at K=8). The turnover happens at R_0.5≈0.7–0.85 ≳ h_K(K=8)=0.71, so the
**delayed onset is reconstruction-resolved** (the robust, K-stable feature). Max
drift-resolution number 0.024 ≪ 2.

**Caution (Codex audit):** the apparent "core focuses while the halo spreads" picture is
**K=8-specific, NOT a robust physical claim.** R_0.8 final = 2.12 (K=8), 0.31 (K=12),
0.24 (K=16): at higher bandwidth the halo *also* collapses (fuller aggregation). At low K
the band-limited drift cannot pull the outer particles inward, so only the core focuses;
better-resolved gradients contract the whole cloud. Report the depth/extent of collapse as
**bandwidth-sensitive**; report only the delayed turnover/onset as resolution-robust.

## Resolution (K) sensitivity at M=96, one seed

| K_dyn | h_K=L/(2K+1) | t_turn | t_focus10 | R_0.5(T)/R_0.5(0) | drift-res max |
|-------|--------------|--------|-----------|-------------------|---------------|
| 8     | 0.706 | 0.20 | 0.60 | 0.42 | 0.024 | R_0.8(T)=2.12 (halo lags) |
| 12    | 0.480 | 0.16 | 0.32 | 0.22 | 0.10  | R_0.8(T)=0.31 (halo collapses) |
| 16    | 0.364 | 0.20 | 0.32 | 0.18 | 0.24  | R_0.8(T)=0.24 (halo collapses) |

- **t_turn ≈ 0.16–0.20 is K-robust** (the delayed turnover is reconstruction-free-stable).
- **t_focus10 converges by K=12** (0.32 = 0.32 at K=16).
- **Collapse depth AND extent are bandwidth-sensitive**: final R_0.5 ratio 0.42→0.22→0.18
  AND final R_0.8 2.12→0.31→0.24. At low K the band-limited drift focuses only the core
  (halo lags at R_0.8=2.12); at higher K the gradient is resolved over more of the cloud and
  the collapse is fuller (halo contracts too). Matches the manuscript's reconstruction story.
  Drift-resolution number stays ≤0.24 ≪ 2 (stable).
- **NOTE (superseded by production):** the R_0.8 halo column above is single-seed at N=20k.
  The 8-seed production (radial_*_M88_M96_K12_8seed) shows R_0.8 is strongly seed-scattered
  near the bimodal split (N=20k seeds span 0.31–2.12), so do NOT read the K-dependence of
  R_0.8 from these single-seed values. The robust K-dependence is in R_0.5 (depth) and the
  K-robust t_turn; R_0.8 is not a reliable diagnostic here.
- **Mass law (Codex audit):** M_v unbiased vs M*(1-e^{-t}); max rel. error 7.75% only at the
  first step t=0.04 (N_v≈845, single-seed integer-injection noise), <1% for t≥0.2, <0.6% by
  t=1. N_v(t) is M-independent at fixed seed (injection prob = (1-e^{-τ}) since ω_u=ω_v), so
  this is pure sampling noise that averages over seeds. (Rigorous unbiasedness already in Expt A.)

## Conclusion / production choices

- **A resolved intermediate (delayed) regime exists at reasonable M (M=96) with
  D_v=1.** No need for the secondary D_v<1 model.
- Production weak/delayed contrast pair (across critical mass):
  - **weak / diffusion-marginal: M=88** (arrest-then-respread; no net focusing),
  - **delayed focusing: M=96** (diffuse → turnover@0.2 → core focuses, halo spreads).
- Primary reconstruction bandwidth **K=12** (converged onset, stable drift); report the
  K=8/12/16 sweep as the resolution diagnostic (turnover-robust / depth-sensitive).

Data: `reference_results/keller_segel_pp3d/msweep_20260619_1710_*` (M=16..128),
`msweep_20260619_1714_*` (M=72..120), `ksens_20260619_1722_*` (M=96 K-sweep + M=92).
Each run dir holds `pilot_protocol.json`, `selection.json`, `pilot_summary.csv`,
and per-run `chi1_M*_seed0/diagnostics.csv`.
