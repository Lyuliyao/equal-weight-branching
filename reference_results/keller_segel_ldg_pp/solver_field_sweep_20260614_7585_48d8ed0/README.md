# Solver-level residual reconstruction (Form I) vs single-K Fourier — results & decision

Tests the solver-hybrid design note's central correction: the reconstruction used
**inside the time step** for the chemotactic drift `b_u = χ ∇v̂`. We compare two
solver fields on the fully parabolic–parabolic KS particle run (`u0=840e^{−84r²}`,
`Np=8×10⁴`, K=10):

- **`current_fourier`** — single-K Fourier `∇P_K μ_v` on the core-adaptive window (the existing drift);
- **`two_level_spectral_residual`** (Form I) — `v̂ = v_lo + χ(v_hi − v_lo)`, `v_lo=P_{Kg=8}`, `v_hi=P_{Kl=24}`, χ a radial cutoff (high bandwidth in the core, low toward the window edge). The drift uses `∇v̂` including the taper term `(v_hi−v_lo)∇χ`.

The hybrid gradient is **FD-verified to 5.6×10⁻⁸** (the taper-gradient term is correct).

## Results (4 seeds)

| | `current_fourier` (K=10) | `two_level` (Kl=24) |
|---|---|---|
| mean drift-CFL **abort time** | `1.9×10⁻⁴` | `1.4×10⁻⁴` |
| inner-core `R_0.2 / h_eff` at end | **1.22** (under-resolved) | **2.87** (resolved) |
| inner-core `R_0.1 / h_eff` at end | 1.12 | 2.46 |
| **mean** `drift_cfl(t)` (diag-sampled) to `1e-4` | rises to **~1.8** | stays **~1.1** (lower) |

(`h_eff = L / K_core`, K_core = 10 for single-K, 24 for the hybrid core. Ratio ≲1
means the inner core is at the reconstruction scale — under-resolved.)

## What this shows

- **Q3 — core-halo under-resolution is real, and the residual fixes the field
  (positive).** Single-K Fourier leaves the inner core unresolved
  (`R_0.2/h_eff ≈ 1.2`): the window scale tracks the outer mass (`R_0.8`) while the
  inner core (`R_0.2, R_0.1`) keeps collapsing. The two-level residual more than
  doubles the inner-core resolution (`R_0.2/h_eff ≈ 2.9`) **and** gives a *smoother*
  background drift (mean `drift_cfl ≈ 1.1` vs `≈ 1.8`). The reconstruction issue is
  genuine for the **field used in the drift**, exactly as the design note argued.

- **Q1 — the hybrid aborts *earlier*, but the cause is high-Kl reconstruction NOISE,
  not intrinsic dynamics (the key correction).** Its *average* `drift_cfl` is lower
  (≈1.1), yet the per-step peak spikes past `θ=5` between diag samples. That
  signature — low background, occasional sharp spikes — is **Monte-Carlo high-mode
  noise**: differentiating the local `P_{Kl}` reconstruction amplifies high-mode
  particle noise by `~Kl`, so a few core particles produce a spurious large `∇v̂`
  that trips the per-step CFL. It is **not** a smoothly sharper drift (the FD
  gradient is exact, 5.6e-8) and **not** an intrinsic-dynamics CFL.
  *Confirming test:* stronger spectral damping (`filter_s 0.3`) **delays** the abort
  to `1.49×10⁻⁴` (from `1.39×10⁻⁴`) and **cuts the seed-to-seed variance ~4×**
  (1.46–1.52 vs 1.10–1.68×10⁻⁴); lower `Kl=16` is similar (`1.42×10⁻⁴`). Damping the
  high modes removes the noise spikes — confirming the early abort is high-Kl
  reconstruction noise, not intrinsic dynamics. (Damping helps but trades off core
  resolution, so it does not fully recover the single-K survival — which is why a
  *smoother local operator* is the right fix, not just damping the spectrum.)

## Decision — Scenario C (local spectral residual gradient is noise-limited)

The residual-reconstruction **principle is correct**: it resolves the inner core
single-K misses and smooths the background drift. The local *spectral* residual,
however, produces a **noisy gradient** at the available particle count — occasional
high-Kl noise spikes trip the drift-CFL and abort the run earlier. Per the design
note this is **Scenario C**: the fix is a **smoother local operator (Gaussian-blob /
KDE residual)** or **stronger spectral damping / lower Kl / more core particles /
the Horvitz–Thompson sketch**, not the local spectrum at high Kl.

**Manuscript positioning:**
- Report the **core-halo under-resolution** of single-K Fourier and the **residual
  reconstruction fix for the field** (inner core single-K cannot represent at a
  fixed bandwidth), used inside the drift — with the smoother background drift.
- State honestly that the **local spectral** residual gradient is **noise-limited**
  at feasible particle counts (occasional high-Kl spikes), so a **smoother local
  operator (blob/KDE) or stronger damping** is the appropriate solver-level form;
  the local spectrum at high Kl is not.
- Keep the **LDG-matched P¹ DG readout** as the quantitative resolution-gap metric.
- Do **not** claim the solver resolves near-blow-up dynamics. Form III (blob
  residual) and an HT-sketched / damped variant are the indicated next steps;
  the local spectral Form I is reported as the noise-limited case.

## Files
`lean_{current_fourier,two_level_spectral_residual}_seed*/diag_*.csv` (with inner-core
radii `R_0.1, R_0.2`), `solver_field_compare.json`. Regenerate the comparison via
`experiments/keller_segel/ldg_comparison/analyze_solver_field.py`. Solver field code:
`hybrid_vfield.py` (FD-verified) + `simulation.py --solver_field`.
