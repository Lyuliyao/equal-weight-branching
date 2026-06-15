# Revision results log — equal-weight branching paper

A single honest record of every experiment in the numerical-section revision,
**positive and negative**. Each entry states the purpose, what was run, the key
numbers, whether it is positive / mixed / negative, where the data live, and
whether it is in the manuscript. Negative and limited results are kept here
deliberately, even when they are not used in the paper.

Status legend: ✅ positive & in paper · 🟢 positive & verified (not yet in paper) ·
🟡 mixed / limited · 🔴 negative (record only) · ⚪ infrastructure / check.

Last updated: 2026-06-14.

---

## 1. Branching vs weighted particles (the core paper claim)

### 1.1 ✅ Stationary localized growth (§5.2)
- **Purpose:** the main branch-vs-weighted comparison under localized multiplicative growth.
- **Result (positive):** weighted develops weight degeneracy (global nESS 0.021, max:mean weight 133). Relative L² error at T=1: **weighted 0.269±0.041, Poisson 0.062±0.035, minimum-variance 0.033±0.011**. At matched particle-step cost (1.9×10⁷), ESS-triggered resampling gives 0.047±0.012 vs minvar 0.033±0.011 — **branching wins reconstructed-field accuracy at matched work**.
- **Data:** `reference_results/branch_vs_weighted/` (+ `cost_match_resample/`). **In paper (Tables 4–5, Figs 2–5).**

### 1.2 ✅ Switching growth (§5.3)
- **Purpose:** structural failure of global resampling when the growth region moves.
- **Result (positive):** at T=1.2, global L² weighted 0.900 / ESS-resample 0.288 / minvar 0.203. In the **new** region `B_B`: ESS 0.297 vs branching 0.174; in the **old** region `B_A`: ESS 0.162 (best) ≈ branching 0.170. Branching keeps ~95% distinct ancestors at the switch vs 63–68% for resampling. **ESS competitive in the old region, branching better in the new region.**
- **Data:** `reference_results/switch/`. **In paper (Table 6, Fig 6).**

### 1.3 ✅ Fourier-bandwidth & KDE robustness audit (Appendix F)
- **Purpose:** check the §5.2/§5.3 ordering is not an artifact of one reconstruction bandwidth.
- **Result (positive):** re-ran the exact production dynamics (validated **byte-faithful**: per-seed E_total reproduces the archived metrics to 2×10⁻¹⁷ for §5.2, 1×10⁻¹⁶ for §5.3). minvar beats weighted at every K∈{8,12,16,24} and h∈{0.10–0.25}; the §5.3 old/new-region mechanism is stable across K∈{32,48,64} and h∈{0.06,0.10,0.15}. No reversal.
- **Data:** `reference_results/reconstruction_audit/`. **In paper (Appendix F, Figs 29–30).**

---

## 2. Multi-island (🔴 negative — records only, NOT in paper)

- **Purpose:** show global ESS can look healthy while local islands fail, and that branching wins per-island mass.
- **Result (negative):** static, staged, and compressive-readout variants were all tested (up to 8 seeds). Branching achieves the lowest **global** L² and by far the largest late-island particle count, **but at matched particle-step cost a cost-matched ESS-resampling baseline attains lower per-island *mass* error** (mass integrals favor weighted particles' continuous weights). The compressive κ=12 reference also under-resolves the core.
- **Conclusion:** "local particle-count advantage ≠ lower per-island-mass error." Did **not** meet the "branching wins the local metric" bar.
- **Data + parameter logs:** `experiments/branch_vs_weighted/{multi_island,staged_multi_island,compressive_multi_island}.py` and `reference_results/{multi_island,staged_multi_island}/`. **Excluded from the paper by design.**

---

## 3. Keller–Segel: fully parabolic–parabolic concentration benchmark (§5.4–§5.5)

System (all of §3): `u_t − div(∇u − u∇v) = 0`, `v_t = Δv + u − v`, χ=1, IC
`u0=840 e^{−84r²}`, `v0=420 e^{−42r²}` (M_u=10π), report at t=6e-5, 1.2e-4, 2e-4.

### 3.1 🟢 Direct LDG reference (Li–Shu–Yang) — VERIFIED
- **Purpose:** the *direct LDG* comparator the benchmark requires (FVM is not acceptable as the LDG reference).
- **Method:** from-scratch P¹ modal LDG following the paper — auxiliary `p=∇u, r=∇v`, alternating fluxes, Lax–Friedrichs chemotaxis flux, Zhang–Shu P¹ positivity limiter, SSP-RK3, Neumann on [−½,½]².
- **Verification (positive):** pure-heat LDG **2nd order** (2.00, 2.03, 2.02); Laplacian operator **symmetric to 5.5×10⁻¹⁷** (negative-definite); chemotaxis `−div(u∇v)` consistency **3rd order**; **u-mass = 10π exactly** (drift ~1e-14); positivity holds (u_min ~ −1e-11). Matches the paper's Table 5.1 structure.
- **🐛 chemotaxis flux bug found & fixed (2026-06-14):** `conv_rhs` had an x-face ξ-mode sign error and a y-face ξ/η modal swap (slope modes only; the cell-average and mass were correct, so the original cell-average-only test missed it). Caught via an x↔y **permutation-symmetry** test (was FAIL → now PASS) and re-verified by full-KS **self-convergence (2nd order, 2.11/2.46)** and exact **radial symmetry**. See `experiments/keller_segel/ldg_reference/LDG_DEBUG_REPORT.md`. The buggy run is kept record-only.
- **Example 5.2 (positive, FIXED flux):** reproduces the concentration — `S_L2(2e-4)` grows **874 (N=80) → 1604 (160) → 3159 (320)**, peak up to ~1×10⁶.
- **Numerical blow-up time `tb(N)=inf{t: S(2N,t)≥1.05 S(N,t)}` (their (5.2)) — 🟢/🟡:** after the fix, `tb(1.05) = 5.95×10⁻⁵` (80→160) → **8.43×10⁻⁵** (160→320) — it now **increases monotonically toward** the reference `1.21×10⁻⁴` as the mesh refines (the paper's expected `tb(N)→T*` trend; the buggy run was frozen at `7.36×10⁻⁶`). Same scale as the FVM baseline (3.5–5.0×10⁻⁵). Still a uniform-mesh resolution-gap indicator, **not** a converged continuum blow-up time (the paper defers adaptive-mesh `tb(N)` for KS to future work).
- **Data:** `experiments/keller_segel/ldg_reference/`, `reference_results/keller_segel_ldg_pp/ldg_<run_id>_fixed_flux/` (and the record-only buggy `ldg_<run_id>/`). **Not yet in paper** (the §5.4 rewrite is pending).

### 3.2 🟢 / ⚪ FVM grid baseline — now a sanity anchor only
- **Purpose:** originally the §5.4 deterministic reference; **demoted** to a sanity check after the LDG reference was built (FVM ≠ LDG).
- **Result (positive as a check):** positivity-preserving upwind FVM (Neumann), mass conserved to ~1e-16, positivity, `t_b(1.05)` = 3.5e-5 (128→256), 5.0e-5 (256→512). Consistent concentration scale with LDG and particle.
- **Data:** `experiments/keller_segel/ldg_pp_baseline/`, `reference_results/keller_segel_ldg_pp/baseline_<run_id>/`. Manuscript §5.4 currently still cites it as the grid reference (to be revised to LDG).

### 3.3 🟢 Particle method on the same equation
- **Result (positive):** the fully-pp particle run (`ldg_comparison`, u-conservative + v decay/injection min-variance kernel) reproduces comparable reporting-time concentration: `S_L2(2e-4)` ≈ 1275 (base) / 2686 (refined), bracketing the LDG (1604–3274) and FVM (1982–2941). Cell mass conserved; chemical mass follows the injection balance to <1e-4. Reconstructed **peak is bandwidth-sensitive**; **core radii R₀.₅,R₀.₈ are reconstruction-free**.
- **🔴 note:** the refined run trips the **drift-CFL guard at t≈1.66e-4** — the un-hybridized global-Fourier reconstruction destabilizes the chemotactic drift as the core concentrates. This motivates the online solver-level hybrid (§3.5, not yet done).
- **Data:** `reference_results/{ldg_comparison/...,keller_segel_ldg_pp/particle_<run_id>}`.

### 3.4 🟡 Core-local / reconstruction-free blow-up-proxy diagnostics (§5.5)
- **Positive:** the reconstruction-free radii `R₀.₅,R₀.₈` collapse robustly and identically across grid resolutions and methods — the reliable concentration signal.
- **🔴 negative/limited (two findings):**
  1. A radius-fit candidate concentration time `R_q² ≈ C_q(T*−t)` gives `T*` of order 1e-4 but **window-sensitive: 1.4×–4.5× spread** across fit windows (radii saturate at the grid/bandwidth floor). **No continuum blow-up time is quoted** — consistent with the project finding that the blow-up *time* is not defensibly computable while the *collapse* is.
  2. **Core-localizing the L² norm does not sharpen the proxy:** for the resolving grid `S_core ≡ S_L2` (the field carries all its L² mass in the core), so `t_b^core ≡ t_b^global`.
- **Data:** `experiments/keller_segel/core_local_proxy/`, `reference_results/keller_segel_ldg_pp/core_proxy_<run_id>/`. **In paper §5.5 (limited result, Fig 8).**

### 3.5 🟡 Particle-adaptive reconstruction (post-processing audit)
- **Purpose:** drive a local reconstruction from the particle cloud instead of a uniform global Fourier bandwidth.
- **Positive:** global Fourier under-resolves the core at every feasible `Kg` (base t=2e-4: peak 2.4k→78k for Kg 5→32), while the **particle-adaptive hybrid** (P_Kg + signed local residual on `B(x_c,3R₀.₈)`, local spectrum/blob) recovers it at low global bandwidth — peak ≈3.9×10⁵, `S_L2` ≈2180, near the FVM anchor (5.95×10⁵, 2941). Consistent with the reconstruction-free radii where global Fourier is not.
- **🔴 negative/limited (caveats, reported not hidden):** the hybrid peak is still **local-`Kl` sensitive** (refined cloud t=1.2e-4: 545k→979k for Kl 24→40) and overshoots the FVM grid baseline; the tapered signed residual is only **~mass-conserving (∫u≈25.3 vs M=31.4, ~19% deficit)**; both high-K global *and* the signed hybrid go **negative (Gibbs)** — the blob variant undershoots least.
- **Conclusion:** a real diagnostic improvement that reduces *global*-bandwidth sensitivity, but it trades it for *local*-bandwidth sensitivity and signed-residual artifacts; **not** a converged peak / blow-up estimate.
- **Data:** `experiments/keller_segel/pp_particle_ldg/adaptive_reconstruct.py`, `reference_results/keller_segel_ldg_pp/adaptive_recon/`. **Not in paper.**

### 3.6 ⚪ Cross-species injection mass-law check (Appendix)
- **Positive:** the parabolic–parabolic chemical equation `v_t=Δv+u−v` uses the injection kernel `μ_v^{n+1}=e^{−τ}μ_v*+(1−e^{−τ})μ_u*` (not a `(u−v)/v` quotient). Chemical mass follows the exact balance law to <1.4% (smooth IC). Validates the §5.4 algorithm.
- **Data:** `experiments/keller_segel/pp_injection/`. **In paper (Appendix F-coupled).**

---

### 3.7 🟢/🟡 LDG-style particle blow-up proxy (reconstruction-operator sweep)
- **Purpose:** define a particle analogue of the LDG `L²`-resolution-gap blow-up indicator tied to an **explicit** reconstruction operator (not a final-time image), and compare to the fixed-flux LDG `tb`.
- **Method:** project the particle cloud onto the **same P¹ DG space and mass-matrix norm as the LDG solver** (Version A; verified — the cross/split estimator converges to the LDG `field_L2`=114.88), plus the Fourier readout (Version B). Three gaps: main `(N_p,n)→(4N_p,2n)`, sampling `(N_p,n)→(4N_p,n)`, recon `(N_p,n)→(N_p,2n)`. Ensemble over seeds + bootstrap CI + persistence rule. Runs: `N_p∈{2e4,8e4,3.2e5}`, 2–4 seeds.
- **Positive:** at **adequate particle counts** the LDG-matched DG main gap `(8e4,80)→(3.2e5,160)` gives `tb=9.2×10⁻⁵` [CI 6.2–11.9e-5] — **on the LDG scale** (5.95–8.43e-5); the same-cloud recon gap is `4.8×10⁻⁵`, near LDG `80→160`. So the particle method reproduces the LDG resolution-gap scale.
- **🔴 negative/caveat:** at **low `N_p`** the metric is **shot-noise limited** — the `2e4`-particle readout opens a spurious gap at `7×10⁻⁶` from early times (the cross estimator cannot fully repair `ppc≲3`). The three-gap decomposition exposes this: a naive low-`N_p` particle "blow-up gap" looks ~10× too early. Fourier (Version B) behaves similarly (`1.3×10⁻⁴` at adequate `N_p`, `7×10⁻⁶` at low). The 8e4/3.2e5 runs hit the drift-CFL guard at `t≈1.5e-4` (curves end early; CIs wide with 2–4 seeds).
- **Decision (Scenario 1/2):** use Version A as the LDG-comparable metric **at adequate ppc with the cross estimator and the stated shot-noise limitation**; Fourier as a reconstruction-sensitivity diagnostic. **Not** a continuum blow-up time. Pair with reconstruction-free radii.
- **Data:** `experiments/keller_segel/ldg_comparison/{particle_dg_readout,analyze_particle_blowup_metric,plot_particle_blowup}.py`, `reference_results/keller_segel_ldg_pp/particle_blowup_<run_id>/` (`particle_tb_summary.csv`, README, figure). **Not in paper.**

### 3.8 🟢/🔴 Solver-level residual reconstruction in the chemotactic drift (Form I)
- **Purpose:** the reconstruction used INSIDE the time step for the drift `b_u=χ∇v̂`, not just diagnostics. Form I two-level spectral residual `v̂=v_lo+χ(v_hi−v_lo)` (`v_lo=P_{Kg=8}`, `v_hi=P_{Kl=24}` on the core-adaptive window), fed into the u-drift. Gradient (incl. the taper term `(v_hi−v_lo)∇χ`) **FD-verified to 5.6e-8**.
- **🟢 positive (Q3):** single-K Fourier under-resolves the inner core (`R_0.2/h_eff≈1.2`) — the window tracks the outer mass `R_0.8` while `R_0.2,R_0.1` keep collapsing; the residual field more than doubles inner-core resolution (`R_0.2/h_eff≈2.9`). The reconstruction issue is real for the drift field. *(The previously reported "smoother background drift, mean `drift_cfl≈1.1` vs `≈1.8`" is **withdrawn**: the logged `drift_cfl` was the single-K **Fourier diagnostic**, not the hybrid solver drift; the hybrid aborted early because its solver `∇v̂` spiked past `cfl_abort=5.0`. See the dual-CFL fix below.)*
- **🔴 negative (Q1) — corrected by the mechanism check:** the hybrid aborts the drift-CFL guard *earlier* (`~1.4e-4` vs `~1.9e-4`). Its **logged** `drift_cfl` (the Fourier diagnostic, not the solver drift) stays low, but the actual solver-field `∇v̂` spikes past `cfl_abort` between diagnostic samples — i.e. **high-Kl Monte-Carlo noise** in `∇v̂` (differentiating `P_{Kl}` amplifies high-mode particle noise ~Kl), **not** intrinsic dynamics and not a bug. **Confirmed:** stronger damping (filter_s 0.3) delays the abort to 1.49e-4 and cuts seed variance ~4×; lower Kl=16 similar.
- **🟢 Scenario-C fix tested (taper sweep, 16-task SLURM, 4 cfg × 4 seeds, N=8e4):** a Gaussian taper of width `h_hi` on the high-Kl core part is exactly an `η_h` blob residual (the blob FT is `exp(−h²k²/2)`); smaller `h_hi` = smoother local operator. The abort time recovers **monotonically** as the operator smooths — `h_hi`: 0.50→0.35→0.25 gives abort `t` = 1.39e-4 → 1.58e-4 → **1.89e-4**, recovering the global-`K` drift's 1.91e-4 (0.99×), and cutting the seed-to-seed abort spread. *(The `drift_cfl` 2.6-vs-4.5 "smoother-drift" comparison is **withdrawn**: at that run commit the logged `drift_cfl` was the single-K Fourier diagnostic of the hybrid-driven cloud, not the hybrid solver drift that drove and aborted the particles. A fresh dual-CFL check shows the hybrid solver drift ≈2.57 far exceeds its Fourier diagnostic ≈1.03; any aborting hybrid spiked its solver `∇v̂` past `cfl_abort=5.0`, which is why it aborted earlier.)* The supported signal is the **monotone abort-time recovery**, which still confirms the early abort was removable high-Kl Monte-Carlo noise in `∇v̂`. The conflation was **found and fixed**: `simulation.py` now logs `drift_cfl_solver_field` (the guard quantity) and `drift_cfl_fourier_diag` separately; the archived pre-fix `drift_cfl` columns are the Fourier diagnostic and are **not** retro-correctable (raw clouds not saved), so only their abort-time columns are used.
- **🔴 honest catch (Q3 re-read):** the `R_0.2/h_core` "resolution gain" (1.0→3.1) is **dominated by the Kl=24-vs-K=10 grid factor**, not by better-tracked concentration. The reconstruction-free `R_0.2` per-seed spread is ~4× (e.g. current 0.0045–0.0197 at `t=1e-4`), so the config means (0.011 vs 0.015) are **within seed noise** — no config shows a clearly tighter actual core. Form I is drift-stable once smoothed, but does **not** demonstrably improve the pre-singular tracked concentration over the plain global-`K` drift, and costs ~2.5×/step.
- **🟢 Physical-space blob residual (Form-I-smooth) implemented + verified (the principled Scenario-C operator):** instead of the high-Kl local *spectrum* (whose ∇ amplifies high-mode noise ~Kl), the local high part is an `η_h` **Gaussian blob** (KDE) — `v̂ = v_lo + χ(η_h*μ_v − η_h*(v_lo dx))`, the signed residual with the low-blob subtracted to avoid double-counting. `BlobResidualVField` (FFT-grid production path + exact kernel-sum verification): FD-verified to **4.4e-9**, no-double-counting to **1e-16**, grid-vs-exact **1.1%**, **5/5** unit tests pass; blob gradient tail < Kl=24 spectral tail (the smoothness hypothesis). Wired as `--solver_field two_level_blob_residual`. A bandwidth analysis: the plan's spacing rule `h~R₀.₈/√N₀.₈` is ~25× finer than `L/Kl`, so the **smoother-than-Kl** regime needs `h` as a fraction of `L` (`frac_L` rule, default `c_h=0.06≈1.4/Kl`). **Lean sweep (current_fourier + spectral Kl=24 + blob frac_L {0.04,0.06,0.09} + 1 spacing-rule, 6×4 seeds) pending.**
- **Decision — Scenario C (interim):** the residual-reconstruction *principle* is correct (resolves the reconstruction grid in the core) and the noise→abort artifact is **fixable** by a smoother local operator (`h_hi≈0.25` spectral, or the `η_h` blob). The taper sweep showed abort-time recovery but **no demonstrable accuracy gain** (Q3 was a grid-density relabeling). The blob sweep will test, with the **corrected dual-CFL diagnostics**, whether the blob's smoother solver drift recovers stability with a genuinely smaller solver-CFL than the spectral hybrid. So far: a **reconstruction/diagnostic option, not a validated accuracy improvement**; consistent with §5.5 (core radii reconstruction-free; reconstructed peak/L2 bandwidth-sensitive). **Not in paper.** Do not claim the solver resolves near-blow-up dynamics.
- **Data:** `experiments/keller_segel/ldg_comparison/{hybrid_vfield.py,blob_residual_vfield.py (FD-verified),test_blob_residual_vfield.py,analyze_solver_field.py,analyze_taper_sweep.py,analyze_blob_sweep.py,plot_taper_sweep.py,run_solver_field_blob_sweep.sb}`, `simulation.py --solver_field {current_fourier,two_level_spectral_residual,two_level_blob_residual}` (now logs `drift_cfl_solver_field` + `drift_cfl_fourier_diag` + `abort_diagnostics.json`), `reference_results/keller_segel_ldg_pp/{solver_field_sweep_<run_id>/,sf_taper_20260614_1983_3c6ae64/}` (diag CSVs + compare json + figures + plot_data). **Not in paper.**

## 4. Other Keller–Segel / high-dim (unchanged this revision)

- ⚪ **Parabolic–elliptic core-adaptive `t_gap` diagnostics** — moved to **appendix (record only)**; the main KS benchmark is the fully-pp system. `experiments/keller_segel/concentration_ldg/`.
- ⚪ **Local reconstruction (hybrid spectrum+residual) on synthetic cores** — moved to **appendix**. `experiments/resolution_hybrid/`.
- ✅ **3D Keller–Segel focusing** (§5.6) and **6D kinetic Keller–Segel** (§5.7): conservative particle-field focusing tests, no blow-up-time claim. Unchanged.

---

## 5. Not done (scoped honestly)

- ~~**Online solver-level hybrid reconstruction**~~ — **done** (§3.8): Form I residual field fed into the drift, FD-verified, swept over the local-operator smoothness. Outcome is drift-stable-but-not-more-accurate (record only, not in paper). No further runs planned unless the design changes.
- **§5.4 manuscript rewrite** around **LDG vs particle-global-Fourier vs particle-adaptive** (now unblocked by the verified LDG reference §3.1). The current manuscript §5.4 still uses the FVM baseline as the grid reference.
- **Pre-existing unrelated issue:** 5 introduction citations are missing from `main.bib` (`vazquez2006porous`, `fisher1937wave`, `williams2018combustion`, `doucet2000sequential`, `liu1998sequential`).

---

## 6. One-line summary

Branching's defensible win is reconstructed-field accuracy on the single-peak (§5.2)
and switching (§5.3) benchmarks, robust to reconstruction bandwidth (Appendix F).
Multi-island and per-island *mass* metrics do **not** favor branching (negative,
record only). For Keller–Segel, a verified direct LDG reference and the particle
method reproduce the same pre-singular concentration; the numerical blow-up time and
the candidate `T*` are resolution/bandwidth/window-sensitive and are **not** quoted
as a continuum blow-up time. Particle-adaptive reconstruction helps the core but
introduces local-bandwidth and signed-residual artifacts (mixed). The solver-level
residual drift (Form I) is now implemented and tested: smoothing the local operator
(`h_hi≈0.25` blob residual) makes it drift-stable to the same time as the global-`K`
drift, but it is not demonstrably more accurate, so it stays a diagnostic record.
The §5.4 LDG manuscript rewrite remains.
