# Particle method on the fully parabolic–parabolic LDG benchmark (§5.4)

> **Blocker / superseding note (2026-06-14):** the current §5.4 workflow is not sufficient for the manuscript.  It compares the particle method mainly against a finite-volume grid baseline, while the desired experiment is a **direct LDG comparison**.  It also reports a particle `(N,K)->(4N,2K)` gap that mixes particle-number refinement with Fourier-bandwidth refinement.  Before §5.4 is used in the paper, follow `LDG_DIRECT_RECONSTRUCTION_TODO.md`: implement or compare directly with LDG, and add particle-adaptive local reconstruction driven by the particle cloud.

The particle method run **on the same equation** as the grid baseline
(`../ldg_pp_baseline/`) for the §5.4 comparison. The solver is **not duplicated
here**; it is `experiments/keller_segel/ldg_comparison/simulation.py`, which already
implements exactly the algorithm the revision plan requires:

- cell density `u`: **conservative** equal-weight particles (no birth/death), Euler–
  Maruyama transport with chemotactic drift `+∇v` reconstructed from the chemical cloud;
- chemical `v`: diffusion + the **cross-species injection kernel**
  `μ_v^{n+1} = e^{-τ} μ_v* + (1-e^{-τ}) μ_u*` — existing `v`-particles survive with
  probability `e^{-τ}`, transported `u`-particles inject new `v`-particles with mean
  `(1-e^{-τ}) ω_u/ω_v` via the **minimum-variance integer kernel** (no `(u-v)/v`
  quotient branching);
- LDG initial data `u0=840 exp(-84r²)`, `v0=420 exp(-42r²)` and reporting times
  `6e-5, 1.2e-4, 2e-4`, on a core-adaptive periodic Fourier window (LDG-aligned, **not**
  strict Neumann — disclosed; the core stays far from the window boundary).

## How the current §5.4 particle results are produced

```bash
cd ../ldg_comparison
python simulation.py --N 20000 --K 5  --tau 1e-6   --n_steps 200 --seed 0 \
    --report_times 6e-5 1.2e-4 2e-4 --outdir <run>/base
python simulation.py --N 80000 --K 10 --tau 2.5e-7 --n_steps 800 --seed 0 \
    --report_times 6e-5 1.2e-4 2e-4 --outdir <run>/refined
```

Each writes a `diag_*.csv` with `t, S_L2, peak_PK_u, R_0.5, R_0.8, M_u, M_v, ...`.
The archived production run used in the paper is
`reference_results/ldg_comparison/20260613_1922_92c0226_pp_inject_ldg_base20k_K5_ref80k_K10/`;
its base/refined diagnostics are copied (for §5.4/§5.5 traceability) into
`reference_results/keller_segel_ldg_pp/particle_<run_id>/`.

## Current comparison to the grid baseline (§5.4) — not final

`../ldg_pp_baseline/plot_baseline.py` overlays the particle `S_L2(t)`, peak, and
`R_0.8(t)` on the grid refinements. Findings (honest scope):

- the particle method reproduces **comparable reporting-time concentration**: at
  `t=2e-4`, particle `S_L2` ≈ 1275 (base) / 2686 (refined) brackets the grid baseline
  `S_L2` ≈ 1982 (n=256) / 2941 (n=512);
- cell mass conserved, chemical mass follows the injection balance to <1e-4;
- the reconstructed **peak is bandwidth-sensitive** (particle 72k→331k as `K` 5→10;
  grid 80k→595k as `n` 128→512), while the **core radii `R_0.5, R_0.8` are
  reconstruction-free** and track the baseline;
- the particle `(N,K)→(4N,2K)` resolution gap `t_gap(1.05)≈8e-6` is a
  **reconstruction-bandwidth** gap (the pair changes both particle number `N` **and**
  Fourier bandwidth `K=5→10`), so it is *not* the same indicator as the grid `t_b`
  and is reported separately.

## Status (2026-06-14)

- **Done — item 3 (post-processing adaptive reconstruction).** The solver now saves raw clouds (`--save_cloud_snapshots`); `adaptive_reconstruct.py` compares global Fourier vs particle-adaptive hybrid (local spectrum/blob, signed residual). Result is **mixed/limited**: the hybrid recovers the concentrating core that global Fourier under-resolves at any feasible global `Kg` (peak/`S_L2` reach the FVM-anchor scale at low global bandwidth — acceptance criterion 3), but trades global- for local-`Kl` bandwidth sensitivity and the signed residual is only ~mass-conserving with Gibbs negativity (blob least). Full write-up: `reference_results/keller_segel_ldg_pp/adaptive_recon/README.md`.
- **Done — items 1–2 (direct LDG reference).** A from-scratch P¹ LDG solver following Li–Shu–Yang (`../ldg_reference/`) is implemented and **verified** (2nd-order diffusion, symmetric operator to 5e-17, 3rd-order chemotaxis consistency, exact 10π mass, positivity), and run on their Example 5.2 (`u0=840e^{-84r²}`) at N=80/160/320, reproducing the concentration. It computes their numerical-blow-up-time `tb(N)` (their (5.2)) for KS — which the paper applied only to a 1-D test. `tb(1.05)≈7e-6` does not converge to the reference 1.21e-4 at uniform resolution, consistent with the paper's own note that adaptive meshes are needed. The FVM baseline is now demoted to a sanity anchor; the LDG is the comparator. See `reference_results/keller_segel_ldg_pp/ldg_<run_id>/`.
- **Not done — item 4 (online solver-level hybrid) and the §5.4 rewrite.** Feeding the hybrid field into the drift (plan §4) and rewriting §5.4 around LDG-vs-particle-global-vs-particle-adaptive remain.

## What must be done before this becomes main-text §5.4

1. Do **not** present the finite-volume baseline as the direct LDG comparator.
2. Add `experiments/keller_segel/ldg_direct/` or an explicitly sourced published-LDG comparison.
3. Add particle-adaptive reconstruction post-processing for the saved particle clouds:
   global coarse Fourier background plus a particle-detected local residual reconstruction in the core window.
4. Recompute `S_L2`, peak, and gap-time diagnostics using the adaptive reconstruction, and compare them to the direct LDG result.
5. Move the finite-volume baseline to appendix/sanity-check status unless it is explicitly labelled as FVM.
