# Validation-closure analysis report — 3D fully parabolic–parabolic Keller–Segel

Neutral, evidence-based summary of the validation-closure pass. **This is not a manuscript
section and makes no recommendation about the paper.** Base commit `dd245f2` (branch
`pp3d-validation-closure` off `main` `6731f15`); all runs on MSU HPCC GPUs (`cuda:0`,
a100/h200/l40s), `JAX_ENABLE_X64=1`, env `/mnt/home/lyuliyao/.conda/envs/heat`.

Convention reminder: `R_q` are **reconstruction-free as a readout** (computed from particle
positions), but their **dynamics depend on `K_dyn`** through the reconstructed drift `∇v`.
All "transitions" are **fixed-bandwidth numerical transitions**, not continuum critical masses.

---

## A. Validation matrix

| Item | Status | Evidence | Limitation |
|---|---|---|---|
| exact mass law `M_v(t)=M(1−e^{−t})` | **PASS** | `test_exact_linear_case`: `M_v(T)=0.394` vs exact `0.392`, within 4 SEM; `M_u` drift `0`. Production: `Mv_relerr_max ≤ 0.039` (radial), `≤ few %` (tetra) | sampling-limited at small `M_v` (early t) |
| `N^{−1/2}` linear modes | **PASS** | `test_exact_linear_case`: `E_u ×2.12`, `E_v ×2.32`, `E_grad_v ×1.80` for 4× `N` (ideal ×2) | MC rate, as expected |
| buffer (fast) equivalence | **PASS** | `test_buffer_equiv`: `fast==slow` asserted to `1e-8` (linear/chi=0) and `1e-7` (radial/chi>0); masked buffer == cloud to `1.8e-15`; padding inert (`0.0`) | floating-point reduction order only |
| axis-permutation symmetry | **PASS** | `test_validation_extra`: field invariant + grad components permute, `<1e-9` | — |
| periodic wrapping | **PASS** | every returned `u,v` coordinate in `[−L/2,L/2)`; `L/2 → −L/2` | — |
| fixed-seed reproducibility | **PASS** | two `fast=True` runs identical: max record diff `0.0` (exact) | same code path |
| no hidden population control | **PASS** | tiny `Nv_cap` raises `RuntimeError`; uncapped grows past `N_u`; `population_control=false` recorded; no run hit its buffer cap | — |
| row-wise injection location | **PASS** | every injected v **row** equals a transported u row (exact 3-tuple membership) | — |
| fast/slow performance | **PASS** | case 1 speedup **5.78×** (slow `2.55s` → fast `0.44s`); case 2 fast `4.93s`/`24.7 ms`-step | "JITted fixed-capacity field reconstruction" only; v-cloud/injection stay NumPy |
| radial N refinement | **PASS** | existing production: `R_0.5(T)/R_0.5(0)=0.219/0.211/0.206` at `N=2e4/1e5/3.2e5` | reconstruction-free readout only |
| radial tau refinement | **PASS (turnover) / LIMITED (depth)** | turnover persists `tau→tau/2`: `t_turn 0.210±0.026 → 0.180±0.028` (overlap), ratio `0.211±0.005 → 0.206±0.007`; curves agree to `max|ΔR_0.5|=0.009` through `t_turn` | final collapse depth not time-converged (`max|ΔR_0.5|=0.115` over full interval) |
| radial K multi-seed | **LIMITED** | turnover exists at all `K`; `t_turn` overlaps for `K=12 (0.210±0.017)` ↔ `K=16 (0.200±0.028)`; `K=8` later (`0.320±0.063`) | final depth strongly `K`-dependent: ratio `0.570/0.209/0.146` (K=8/12/16) — NOT resolution-independent |
| drift resolution near turnover | **LIMITED (resolution-limited)** | same-cloud `δ_{12,16}` already `≥20%` from `t≈0.04`; at `t_turn=0.20` `δ_{12,16}=0.263`; `δ→0.50` by `T` | no interval has `K=12→16`-stable drift; radial result is resolution-limited |
| `Q_0.2≥3` gate | **AUDITED → INVALID** | `Q_0.2(0)=0.64/0.94/1.24` (K=8/12/16), max `0.72/1.03/1.36`, **never ≥3** at any `K` | gate incompatible with the concentrated `σ=0.45` IC; no empirical relation to true `K`-sensitivity |
| tetra N refinement | **PASS (time-averaged) / LIMITED (final instant)** | time-averaged gap `+0.188` (`−13.3%` vs baseline, still positive → attraction) | single-seed N=160k control `d_min(T)=2.204 < active 2.311` (`final_instant_gap=−0.108`); per-cluster `R_0.5` bandwidth-limited; control `d_min` noisy late |
| tetra K refinement | **PASS (attraction) / LIMITED (depth)** | attraction holds; gap `+16.1%` (<20%) | per-cluster `R_0.5` deepens `0.165→0.108` (K=12→16), bandwidth-limited |
| tetra tau refinement | **PASS** | seed-0 gap `+38%` (>20%) triggered seed-1–3 expansion; **4-seed gap `+2.0%`** vs baseline — the `+38%` was single-seed scatter; attraction sign holds | single-seed control `d_min` noisy late |
| centroid reliability | **PASS** | control `A_min` stays `0.43–0.99 ≥ 0.2` over `[0,3]`; active `A_min≈0.93` | control `d_min` still single-seed-noisy late at refinements |

---

## B. What the numerical data support

- **`v` can be created from `v0=0` by cross-species injection.** Confirmed: `N_v: 0 → 18083 → 35691 → 63011 → 86679` at `t=0/0.20/0.44/1/2` (state run); `test_validation_extra`/`test_injection_kernel` confirm row-wise injection at transported `u` locations.
- **The chemical mass follows the exact bounded law** `M_v(t)=M(1−e^{−t})`: relative error `≤ 3.9%` (radial, worst at small `M_v`), `≤` a few % (tetra); `0.39%` typical at large `N`.
- **A delayed turnover exists for `M=96`** (initial expansion then focusing): `t_turn≈0.18–0.21`, peak expansion `×1.26` before turnover; appears at every tested `K∈{8,12,16}` and both `tau∈{1e-3,5e-4}`. **The phenomenon is robust.**
- **The turnover is stable w.r.t. `N` and `tau`**, and w.r.t. `K` *in timing* for `K=12↔16` (`t_turn` distributions overlap).
- **Tetrahedral clusters exhibit attraction relative to the diffusion control**: the **time-averaged** active–control `d_min` gap over the centroid-reliable interval `[0,3]` is **positive for every refinement** (`tetra_gap_summary.csv`: baseline `0.217`, tau=5e-4 `0.221` `+2.0%`, N=160k `0.188` `−13.3%`, K=16 `0.252` `+16.1%`), and active per-cluster `R_0.5 ≈ 0.11–0.17` collapses while control spreads to `≈3.8` — a `~23×` contrast. The gap percentages are now traced in `tetra_refinement/tetra_gap_summary.csv`. **Scope caveat:** the gap is positive *in the time-average*; at the **final instant** the single-seed N=160k control `d_min(T)=2.204` dips *below* active `2.311` (`final_instant_gap=−0.108`, single-seed control-centroid noise — see Limitation 3), so the instantaneous-final attraction is not preserved for that one noisy seed. The robust statement is the time-averaged gap, not the final instant. (The seed-0-only tau gap read `+38%`; the required 4-seed expansion shows this was single-seed scatter → `+2.0%`.)

## C. What the data do NOT support

- **No universal/continuum critical mass.** The `M*` transition is a fixed-bandwidth numerical one (`M*` shifts with `K`).
- **No blow-up time / no converged singular peak.** Not computed; reconstructed `|∇v|` is bandwidth-sensitive — the **same-cloud** probe on the `K=12` cloud gives `G_v^{K=8/12/16} ≈ 8/21/42` at `T` (`radial_resolution_audit.csv`).
- **No resolution-independent final core radius.** Radial `R_0.5(T)/R_0.5(0)` is strongly `K`-dependent (`0.570/0.209/0.146`); tetra per-cluster `R_0.5` deepens with `K` (`0.165→0.108`). The deep core (`<h_K`) is bandwidth-limited.
- **No single-central-cluster merging in the tetra test.** Active clusters collapse *individually* at the vertices (overlap stays `≫1`, `m_center` low); they attract but do not merge on `T=3`.
- **No branching-vs-weighted superiority claim** is made or supported by these runs (this experiment has no weighted comparator).
- **The radial chemotactic drift is not `K`-stable.** `δ_{12,16}≥20%` for essentially all `t>0`, so the radial quantitative result is **resolution-limited**; only the qualitative delayed-turnover phenomenon is bandwidth-robust.

## D. Exact outstanding limitations / failed gates

1. **`Q_0.2≥3` gate is invalid** for this IC and is reported as audited, not passed (it never reaches 3 at any `K`). It is replaced by the same-cloud drift diagnostic.
2. **Radial drift is resolution-limited** (`δ_{12,16}≥20%` from `t≈0.04`); the final collapse depth is `K`- and (mildly) `tau`-sensitive. Only the qualitative turnover is robust.
3. **Single-seed control `d_min` is noisy late** in the tetra refinements (diffuse control centroids); mitigated by the circular-resultant reliability score (`A_min≥0.43` shows the centroid is still defined, but the late `d_min` ripples are single-seed scatter). The seed-0 tau gap `+38%` was such scatter — the 4-seed expansion (required by the >20% rule) gives `+2.0%`.
4. **No outstanding failed gate.** Every gate is PASS or an explicitly disclosed LIMITED (radial drift resolution-limited; final core radius not resolution-independent; `Q_0.2` gate invalid/audited). No gate was rescued by parameter tuning.

---

## E. Reproduction

All figures regenerate from saved CSV/NPZ without the solver — see `README.md` in this
directory. Raw per-run `diagnostics.csv` + `manifest.json` + `command.txt` are under each
`radial_K/`, `radial_tau/`, `tetra_refinement/`, `radial_state_figure/` subdirectory;
metrics CSVs are `radial_K/radial_K_metrics.csv`, `radial_tau/radial_tau_metrics.csv`,
`radial_resolution_audit/radial_resolution_audit.csv`,
`tetra_refinement/tetra_refinement_metrics.csv` and `tetra_refinement/tetra_gap_summary.csv`
(active–control `d_min` gap, mean-over-interval + final-instant, traceable); benchmark in
`performance/benchmark.{csv,json}`; test logs in `test_logs/`.
