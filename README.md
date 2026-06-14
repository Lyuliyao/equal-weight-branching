# Equal-weight branching particle methods for non-conservative transport–reaction–diffusion equations

Reference implementation and reproduction scripts for the manuscript

> *Adaptive equal-weight branching particle methods for non-conservative transport–reaction–diffusion equations*,
> Liyao Lyu and Huan Lei.

**Idea in one paragraph.** Non-conservative reaction terms give particle methods exponential weights: a weighted
update is unbiased, but under localized growth the represented mass concentrates onto a few particles and the
local effective sample size collapses exactly where the solution grows. This code implements an *equal-weight
stochastic branching* reaction step that converts mass creation into integer particle count (with a Poisson
birth–death kernel and a minimum-variance / stochastic-rounding integer kernel), and compares it head-to-head
with the weighted representation under shared transport randomness. Beyond the controlled comparisons, the same
branching step is stress-tested on a moving growth region, a genuinely nonlinear six-dimensional field-coupled
kinetic Keller–Segel model (only a three-dimensional field solve), and two- and three-dimensional Keller–Segel
concentration/focusing regimes.

## Repository layout

| Path | Paper section | Contents |
|---|---|---|
| `experiments/mms/` | §5.1 | Manufactured-solution verification (error vs `N`, `τ`, `K`; Harris-coupled splitting-bias projection) |
| `experiments/branch_vs_weighted/` (`experiment.py`) | §5.2 | Stationary localized growth: branching vs weighted and systematic resampling at matched particle-step cost |
| `experiments/branch_vs_weighted/` (`experiment_switch.py`) | §5.3 | Switching growth: structural failure mode of global resampling (lineage-diversity collapse) |
| `experiments/branch_vs_weighted/` (`staged_multi_island.py`) | not in paper (record only) | Staged 16-island benchmark. Branching keeps the lowest global `L²` and by far the largest late-island count, but at matched particle-step cost a cost-matched resampling baseline attains lower per-island *mass* error. This did not meet the "branching wins the local metric" bar, so it is **not in the paper**; the full record is in `staged_parameter_log.md` / `reference_results/staged_multi_island/<run_id>/parameter_log.md` |
| `experiments/branch_vs_weighted/` (`multi_island.py`) | not in paper (record only) | Static separated-island diagnostic (global ESS vs local ESS); superseded by the staged variant |
| `experiments/branch_vs_weighted/` (`compressive_multi_island.py`) | not in paper (negative record) | Staged islands + compressive drift, with local field/peak/narrow-Gaussian metrics. Branching keeps the largest late local count but does not win the local-field metrics; at strong compression the reference under-resolves. Negative result; see `compressive_parameter_log.md` |
| `experiments/kinetic_ks/` | §5.4 | Six-dimensional field-coupled kinetic Keller–Segel (3D screened-Poisson field from the spatial marginal) |
| `experiments/keller_segel/mass_balance/` | §5.5 | Conservative KS implementation check (exact mass balance) |
| `experiments/keller_segel/concentration/` | §5.5 | Supercritical 2D concentration (parabolic–parabolic, finite-difference comparison) |
| `experiments/keller_segel/concentration_ldg/` | §5.5 | Core-local & resolution-gap (LDG-style) diagnostics: parabolic–elliptic reduction on a core-adaptive Fourier window |
| `experiments/keller_segel/pp_injection/` | §5.4 | Cross-species injection for the parabolic–parabolic chemical equation `v_t=Δv+u−v` (mass-law check) |
| `experiments/keller_segel/ldg_comparison/` | §5.5 | LDG-aligned parabolic–parabolic concentration (Li–Shu–Yang IC + report times, `t_gap` resolution-gap indicator) |
| `experiments/resolution_hybrid/` | §5.x | Local reconstruction diagnostics: global spectrum + local residual window/blob/particles |
| `experiments/reconstruction_audit/` | Appendix | Fourier-bandwidth & periodic-Gaussian-KDE robustness audit of the §5.2/§5.3 error ordering (reruns exact production dynamics, validated byte-faithful) |
| `experiments/keller_segel/focusing_3d/` | §5.6 | Three-dimensional Keller–Segel focusing transition (radial mass sweep + tetrahedral clusters) |
| `experiments/highdim/` | Appendix | Dense 4D/6D reconstruction on a separable manufactured solution + FHT low-rank marginal diagnostics |
| `experiments/keller_segel/logistic/` | Appendix | Non-conservative logistic KS (coupled-system check) |
| `experiments/allen_cahn/` | Appendix | Allen–Cahn sanity check |
| `reference_results/` | Tables & Figures | CSV outputs of the production runs used in the paper, with the exact configs |
| `paper/` | Manuscript | Self-contained LaTeX source (`cmame-main.tex`) + figures + compiled PDF |

Each experiment directory is self-contained: shared modules are vendored in rather than imported across
directories (`kinetic_ks/` vendors `common_highdim.py`; `keller_segel/concentration_ldg/` vendors the
core-adaptive-window solver `adaptive_window.py` + `simulation_blowup.py`). Each has its own `README.md` and a
SLURM submission script (`run_me` / `run_*.sb`).

## Environment

Python ≥ 3.10 with [JAX](https://github.com/jax-ml/jax) (CPU is sufficient; `x64` is enabled by the scripts):

```bash
pip install -r requirements.txt
```

The SLURM scripts assume a conda environment named `heat`, except `keller_segel/focusing_3d/` which uses
`jax-baseline`; edit the `run_*` scripts for your cluster.

## Quick start (smoke tests, a few minutes on CPU)

```bash
cd experiments/mms                         && python experiment.py --test     # §5.1 unit tests: kernel unbiasedness + MMS residual
cd ../branch_vs_weighted                   && python experiment.py --smoke    # §5.2 stationary localized growth
cd ../branch_vs_weighted                   && python experiment_switch.py --smoke   # §5.3 switching growth
cd ../kinetic_ks                           && python experiment_kinetic.py --smoke  # §5.4 6D field-coupled kinetic KS
cd ../keller_segel/focusing_3d             && python simulation_focusing.py --ic_type radial --N 20000 --M 60 --sigma 0.45 --H 12 --tau 1e-4 --T 0.005 --L 12 --kappa 0.1 --chi 1.0 --seed 0 --n_report 3 --out_dir scratch_smoke   # §5.6 (jax-baseline env)
cd ../../highdim                           && python experiment.py --smoke    # appendix: dense high-dim reconstruction
```

## Reproducing the paper figures and tables

| Paper item | Command |
|---|---|
| MMS errors vs `N`/`τ`/`K`, splitting-bias table (§5.1) | `cd experiments/mms && python experiment.py && python plot.py` |
| Branch-vs-weighted table + snapshots/nESS/L²/boxplot (§5.2) | `cd experiments/branch_vs_weighted && python experiment.py && python plot.py` |
| Cost-matched resampling rows of the §5.2 table | `cd experiments/branch_vs_weighted && python cost_match.py` (weighted cost-match) and `python cost_match_resample.py --N0 38000` (the ESS-resample cost-matched row; CSV under `reference_results/branch_vs_weighted/cost_match_resample/`) |
| Switching-growth table + ancestor diversity (§5.3) | `cd experiments/branch_vs_weighted && python experiment_switch.py` |
| Switching-growth Figure 6 (full domain + A/B microscope zooms) | `cd experiments/branch_vs_weighted && python plot_switch.py` (reads `reference_results/switch/` saved data only; overwrites `paper/figure/switch_snapshots.pdf`; no-clip row-wise color scale recorded in the plot_data) |
| 6D field-coupled kinetic KS table + figures (§5.4) | `cd experiments/kinetic_ks && sbatch run_me "--config config_pilot.json" && python plot_kinetic.py --results_dir results/pilot` |
| KS mass balance (§5.5) | `cd experiments/keller_segel/mass_balance && python simulation.py <N>` |
| KS 2D concentration vs finite difference (§5.5) | `cd experiments/keller_segel/concentration && python simulation.py <N>` (reference: `python finite_difference.py`) |
| KS parabolic–elliptic core/resolution-gap diagnostics + `t_gap` (§5.4) | `cd experiments/keller_segel/concentration_ldg && bash submit_focused.sh` then `python tgap.py --pairs ...` and `python plot_ldg.py` |
| KS cross-species injection mass-law check (§5.4) | `cd experiments/keller_segel/pp_injection && python simulation.py` then `python plot.py --results_dir results` |
| KS LDG-aligned parabolic–parabolic concentration + `t_gap` (§5.4) | `cd experiments/keller_segel/ldg_comparison && python simulation.py --report_times 6e-5 1.2e-4 2e-4 ...` then `python tgap.py --pairs ...` and `python plot_ldg_style.py --results_dir ...` |
| Local reconstruction diagnostics (§5.4) | `cd experiments/resolution_hybrid && python core_window_demo.py --blob && python plot_hybrid_reconstruction.py --results_dir ...` |
| Fourier/KDE reconstruction audit (Appendix) | `cd experiments/reconstruction_audit && python audit_fourier_kde.py --seeds 0 1 2 && python plot_audit.py` (validation tolerances in each `reference_results/reconstruction_audit/*/manifest.json`) |
| Staged / static multi-island (not in paper; diagnostic records) | `cd experiments/branch_vs_weighted && python staged_multi_island.py --config config_staged_multi_island.json --smoke` (see `staged_parameter_log.md`) |
| 3D KS focusing: mass sweep, self-convergence, tetrahedral (§5.6) | `cd experiments/keller_segel/focusing_3d && bash submit_focused.sh` then `python plot_focusing.py --runs results/* && python plot_selfconv_NH.py` |
| Dense 4D/6D reconstruction MMS (Appendix) | `cd experiments/highdim && python experiment.py && python experiment.py --d6 && python plot.py` |
| Logistic KS (Appendix) | `cd experiments/keller_segel/logistic && python simulation.py <N>` |
| Allen–Cahn (Appendix) | `cd experiments/allen_cahn && python simulation.py <N>` |

On SLURM: `sbatch run_me` (full config) or `sbatch run_me --smoke` inside each experiment directory.
Production runs in the paper used fixed seed lists declared in each `experiment.py` `CONFIG`; the exact
configurations of the production runs are archived under `reference_results/*/config_used.json` together with
the resulting metrics CSVs.

## Notes

- All experiments run on **fixed-size particle buffers with an active mask** so that JAX/XLA compiles each time
  step once; branching compaction is performed host-side with `numpy.repeat`.
- In the branching-vs-weighted comparisons, all methods share the **same initial particles and the same
  Brownian-increment stream** per seed; only the reaction representation differs.
- `experiments/highdim/` vendors three files of the functional-hierarchical-tensor (FHT) reconstruction from
  [Xun-Tang123/FHT_for_deans_equation](https://github.com/Xun-Tang123/FHT_for_deans_equation)
  (Tang & Ying, arXiv:2503.22816, MIT license); see `THIRD_PARTY.md`. The FHT is used only as a low-rank
  *diagnostic* — the dynamics never depend on it.

## License

MIT — see `LICENSE`. The vendored FHT files retain their upstream MIT license and attribution.
