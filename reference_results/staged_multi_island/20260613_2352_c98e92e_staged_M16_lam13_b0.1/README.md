# Staged separated growth-island benchmark (paper §5.2) — production run

Production reference for the **staged** local-degeneracy benchmark
(`experiments/branch_vs_weighted/staged_multi_island.py`). This replaces the
static multi-island run as the main §5.2 benchmark: it shows branching is
quantitatively *better* (not just a diagnostic) when separated growth regions
become important at different times.

## Mechanism

`r(t,x) = λ Σ_g s_g(t) Σ_{m∈G_g} a_m G(x;c_m) − β` with M=16 islands in G=4
spatially-separated activation groups (2×2 checkerboard sub-lattices) and smooth
tanh windows. The **late group** (g=4, window [0.8,1.2]) turns on last. Global
ESS-triggered resampling keeps a healthy global nESS but, by repeatedly
concentrating lineages on the earlier-active groups, depletes the late-island
lineages; equal-weight branching creates particles where the late source turns on
and keeps the late islands resolved.

## Configuration (tuned; see `staged_parameter_log.md`)

| | |
|---|---|
| λ, β | 13, 0.1 (growth regime; islands accumulate, weighted degenerates) |
| amplitudes | uniform (`amp_var=0`): isolates the activation-timing mechanism |
| σ, D, T, τ | 0.16, 0.01, 1.2, 1e-3 |
| windows | [0,0.35], [0.25,0.6], [0.5,0.9], [0.8,1.2]; δ=0.03 |
| initial cloud | **stratified uniform** (equal particles per `B_m`), shared across same-budget methods |
| particles | N0=2×10⁴, K=64, grid 512²; 8 seeds |
| methods | weighted, weighted+ESS, cost-matched weighted+ESS, min-var branching |

The cost-matched ESS particle count is `N0_cm = round(C_branch / (T/τ))` so its
integrated particle-steps match branching. Exact config / git hash / package
versions are in `config.json` / `manifest.json`.

## How it was run

```bash
RUNDIR=<this dir> CFG=<...>/config_staged_multi_island.json \
  sbatch experiments/branch_vs_weighted/run_staged_multi_island.sb
python experiments/branch_vs_weighted/merge_staged_multi_island.py --in_dir <this dir> --seeds 0 1 2 3 4 5 6 7
```

## Primary metrics (`metrics_summary.csv`, `late_group_metrics.csv`)

Per-method, over seeds: particle-steps, global L², all/late-island mean & max
`E_m`, `#{E_m>20%}`, max late local `L²(B_m)`, min (all/late) island local
effective count, final active count. The **late** columns are the discriminating
ones. `E_m = |μ_T(B_m) − ∫_{B_m} u_ref| / ∫_{B_m} u_ref`.

## Regenerate the figures (no solver rerun)

```bash
python experiments/branch_vs_weighted/plot_staged_multi_island.py --results_dir <this dir>
```

Figure 1: reference / weighted+ESS / cost-matched weighted+ESS / branching, late
islands marked, with one diagnostic-selected magnifier on the worst late island of
the cost-matched ESS baseline (`m* = argmax_{m∈late} E_m`). Figure 2: global nESS,
min late-island local count, and max late local `L²(B_m)` versus time.

## One-line result (honest; see `parameter_log.md`)

This is reported as a **diagnostic, not a main-paper "branching wins" benchmark.**
With 8-seed statistics, branching wins the **global** `L²` (0.343, best) and keeps
the highest late-island local count (1517 vs 368), but the **cost-matched
weighted+ESS baseline beats branching on every per-island metric** at matched
particle-step cost (late mean `E_m` 0.137 vs 0.202; late local `L²` 0.465 vs
0.511). The per-island *mass* is a forgiving integral that favors weighted
particles — branching's mass carries reproduction variance set by the ancestor
count, which extra particles do not reduce. Branching's genuine advantage is the
**global `L²` / peak resolution**, demonstrated by the single-peak (§5.2) and
switching (§5.3) benchmarks; this staged run did not meet the success criterion of
beating cost-matched resampling on the late per-island metrics, so per the project
guardrails the multi-island benchmark is kept here as a diagnostic rather than
placed in the main paper.
