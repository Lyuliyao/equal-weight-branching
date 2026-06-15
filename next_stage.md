# Solver-field comparison using the same LDG-style blow-up proxy

**Purpose.**  The previous `sf_blob` sweep showed that a smoother blob/KDE residual field can enter the drift stably, but the reported mean abort/final time is only a numerical-stability diagnostic.  It cannot by itself demonstrate that the particle dynamics captures the Keller--Segel concentration or blow-up onset more accurately.  The next step is to compare solver-field variants using the **same LDG-style resolution-gap time** already used for the particle method and the fixed-flux LDG reference.

This note specifies the exact experiment, definitions, run matrix, analysis script, outputs, and decision rules.

---

## 1. What must be compared

We compare different **solver-field reconstructions**, i.e. reconstructions used inside the time step to compute the chemotactic drift

\[
X_{u,i}^{n+1}
= X_{u,i}^n + \tau\,\nabla \widehat v_n(X_{u,i}^n) + \sqrt{2\tau}\,\xi_i^n.
\]

The comparison must not be based on final-time visualization or abort time.  Each solver field generates its own particle trajectory, and all trajectories are evaluated with the same diagnostic norm.

### Solver fields to compare

Use the following configurations:

1. **Baseline:** `current_fourier`

   Current single-window Fourier field used in the solver drift.

2. **Blob residual, moderate bandwidth:** `two_level_blob_residual`, `blob_ch=0.06`

   Solver field:

   \[
   \widehat v
   = v_{\rm lo}
   + \chi\left[\eta_h*\mu_v - \eta_h*(v_{\rm lo}\,dx)\right],
   \qquad
   v_{\rm lo}=P_{K_g}\mu_v,
   \qquad
   h=0.06L.
   \]

3. **Blob residual, smoother bandwidth:** `two_level_blob_residual`, `blob_ch=0.09`

   Same formula, but with

   \[
   h=0.09L.
   \]

Optional record-only configuration:

4. **Spectral residual:** `two_level_spectral_residual`, `Kl=24`, `hybrid_taper_hi=0.25`

   Keep this only as a noise-limited reference.  It should not be promoted unless it beats the blob residual under the same blow-up proxy.

---

## 2. Main metric: LDG-matched particle resolution-gap time

For a particle trajectory with total particle count \(N_p\), define the diagnostic norm by projecting the particle measure onto the same \(P^1\) DG space used by the LDG reference:

\[
S^{\rm DG}_{N_p,n}(t)
=
\left\|\Pi_n^{P^1{\rm DG}}\mu^u_{N_p}(t)\right\|_{L^2},
\]

where the \(L^2\) norm is computed with the same mass matrix as the LDG solver.  Use the cross/split estimator

\[
\left(S^{\rm DG,cross}_{N_p,n}\right)^2
=
\left\langle
\Pi_n^{P^1{\rm DG}}\mu^{u,a}_{N_p},
\Pi_n^{P^1{\rm DG}}\mu^{u,b}_{N_p}
\right\rangle
\]

to reduce the empirical self-term bias of the raw projected Dirac measure.

For the particle analogue of LDG grid refinement, use

\[
(N_p,n)\longrightarrow (4N_p,2n),
\]

because in two dimensions halving the particle spacing requires four times as many particles.

For each solver field \(m\), define

\[
\overline S^{m}_{N_p,n}(t)
=
\frac1{M}\sum_{s=1}^M
S^{m,{\rm DG,cross},(s)}_{N_p,n}(t),
\]

and

\[
 t_b^m(N_p,n;\theta)
 =
 \inf\left\{
 t:
 \frac{\overline S^m_{4N_p,2n}(t)}{\overline S^m_{N_p,n}(t)}
 \ge \theta
 \right\},
 \qquad \theta=1.05.
\]

Use a persistence rule: the ratio must remain above \(\theta\) for at least

\[
\Delta t_{\rm persist}=5\times10^{-6}.
\]

The main comparison is therefore

\[
(8\times10^4,80)\to(3.2\times10^5,160).
\]

The fixed-flux LDG reference values are

\[
t_b^{\rm LDG}(80\to160;1.05)=5.95\times10^{-5},
\]

\[
t_b^{\rm LDG}(160\to320;1.05)=8.43\times10^{-5}.
\]

The particle value should be interpreted as a **numerical resolution-gap indicator**, not a continuum blow-up time.

---

## 3. Why mean abort/final time is not the main metric

For each seed,

\[
 t_{\rm end}^{(s)} =
 \begin{cases}
 T, & \text{if the run reaches the final horizon},\\
 t_{\rm abort}^{(s)}, & \text{if a guard stops the run early}.
 \end{cases}
\]

The previously reported “mean abort/final time” is just

\[
 \frac1M\sum_s t_{\rm end}^{(s)}.
\]

This is useful for stability, but not accuracy.  A smoother solver field can have a later abort time simply because it reduces \(\max|\nabla \widehat v|\), and a sharper solver field can abort earlier even if it captures a stronger core.  Therefore:

- use **mean abort/final time** only as a secondary stability diagnostic;
- use **LDG-style \(t_b\)** as the primary concentration / blow-up-onset proxy;
- do not claim improved dynamics from stability alone.

---

## 4. Required simulation matrix

### Main production matrix

Run all combinations:

| solver field | parameters | \(N_p\) | DG readout | seeds |
|---|---|---:|---|---|
| `current_fourier` | `K=10` | `80000` | `80 160` | `0 1 2 3` |
| `current_fourier` | `K=10` | `320000` | `80 160` | `0 1 2 3` |
| `two_level_blob_residual` | `Kg=8`, `blob_h_rule=frac_L`, `blob_ch=0.06` | `80000` | `80 160` | `0 1 2 3` |
| `two_level_blob_residual` | `Kg=8`, `blob_h_rule=frac_L`, `blob_ch=0.06` | `320000` | `80 160` | `0 1 2 3` |
| `two_level_blob_residual` | `Kg=8`, `blob_h_rule=frac_L`, `blob_ch=0.09` | `80000` | `80 160` | `0 1 2 3` |
| `two_level_blob_residual` | `Kg=8`, `blob_h_rule=frac_L`, `blob_ch=0.09` | `320000` | `80 160` | `0 1 2 3` |

This is \(3\) solver-field configurations \(\times\) \(2\) particle resolutions \(\times\) \(4\) seeds = **24 runs**.

### Optional record-only matrix

Add spectral residual only if resources allow:

| solver field | parameters | \(N_p\) | DG readout | seeds |
|---|---|---:|---|---|
| `two_level_spectral_residual` | `Kg=8`, `Kl=24`, `hybrid_taper_hi=0.25` | `80000` | `80 160` | `0 1 2 3` |
| `two_level_spectral_residual` | `Kg=8`, `Kl=24`, `hybrid_taper_hi=0.25` | `320000` | `80 160` | `0 1 2 3` |

---

## 5. Use the same time step and output grid

To avoid mixing time-discretization effects with resolution-gap effects, use the same time step for all solver fields and both particle counts:

\[
\tau=2.0\times10^{-7},
\qquad
T=2.0\times10^{-4},
\qquad
n_{\rm steps}=1000.
\]

Use

\[
\Delta t_{\rm output}=10^{-6}.
\]

In `simulation.py`, this is achieved with

```bash
--tau 2e-7 --n_steps 1000 --diag_every 5
```

All runs should use:

```bash
--K 10
--dg_readout_n 80 160
--cfl_abort 5.0
--filter_s 0.5
--q_window 0.8
--report_times 6e-5 1.2e-4 2e-4
```

The solver fields differ only through `--solver_field` and the blob/spectral parameters.

---

## 6. Concrete run commands

Assume the working directory is the repository root.

Create an output directory:

```bash
RUNID=$(date +%Y%m%d_%H%M)_solverfield_tb
OUT=reference_results/keller_segel_ldg_pp/solver_field_tb_${RUNID}
mkdir -p "$OUT"
```

### 6.1 Current Fourier baseline

```bash
for N in 80000 320000; do
  for seed in 0 1 2 3; do
    python experiments/keller_segel/ldg_comparison/simulation.py \
      --N ${N} \
      --K 10 \
      --tau 2e-7 \
      --n_steps 1000 \
      --diag_every 5 \
      --seed ${seed} \
      --solver_field current_fourier \
      --dg_readout_n 80 160 \
      --cfl_abort 5.0 \
      --filter_s 0.5 \
      --q_window 0.8 \
      --report_times 6e-5 1.2e-4 2e-4 \
      --outdir "${OUT}/current_fourier_N${N}_seed${seed}"
  done
done
```

### 6.2 Blob residual, `c_h=0.06`

```bash
for N in 80000 320000; do
  for seed in 0 1 2 3; do
    python experiments/keller_segel/ldg_comparison/simulation.py \
      --N ${N} \
      --K 10 \
      --tau 2e-7 \
      --n_steps 1000 \
      --diag_every 5 \
      --seed ${seed} \
      --solver_field two_level_blob_residual \
      --Kg 8 \
      --blob_h_rule frac_L \
      --blob_ch 0.06 \
      --blob_min_count 100 \
      --dg_readout_n 80 160 \
      --cfl_abort 5.0 \
      --filter_s 0.5 \
      --q_window 0.8 \
      --report_times 6e-5 1.2e-4 2e-4 \
      --outdir "${OUT}/blob_ch006_N${N}_seed${seed}"
  done
done
```

### 6.3 Blob residual, `c_h=0.09`

```bash
for N in 80000 320000; do
  for seed in 0 1 2 3; do
    python experiments/keller_segel/ldg_comparison/simulation.py \
      --N ${N} \
      --K 10 \
      --tau 2e-7 \
      --n_steps 1000 \
      --diag_every 5 \
      --seed ${seed} \
      --solver_field two_level_blob_residual \
      --Kg 8 \
      --blob_h_rule frac_L \
      --blob_ch 0.09 \
      --blob_min_count 100 \
      --dg_readout_n 80 160 \
      --cfl_abort 5.0 \
      --filter_s 0.5 \
      --q_window 0.8 \
      --report_times 6e-5 1.2e-4 2e-4 \
      --outdir "${OUT}/blob_ch009_N${N}_seed${seed}"
  done
done
```

### 6.4 Optional spectral residual reference

```bash
for N in 80000 320000; do
  for seed in 0 1 2 3; do
    python experiments/keller_segel/ldg_comparison/simulation.py \
      --N ${N} \
      --K 10 \
      --tau 2e-7 \
      --n_steps 1000 \
      --diag_every 5 \
      --seed ${seed} \
      --solver_field two_level_spectral_residual \
      --Kg 8 \
      --Kl 24 \
      --hybrid_taper_hi 0.25 \
      --dg_readout_n 80 160 \
      --cfl_abort 5.0 \
      --filter_s 0.5 \
      --q_window 0.8 \
      --report_times 6e-5 1.2e-4 2e-4 \
      --outdir "${OUT}/spectral_taper025_N${N}_seed${seed}"
  done
done
```

---

## 7. SLURM array version

Create `experiments/keller_segel/ldg_comparison/run_solver_field_tb_sweep.sb`:

```bash
#!/bin/bash
#SBATCH -J sf_tb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH --array=0-23
#SBATCH -o sf_tb_%A_%a.out

set -euo pipefail

REPO=${REPO:-$PWD}
OUT=${OUT:-$REPO/reference_results/keller_segel_ldg_pp/solver_field_tb_${SLURM_JOB_ID}}
mkdir -p "$OUT"

# 3 configs x 2 resolutions x 4 seeds = 24 tasks
CONFIGS=(current_fourier blob006 blob009)
NS=(80000 320000)
SEEDS=(0 1 2 3)

idx=$SLURM_ARRAY_TASK_ID
seed=${SEEDS[$((idx % 4))]}
idx=$((idx / 4))
N=${NS[$((idx % 2))]}
idx=$((idx / 2))
cfg=${CONFIGS[$idx]}

COMMON="--N ${N} --K 10 --tau 2e-7 --n_steps 1000 --diag_every 5 \
  --seed ${seed} --dg_readout_n 80 160 --cfl_abort 5.0 --filter_s 0.5 \
  --q_window 0.8 --report_times 6e-5 1.2e-4 2e-4"

case "$cfg" in
  current_fourier)
    FIELD="--solver_field current_fourier"
    ;;
  blob006)
    FIELD="--solver_field two_level_blob_residual --Kg 8 --blob_h_rule frac_L --blob_ch 0.06 --blob_min_count 100"
    ;;
  blob009)
    FIELD="--solver_field two_level_blob_residual --Kg 8 --blob_h_rule frac_L --blob_ch 0.09 --blob_min_count 100"
    ;;
esac

python $REPO/experiments/keller_segel/ldg_comparison/simulation.py \
  $COMMON $FIELD \
  --outdir "$OUT/${cfg}_N${N}_seed${seed}"
```

Run with:

```bash
REPO=$PWD OUT=$PWD/reference_results/keller_segel_ldg_pp/solver_field_tb_<runid> \
  sbatch experiments/keller_segel/ldg_comparison/run_solver_field_tb_sweep.sb
```

---

## 8. Analysis script specification

Create a new script:

```text
experiments/keller_segel/ldg_comparison/analyze_solver_field_tb.py
```

It should not reuse abort time as the main criterion.  It should compute the same LDG-style crossing for every solver field.

### 8.1 Inputs

Directory layout:

```text
solver_field_tb_<runid>/
  current_fourier_N80000_seed0/diag_*.csv
  current_fourier_N80000_seed1/diag_*.csv
  ...
  current_fourier_N320000_seed3/diag_*.csv
  blob_ch006_N80000_seed0/diag_*.csv
  ...
  blob_ch009_N320000_seed3/diag_*.csv
```

Required columns from each `diag_*.csv`:

```text
t
S_dg_cross_80
S_dg_cross_160
S_dg_raw_80
S_dg_raw_160
R_0.1
R_0.2
R_0.5
R_0.8
drift_cfl_solver_field
drift_cfl_fourier_diag
max_grad_solver_field
max_grad_fourier_diag
solver_field_h
solver_field_residual_E
solver_field_mode
```

If any required DG column is missing, the run should be marked invalid for the main \(t_b\) computation.

### 8.2 Main crossing computation

For each solver-field config `m`:

- low group: `N=80000`, column `S_dg_cross_80`;
- high group: `N=320000`, column `S_dg_cross_160`.

Build a common time grid:

\[
 t_k = k\times 10^{-6}.
\]

Use only the interval where all required seed curves exist.  Define

\[
 t_{\max}^{m}=\min_s t_{\rm end}^{m,N=80000,s}\wedge
              \min_s t_{\rm end}^{m,N=320000,s}.
\]

If \(t_{\max}^{m}<5\times10^{-5}\), mark the config invalid for the main comparison.

Interpolate each seed curve to the common grid up to \(t_{\max}^{m}\), then compute

\[
 \overline S_{\rm low}^{m}(t),
 \qquad
 \overline S_{\rm high}^{m}(t).
\]

Define

\[
 R^m(t)=
 \frac{\overline S_{\rm high}^{m}(t)}{\overline S_{\rm low}^{m}(t)}.
\]

Then

\[
 t_b^m(1.05)=
 \inf\{t:R^m(t)\ge1.05\text{ for at least }5\times10^{-6}\}.
\]

### 8.3 Bootstrap confidence interval

Bootstrap over seeds:

1. Resample low-resolution seeds with replacement.
2. Resample high-resolution seeds with replacement.
3. Recompute the ensemble mean ratio and \(t_b\).
4. Repeat 1000 times.
5. Report 5th and 95th percentiles.

Use independent resampling unless the Brownian/random seeds are deliberately coupled across low/high runs.  If coupled seeds are used, also report a paired bootstrap as sensitivity.

### 8.4 Secondary diagnostics

For each config, report:

```text
t_end_mean, t_end_std
fraction_reached_T
max drift_cfl_solver_field mean/std
max drift_cfl_fourier_diag mean/std
solver/Fourier CFL ratio
R_0.2(t=1e-4) mean/std
R_0.1(t=1e-4) mean/std
R_0.8(t=1e-4) mean/std
residual_E(t=1e-4) mean/std
```

These are secondary.  They should never replace \(t_b\).

---

## 9. Expected output files

The analysis script should create:

```text
solver_field_tb_summary.csv
solver_field_tb_summary.json
README.md
figures/solver_field_tb_ratio.pdf
figures/solver_field_tb_ratio.png
figures/solver_field_S_curves.pdf
figures/solver_field_core_radii.pdf
figures/solver_field_dual_cfl.pdf
plot_data/*.csv
```

### 9.1 `solver_field_tb_summary.csv`

Required columns:

```text
config
theta
low_group
high_group
n_low
n_high
n_seed_low
n_seed_high
tmax_complete
tb
ci_low
ci_high
ratio_max
ldg_ref_80_160
ldg_ref_160_320
on_ldg_scale
fraction_reached_T_low
fraction_reached_T_high
t_end_mean_low
t_end_mean_high
cfl_solver_max_mean_low
cfl_solver_max_mean_high
cfl_fourier_max_mean_low
cfl_fourier_max_mean_high
R02_tm_mean_low
R02_tm_mean_high
residual_E_tm_mean_low
residual_E_tm_mean_high
valid_main_tb
invalid_reason
```

### 9.2 README content

The generated README must contain:

1. the definition of \(S^{\rm DG}_{N_p,n}\);
2. the definition of \(t_b^{\rm part}\);
3. a table comparing solver fields;
4. a statement that abort/final time is only stability;
5. a statement that the comparison is a numerical resolution-gap indicator, not continuum blow-up time;
6. a decision section.

---

## 10. Interpretation rules

### Scenario A: blob changes stability but not \(t_b\)

If

\[
 t_b^{\rm blob}\approx t_b^{\rm current}
\]

within bootstrap CI, but blob has smaller real solver CFL or higher fraction reaching \(T\), then conclusion:

```text
Blob residual improves numerical stability / smoothness of the drift, but does not change the LDG-style concentration proxy. It is not an accuracy improvement on this benchmark.
```

This is the most likely scenario based on the existing `sf_blob` sweep.

### Scenario B: blob gives a significantly more LDG-like \(t_b\)

If blob has a \(t_b\) closer to the fixed-flux LDG interval \([5.95,8.43]\times10^{-5}\) with narrower CI, and the core radii are consistent, then conclusion:

```text
Blob residual improves the stability of the LDG-style particle blow-up proxy.
```

Still do not call it continuum blow-up time.

### Scenario C: blob pushes \(t_b\) later

If blob moves \(t_b\) to \(1.2\times10^{-4}\) or later while residual energy is very small or core radii do not change, likely interpretation:

```text
The blob may be oversmoothing the chemotactic drift; the later gap is not automatically better.
```

Check radii and LDG snapshots before making any claim.

### Scenario D: blob makes \(t_b\) earlier

If blob makes \(t_b\) much earlier, inspect:

```text
drift_cfl_solver_field spikes
abort_diagnostics.json
residual_E
R_0.1/R_0.2 seed variance
```

Do not call this improved concentration unless the change is robust across seeds and consistent with LDG reference.

### Scenario E: many runs abort before \(t_b\)

If a solver field does not survive beyond the crossing window plus persistence, mark it invalid:

```text
valid_main_tb = false
invalid_reason = aborted before crossing window
```

Do not extrapolate.

---

## 11. Paper-use decision

Only include solver-field variants in the paper if they improve the same LDG-style \(t_b\) metric or reduce uncertainty without compromising concentration diagnostics.

Current paper-safe result remains:

```text
Current particle method + LDG-matched DG readout gives a particle resolution-gap time on the same scale as fixed-flux LDG at adequate particle count.
```

Do not include:

```text
Blob residual solves near-blow-up dynamics.
```

unless the new same-definition \(t_b\) comparison proves it.

If blob only improves stability but not \(t_b\), keep it as a record/appendix note:

```text
A solver-level blob residual field can be inserted into the drift and reduces real solver-CFL, but did not produce a demonstrable accuracy gain in the LDG-style concentration proxy.
```

---

## 12. Immediate checklist

Before running production:

- [ ] Verify `test_blob_residual_vfield.py` passes.
- [ ] Verify `simulation.py --solver_field two_level_blob_residual --smoke` advances and writes `drift_cfl_solver_field`.
- [ ] Verify all production runs write `S_dg_cross_80` and `S_dg_cross_160`.
- [ ] Confirm all runs use the same `tau`, `n_steps`, `K`, `filter_s`, and `q_window`.
- [ ] Confirm output spacing is \(10^{-6}\).
- [ ] Confirm `abort_diagnostics.json` is written when a run aborts.

After running:

- [ ] Run `analyze_solver_field_tb.py`.
- [ ] Inspect ratio curves before trusting the CSV.
- [ ] Check every config reaches at least the crossing time plus persistence.
- [ ] Compare \(t_b\) and CI against LDG reference.
- [ ] Check core radii and residual energy to detect oversmoothing.
- [ ] Update `REVISION_RESULTS.md` with positive and negative outcomes.

---

## 13. Minimal conclusion template

Use this structure after the run:

```text
We compared solver-field reconstructions using the same LDG-style particle resolution-gap definition.  The diagnostic norm was the LDG-matched P1 DG cross estimator, and the main pair was (8e4,80)->(3.2e5,160).  The current Fourier solver gave tb = ... with CI ....  The blob residual c_h=0.06 gave tb = ... with CI ..., and c_h=0.09 gave tb = ... with CI ....  Abort/final time and solver-CFL are reported only as stability diagnostics.  Therefore, [decision].
```

Possible decisions:

```text
Decision 1: Blob residual improves stability but not the LDG-style blow-up proxy; keep record-only.
Decision 2: Blob residual improves the LDG-style proxy and can be used in §5.4.
Decision 3: Blob residual oversmooths or is inconclusive; keep current Fourier + DG readout as main result.
```
