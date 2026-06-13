#!/bin/bash
# ============================================================================
# SLURM SUBMISSION TEMPLATES for the KS 2D LDG-diagnostics runs.
#   *** DO NOT AUTO-SUBMIT FROM THE AGENT. ***  The orchestrator Codex-verifies
#   the SMOKE command first, runs it, and only then submits the production grid.
#
# Same conventions as ../blowup_time/run_production.sh:
#   -A Multiscaleml -C amr, env `heat`, jax CPU fallback, sbatch --wrap.
# Each job writes diag_*.csv + snapshots/*.npz + config.json into --outdir.
# ============================================================================
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
RUNDIR=/mnt/gs21/scratch/lyuliyao/SDE_PDE/equal-weight-branching/experiments/keller_segel/concentration_ldg
ENVPREFIX="MPLBACKEND=Agg JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=16"
cd "$RUNDIR" || exit 1

# ---------------------------------------------------------------------------
# SMOKE (run this AFTER Codex approval, BEFORE submitting the grid).
# Small: N=2e4, K=5, dt=1e-7, n_steps=200, 1 seed.  CPU, a few minutes.
# Produces results_smoke/diag_*.csv + results_smoke/snapshots/*.npz.
# ---------------------------------------------------------------------------
# SMOKE:
#   cd $RUNDIR && MPLBACKEND=Agg JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=16 \
#     /mnt/home/lyuliyao/.conda/envs/heat/bin/python simulation_ldg.py \
#     --N 20000 --K 5 --dt 1e-7 --n_steps 200 --diag_every 10 --seed 0 \
#     --q_window 0.8 --verbose --outdir results_smoke
#
#   # then plots:
#   cd $RUNDIR && MPLBACKEND=Agg JAX_PLATFORM_NAME=cpu \
#     /mnt/home/lyuliyao/.conda/envs/heat/bin/python plot_ldg.py \
#     --out_dir results_smoke

# ---------------------------------------------------------------------------
# PRODUCTION GRID:  N in {4e4,1.6e5,6.4e5,2.56e6}, K in {5,7,9,11},
#                   tau in {1e-7, 5e-8}.
# n_steps chosen so the time span covers the canonical report times
# (5e-5,1e-4,1.5e-4): tau=1e-7 -> >=1500 steps (T=1.5e-4); tau=5e-8 -> >=3000.
# ---------------------------------------------------------------------------
for N in 40000 160000 640000 2560000; do
  for K in 5 7 9 11; do
    # tau = 1e-7 (n_steps 1500 -> T=1.5e-4)
    sbatch -A Multiscaleml -C amr --job-name=ks_ldg_N${N}_K${K}_t1e7 \
           --time=16:00:00 --mem=48G --cpus-per-task=16 \
           --output="$RUNDIR/slurm_N${N}_K${K}_t1e7_%j.out" \
           --wrap="cd $RUNDIR && $ENVPREFIX $PY simulation_ldg.py \
                   --N $N --K $K --dt 1e-7 --n_steps 1500 --diag_every 10 \
                   --q_window 0.8 --gamma 3.0 --seed 0 --verbose \
                   --outdir results"
    # tau = 5e-8 (n_steps 3000 -> T=1.5e-4); temporal-refinement check
    sbatch -A Multiscaleml -C amr --job-name=ks_ldg_N${N}_K${K}_t5e8 \
           --time=16:00:00 --mem=48G --cpus-per-task=16 \
           --output="$RUNDIR/slurm_N${N}_K${K}_t5e8_%j.out" \
           --wrap="cd $RUNDIR && $ENVPREFIX $PY simulation_ldg.py \
                   --N $N --K $K --dt 5e-8 --n_steps 3000 --diag_every 20 \
                   --q_window 0.8 --gamma 3.0 --seed 0 --verbose \
                   --outdir results"
  done
done

# ---------------------------------------------------------------------------
# PAIRED (N,K) + (4N,2K) runs for the t_gap resolution-gap indicator.
# Pairs: (4e4,5)->(1.6e5,10); (1.6e5,5)->(6.4e5,10); (6.4e5,5)->(2.56e6,10).
# (K=10 is the 2K partner of K=5; it is the refined run only, not in the main
#  grid above which uses odd K.)  All at tau=1e-7, n_steps 1500, seed 0.
# ---------------------------------------------------------------------------
PAIRS=( "40000 5 160000 10" "160000 5 640000 10" "640000 5 2560000 10" )
for quad in "${PAIRS[@]}"; do
  read -r Nb Kb Nr Kr <<< "$quad"
  for cfg in "$Nb $Kb" "$Nr $Kr"; do
    read -r N K <<< "$cfg"
    sbatch -A Multiscaleml -C amr --job-name=ks_ldg_pair_N${N}_K${K} \
           --time=16:00:00 --mem=48G --cpus-per-task=16 \
           --output="$RUNDIR/slurm_pair_N${N}_K${K}_%j.out" \
           --wrap="cd $RUNDIR && $ENVPREFIX $PY simulation_ldg.py \
                   --N $N --K $K --dt 1e-7 --n_steps 1500 --diag_every 10 \
                   --q_window 0.8 --gamma 3.0 --seed 0 --verbose \
                   --outdir results"
  done
done

# ---------------------------------------------------------------------------
# POST-PROCESSING (after all jobs finish; run on a login/compute node).
#   t_gap table from the paired runs:
#   $ENVPREFIX $PY tgap.py \
#     --pairs \
#       results/diag_N40000_K5_dt1e-07_q0.8_seed0.csv:results/diag_N160000_K10_dt1e-07_q0.8_seed0.csv \
#       results/diag_N160000_K5_dt1e-07_q0.8_seed0.csv:results/diag_N640000_K10_dt1e-07_q0.8_seed0.csv \
#       results/diag_N640000_K5_dt1e-07_q0.8_seed0.csv:results/diag_N2560000_K10_dt1e-07_q0.8_seed0.csv \
#     --out results/tgap_table
#   plots:
#   $ENVPREFIX $PY plot_ldg.py --out_dir results

echo "Templates only. Submit the production grid manually AFTER the SMOKE run is"
echo "Codex-approved and verified.  SMOKE command is in the commented block above."
