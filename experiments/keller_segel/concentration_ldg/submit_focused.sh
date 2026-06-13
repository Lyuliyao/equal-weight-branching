#!/bin/bash
# Focused LDG production: the two lower-resolution t_gap pairs (decisive for the
# resolution-gap indicator + give high-res snapshots/curves). The heaviest pair
# (6.4e5->2.56e6) is deferred (2.56e6 ~ 11h) and noted as a confirmation run.
#   pair1: (4e4,K5) -> (1.6e5,K10)
#   pair2: (1.6e5,K5) -> (6.4e5,K10)
# Runs: (4e4,5),(1.6e5,10),(1.6e5,5),(6.4e5,10), seed 0.
# Time horizon: tau=1e-6, 4000 steps -> T=4e-3, which REACHES the core-collapse /
# resolution-gap regime (Codex LDG check: this 10pi-mass benchmark concentrates
# around t~1e-3..3e-3; an earlier T=1.5e-4 run is far too short for t_gap).
# Snapshots at early {5e-5,1e-4,1.5e-4} (auto) + later {5e-4,1e-3,2e-3,4e-3}.
# Each job writes to its OWN outdir results_N${N}_K${K} so config.json/README and
# the diag CSV/snapshots never collide across jobs.
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
RUNDIR=/mnt/gs21/scratch/lyuliyao/SDE_PDE/equal-weight-branching/experiments/keller_segel/concentration_ldg
ENVPREFIX="MPLBACKEND=Agg JAX_PLATFORMS=cpu OMP_NUM_THREADS=16"
cd "$RUNDIR" || exit 1

submit () {  # $1=N $2=K
  local N=$1 K=$2
  local OUT="results_N${N}_K${K}"
  sbatch -A Multiscaleml -C amr --job-name=ks_ldg_N${N}_K${K} \
         --time=12:00:00 --mem=48G --cpus-per-task=16 \
         --output="$RUNDIR/slurm_N${N}_K${K}_%j.out" \
         --wrap="cd $RUNDIR && $ENVPREFIX $PY simulation_ldg.py \
                 --N $N --K $K --dt 1e-6 --n_steps 4000 --diag_every 20 \
                 --q_window 0.8 --gamma 3.0 --seed 0 \
                 --report_times 5e-4 1e-3 2e-3 4e-3 \
                 --verbose --outdir $OUT"
}

submit 40000  5
submit 160000 10
submit 160000 5
submit 640000 10
echo "submitted 4 LDG focused runs (outdirs results_N*_K*)"
# t_gap post-processing (after completion):
#   tgap.py pairs: results_N40000_K5/diag_*.csv : results_N160000_K10/diag_*.csv
#                  results_N160000_K5/diag_*.csv : results_N640000_K10/diag_*.csv
