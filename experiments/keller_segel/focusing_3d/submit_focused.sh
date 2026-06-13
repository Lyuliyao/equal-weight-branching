#!/bin/bash
# Focused 3D KS focusing production. REDUCED scales vs plan (which lists N up to
# 3.2e6, H up to 32, tau=1e-4). The particle-Fourier field solve is O(N*H^3) per
# step, so the plan's tau=1e-4 (1e4 steps) at large N/H is days on CPU. Chosen:
# tau=5e-4 (2000 steps, T=1). Per-step ~ 0.15s*(N*H^3)/(2e4*12^3) from the smoke.
# All jobs below are <~4h, within the 8h walltime in run_focusing.sb.
# Worst feasible: N=2e5/H24 ~6.7h (kept at N=1e5/H24 ~3.3h here). Larger-N /
# finer-H / finer-tau are confirmation runs (deferred; recorded in manifest).
# drift_cfl is logged each snapshot (smoke ~0.036 at tau=1e-4; ~0.08 at 5e-4).
# Repo-relative: resolve run_focusing.sb next to this submit script.
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_focusing.sb"
TAU=5e-4
T=1.0

# (A) radial mass sweep at production resolution N=2e5,H=16 (focusing-vs-mass fig)
for M in 20 40 60 80 100; do
  sbatch $RUN radial 200000 $M 16 $TAU $T 0 results/radial_M${M}_N2e5_H16_s0
done

# (B) bandwidth self-convergence at M=80, fixed N=1e5: H in {12,16,24}
for H in 12 16 24; do
  sbatch $RUN radial 100000 80 $H $TAU $T 0 results/selfconvH_M80_N1e5_H${H}_s0
done

# (C) particle-number self-convergence at M=80, fixed H=16: N in {1e5,4e5}
#     (N=2e5,H16,M80 is already in (A))
sbatch $RUN radial 100000 80 16 $TAU $T 0 results/selfconvN_M80_N1e5_H16_s0
sbatch $RUN radial 400000 80 16 $TAU $T 0 results/selfconvN_M80_N4e5_H16_s0

# (D) tetrahedral 4-cluster, M=80, N=2e5, H=16
sbatch $RUN tetra  200000 80 16 $TAU $T 0 results/tetra_M80_N2e5_H16_s0

echo "submitted 11 focused 3D KS runs"
