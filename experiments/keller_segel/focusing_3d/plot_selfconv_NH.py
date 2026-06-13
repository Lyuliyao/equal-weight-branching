"""
Dedicated self-convergence figure for the 3D Keller-Segel focusing study at M=80.

Two panels, each with a clean 3-entry legend:
  left  : R_0.5(t) for N = 1e5, 2e5, 4e5 (H fixed = 16)  -- particle-number
          convergence of the (reconstruction-free) inner half-mass core radius.
  right : P_H(t)   for H = 12, 16, 24   (N fixed = 1e5)  -- reconstructed peak
          still grows with bandwidth.

Reads diagnostics.csv produced by simulation_focusing.py.  Does not touch the
mass-sweep runs.  Output: <out_dir>/ks3d_selfconv.pdf
"""
import os
import csv
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_run(run_dir):
    with open(os.path.join(run_dir, "config_used.json")) as f:
        cfg = json.load(f)
    rows = []
    with open(os.path.join(run_dir, "diagnostics.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if v != "" else np.nan)
                         for k, v in r.items()})
    return cfg, rows


def col(rows, name):
    return np.array([r.get(name, np.nan) for r in rows], dtype=float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_runs", nargs="+", required=True,
                   help="run dirs for the N-sweep (fixed H=16), in increasing N")
    p.add_argument("--h_runs", nargs="+", required=True,
                   help="run dirs for the H-sweep (fixed N=1e5), in increasing H")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    n_runs = [load_run(d) for d in args.n_runs]
    h_runs = [load_run(d) for d in args.h_runs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for cfg, rows in sorted(n_runs, key=lambda cr: cr[0]["n_particles"]):
        t = col(rows, "t")
        ax1.plot(t, col(rows, "R_0.5"), marker="o", ms=3,
                 label=f"$N={cfg['n_particles']:.0g}$")
    ax1.set_xlabel("$t$")
    ax1.set_ylabel(r"$R_{0.5}(t)$")
    ax1.set_title("Core radius: particle-number convergence ($H=16$)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    for cfg, rows in sorted(h_runs, key=lambda cr: cr[0]["H"]):
        t = col(rows, "t")
        ax2.semilogy(t, col(rows, "P_H"), marker="s", ms=3,
                     label=f"$H={cfg['H']}$")
    ax2.set_xlabel("$t$")
    ax2.set_ylabel(r"$\|P_H\rho\|_\infty$")
    ax2.set_title("Reconstructed peak: bandwidth growth ($N=10^5$)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
