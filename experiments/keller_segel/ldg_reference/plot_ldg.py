"""
Figure for the direct LDG reference (Example 5.2 blow-up) and its comparison to
the particle method and the FVM sanity baseline.
================================================================================

Reads saved data only:
  * LDG  reference_results/keller_segel_ldg_pp/ldg_<run_id>/N{80,160,320}/
        (S_curves.csv + N160/snapshots.npz),
  * FVM  reference_results/keller_segel_ldg_pp/baseline_<run_id>/n512/S_curves.csv,
  * particle ldg_comparison base/refined diag_*.csv.

Figure (2 rows): top = LDG u snapshots at the LDG report times (N=160, core zoom);
bottom = S_L2(t) and peak(t) for the LDG refinements (with tb markers) overlaid
with the FVM baseline and the particle method on the same axes.

Usage:
  python plot_ldg.py --ldg_dir <ldg_run> --fvm <baseline_run> \
      --particle_base <diag.csv> --particle_refined <diag.csv> --out_dir <ldg_run>/figures
"""
import os
import sys
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common_plot_style as cps                       # noqa: E402
from common_plot_style import TEXTWIDTH_IN, savefig_multi  # noqa: E402

cps.apply_style()
REPORTS = [6e-5, 1.2e-4, 2.0e-4]
ZOOM = 0.08


def load_S(path, cols):
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r["t"]) for r in rows])
    return t, {c: np.array([float(r[c]) for r in rows]) for c in cols}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ldg_dir", required=True)
    ap.add_argument("--fvm", default="")
    ap.add_argument("--particle_base", default="")
    ap.add_argument("--particle_refined", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ldg_Ns", type=int, nargs="+", default=[80, 160, 320])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ldg = {}
    for N in args.ldg_Ns:
        p = os.path.join(args.ldg_dir, f"N{N}", "S_curves.csv")
        if os.path.exists(p):
            ldg[N] = load_S(p, ("S_L2", "peak"))
    snap = np.load(os.path.join(args.ldg_dir, "N160", "snapshots.npz"))
    xc = snap["xc"]

    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH_IN, 0.62 * TEXTWIDTH_IN))

    zi = np.where(np.abs(xc) <= ZOOM)[0]
    sl = slice(zi[0], zi[-1] + 1)
    ext = [xc[zi[0]], xc[zi[-1]], xc[zi[0]], xc[zi[-1]]]
    for j, rt in enumerate(REPORTS):
        ax = axes[0][j]
        U = snap[f"u_{rt:.2e}"][sl, sl]
        im = ax.imshow(U, origin="lower", extent=ext, cmap="magma", interpolation="nearest")
        mant, exp = f"{rt:.1e}".split("e")
        ax.set_title(rf"$t={mant}\times10^{{{int(exp)}}}$", fontsize=6.8, pad=2)
        ax.text(0.04, 0.05, rf"peak ${U.max():.0f}$", transform=ax.transAxes,
                fontsize=5.6, color="white", va="bottom")
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(r"LDG $u$ (core, $N{=}160$)", fontsize=6.5)
        fig.colorbar(im, ax=ax, shrink=0.80, pad=0.01, aspect=12)

    cols = {80: "#9ecae1", 160: "#4292c6", 320: "#08519c"}
    # S_L2
    axS = axes[1][0]
    for N in sorted(ldg):
        axS.plot(ldg[N][0], ldg[N][1]["S_L2"], "-", color=cols.get(N, "0.3"),
                 lw=1.1, label=f"LDG $N={N}$")
    if args.fvm:
        t, d = load_S(os.path.join(args.fvm, "n512", "S_curves.csv"), ("S_L2",))
        axS.plot(t, d["S_L2"], ":", color="tab:green", lw=1.0, label="FVM $n{=}512$")
    if args.particle_refined:
        t, d = load_S(args.particle_refined, ("S_L2",))
        axS.plot(t, d["S_L2"], "--", color="tab:red", lw=1.0, label="particle ref.")
    axS.set_xlabel(r"$t$"); axS.set_ylabel(r"$S(t)=\|u\|_{L^2}$")
    axS.set_yscale("log"); axS.legend(fontsize=4.6, loc="lower right")
    axS.set_title("concentration norm", fontsize=7)

    axP = axes[1][1]
    for N in sorted(ldg):
        axP.plot(ldg[N][0], ldg[N][1]["peak"], "-", color=cols.get(N, "0.3"), lw=1.1)
    if args.particle_refined:
        t, d = load_S(args.particle_refined, ("peak_PK_u",))
        axP.plot(t, d["peak_PK_u"], "--", color="tab:red", lw=1.0)
    axP.set_xlabel(r"$t$"); axP.set_ylabel(r"peak $\|u\|_\infty$")
    axP.set_yscale("log"); axP.set_title("peak (resolution-sensitive)", fontsize=6.6)

    # tb(N) vs refinement pair
    axT = axes[1][2]
    import json
    tbp = os.path.join(args.ldg_dir, "tb_ldg.json")
    if os.path.exists(tbp):
        d = json.load(open(tbp))
        labels = list(d.keys()); vals = [d[k]["tb_1_05"] for k in labels]
        axT.plot(range(len(labels)), vals, "o-", color="#08519c", ms=4)
        axT.set_xticks(range(len(labels))); axT.set_xticklabels(labels, fontsize=5)
    axT.axhline(1.21e-4, color="0.4", ls="--", lw=0.8)
    axT.text(0.02, 1.21e-4 * 1.04, r"ref $1.21{\times}10^{-4}$",
             transform=axT.get_yaxis_transform(), fontsize=4.8, color="0.4", va="bottom")
    axT.set_ylabel(r"$t_b(N;1.05)$"); axT.set_xlabel("refinement pair")
    axT.set_title(r"numerical blow-up time $t_b$", fontsize=6.4)

    fig.tight_layout(pad=0.4)
    out = os.path.join(args.out_dir, "ldg_example52")
    savefig_multi(fig, out, close=True)
    print("wrote", out + ".pdf/.png")


if __name__ == "__main__":
    main()
