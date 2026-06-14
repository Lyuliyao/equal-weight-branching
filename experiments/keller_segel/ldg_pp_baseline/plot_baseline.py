"""
§5.4 figure: fully parabolic-parabolic Keller-Segel LDG-aligned concentration.
==============================================================================

Reads ONLY saved data (no solver run):
  * FVM baseline snapshots.npz + S_curves.csv at grids 128/256/512
    (reference_results/keller_segel_ldg_pp/baseline_<run_id>/);
  * particle pp diag CSVs (ldg_comparison base/refined), passed explicitly.

Figure (2 rows):
  row 1: baseline cell density u at the LDG reporting times 6e-5,1.2e-4,2e-4
         (n=512, zoomed to the concentrating core), each with its own colorbar
         and annotated peak (peak is bandwidth/resolution-sensitive).
  row 2: S_L2(t)=||u||_L2 and peak(t) for the grid refinements (solid) and the
         particle method (dashed), log-y, with t_b(1.05) resolution-gap markers.

Usage:
  python plot_baseline.py --baseline_dir <baseline_run> \
      --particle_base <diag.csv> --particle_refined <diag.csv> \
      --out_dir <baseline_run>/figures
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
    out = {c: np.array([float(r[c]) for r in rows]) for c in cols}
    return t, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", required=True)
    ap.add_argument("--particle_base", required=True)
    ap.add_argument("--particle_refined", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    pdir = os.path.join(os.path.dirname(args.out_dir.rstrip("/")), "plot_data")
    os.makedirs(pdir, exist_ok=True)

    Sg = {n: load_S(os.path.join(args.baseline_dir, f"n{n}", "S_curves.csv"),
                    ("S_L2", "peak", "R_0_8")) for n in (128, 256, 512)}
    tpb, pb = load_S(args.particle_base, ("S_L2", "peak_PK_u", "R_0.8"))
    tpr, pr = load_S(args.particle_refined, ("S_L2", "peak_PK_u", "R_0.8"))

    # Snapshots: prefer the raw solver output; fall back to the committed
    # lean plot_data (core zoom only) so the figure regenerates from saved data.
    raw = os.path.join(args.baseline_dir, "n512", "snapshots.npz")
    pdfile = os.path.join(pdir, "ks_pp_baseline.npz")
    zoom = {}
    if os.path.exists(raw):
        snap = np.load(raw)
        xs = snap["xs"]
        zi = np.where(np.abs(xs) <= ZOOM)[0]
        sl = slice(zi[0], zi[-1] + 1)
        xs_zoom = xs[zi]
        for rt in REPORTS:
            zoom[rt] = snap[f"u_{rt:.2e}"][sl, sl].astype(np.float32)
    else:
        d0 = np.load(pdfile)
        xs_zoom = d0["xs_zoom"]
        for rt in REPORTS:
            zoom[rt] = d0[f"uzoom_{rt:.2e}"]
    ext = [xs_zoom[0], xs_zoom[-1], xs_zoom[0], xs_zoom[-1]]

    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH_IN, 0.62 * TEXTWIDTH_IN))

    # ---- row 1: baseline snapshots (zoom to core) ----
    for j, rt in enumerate(REPORTS):
        ax = axes[0][j]
        U = zoom[rt]
        im = ax.imshow(U, origin="lower", extent=ext, cmap="magma",
                       interpolation="nearest")
        mant, exp = f"{rt:.1e}".split("e")
        ax.set_title(rf"$t={mant}\times10^{{{int(exp)}}}$", fontsize=6.8, pad=2)
        ax.text(0.04, 0.05, rf"peak ${U.max():.0f}$", transform=ax.transAxes,
                fontsize=5.6, color="white", va="bottom")
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(r"baseline $u$ (core)", fontsize=6.5)
        fig.colorbar(im, ax=ax, shrink=0.80, pad=0.01, aspect=12)

    grids = [(128, "#9ecae1"), (256, "#4292c6"), (512, "#08519c")]
    # ---- row 2 left: S_L2(t) ----
    axS = axes[1][0]
    for n, c in grids:
        axS.plot(Sg[n][0], Sg[n][1]["S_L2"], "-", color=c, lw=1.1, label=f"FVM $n={n}$")
    axS.plot(tpb, pb["S_L2"], "--", color="tab:orange", lw=1.0, label="particle base")
    axS.plot(tpr, pr["S_L2"], "--", color="tab:red", lw=1.0, label="particle ref.")
    axS.axvline(5.0e-5, color="0.5", ls=":", lw=0.8)
    axS.set_xlabel(r"$t$"); axS.set_ylabel(r"$S(t)=\|u\|_{L^2}$")
    axS.set_yscale("log"); axS.legend(fontsize=4.8, loc="lower right")
    axS.set_title(r"concentration $S(t)$", fontsize=7)

    # ---- row 2 mid: peak(t) (bandwidth/resolution-sensitive) ----
    axP = axes[1][1]
    for n, c in grids:
        axP.plot(Sg[n][0], Sg[n][1]["peak"], "-", color=c, lw=1.1)
    axP.plot(tpb, pb["peak_PK_u"], "--", color="tab:orange", lw=1.0)
    axP.plot(tpr, pr["peak_PK_u"], "--", color="tab:red", lw=1.0)
    axP.set_xlabel(r"$t$"); axP.set_ylabel(r"peak $\|u\|_\infty$")
    axP.set_yscale("log")
    axP.set_title("peak (bandwidth-sensitive)", fontsize=6.6)

    # ---- row 2 right: reconstruction-free R_0.8(t) ----
    axR = axes[1][2]
    for n, c in grids:
        axR.plot(Sg[n][0], Sg[n][1]["R_0_8"], "-", color=c, lw=1.1)
    axR.plot(tpb, pb["R_0.8"], "--", color="tab:orange", lw=1.0)
    axR.plot(tpr, pr["R_0.8"], "--", color="tab:red", lw=1.0)
    axR.set_xlabel(r"$t$"); axR.set_ylabel(r"$R_{0.8}(t)$")
    axR.set_title(r"$R_{0.8}$ (reconstruction-free)", fontsize=6.6)

    fig.tight_layout(pad=0.4)
    out = os.path.join(args.out_dir, "ks_pp_baseline")
    savefig_multi(fig, out, close=False)
    # save LEAN plot_data: core-zoom snapshots only (figure-sufficient) + curves
    np.savez(os.path.join(pdir, "ks_pp_baseline.npz"),
             xs_zoom=xs_zoom, **{f"uzoom_{rt:.2e}": zoom[rt] for rt in REPORTS},
             t128=Sg[128][0], S128=Sg[128][1]["S_L2"], peak128=Sg[128][1]["peak"],
             t256=Sg[256][0], S256=Sg[256][1]["S_L2"], peak256=Sg[256][1]["peak"],
             t512=Sg[512][0], S512=Sg[512][1]["S_L2"], peak512=Sg[512][1]["peak"],
             tpb=tpb, Spb=pb["S_L2"], peakpb=pb["peak_PK_u"],
             tpr=tpr, Spr=pr["S_L2"], peakpr=pr["peak_PK_u"])
    plt.close(fig)
    print("wrote", out + ".pdf/.png and plot_data/ks_pp_baseline.npz")


if __name__ == "__main__":
    main()
