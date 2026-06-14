"""Switching-growth Figure 6: full domain + A/B microscope zooms (CLAUDE.md §3).
================================================================================

Reads ONLY saved data (no solver run):
  * reference_results/switch/fields_seed<seed>.npz  (reference/weighted/weighted_ess/minvar fields)
  * reference_results/switch/metrics.csv            (final-time local L2 in B_A, B_B, mean over seeds)

Builds a 4-column x 3-row figure that visually explains Table 6:
  columns: reference | weighted | weighted+ESS | min.-variance branching
  row 1: full domain
  row 2: microscope zoom around the OLD growth region A (x_A)
  row 3: microscope zoom around the NEW growth region B (x_B)
The diagnostic sets B_A, B_B are drawn in every row; the A/B zoom rows are
annotated with the final-time local L2 (from metrics.csv).  Row-wise shared color
scales.  Idempotent: rerunning reproduces the figure from saved data.

Usage:  python plot_switch.py [--results_dir reference_results/switch] [--seed 0]
"""
import os
import sys
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common_plot_style as cps
from common_plot_style import TEXTWIDTH_IN, savefig_multi

cps.apply_style()

# switching benchmark geometry (CLAUDE.md §3.2)
CA = (-1.2, 0.0); CB = (1.2, 0.0); SIGMA = 0.25; ETA = 0.5
R_B = SIGMA * np.sqrt(-2.0 * np.log(ETA))       # half-height radius ~0.294
R_ZOOM = 0.60
COLS = [("reference", "reference"), ("weighted", "weighted"),
        ("weighted_ess", "weighted+ESS"), ("minvar", "min.-var. branching")]


def final_local_L2(metrics_csv):
    """Mean over seeds of final-time L2_rel_err_BA / _BB per method."""
    rows = list(csv.DictReader(open(metrics_csv)))
    tmax = max(float(r["t"]) for r in rows)
    out = {}
    for method in ["weighted", "weighted_ess", "minvar"]:
        sub = [r for r in rows if r["method"] == method and abs(float(r["t"]) - tmax) < 1e-9]
        def mn(c):
            v = [float(r[c]) for r in sub if r.get(c) not in (None, "", "nan")]
            return float(np.mean(v)) if v else np.nan
        out[method] = dict(BA=mn("L2_rel_err_BA"), BB=mn("L2_rel_err_BB"), n=len(sub))
    return out


def draw_regions(ax):
    ax.add_patch(Circle(CA, R_B, fill=False, edgecolor="cyan", lw=0.7, ls="-"))
    ax.add_patch(Circle(CB, R_B, fill=False, edgecolor="yellow", lw=0.7, ls="--"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="reference_results/switch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--paper_fig", default="paper/figure/switch_snapshots.pdf")
    args = ap.parse_args()
    repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    RD = args.results_dir if os.path.isabs(args.results_dir) else os.path.join(repo, args.results_dir)
    d = np.load(os.path.join(RD, f"fields_seed{args.seed}.npz"))
    XX, YY = d["XX"], d["YY"]
    extent = [-np.pi, np.pi, -np.pi, np.pi]
    locL2 = final_local_L2(os.path.join(RD, "metrics.csv"))

    fields = {k: d[k] for k, _ in COLS}
    # row-wise color scales
    full_vmax = max(np.percentile(fields[k], 99.7) for k, _ in COLS)

    def zoom_mask(cx):
        return ((np.abs(XX - cx) <= R_ZOOM) & (np.abs(YY) <= R_ZOOM))
    mA = zoom_mask(CA[0]); mB = zoom_mask(CB[0])
    a_vmax = max(np.max(fields[k][mA]) for k, _ in COLS)
    b_vmax = max(np.max(fields[k][mB]) for k, _ in COLS)

    fig, axes = plt.subplots(3, 4, figsize=(TEXTWIDTH_IN, 0.80 * TEXTWIDTH_IN),
                             constrained_layout=True)
    rowinfo = [("full domain", extent, full_vmax, None),
               (r"zoom: old region $B_A$", [CA[0] - R_ZOOM, CA[0] + R_ZOOM, -R_ZOOM, R_ZOOM], a_vmax, "BA"),
               (r"zoom: new region $B_B$", [CB[0] - R_ZOOM, CB[0] + R_ZOOM, -R_ZOOM, R_ZOOM], b_vmax, "BB")]
    for ri, (rlabel, ext, vmax, which) in enumerate(rowinfo):
        for ci, (key, ctitle) in enumerate(COLS):
            ax = axes[ri][ci]
            im = ax.imshow(fields[key], origin="lower", extent=extent, vmin=0, vmax=vmax,
                           cmap="magma", interpolation="nearest")
            draw_regions(ax)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if ri == 0:
                ax.set_title(ctitle, fontsize=6.5, pad=2)
            if ci == 0:
                ax.set_ylabel(rlabel, fontsize=6.5)
            # local-L2 annotation in the zoom rows (not for the reference column)
            if which and key in locL2:
                val = locL2[key][which]
                ax.text(0.04, 0.92, f"$L^2={val:.3f}$", transform=ax.transAxes,
                        fontsize=6, color="white", va="top",
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.45))
            elif which and key == "reference":
                ax.text(0.04, 0.92, "reference", transform=ax.transAxes, fontsize=6,
                        color="white", va="top",
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.45))
        fig.colorbar(im, ax=axes[ri], shrink=0.82, pad=0.008, aspect=18)

    out_pdf = os.path.join(RD, "switch_snapshots_zoom")
    savefig_multi(fig, out_pdf, close=False)
    # also overwrite the manuscript figure so the paper picks it up
    paper_fig = args.paper_fig if os.path.isabs(args.paper_fig) else os.path.join(repo, args.paper_fig)
    os.makedirs(os.path.dirname(paper_fig), exist_ok=True)
    fig.savefig(paper_fig)
    plt.close(fig)

    pd = os.path.join(RD, "plot_data"); os.makedirs(pd, exist_ok=True)
    np.savez(os.path.join(pd, "switch_snapshots_zoom.npz"),
             XX=XX, YY=YY, extent=np.array(extent),
             reference=fields["reference"], weighted=fields["weighted"],
             weighted_ess=fields["weighted_ess"], minvar=fields["minvar"],
             cA=np.array(CA), cB=np.array(CB), sigma=SIGMA, eta=ETA, R_B=R_B, R_zoom=R_ZOOM,
             full_vmax=full_vmax, a_vmax=a_vmax, b_vmax=b_vmax,
             locL2_weighted_BA=locL2["weighted"]["BA"], locL2_weighted_BB=locL2["weighted"]["BB"],
             locL2_ess_BA=locL2["weighted_ess"]["BA"], locL2_ess_BB=locL2["weighted_ess"]["BB"],
             locL2_minvar_BA=locL2["minvar"]["BA"], locL2_minvar_BB=locL2["minvar"]["BB"])
    print(f"wrote {out_pdf}.pdf/.png, {paper_fig}, and plot_data/switch_snapshots_zoom.npz")
    print("local L2 (final, mean over seeds):", {k: {kk: round(vv, 3) for kk, vv in v.items() if kk != 'n'} for k, v in locL2.items()})


if __name__ == "__main__":
    main()
