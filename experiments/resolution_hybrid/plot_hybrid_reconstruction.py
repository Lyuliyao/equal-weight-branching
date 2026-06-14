"""plot_hybrid_reconstruction.py -- the hybrid-reconstruction figure (Sec. 5.x).
================================================================================

Reads ONLY the saved plot data of `core_window_demo.py`
(plot_data/figure_core_reconstruction.npz + core_demo_metrics.csv +
residual_acceptance.csv) and builds the 5-panel figure (CLAUDE.md Sec. 6):

  Panel 1: full-domain global low-K reconstruction
  Panel 2: full-domain hybrid reconstruction
  Panel 3: zoomed core line profile through the centroid (low-K, high-K, hybrid,
           and the exact profile when known)
  Panel 4: retained residual particles overlaid on the detected window
  Panel 5: bar chart of reconstruction cost (global modes) vs peak error

Usage:  python plot_hybrid_reconstruction.py --results_dir results/core_demo
"""
import os
import sys
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common_plot_style as cps
from common_plot_style import TEXTWIDTH_IN, savefig_multi

cps.apply_style()


def load_metrics(path):
    return list(csv.DictReader(open(path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/core_demo")
    ap.add_argument("--a", type=float, default=84.0, help="exact-Gaussian width for the profile")
    ap.add_argument("--mass", type=float, default=10.0 * np.pi)
    args = ap.parse_args()
    RD = args.results_dir
    d = np.load(os.path.join(RD, "plot_data", "figure_core_reconstruction.npz"))
    XX, YY = d["XX"], d["YY"]
    center = d["center"]; half = float(d["half"])
    box = d["box"].tolist()
    extent = [box[0][0], box[0][1], box[1][0], box[1][1]]
    metrics = load_metrics(os.path.join(RD, "core_demo_metrics.csv"))
    fig_dir = os.path.join(RD, "figures"); os.makedirs(fig_dir, exist_ok=True)

    u_lo = d["u_global_low"]; u_hi = d["u_global_high"]; u_hyb = d["u_hybrid"]
    vmax = float(np.max(u_hi))
    norm = PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax)

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 0.62 * TEXTWIDTH_IN), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1:])]

    # Panel 1: global low-K
    axes[0].imshow(u_lo, origin="lower", extent=extent, norm=norm, cmap="magma")
    axes[0].set_title(r"global low-$K_g$", fontsize=7)
    # Panel 2: hybrid
    im = axes[1].imshow(u_hyb, origin="lower", extent=extent, norm=norm, cmap="magma")
    axes[1].set_title(r"hybrid ($K_g$ + local window)", fontsize=7)
    for ax in axes[:2]:
        ax.add_patch(Rectangle((center[0] - half, center[1] - half), 2 * half, 2 * half,
                               fill=False, edgecolor="cyan", lw=0.6))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_xlim(center[0] - 4 * half, center[0] + 4 * half)
        ax.set_ylim(center[1] - 4 * half, center[1] + 4 * half)
    fig.colorbar(im, ax=axes[1], shrink=0.8, pad=0.01)

    # Panel 3: core line profile through the centroid
    iy = np.argmin(np.abs(YY[:, 0] - center[1]))
    xs = XX[iy, :]
    axes[2].plot(xs, u_lo[iy, :], color="#1f77b4", label=r"low $K_g$")
    axes[2].plot(xs, u_hi[iy, :], color="#2ca02c", label=r"high $K$")
    axes[2].plot(xs, u_hyb[iy, :], color="#d62728", ls="--", label="hybrid")
    if "exact_peak" in d.files:
        A = float(d["exact_peak"])
        prof = A * np.exp(-args.a * ((xs - center[0]) ** 2))
        axes[2].plot(xs, prof, color="k", lw=0.8, ls=":", label="exact")
    axes[2].set_xlim(center[0] - 3 * half, center[0] + 3 * half)
    axes[2].set_title("core profile", fontsize=7)
    axes[2].set_xlabel(r"$x$"); axes[2].legend(fontsize=5.5, loc="upper right")
    axes[2].tick_params(labelsize=6)

    # Panel 4: retained residual particles overlaid on the detected window
    if "retained_particles" in d.files and d["retained_particles"].size:
        allp = d["all_particles_sub"]; ret = d["retained_particles"]
        axes[3].scatter(allp[:, 0], allp[:, 1], s=0.4, color="0.7", rasterized=True)
        axes[3].scatter(ret[:, 0], ret[:, 1], s=0.8, color="#d62728", rasterized=True)
        axes[3].add_patch(Rectangle((center[0] - half, center[1] - half), 2 * half, 2 * half,
                                    fill=False, edgecolor="k", lw=0.6))
        axes[3].set_xlim(center[0] - 2.5 * half, center[0] + 2.5 * half)
        axes[3].set_ylim(center[1] - 2.5 * half, center[1] + 2.5 * half)
    axes[3].set_title("retained residual particles", fontsize=7)
    axes[3].set_xticks([]); axes[3].set_yticks([]); axes[3].grid(False)
    axes[3].set_aspect("equal")

    # Panel 5: cost (global modes) vs peak error bar chart
    schemes, gmodes, perr = [], [], []
    short = {"A_global_low": "low-$K$", "B_global_high": "high-$K$",
             "C_hybrid_window": "hybrid-W", "D_hybrid_blob": "hybrid-blob",
             "E_ht_residual": "HT-resid", "E_positive_residual": "pos-resid"}
    for r in metrics:
        schemes.append(short.get(r["scheme"], r["scheme"]))
        gmodes.append(int(r["global_modes"]))
        perr.append(float(r.get("peak_relerr", "nan")) if r.get("peak_relerr") else np.nan)
    ax5 = axes[4]
    x = np.arange(len(schemes))
    ax5.bar(x - 0.2, gmodes, width=0.4, color="#1f77b4", label="global modes")
    ax5.set_yscale("log"); ax5.set_ylabel("global modes", color="#1f77b4", fontsize=6)
    ax5.set_xticks(x); ax5.set_xticklabels(schemes, fontsize=5.5, rotation=20)
    ax52 = ax5.twinx()
    ax52.plot(x, np.array(perr) * 100, "o-", color="#d62728", ms=3, label="peak err")
    ax52.set_ylabel("peak rel. err [%]", color="#d62728", fontsize=6)
    ax52.tick_params(labelsize=6); ax5.tick_params(labelsize=6)
    ax5.set_title("cost vs peak error", fontsize=7)

    savefig_multi(fig, os.path.join(fig_dir, "figure_hybrid_reconstruction"))
    print("wrote", os.path.join(fig_dir, "figure_hybrid_reconstruction.pdf"))


if __name__ == "__main__":
    main()
