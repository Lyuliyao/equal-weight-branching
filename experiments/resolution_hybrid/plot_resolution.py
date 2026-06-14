"""plot_resolution.py -- multi-island local-reconstruction diagnostics (Demo 2).
================================================================================

Reads ONLY the saved data of `reconstruct_from_snapshot.py`
(island_reconstruction.csv + residual_acceptance.csv + plot_data npz) and builds
the multi-island reconstruction figure (CLAUDE.md Sec. 6, Demo 2):

  Panel 1: per-island mass error E_m vs amplitude a_m, for each reconstruction
           (particle count, global low-K, global high-K, local window).  Shows
           that reconstructing-then-integrating at low global K fails every
           island, while particle counting and local windows recover accuracy.
  Panel 2: per-island residual accept rate (HT) vs amplitude -- the
           reconstruction-enrichment indicator.

Usage:  python plot_resolution.py --results_dir results/island_demo
"""
import os
import sys
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common_plot_style as cps
from common_plot_style import TEXTWIDTH_IN, savefig_multi

cps.apply_style()

COLORS = {"A_count": "k", "B_global_low": "#1f77b4", "B_global_high": "#9467bd",
          "C_local_window": "#2ca02c", "E_ht_residual": "#ff7f0e"}
LABELS = {"A_count": "particle count (recon-free)", "B_global_low": r"global low-$K$",
          "B_global_high": r"global high-$K$", "C_local_window": "local window",
          "E_ht_residual": "HT residual"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/island_demo")
    args = ap.parse_args()
    RD = args.results_dir
    rows = list(csv.DictReader(open(os.path.join(RD, "island_reconstruction.csv"))))
    amp = np.array([float(r["amplitude"]) for r in rows])
    o = np.argsort(amp)
    schemes = ["A_count", "B_global_low", "B_global_high", "C_local_window"]
    fig_dir = os.path.join(RD, "figures"); os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 0.40 * TEXTWIDTH_IN),
                             constrained_layout=True)
    for a in axes:
        a.set_box_aspect(0.85)
    for s in schemes:
        E = np.array([float(r[f"Em_{s}"]) for r in rows])
        axes[0].plot(amp[o], E[o], "o-", ms=3, color=COLORS[s], label=LABELS[s], lw=1.0)
    axes[0].axhline(0.20, color="gray", lw=0.6, ls=":")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"island amplitude $a_m$")
    axes[0].set_ylabel(r"island mass error $E_m$")
    axes[0].legend(fontsize=5.5, loc="center right")

    # Panel 2: accept rate vs amplitude
    apath = os.path.join(RD, "residual_acceptance.csv")
    if os.path.exists(apath):
        ar = list(csv.DictReader(open(apath)))
        a_amp = np.array([float(r["amplitude"]) for r in ar])
        mass_ar = np.array([float(r["mass_accept_rate"]) for r in ar])
        oo = np.argsort(a_amp)
        axes[1].plot(a_amp[oo], mass_ar[oo], "s-", ms=3, color="#ff7f0e")
        axes[1].set_xlabel(r"island amplitude $a_m$")
        axes[1].set_ylabel("residual mass-accept rate")
        axes[1].set_title("enrichment indicator", fontsize=7)
    savefig_multi(fig, os.path.join(fig_dir, "figure_island_reconstruction"))
    print("wrote", os.path.join(fig_dir, "figure_island_reconstruction.pdf"))


if __name__ == "__main__":
    main()
