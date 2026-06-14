"""
plot_audit.py -- figures for the Fourier-bandwidth / KDE reconstruction audit.
==============================================================================

Reads ONLY the aggregated sweep data written by audit_fourier_kde.py
(reference_results/reconstruction_audit/<exp>/plot_data/<exp>_sweeps.npz) and
draws, per experiment:

  * Fourier K-sweep:  E_total, E_particle, E_proj vs K (per method; for S5.3 the
    global / B_A / B_B regions in separate panels).
  * KDE h-sweep:      E_KDE_rep vs h (per method) and E_bias vs h (shared).

The point of the figure is robustness: the method ordering should be stable over
the moderate bandwidths, with E_proj (Fourier truncation bias of the reference)
and E_bias (Gaussian smoothing bias) marked so the reader can see the regime
where differences are real rather than dominated by reconstruction bias/noise.

Usage:  python plot_audit.py [--exp both|localized|switching]
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common_plot_style as cps                      # noqa: E402
from common_plot_style import TEXTWIDTH_IN, savefig_multi  # noqa: E402

cps.apply_style()

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
REFDIR = os.path.join(REPO, "reference_results", "reconstruction_audit")

MLABEL = {"weighted": "weighted", "weighted_ess": "weighted+ESS",
          "minvar": "min.-var. branching"}
MCOLOR = {"weighted": "tab:red", "weighted_ess": "tab:orange",
          "minvar": "tab:blue"}
RLABEL = {"global": "global", "B": "growth region $B$",
          "B_A": "old region $B_A$", "B_B": "new region $B_B$"}


def load(exp):
    sub = "localized_growth" if exp == "localized" else "switching_growth"
    d = np.load(os.path.join(REFDIR, sub, "plot_data", f"{exp}_sweeps.npz"),
                allow_pickle=True)
    return d, sub


def _line(ax, x, mean, std, color, label, ls="-", marker="o"):
    m = np.asarray(mean, float)
    s = np.asarray(std, float)
    ok = ~np.isnan(m)
    ax.plot(np.asarray(x)[ok], m[ok], ls=ls, marker=marker, ms=3.5, color=color,
            label=label, lw=1.2)
    ax.fill_between(np.asarray(x)[ok], (m - s)[ok], (m + s)[ok], color=color, alpha=0.15)


def plot_localized():
    d, sub = load("localized")
    K = d["K_list"]; h = d["h_list"]; methods = list(d["methods"])
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 0.34 * TEXTWIDTH_IN),
                             constrained_layout=True)
    # report the global region (matches the main §5.2 table and the validation)
    cand = [r for r in ["global", "B"] if f"fourier_E_total_weighted_{r}_mean" in d.files]
    rg = cand[0] if cand else "global"

    # panel 1: Fourier E_total
    ax = axes[0]
    for m in methods:
        _line(ax, K, d[f"fourier_E_total_{m}_{rg}_mean"], d[f"fourier_E_total_{m}_{rg}_std"],
              MCOLOR[m], MLABEL[m])
    _line(ax, K, d[f"fourier_E_proj_{methods[0]}_{rg}_mean"],
          d[f"fourier_E_proj_{methods[0]}_{rg}_std"], "0.5", r"$E_{\rm proj}$ (ref. trunc.)",
          ls="--", marker="s")
    ax.axvline(16, color="0.8", lw=0.8, zorder=0)
    ax.set_xlabel(r"Fourier bandwidth $K$"); ax.set_ylabel(r"relative $L^2$ error")
    ax.set_title(r"$E_{\rm total}(K)$", fontsize=7)
    ax.set_yscale("log"); ax.legend(fontsize=5.5, loc="best")

    # panel 2: Fourier E_particle
    ax = axes[1]
    for m in methods:
        _line(ax, K, d[f"fourier_E_particle_{m}_{rg}_mean"],
              d[f"fourier_E_particle_{m}_{rg}_std"], MCOLOR[m], MLABEL[m])
    ax.axvline(16, color="0.8", lw=0.8, zorder=0)
    ax.set_xlabel(r"Fourier bandwidth $K$")
    ax.set_title(r"$E_{\rm particle}(K)$ (fixed scale)", fontsize=7)
    ax.set_yscale("log")

    # panel 3: KDE
    ax = axes[2]
    for m in methods:
        _line(ax, h, d[f"kde_E_KDE_rep_{m}_{rg}_mean"], d[f"kde_E_KDE_rep_{m}_{rg}_std"],
              MCOLOR[m], MLABEL[m])
    _line(ax, h, d[f"kde_E_bias_{methods[0]}_{rg}_mean"], d[f"kde_E_bias_{methods[0]}_{rg}_std"],
          "0.5", r"$E_{\rm bias}$ (smoothing)", ls="--", marker="s")
    ax.set_xlabel(r"Gaussian smoothing $h$")
    ax.set_title(r"$E_{\rm KDE}^{\rm rep}(h)$", fontsize=7)
    ax.set_yscale("log")
    fig.suptitle("Localized-growth reconstruction audit (mean$\\pm$std over seeds)",
                 fontsize=8)
    out = os.path.join(REFDIR, sub, "figures", "localized_audit")
    savefig_multi(fig, out, close=True)
    print("wrote", out + ".pdf/.png")


def plot_switching():
    d, sub = load("switching")
    K = d["K_list"]; h = d["h_list"]; methods = list(d["methods"])
    regions = ["global", "B_A", "B_B"]
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH_IN, 0.62 * TEXTWIDTH_IN),
                             constrained_layout=True)
    # row 1: Fourier E_total per region.  Here the reference is smooth, so the
    # truncation bias E_proj is negligible (< 1.6e-4, down to ~1e-11 at K=64) --
    # plotting it would crush the method curves, so it is reported in the caption
    # instead and E_total = E_particle to plotting accuracy.
    for j, rg in enumerate(regions):
        ax = axes[0][j]
        for m in methods:
            _line(ax, K, d[f"fourier_E_total_{m}_{rg}_mean"],
                  d[f"fourier_E_total_{m}_{rg}_std"], MCOLOR[m], MLABEL[m])
        ax.axvline(48, color="0.8", lw=0.8, zorder=0)
        ax.set_yscale("log"); ax.set_title(RLABEL[rg], fontsize=7)
        if j == 0:
            ax.set_ylabel(r"$E_{\rm total}(K)$")
            ax.legend(fontsize=5.5, loc="best")
        ax.set_xlabel(r"$K$")
    # row 2: KDE E_KDE_rep per region
    for j, rg in enumerate(regions):
        ax = axes[1][j]
        for m in methods:
            _line(ax, h, d[f"kde_E_KDE_rep_{m}_{rg}_mean"],
                  d[f"kde_E_KDE_rep_{m}_{rg}_std"], MCOLOR[m], MLABEL[m])
        _line(ax, h, d[f"kde_E_bias_{methods[0]}_{rg}_mean"],
              d[f"kde_E_bias_{methods[0]}_{rg}_std"], "0.5", r"$E_{\rm bias}$",
              ls="--", marker="s")
        ax.set_yscale("log")
        if j == 0:
            ax.set_ylabel(r"$E_{\rm KDE}^{\rm rep}(h)$")
        ax.set_xlabel(r"$h$")
    fig.suptitle("Switching-growth reconstruction audit: global / $B_A$ / $B_B$ "
                 "(mean$\\pm$std over seeds)", fontsize=8)
    out = os.path.join(REFDIR, sub, "figures", "switching_audit")
    savefig_multi(fig, out, close=True)
    print("wrote", out + ".pdf/.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["localized", "switching", "both"], default="both")
    args = ap.parse_args()
    if args.exp in ("localized", "both"):
        plot_localized()
    if args.exp in ("switching", "both"):
        plot_switching()


if __name__ == "__main__":
    main()
