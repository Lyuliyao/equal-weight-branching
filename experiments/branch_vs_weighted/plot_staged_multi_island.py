"""Publication figures for the staged separated-growth-island benchmark (Sec. 5.2).
================================================================================

Reads ONLY the saved data products of `staged_multi_island.py` and regenerates
the figures WITHOUT rerunning the solver.

Figures (CLAUDE.md figure-design section):
  1. final-field comparison: reference, weighted+ESS, cost-matched weighted+ESS,
     min-var branching; LATE-stage islands marked; one diagnostic-selected
     magnifier on the worst late island of the cost-matched ESS baseline
     (m* = argmax_{m in late} E_m^{costmatched}), NOT chosen by eye.
  2. late-group degeneracy: late-island local L2 / local effective count vs time.

Usage:  python plot_staged_multi_island.py --results_dir results/staged
"""
import os
import sys
import csv
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common_plot_style as cps
from common_plot_style import TEXTWIDTH_IN, METHOD_COLORS, savefig_multi, add_zoom_inset

cps.apply_style()

PANEL = {
    "reference": "reference",
    "weighted_ess_resample": "weighted+ESS",
    "weighted_ess_resample_costmatched": "cost-matched weighted+ESS",
    "minvar_branch": "min.-var. branching",
}
SERIES_STYLE = {
    "weighted": ("-", "#1f77b4", "weighted"),
    "weighted_ess_resample": ("-", "#9467bd", "weighted+ESS"),
    "weighted_ess_resample_costmatched": ("--", "#8c564b", "cost-matched ESS"),
    "minvar_branch": ("-", "#2ca02c", "min.-var. branch"),
}


def load_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def f(r, k):
    v = r[k]
    return float(v) if v not in ("", "nan") else np.nan


def fig_fields(RD, cfg, island_rows, fig_dir, plot_dir):
    ref = np.load(os.path.join(RD, "fields_ref.npz"))
    seed0 = np.load(os.path.join(RD, "fields_seed0.npz"))
    centers = ref["centers"]; late_idx = ref["late_idx"]
    extent = [-np.pi, np.pi, -np.pi, np.pi]
    keys = ["reference", "weighted_ess_resample",
            "weighted_ess_resample_costmatched", "minvar_branch"]
    panels = [("reference", ref["reference"])]
    for k in keys[1:]:
        if k in seed0.files:
            panels.append((k, seed0[k]))
    vmax = max(np.percentile(p, 99.8) for _, p in panels)
    norm = PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax)

    # worst late island of cost-matched ESS (diagnostic-selected, not by eye)
    M = cfg["M"]
    cm = "weighted_ess_resample_costmatched"
    Em_cm = {int(r["m"]): f(r, "E_m") for r in island_rows
             if r["method"] == cm and int(r["is_late"]) == 1}
    if Em_cm:
        m_star = max(Em_cm, key=Em_cm.get)
    else:
        m_star = int(late_idx[0])
    cx, cy = centers[m_star]
    half = max(3.0 * cfg["sigma"], 0.5)
    zbox = (cx - half, cx + half, cy - half, cy + half)

    npan = len(panels)
    fig, axes = plt.subplots(1, npan,
                             figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN / npan * 1.16),
                             constrained_layout=True)
    for ax, (key, fld) in zip(axes, panels):
        im = ax.imshow(fld, origin="lower", extent=extent, norm=norm, cmap="magma",
                       interpolation="nearest")
        ax.set_title(PANEL.get(key, key), fontsize=6.0, pad=2)
        # mark late islands (cyan) and others (white)
        for m in range(M):
            col = "cyan" if m in late_idx else "white"
            ax.scatter(centers[m, 0], centers[m, 1], s=3, facecolors="none",
                       edgecolors=col, linewidths=0.4)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        # inset on the worst late island, same zoom box in every panel
        add_zoom_inset(ax, fld, extent, zbox, vmin=None, vmax=None, cmap="magma",
                       loc="lower left", zoom_frac=0.42)
        for child in ax.child_axes:
            for cim in child.images:
                cim.set_norm(norm)
    fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01, aspect=24, label=r"$u(T,x)$")
    fig.suptitle(
        f"late islands (cyan); inset = worst late island of cost-matched ESS "
        f"($m={m_star}$, $E_m={Em_cm.get(m_star, np.nan):.2f}$)", fontsize=6)
    savefig_multi(fig, os.path.join(fig_dir, "figure_staged_fields"))
    np.savez(os.path.join(plot_dir, "figure_staged_fields_data.npz"),
             reference=ref["reference"], centers=centers, late_idx=late_idx,
             m_star=m_star, zbox=np.array(zbox), vmax=vmax,
             **{k: seed0[k] for k in seed0.files})


def fig_late_timeseries(RD, cfg, ts_rows, fig_dir):
    methods = [m for m in SERIES_STYLE if any(r["method"] == m for r in ts_rows)]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 0.32 * TEXTWIDTH_IN),
                             constrained_layout=True)
    for a in axes:
        a.set_box_aspect(0.9)

    def series(method, col):
        ts = sorted({f(r, "t") for r in ts_rows if r["method"] == method})
        return np.array(ts), np.array([np.nanmean([f(r, col) for r in ts_rows
                if r["method"] == method and abs(f(r, "t") - t) < 1e-9]) for t in ts])

    for m in methods:
        ls, c, lab = SERIES_STYLE[m]
        t, v = series(m, "global_nESS")
        axes[0].plot(t, v, ls=ls, color=c, label=lab)
    axes[0].axhline(0.5, color="gray", lw=0.6, ls=":")
    axes[0].set_xlabel(r"$t$"); axes[0].set_ylabel("global nESS"); axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=5, loc="lower left")

    for m in methods:
        ls, c, lab = SERIES_STYLE[m]
        t, v = series(m, "min_late_local_eff")
        axes[1].semilogy(t, np.maximum(v, 0.5), ls=ls, color=c)
    axes[1].set_xlabel(r"$t$"); axes[1].set_ylabel("min late-island local count")

    for m in methods:
        ls, c, lab = SERIES_STYLE[m]
        t, v = series(m, "max_late_localL2")
        axes[2].plot(t, v, ls=ls, color=c)
    axes[2].set_xlabel(r"$t$"); axes[2].set_ylabel(r"max late local $L^2(B_m)$")
    # mark late activation
    t_on = cfg["windows"][cfg["G"] - 1][0]
    for a in axes:
        a.axvline(t_on, color="0.6", lw=0.5, ls="--")
    savefig_multi(fig, os.path.join(fig_dir, "figure_staged_late_timeseries"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/staged_multi_island")
    args = ap.parse_args()
    RD = args.results_dir
    with open(os.path.join(RD, "config.json")) as fj:
        cfg = json.load(fj)
    fig_dir = os.path.join(RD, "figures"); os.makedirs(fig_dir, exist_ok=True)
    plot_dir = os.path.join(RD, "plot_data"); os.makedirs(plot_dir, exist_ok=True)
    island_rows = load_csv(os.path.join(RD, "island_masses.csv"))
    ts_rows = load_csv(os.path.join(RD, "time_series.csv"))
    fig_fields(RD, cfg, island_rows, fig_dir, plot_dir)
    fig_late_timeseries(RD, cfg, ts_rows, fig_dir)
    print("wrote figures to", fig_dir)


if __name__ == "__main__":
    main()
