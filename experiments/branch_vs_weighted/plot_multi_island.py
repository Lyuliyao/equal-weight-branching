"""Publication figures for the separated-growth-island benchmark (Sec. 5.2).
================================================================================

Reads ONLY the saved data products of `multi_island.py` (CSV + field npz) and
regenerates every figure WITHOUT rerunning the solver.  Each figure also writes a
small `plot_data/figure_<name>.npz` so it can be re-styled later.

Figures (CLAUDE.md Sec. 2):
  1. final-field comparison  : reference, weighted, weighted+ESS-resample, min-var
  2. per-island E_m heatmaps : 4x4 grid, shared 0-50% scale, 20% contour
  3. degeneracy time series  : global vs local ESS / local particle counts
  4. zoom/inset on the worst island (chosen by argmax_m E_m for the resampled run)
  5. E_m vs amplitude         : weaker islands are the ones lost by global resampling

Usage:  python plot_multi_island.py --results_dir results/multi_island
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common_plot_style as cps
from common_plot_style import (TEXTWIDTH_IN, METHOD_COLORS, savefig_multi,
                               add_zoom_inset, ERR_CMAP, err_norm, ERR_THRESH)

cps.apply_style()

# panel titles per method key (compact, to fit 4 narrow panels across textwidth)
PANEL_TITLE = {
    "reference": "reference",
    "weighted": "weighted",
    "weighted_ess_resample": "weighted+resample",
    "minvar_branch": "min.-var. branching",
    "poisson_branch": "Poisson branching",
    "minvar_branch_cap": "min.-var. (capped)",
}
SHORT = {
    "weighted": "weighted",
    "weighted_ess_resample": "weighted+resample",
    "minvar_branch": "min.-var. branch",
    "poisson_branch": "Poisson branch",
    "minvar_branch_cap": "min.-var. (capped)",
}


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def f(row, k):
    v = row[k]
    return float(v) if v not in ("", "nan") else np.nan


def island_E_by_method(island_rows, M):
    """Return {method: (E_mean[M], E_seedstd[M], amplitude[M], cx[M], cy[M])}."""
    methods = sorted({r["method"] for r in island_rows},
                     key=lambda m: list(PANEL_TITLE).index(m) if m in PANEL_TITLE else 99)
    out = {}
    for method in methods:
        E = np.full((M,), np.nan); Estd = np.full((M,), np.nan)
        amp = np.full((M,), np.nan); cx = np.full((M,), np.nan); cy = np.full((M,), np.nan)
        for m in range(M):
            vals = [f(r, "E_m") for r in island_rows
                    if r["method"] == method and int(r["m"]) == m]
            if vals:
                E[m] = np.mean(vals); Estd[m] = np.std(vals)
            anyrow = next((r for r in island_rows
                           if r["method"] == method and int(r["m"]) == m), None)
            if anyrow:
                amp[m] = f(anyrow, "amplitude"); cx[m] = f(anyrow, "cx"); cy[m] = f(anyrow, "cy")
        out[method] = (E, Estd, amp, cx, cy)
    return out


# ---------------------------------------------------------------------------
def fig_fields(RD, cfg, plot_dir):
    ref = np.load(os.path.join(RD, "fields_ref.npz"))
    seed0 = np.load(os.path.join(RD, "fields_seed0.npz"))
    XX, YY = ref["XX"], ref["YY"]
    extent = [-np.pi, np.pi, -np.pi, np.pi]
    panels = [("reference", ref["reference"])]
    for key in ["weighted", "weighted_ess_resample", "minvar_branch"]:
        if key in seed0.files:
            panels.append((key, seed0[key]))
    vmax = max(np.max(p) for _, p in panels)
    norm = PowerNorm(gamma=0.40, vmin=0.0, vmax=vmax)
    npan = len(panels)
    fig, axes = plt.subplots(1, npan,
                             figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN / npan * 1.18),
                             constrained_layout=True)
    for ax, (key, fld) in zip(axes, panels):
        im = ax.imshow(fld, origin="lower", extent=extent, norm=norm,
                       cmap="magma", interpolation="nearest")
        ax.set_title(PANEL_TITLE.get(key, key), fontsize=6.0, pad=2)
        ax.scatter(ref["centers"][:, 0], ref["centers"][:, 1], s=2,
                   facecolors="none", edgecolors="cyan", linewidths=0.4)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.82, pad=0.012, aspect=22,
                 label=r"$u(T,x)$")
    savefig_multi(fig, os.path.join(plot_dir, "figure_fields"))
    np.savez(os.path.join(plot_dir, "figure_fields_data.npz"),
             reference=ref["reference"],
             **{k: seed0[k] for _, k in [(0, kk) for kk in seed0.files]},
             centers=ref["centers"], vmax=vmax, extent=np.array(extent))


def fig_heatmap(RD, cfg, island_rows, plot_dir):
    M = cfg["M"]; n = int(round(np.sqrt(M)))
    by = island_E_by_method(island_rows, M)
    methods = [m for m in ["weighted", "weighted_ess_resample",
                           "minvar_branch", "poisson_branch", "minvar_branch_cap"]
               if m in by]
    # per-seed failure counts (matching the paper table), from metrics_summary
    persd = {}
    msum = os.path.join(RD, "metrics_summary.csv")
    if os.path.exists(msum):
        for r in load_csv(msum):
            persd[r["method"]] = f(r, "num_Em_gt_20pct")
    extent = [0.5, n + 0.5, 0.5, n + 0.5]
    fig, axes = plt.subplots(1, len(methods),
                             figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN / len(methods) * 1.12),
                             constrained_layout=True)
    if len(methods) == 1:
        axes = [axes]
    store = {}
    for ax, method in zip(axes, methods):
        E = by[method][0].reshape(n, n)
        store[method] = E
        im = ax.imshow(E, origin="upper", cmap=ERR_CMAP, norm=err_norm(),
                       extent=extent, interpolation="nearest")
        cps.annotate_threshold_contour(ax, E, extent, thresh=ERR_THRESH)
        ax.set_title(SHORT.get(method, method), fontsize=7)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        # annotate with the per-seed failure count (matches the paper table)
        nfail = persd.get(method, float(np.sum(by[method][0] > ERR_THRESH)))
        ax.set_xlabel(f"$\\langle\\#\\{{E_m{{>}}20\\%\\}}\\rangle={nfail:.1f}/{M}$", fontsize=6)
    cb = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.012, aspect=22)
    cb.set_label(r"seed-mean $E_m$")
    cb.ax.axhline(ERR_THRESH, color="cyan", lw=0.8)
    savefig_multi(fig, os.path.join(plot_dir, "figure_island_heatmap"))
    np.savez(os.path.join(plot_dir, "figure_island_heatmap_data.npz"),
             **{f"E_{k}": v for k, v in store.items()})


def fig_timeseries(RD, ts_rows, plot_dir):
    methods = sorted({r["method"] for r in ts_rows})
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 0.34 * TEXTWIDTH_IN),
                             constrained_layout=True)
    for a in axes:
        a.set_box_aspect(1)

    def series(method, col):
        ts = sorted({f(r, "t") for r in ts_rows if r["method"] == method})
        m = []
        for t in ts:
            v = [f(r, col) for r in ts_rows
                 if r["method"] == method and abs(f(r, "t") - t) < 1e-9]
            m.append(np.nanmean(v))
        return np.array(ts), np.array(m)

    # panel 0: global nESS for weighted methods (looks acceptable / resamples back up)
    for method in ["weighted", "weighted_ess_resample"]:
        if method in methods:
            t, g = series(method, "global_nESS")
            axes[0].plot(t, g, color=METHOD_COLORS["weighted"] if method == "weighted"
                         else METHOD_COLORS["resampled"],
                         ls="-" if method == "weighted" else "--",
                         label=SHORT[method])
    axes[0].axhline(0.5, color="gray", lw=0.6, ls=":")
    axes[0].set_xlabel(r"$t$"); axes[0].set_ylabel(r"global nESS")
    axes[0].set_ylim(0, 1.05); axes[0].legend(loc="lower left", fontsize=6)

    # panel 1: min island-local effective count (the local degeneracy)
    style = {"weighted": ("-", METHOD_COLORS["weighted"]),
             "weighted_ess_resample": ("--", METHOD_COLORS["resampled"]),
             "minvar_branch": ("-", METHOD_COLORS["minvar"]),
             "poisson_branch": ("-", METHOD_COLORS["poisson"])}
    for method in methods:
        if method in style:
            t, mn = series(method, "min_local_eff")
            ls, c = style[method]
            axes[1].semilogy(t, np.maximum(mn, 0.5), color=c, ls=ls,
                             label=SHORT.get(method, method))
    axes[1].set_xlabel(r"$t$")
    axes[1].set_ylabel(r"min island-local eff. count")
    axes[1].legend(loc="lower left", fontsize=5.5)

    # panel 2: max island mass error E_m
    for method in methods:
        if method in style:
            t, mx = series(method, "max_Em")
            ls, c = style[method]
            axes[2].plot(t, mx, color=c, ls=ls)
    axes[2].axhline(0.20, color="gray", lw=0.6, ls=":")
    axes[2].set_xlabel(r"$t$"); axes[2].set_ylabel(r"$\max_m E_m$")

    savefig_multi(fig, os.path.join(plot_dir, "figure_degeneracy_timeseries"))


def fig_zoom(RD, cfg, island_rows, plot_dir):
    ref = np.load(os.path.join(RD, "fields_ref.npz"))
    seed0 = np.load(os.path.join(RD, "fields_seed0.npz"))
    M = cfg["M"]
    by = island_E_by_method(island_rows, M)
    # choose the worst island for the resampled run (diagnostic-driven, not by eye)
    pick_method = "weighted_ess_resample" if "weighted_ess_resample" in by else "weighted"
    E = by[pick_method][0]
    m_star = int(np.nanargmax(E))
    cx, cy = by[pick_method][3][m_star], by[pick_method][4][m_star]
    half = max(3.0 * cfg["sigma"], 0.45)
    zbox = (cx - half, cx + half, cy - half, cy + half)
    extent = [-np.pi, np.pi, -np.pi, np.pi]
    panels = [("reference", ref["reference"])]
    for key in [pick_method, "minvar_branch"]:
        if key in seed0.files:
            panels.append((key, seed0[key]))
    vmax = max(np.max(p) for _, p in panels)
    norm = PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax)
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN / len(panels) * 1.08),
                             constrained_layout=True)
    for ax, (key, fld) in zip(axes, panels):
        ax.imshow(fld, origin="lower", extent=extent, norm=norm, cmap="magma",
                  interpolation="nearest")
        ax.set_title(PANEL_TITLE.get(key, key), fontsize=7)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        add_zoom_inset(ax, fld, extent, zbox, vmin=None, vmax=None, cmap="magma",
                       loc="upper right", zoom_frac=0.5)
        # match inset norm to panel norm: redraw inset image with same norm
        for child in ax.child_axes:
            for im in child.images:
                im.set_norm(norm)
    fig.suptitle(
        f"zoom on island $m={m_star}$ (amplitude $a={by[pick_method][2][m_star]:.3f}$, "
        f"$E_m^{{resample}}={E[m_star]:.2f}$)", fontsize=7)
    savefig_multi(fig, os.path.join(plot_dir, "figure_zoom_worst_island"))
    np.savez(os.path.join(plot_dir, "figure_zoom_data.npz"),
             m_star=m_star, zbox=np.array(zbox), cx=cx, cy=cy)


def fig_Em_vs_amp(RD, cfg, island_rows, plot_dir):
    M = cfg["M"]
    by = island_E_by_method(island_rows, M)
    methods = [m for m in ["weighted", "weighted_ess_resample",
                           "minvar_branch", "poisson_branch"] if m in by]
    fig, ax = plt.subplots(figsize=(0.62 * TEXTWIDTH_IN, 0.52 * TEXTWIDTH_IN),
                           constrained_layout=True)
    cmap = {"weighted": METHOD_COLORS["weighted"],
            "weighted_ess_resample": METHOD_COLORS["resampled"],
            "minvar_branch": METHOD_COLORS["minvar"],
            "poisson_branch": METHOD_COLORS["poisson"]}
    for method in methods:
        E, Estd, amp, _, _ = by[method]
        o = np.argsort(amp)
        ax.plot(amp[o], E[o], "o-", ms=3, color=cmap.get(method, "k"),
                label=SHORT.get(method, method), lw=1.0)
    ax.axhline(ERR_THRESH, color="gray", lw=0.6, ls=":")
    ax.set_xlabel(r"island amplitude $a_m$")
    ax.set_ylabel(r"per-island mass error $E_m$")
    ax.legend(fontsize=6, loc="upper right")
    savefig_multi(fig, os.path.join(plot_dir, "figure_Em_vs_amplitude"))
    np.savez(os.path.join(plot_dir, "figure_Em_vs_amplitude_data.npz"),
             **{f"E_{m}": by[m][0] for m in methods},
             amplitude=by[methods[0]][2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/multi_island")
    args = ap.parse_args()
    RD = args.results_dir
    with open(os.path.join(RD, "config.json")) as fjson:
        cfg = json.load(fjson)
    plot_dir = os.path.join(RD, "plot_data")
    os.makedirs(plot_dir, exist_ok=True)
    fig_dir = os.path.join(RD, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    island_rows = load_csv(os.path.join(RD, "island_masses.csv"))
    ts_rows = load_csv(os.path.join(RD, "time_series.csv"))

    # figures into figures/, plot data into plot_data/
    fig_fields(RD, cfg, fig_dir)
    fig_heatmap(RD, cfg, island_rows, fig_dir)
    fig_timeseries(RD, ts_rows, fig_dir)
    fig_zoom(RD, cfg, island_rows, fig_dir)
    fig_Em_vs_amp(RD, cfg, island_rows, fig_dir)
    # also drop the plot_data npz copies in plot_data/
    for name in ["figure_fields_data.npz", "figure_island_heatmap_data.npz",
                 "figure_zoom_data.npz", "figure_Em_vs_amplitude_data.npz"]:
        src = os.path.join(fig_dir, name)
        if os.path.exists(src):
            os.replace(src, os.path.join(plot_dir, name))
    print("wrote figures to", fig_dir, "and plot data to", plot_dir)


if __name__ == "__main__":
    main()
