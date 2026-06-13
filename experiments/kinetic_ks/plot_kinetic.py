"""Figures for the 6D field-coupled kinetic Keller-Segel experiment.

Reads metrics.csv (one row per seed/method/snapshot) and produces 3 PDFs:
  kinetic_field.pdf  : seed-averaged mass, ||c||_inf, ||rho||_inf, R_0.5/R_0.9 vs t
  kinetic_repr.pdf   : weighted nESS / max:mean w (degeneracy) and branching
                       N_active; local ESS (weighted) vs local count (branching) in B
  kinetic_coupling.pdf: corr(r,c), corr(r,rho), mean_Sc vs t (field-coupling evidence)

Usage:
  python plot_kinetic.py --results_dir <dir-with-metrics.csv> [--out_dir <dir>]
"""
import os
import argparse
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["weighted", "weighted_resample", "poisson", "minvar"]
LABEL = {"weighted": "weighted", "weighted_resample": "weighted+resample",
         "poisson": "Poisson branch", "minvar": "min-var branch"}
COLOR = {"weighted": "tab:red", "weighted_resample": "tab:orange",
         "poisson": "tab:blue", "minvar": "tab:green"}


def load(results_dir):
    path = os.path.join(results_dir, "metrics.csv")
    rows = list(csv.DictReader(open(path)))
    # group by (method, t) -> list over seeds, then average
    agg = {m: defaultdict(list) for m in METHODS}
    for r in rows:
        m = r["method"]
        if m not in agg:
            continue
        agg[m][float(r["t"])].append(r)
    out = {}
    for m in METHODS:
        ts = sorted(agg[m].keys())
        out[m] = {"t": np.array(ts)}
        if not ts:
            continue
        keys = [k for k in rows[0].keys() if k not in ("method", "d", "seed", "step")]
        for k in keys:
            vals = []
            for t in ts:
                col = []
                for r in agg[m][t]:
                    try:
                        col.append(float(r[k]))
                    except (ValueError, TypeError):
                        col.append(np.nan)
                vals.append(np.nanmean(col))
            out[m][k] = np.array(vals)
    return out


def plot_field(d, out_dir):
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.5))
    panels = [("total_mass", "total mass $M(t)$"),
              ("c_inf_grid", r"$\|c\|_\infty$"),
              ("rho_inf_grid", r"$\|\rho\|_\infty$"),
              ("R_core_0p5", r"core radius $R_{0.5}$ (solid), $R_{0.9}$ (dashed)")]
    for a, (key, title) in zip(ax.ravel(), panels):
        for m in METHODS:
            if key in d[m] and len(d[m]["t"]):
                a.plot(d[m]["t"], d[m][key], color=COLOR[m], label=LABEL[m])
                if key == "R_core_0p5" and "R_core_0p9" in d[m]:
                    a.plot(d[m]["t"], d[m]["R_core_0p9"], color=COLOR[m], ls="--")
        a.set_xlabel("t"); a.set_title(title); a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "kinetic_field.pdf"))
    plt.close(fig)


def plot_repr(d, out_dir):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    # global nESS (weighted methods)
    for m in ["weighted", "weighted_resample"]:
        if "global_nESS" in d[m] and len(d[m]["t"]):
            ax[0].plot(d[m]["t"], d[m]["global_nESS"], color=COLOR[m], label=LABEL[m])
    ax[0].axhline(1.0, color="k", ls=":", lw=0.8, label="equal weights (branch)")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("global nESS"); ax[0].set_title("weight degeneracy")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    # max:mean weight (weighted)
    for m in ["weighted", "weighted_resample"]:
        if "max_w_over_mean_w" in d[m] and len(d[m]["t"]):
            ax[1].plot(d[m]["t"], d[m]["max_w_over_mean_w"], color=COLOR[m], label=LABEL[m])
    ax[1].set_xlabel("t"); ax[1].set_ylabel("max:mean weight"); ax[1].set_title("weight concentration")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    # local resolution: weighted ABSOLUTE local ESS (= normalized nESS_B * N_B)
    # vs branching equal-weight local count, both absolute and comparable.
    for m in METHODS:
        if "N_local_B" in d[m] and len(d[m]["t"]):
            if "weight" in m:
                y = d[m]["local_ESS_B"] * d[m]["N_local_B"]   # absolute effective count
                lab = LABEL[m] + " (local ESS)"
            else:
                y = d[m]["N_local_B"]
                lab = LABEL[m] + " (local count)"
            ax[2].plot(d[m]["t"], y, color=COLOR[m], label=lab)
    ax[2].set_xlabel("t"); ax[2].set_ylabel("local resolution in B")
    ax[2].set_title("local resolution"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "kinetic_repr.pdf"))
    plt.close(fig)


def plot_coupling(d, out_dir):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    panels = [("corr_r_c", r"corr$(r,c)$"), ("corr_r_rho", r"corr$(r,\rho)$"),
              ("mean_Sc", r"mean $S_c(c)$ (activation)")]
    for a, (key, title) in zip(ax, panels):
        for m in METHODS:
            if key in d[m] and len(d[m]["t"]):
                a.plot(d[m]["t"], d[m][key], color=COLOR[m], label=LABEL[m])
        a.set_xlabel("t"); a.set_title(title); a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "kinetic_coupling.pdf"))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()
    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)
    d = load(args.results_dir)
    plot_field(d, out_dir)
    plot_repr(d, out_dir)
    plot_coupling(d, out_dir)
    print(f"wrote kinetic_field.pdf, kinetic_repr.pdf, kinetic_coupling.pdf to {out_dir}")


if __name__ == "__main__":
    main()
