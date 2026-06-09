"""
Plotting for Experiment 1 (branch vs weighted).
Reads results/branch_vs_weighted/{metrics.csv, fields_seed*.npz} and writes
publication-style PDFs into the SAME results dir:
  (i)   final-time snapshots: reference | weighted | poisson | minvar
  (ii)  global L2-rel error vs t for the 3 methods
  (iii) weighted global+local nESS and max_w/mean_w vs t, branching local count overlaid
  (iv)  seed boxplots of final L2

Run:  python plot.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = "results/branch_vs_weighted"
METHODS = ["weighted", "poisson", "minvar"]
COLORS = {"weighted": "tab:red", "poisson": "tab:blue", "minvar": "tab:green"}


def load_metrics():
    rows = list(csv.DictReader(open(os.path.join(RD, "metrics.csv"))))
    def f(x):
        try:
            return float(x)
        except (ValueError, TypeError):
            return np.nan
    for r in rows:
        for k in r:
            if k != "method":
                r[k] = f(r[k])
    return rows


def plot_snapshots():
    seeds = sorted(int(fn.split("seed")[1].split(".")[0])
                   for fn in os.listdir(RD) if fn.startswith("fields_seed"))
    if not seeds:
        return
    seed = seeds[0]
    d = np.load(os.path.join(RD, f"fields_seed{seed}.npz"))
    XX, YY = d["XX"], d["YY"]
    panels = [("reference", d["reference"]), ("weighted", d["weighted"]),
              ("poisson", d["poisson"]), ("minvar", d["minvar"])]
    vmax = max(np.max(p[1]) for p in panels)
    vmin = min(np.min(p[1]) for p in panels)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for ax, (name, u) in zip(axes, panels):
        im = ax.pcolormesh(XX, YY, u, shading="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
    fig.colorbar(im, ax=axes, shrink=0.8, label="u")
    fig.suptitle(f"Final-time density (seed {seed})")
    fig.savefig(os.path.join(RD, "snapshots_final.pdf"))
    plt.close(fig)


def plot_l2_vs_t(rows):
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    for m in METHODS:
        ts = sorted(set(r["t"] for r in rows if r["method"] == m))
        means, stds = [], []
        for t in ts:
            vals = [r["L2_rel_err"] for r in rows if r["method"] == m and r["t"] == t]
            means.append(np.nanmean(vals)); stds.append(np.nanstd(vals))
        means, stds = np.array(means), np.array(stds)
        ax.plot(ts, means, "-o", color=COLORS[m], label=m, ms=3)
        ax.fill_between(ts, means - stds, means + stds, color=COLORS[m], alpha=0.2)
    ax.set_xlabel("t"); ax.set_ylabel("global L2 relative error")
    ax.set_title("L2 relative error vs time")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(RD, "l2_vs_t.pdf"))
    plt.close(fig)


def plot_ness(rows):
    seed = sorted(set(int(r["seed"]) for r in rows))[0]
    wr = sorted([r for r in rows if r["method"] == "weighted" and int(r["seed"]) == seed],
                key=lambda r: r["t"])
    pr = sorted([r for r in rows if r["method"] == "poisson" and int(r["seed"]) == seed],
                key=lambda r: r["t"])
    ts = [r["t"] for r in wr]
    fig, ax1 = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax1.plot(ts, [r["global_nESS"] for r in wr], "-o", color="tab:red",
             label="weighted global nESS", ms=3)
    ax1.plot(ts, [r["local_nESS_B"] for r in wr], "--s", color="tab:orange",
             label="weighted local nESS (B)", ms=3)
    ax1.set_xlabel("t"); ax1.set_ylabel("nESS")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(ts, [r["max_w_over_mean_w"] for r in wr], "-^", color="tab:purple",
             label="max_w/mean_w", ms=3)
    ax2.plot(ts, [r["N_local_B"] for r in pr], "-d", color="tab:blue",
             label="poisson local count (B)", ms=3)
    ax2.set_ylabel("max_w/mean_w  &  branching local count")
    ax2.set_yscale("log")
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, fontsize=8, loc="center left")
    ax1.set_title(f"Weight degeneracy vs branching (seed {seed})")
    fig.savefig(os.path.join(RD, "ness_vs_t.pdf"))
    plt.close(fig)


def plot_boxplots(rows):
    tmax = max(r["t"] for r in rows)
    data = []
    for m in METHODS:
        vals = [r["L2_rel_err"] for r in rows if r["method"] == m and r["t"] == tmax]
        data.append(vals)
    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    bp = ax.boxplot(data, labels=METHODS, patch_artist=True)
    for patch, m in zip(bp["boxes"], METHODS):
        patch.set_facecolor(COLORS[m]); patch.set_alpha(0.5)
    ax.set_ylabel("final-time L2 relative error")
    ax.set_title("Seed distribution of final L2 error")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(os.path.join(RD, "boxplot_final_l2.pdf"))
    plt.close(fig)


def main():
    rows = load_metrics()
    plot_snapshots()
    plot_l2_vs_t(rows)
    plot_ness(rows)
    plot_boxplots(rows)
    print("wrote PDFs to", RD)


if __name__ == "__main__":
    main()
