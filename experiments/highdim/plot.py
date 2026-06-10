"""Publication figures for the high-dimensional kinetic stress test (Sec. 5.4).

highdim_metrics_d{4,6}.pdf : 2x3 panels, full width (4.773 in).
highdim_marginals_d6.pdf   : histogram vs FHT 1D marginals, width 0.9*4.773 in.
"""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paper_style import apply_style, TEXTWIDTH_IN, METHOD_COLORS, METHOD_LABELS

apply_style()
RD = "results/highdim"
METHODS = ["weighted", "poisson", "minvar"]


def rows_d(rows, d):
    return [r for r in rows if r["d"] == str(d)]


def seed_avg(rows, method, col):
    ts = sorted({float(r["t"]) for r in rows})
    T, M = [], []
    for t in ts:
        v = [float(r[col]) for r in rows
             if r["method"] == method and abs(float(r["t"]) - t) < 1e-12
             and r[col] not in ("", "nan")]
        if v:
            T.append(t); M.append(np.mean(v))
    return np.array(T), np.array(M)


def metrics_figure(rows, d):
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH_IN, 0.62 * TEXTWIDTH_IN),
                             constrained_layout=True)
    panels = [("moment_m", r"$m(t)$", METHODS, "linear"),
              ("total_mass", r"total mass $M(t)$", METHODS, "linear"),
              ("N_active", r"$N_{\mathrm{act}}$", ["poisson", "minvar"], "linear"),
              ("N_local_B", r"$N_B$", ["poisson", "minvar"], "linear"),
              ("global_nESS", r"$\mathrm{nESS}$ (weighted)", ["weighted"], "linear"),
              ("max_w_over_mean_w", r"$\max_i w_i/\overline{w}$ (weighted)", ["weighted"], "log")]
    for ax, (col, ylab, ms, sc) in zip(axes.flat, panels):
        for m in ms:
            t, mu = seed_avg(rows, m, col)
            ax.plot(t, mu, color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        if sc == "log":
            ax.set_yscale("log")
        ax.set_ylabel(ylab); ax.set_xlabel(r"$t$")
    axes[0, 0].legend(fontsize=7)
    fig.savefig(os.path.join(RD, f"highdim_metrics_d{d}.pdf").replace("highdim_metrics", "metrics") if False
                else os.path.join(RD, f"metrics_d{d}.pdf"))
    plt.close(fig)


def marginals_figure(d=6, seed=0):
    m = np.load(os.path.join(RD, f"marginals_d{d}_seed{seed}.npz"))
    f = np.load(os.path.join(RD, f"fht_d{d}_seed{seed}.npz"), allow_pickle=True)
    centers = m["centers"]; h = m["hist1d_minvar"]
    zg = f["zgrid"]; fm = f["fht_marg1d"]
    ncol = 3; nrow = int(np.ceil(d / ncol))
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(0.9 * TEXTWIDTH_IN, 0.30 * TEXTWIDTH_IN * nrow),
                             constrained_layout=True, sharex=True, sharey=True)
    width = centers[1] - centers[0]
    for j, ax in enumerate(axes.flat):
        if j >= d:
            ax.axis("off"); continue
        ax.bar(centers, h[j], width=width, color="0.78", lw=0, label="histogram")
        ax.plot(zg, fm[j], color="#d62728", lw=1.2, label="FHT")
        ax.set_title(rf"$z_{{{j+1}}}$", fontsize=8)
        ax.grid(False)
    axes.flat[0].legend(fontsize=7)
    fig.savefig(os.path.join(RD, f"marginals_d{d}.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    rows = list(csv.DictReader(open(os.path.join(RD, "metrics.csv"))))
    for d in (4, 6):
        sub = rows_d(rows, d)
        if sub:
            metrics_figure(sub, d)
    marginals_figure(6, 0)
    print("wrote restyled PDFs to", RD)
