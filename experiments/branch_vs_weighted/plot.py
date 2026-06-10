"""Publication figures for the branch-vs-weighted experiment (paper Sec. 5.2).

Sizes follow paper_style: physical width = include-fraction x 4.773 in, so the
fonts print at face size. Include fractions: snapshots 1.0, ness 0.85,
l2-vs-t 0.70, boxplot 0.60 (keep in sync with the .tex).
"""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paper_style import apply_style, fig_size, TEXTWIDTH_IN, METHOD_COLORS, METHOD_LABELS

apply_style()
RD = "results/branch_vs_weighted"
METHODS = ["weighted", "poisson", "minvar"]


def load_metrics():
    with open(os.path.join(RD, "metrics.csv")) as f:
        return list(csv.DictReader(f))


def seed_avg(rows, method, col):
    ts = sorted({float(r["t"]) for r in rows})
    out_t, out_m, out_s = [], [], []
    for t in ts:
        v = [float(r[col]) for r in rows
             if r["method"] == method and abs(float(r["t"]) - t) < 1e-12
             and r[col] not in ("", "nan")]
        if v:
            out_t.append(t); out_m.append(np.mean(v)); out_s.append(np.std(v))
    return np.array(out_t), np.array(out_m), np.array(out_s)


def plot_snapshots(rows):
    d = np.load(os.path.join(RD, "fields_seed0.npz"))
    fields = [("reference", d["reference"]), ("weighted", d["weighted"]),
              ("poisson", d["poisson"]), ("minvar", d["minvar"])]
    vmax = max(np.max(f) for _, f in fields)
    fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH_IN, 0.30 * TEXTWIDTH_IN),
                             constrained_layout=True)
    for ax, (name, f) in zip(axes, fields):
        im = ax.imshow(f, origin="lower", extent=[-np.pi, np.pi, -np.pi, np.pi],
                       vmin=0, vmax=vmax, cmap="viridis")
        ax.set_title({"reference": "reference", "weighted": "weighted", "poisson": "Poisson", "minvar": "min.-variance"}[name], fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015, aspect=14)
    fig.savefig(os.path.join(RD, "snapshots_final.pdf"))
    plt.close(fig)


def plot_ness(rows):
    fig, axes = plt.subplots(1, 3, figsize=(0.85 * TEXTWIDTH_IN, 0.40 * TEXTWIDTH_IN),
                             constrained_layout=True)
    for a in axes:
        a.set_box_aspect(1)
    t, g, _ = seed_avg(rows, "weighted", "global_nESS")
    tb, lb, _ = seed_avg(rows, "weighted", "local_nESS_B")
    axes[0].plot(t, g, color=METHOD_COLORS["weighted"])
    axes[0].plot(tb, lb, color=METHOD_COLORS["weighted"], ls="--")
    axes[0].set_xlabel(r"$t$"); axes[0].set_ylabel(r"$\mathrm{nESS}$")
    axes[0].set_ylim(0, 1.05)
    axes[0].text(0.62, 0.10, "global", color=METHOD_COLORS["weighted"], fontsize=7,
                 transform=axes[0].transAxes)
    axes[0].text(0.40, 0.72, r"local ($B$)", color=METHOD_COLORS["weighted"], fontsize=7,
                 transform=axes[0].transAxes)
    t, mw, _ = seed_avg(rows, "weighted", "max_w_over_mean_w")
    axes[1].semilogy(t, mw, color=METHOD_COLORS["weighted"])
    axes[1].set_xlabel(r"$t$"); axes[1].set_ylabel(r"$\max_i w_i\,/\,\overline{w}$")
    for m in ["poisson", "minvar"]:
        t, nb, sb = seed_avg(rows, m, "N_local_B")
        axes[2].semilogy(t, nb, color=METHOD_COLORS[m])
    axes[2].set_xlabel(r"$t$"); axes[2].set_ylabel(r"$N_B$")
    axes[2].text(0.05, 0.90, "Poisson", color=METHOD_COLORS["poisson"], fontsize=7,
                 ha="left", va="top", transform=axes[2].transAxes)
    axes[2].text(0.05, 0.78, "min.-var.", color=METHOD_COLORS["minvar"], fontsize=7,
                 ha="left", va="top", transform=axes[2].transAxes)
    fig.savefig(os.path.join(RD, "ness_vs_t.pdf"))
    plt.close(fig)


def plot_l2(rows):
    fig, ax = plt.subplots(figsize=fig_size(0.70, aspect=0.95),
                           constrained_layout=True)
    ax.set_box_aspect(1)
    for m in METHODS:
        t, mu, sd = seed_avg(rows, m, "L2_rel_err")
        ax.semilogy(t, mu, color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        ax.fill_between(t, np.maximum(mu - sd, 1e-12), mu + sd,
                        color=METHOD_COLORS[m], alpha=0.2, lw=0)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"relative $L^2$ error")
    ax.legend(loc="lower left", fontsize=7)
    fig.savefig(os.path.join(RD, "l2_vs_t.pdf"))
    plt.close(fig)


def plot_box(rows):
    tmax = max(float(r["t"]) for r in rows)
    data = [[float(r["L2_rel_err"]) for r in rows
             if r["method"] == m and abs(float(r["t"]) - tmax) < 1e-12] for m in METHODS]
    fig, ax = plt.subplots(figsize=fig_size(0.60, aspect=0.95),
                           constrained_layout=True)
    ax.set_box_aspect(1)
    bp = ax.boxplot(data, tick_labels=["weighted", "Poisson", "min.-var."],
                    patch_artist=True, widths=0.55)
    for patch, m in zip(bp["boxes"], METHODS):
        patch.set_facecolor(METHOD_COLORS[m]); patch.set_alpha(0.45)
    for med in bp["medians"]:
        med.set_color("k")
    ax.set_yscale("log"); ax.set_ylabel(r"relative $L^2$ error at $T$")
    fig.savefig(os.path.join(RD, "boxplot_final_l2.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    rows = load_metrics()
    plot_snapshots(rows)
    plot_ness(rows)
    plot_l2(rows)
    plot_box(rows)
    print("wrote restyled PDFs to", RD)
