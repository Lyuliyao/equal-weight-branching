"""Publication figures for the branch-vs-weighted experiment (paper Sec. 5.2).

Fig. 2 (fig:bw_snap, snapshots_final.pdf): 2x2 final-time reconstructed fields
  -- reference, weighted, weighted+ESS resampling, min.-variance branching (N0=2e4).
Fig. 4 (fig:bw_l2, l2_vs_t.pdf): relative L2 error vs time for the same-initial-budget
  weighted, weighted+ESS, and min.-variance branching.

Data (existing saved results; the ESS pieces come from figdata_52.py, run once):
  reference_results/branch_vs_weighted/metrics.csv             weighted/minvar L2-vs-t.
  reference_results/branch_vs_weighted/fig52_ess_l2_vs_t.csv   weighted+ESS L2-vs-t.
  reference_results/branch_vs_weighted/fig52_fields_seed0.npz  seed-0 final fields.
Writes the two PDFs to the data dir AND paper/figure/.  (ness_vs_t / boxplot are no
longer part of the main Sec. 5.2 figure set and are left untouched.)
"""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments"))
from paper_style import apply_style, fig_size, TEXTWIDTH_IN, METHOD_COLORS, METHOD_LABELS

apply_style()
RD = os.path.join(REPO, "reference_results", "branch_vs_weighted")
FIGDIR = os.path.join(REPO, "paper", "figure")
METHODS = ["weighted", "poisson", "minvar"]
ESS_COLOR = METHOD_COLORS["resampled"]      # purple
ESS_LABEL = "weighted + ESS"


def _save(fig, name):
    for d in (RD, FIGDIR):
        fig.savefig(os.path.join(d, name))


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


def plot_snapshots(rows=None):
    """Fig. 2: 2x2 final-time fields -- reference, weighted, weighted+ESS, min.-variance
    (same initial budget N0=2e4, same final time T=1, one shared color scale)."""
    d = np.load(os.path.join(RD, "fig52_fields_seed0.npz"))
    ext = d["extent"] if "extent" in d.files else [-np.pi, np.pi, -np.pi, np.pi]
    panels = [("reference", "reference"), ("weighted", "weighted"),
              ("weighted_ess", "weighted + ESS"), ("minvar", "min.-variance")]
    vmax = max(float(np.max(d[k])) for k, _ in panels)
    fig, axes = plt.subplots(2, 2, figsize=(0.62 * TEXTWIDTH_IN, 0.66 * TEXTWIDTH_IN),
                             constrained_layout=True)
    for ax, (key, title) in zip(axes.ravel(), panels):
        im = ax.imshow(d[key], origin="lower", extent=ext, vmin=0, vmax=vmax,
                       cmap="viridis", aspect="equal")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02, aspect=22)
    cb.ax.tick_params(labelsize=7)
    _save(fig, "snapshots_final.pdf")
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


def _ess_seed_avg():
    """Seed-mean +/- std of the weighted+ESS L2-vs-t (from figdata_52.py)."""
    rows = list(csv.DictReader(open(os.path.join(RD, "fig52_ess_l2_vs_t.csv"))))
    ts = sorted({float(r["t"]) for r in rows})
    t, mu, sd = [], [], []
    for tt in ts:
        v = [float(r["L2_rel_err"]) for r in rows if abs(float(r["t"]) - tt) < 1e-9]
        if v:
            t.append(tt); mu.append(np.mean(v)); sd.append(np.std(v))
    return np.array(t), np.array(mu), np.array(sd)


def plot_l2(rows):
    """Fig. 4: relative L2 error vs time for weighted, weighted+ESS, min.-variance
    (same initial budget N0=2e4).  Weighted/minvar from metrics.csv; ESS from
    fig52_ess_l2_vs_t.csv."""
    fig, ax = plt.subplots(figsize=fig_size(0.5, aspect=0.92), constrained_layout=True)
    ax.set_box_aspect(1)
    series = [("weighted", METHOD_COLORS["weighted"], METHOD_LABELS["weighted"]),
              ("__ess__", ESS_COLOR, ESS_LABEL),
              ("minvar", METHOD_COLORS["minvar"], METHOD_LABELS["minvar"])]
    for m, color, label in series:
        if m == "__ess__":
            t, mu, sd = _ess_seed_avg()
        else:
            t, mu, sd = seed_avg(rows, m, "L2_rel_err")
        ax.semilogy(t, mu, color=color, lw=1.3, label=label)
        ax.fill_between(t, np.maximum(mu - sd, 1e-12), mu + sd, color=color, alpha=0.18, lw=0)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"relative $L^2$ error")
    ax.legend(loc="lower left", fontsize=7.5, frameon=False, handlelength=1.4,
              borderaxespad=0.5, labelspacing=0.3)
    _save(fig, "l2_vs_t.pdf")
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
    plot_snapshots(rows)          # Fig. 2 (2x2 fields, incl. weighted+ESS)
    plot_l2(rows)                 # Fig. 4 (L2-vs-t, incl. weighted+ESS)
    # ness_vs_t / boxplot_final_l2 are no longer in the main Sec. 5.2 figure set; the
    # plot_ness/plot_box helpers are retained but not regenerated here.
    print("wrote snapshots_final.pdf + l2_vs_t.pdf to", RD, "and", FIGDIR)
