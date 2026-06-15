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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments"))
from paper_style import apply_style, fig_size, TEXTWIDTH_IN, METHOD_COLORS, METHOD_LABELS

apply_style()
# Both figures are included at 0.4\linewidth side by side, so generate BOTH at the same
# SQUARE physical size -> equal displayed size and consistent font scaling at 0.4 width.
mpl.rcParams.update({"axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 7.5})
FIG_S = 0.46 * TEXTWIDTH_IN          # square side, in inches
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


# Plot box (figure fraction) produced by plot_l2's constrained_layout.  The snapshot
# 2x2 field block is placed in the SAME square bbox so the two figures read at equal
# size at 0.4\linewidth; the colorbar sits in the left margin, exactly where the l2
# y-label and ticks sit.  (If plot_l2's layout changes, re-measure and update this.)
L2_BOX = (0.2166, 0.1799, 0.9746, 0.9379)     # x0, y0, x1, y1


def plot_snapshots(rows=None):
    """Fig. 2: 2x2 final-time fields -- reference, weighted, weighted+ESS, min.-variance
    (same initial budget N0=2e4, same final time T=1, one shared color scale).

    Manual layout (no constrained_layout): the 2x2 colored block fills the same square
    bbox as the l2 plot box, titles sit above each panel, a small inter-row gap and a
    balanced bottom margin keep the two figures visually the same size at 0.4\\linewidth.
    """
    d = np.load(os.path.join(RD, "fig52_fields_seed0.npz"))
    ext = d["extent"] if "extent" in d.files else [-np.pi, np.pi, -np.pi, np.pi]
    panels = [("reference", "reference"), ("weighted", "weighted"),
              ("weighted_ess", "weighted + ESS"), ("minvar", "min.-variance")]
    vmax = max(float(np.max(d[k])) for k, _ in panels)

    x0, y0, x1, y1 = L2_BOX
    side = min(x1 - x0, y1 - y0)
    bx0 = x1 - side                              # right edge aligned with l2 plot box
    by0 = 0.5 * (y0 + y1) - 0.5 * side           # vertical center matches l2 plot box
    gap = 0.065                                  # small inter-row/col gap (tighter than A)
    ps = 0.5 * (side - gap)                      # square panel side
    cols = [bx0, bx0 + ps + gap]
    rows_y = [by0 + ps + gap, by0]               # top row, bottom row

    fig = plt.figure(figsize=(FIG_S, FIG_S))
    im = None
    for (key, title), (r, c) in zip(panels, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        ax = fig.add_axes([cols[c], rows_y[r], ps, ps])
        im = ax.imshow(d[key], origin="lower", extent=ext, vmin=0, vmax=vmax,
                       cmap="viridis", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(title, fontsize=8, pad=2)
    # colorbar in the left margin (where the l2 y-label/ticks live), spanning the block
    cax = fig.add_axes([bx0 - 0.11, by0, 0.045, side])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.yaxis.set_ticks_position("left"); cb.ax.yaxis.set_label_position("left")
    cb.set_ticks([0, 200, 400, 600]); cb.ax.tick_params(labelsize=7)
    # save at the full SQUARE figsize (no tight crop) so it is the SAME canvas as l2
    with mpl.rc_context({"savefig.bbox": "standard"}):
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
    # square AXES BOX (set_box_aspect) so the plot reads the same shape/size as the square
    # snapshot panels at 0.4\linewidth.  Short labels keep the legend inside the box.
    fig, ax = plt.subplots(figsize=(FIG_S, FIG_S), constrained_layout=True)
    ax.set_box_aspect(1)
    series = [("weighted", METHOD_COLORS["weighted"], "weighted"),
              ("__ess__", ESS_COLOR, "weighted + ESS"),
              ("minvar", METHOD_COLORS["minvar"], "min.-variance")]
    for m, color, label in series:
        if m == "__ess__":
            t, mu, sd = _ess_seed_avg()
        else:
            t, mu, sd = seed_avg(rows, m, "L2_rel_err")
        ax.semilogy(t, mu, color=color, lw=1.3, label=label)
        ax.fill_between(t, np.maximum(mu - sd, 1e-12), mu + sd, color=color, alpha=0.18, lw=0)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"relative $L^2$ error")
    # log-y with several labelled dec'l ticks (data span ~0.04--0.35 -> one decade only,
    # so set explicit math-formatted ticks instead of a single 10^-1)
    ax.set_ylim(0.035, 0.42)
    ax.set_yticks([0.04, 0.06, 0.1, 0.2, 0.3])
    ax.set_yticklabels([r"$0.04$", r"$0.06$", r"$0.1$", r"$0.2$", r"$0.3$"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc="lower left", fontsize=7.5, frameon=False, handlelength=1.4,
              borderaxespad=0.4, labelspacing=0.3)
    # save at the full SQUARE figsize (no tight crop) so the square box is centred and the
    # two figures display at the same size at 0.4\linewidth
    with mpl.rc_context({"savefig.bbox": "standard"}):
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
