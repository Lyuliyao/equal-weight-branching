"""Switching-growth figure candidates: in-panel A/B magnifiers, 2x2 OR 4x1 layout.
================================================================================

Compact "magnifying-glass" redesign of the switching-growth figure (replaces the old
4-col x 3-row full/old-zoom/new-zoom layout).  Each method panel shows the FULL-DOMAIN
field at T=1.2 with two in-panel magnifier insets:
    inset A : old growth region B_A around x_A = (-1.2, 0)   SOLID red border + circle
    inset B : new growth region B_B around x_B = ( 1.2, 0)   DASHED magenta border/circle
each linked to its on-panel circle by matching colour/linestyle connector lines.

Two candidate layouts are generated for visual comparison (choose later which one
replaces the manuscript figure -- this script does NOT overwrite switch_snapshots.pdf
unless --write_main <layout> is given):
    2x2 :  reference        weighted
           weighted+ESS     min.-var. branching        (prioritises readability)
    4x1 :  reference | weighted | weighted+ESS | min.-var. branching   (linear order;
           more cramped at \\linewidth -- insets are smaller; generated anyway)

Colour-scale policy (both layouts): full-domain panels share ONE scale (reference
global peak; over-concentration such as the weighted degeneracy spike saturates); all
B_A insets share one local scale (reference peak in B_A); all B_B insets share one local
scale (reference peak in B_B).  This makes the mechanism visible: weighted+ESS resolves
the OLD region B_A but is grainier in the NEW region B_B, while branching resolves both.
Non-reference panels are annotated with final-time mean local L2 errors E_A, E_B.

Reads ONLY saved data (no solver run):
  reference_results/switch/fields_seed<seed>.npz   (reference/weighted/weighted_ess/minvar)
  reference_results/switch/metrics.csv             (final-time mean local L2 in B_A, B_B)

Outputs (per layout L in {2x2, 4x1}):
  reference_results/switch/switch_snapshots_magnifier_<L>.pdf / .png
  reference_results/switch/plot_data/switch_snapshots_magnifier_<L>.npz
  paper/figure/switch_snapshots_magnifier_<L>.pdf        (copy, unless --no_paper_copy)

Usage:  python plot_switch.py [--layout {2x2,4x1,both}] [--seed 0] [--write_main {2x2,4x1}]
"""
import os
import sys
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common_plot_style as cps
from common_plot_style import TEXTWIDTH_IN, savefig_multi

cps.apply_style()

# switching benchmark geometry (CLAUDE.md §3.2)
CA = (-1.2, 0.0); CB = (1.2, 0.0); SIGMA = 0.25; ETA = 0.5
R_B = SIGMA * np.sqrt(-2.0 * np.log(ETA))       # half-height radius ~0.294
R_ZOOM = 0.60                                   # magnifier field-of-view half-width

PANELS = [("reference", "reference"), ("weighted", "weighted"),
          ("weighted_ess", "weighted+ESS"), ("minvar", "min.-var. branching")]
# shorter titles used in the cramped 4x1 layout (two lines for the long one)
TITLE_4x1 = {"reference": "reference", "weighted": "weighted",
             "weighted_ess": "weighted+ESS", "minvar": "min.-var.\nbranching"}

# region A = solid border, region B = dashed border; colours read on viridis (incl. its
# bright-yellow cores) and match the Sec. 5.2 snapshot figure's colormap.
A_EDGE = "#ff2d2d"; A_LS = "solid"                  # red, old region
B_EDGE = "#ff45ec"; B_LS = (0, (4, 2))             # magenta, dashed, new region
CMAP = "viridis"


def final_local_L2(metrics_csv):
    """Mean over seeds of final-time L2_rel_err_BA / _BB per method."""
    rows = list(csv.DictReader(open(metrics_csv)))
    tmax = max(float(r["t"]) for r in rows)
    out = {}
    for method in ["weighted", "weighted_ess", "minvar"]:
        sub = [r for r in rows if r["method"] == method and abs(float(r["t"]) - tmax) < 1e-9]

        def mn(c):
            v = [float(r[c]) for r in sub if r.get(c) not in (None, "", "nan")]
            return float(np.mean(v)) if v else np.nan
        out[method] = dict(BA=mn("L2_rel_err_BA"), BB=mn("L2_rel_err_BB"), n=len(sub))
    return out


def add_magnifier(ax, field, extent, center, vmax, loc, edge, ls, label,
                  frac=42, label_fs=6.0):
    """A magnifier inset of `field` over the square region around `center` (half=R_ZOOM).

    Shares the supplied local `vmax` (over-concentration saturates), draws the diagnostic
    circle inside the inset, borders the inset in `edge`/`ls`, and links it to the
    on-panel region with matching connector lines from the source down to the inset's two
    TOP corners (a clean wedge that does not cross the inset interior; auto rectangle
    hidden).  `label` is a small caption inside the inset top.
    """
    cx = center[0]
    axins = inset_axes(ax, width=f"{frac}%", height=f"{frac}%", loc=loc, borderpad=0.7)
    axins.imshow(field, origin="lower", extent=extent, vmin=0, vmax=vmax,
                 cmap=CMAP, interpolation="bilinear")
    axins.set_xlim(cx - R_ZOOM, cx + R_ZOOM); axins.set_ylim(-R_ZOOM, R_ZOOM)
    axins.set_xticks([]); axins.set_yticks([]); axins.grid(False)
    for s in axins.spines.values():
        s.set_color(edge); s.set_linewidth(1.3); s.set_linestyle(ls)
    axins.add_patch(Circle(center, R_B, fill=False, edgecolor=edge, lw=1.0, ls=ls))
    try:
        pp, _, _ = mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec=edge,
                              lw=0.8, ls=ls, alpha=0.9)
        pp.set_visible(False)
    except Exception:
        pass
    if label:
        axins.text(0.5, 0.95, label, transform=axins.transAxes, fontsize=label_fs,
                   color="white", ha="center", va="top",
                   bbox=dict(boxstyle="round,pad=0.12", fc="black", ec="none", alpha=0.5))
    return axins


def build_figure(layout, fields, extent, locL2, full_vmax, a_vmax, b_vmax):
    """Return a Figure for the given layout ('2x2' or '4x1')."""
    if layout == "2x2":
        fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 1.04 * TEXTWIDTH_IN),
                                 constrained_layout=True)
        axlist = axes.ravel()
        title_fs, frac, label_fs = 8.0, 42, 6.0
        titles = {k: t for k, t in PANELS}
        cbar_kw = dict(location="right", shrink=0.6, pad=0.012, aspect=30)
    elif layout == "4x1":
        fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH_IN, 0.42 * TEXTWIDTH_IN),
                                 constrained_layout=True)
        axlist = axes.ravel()
        title_fs, frac, label_fs = 6.5, 46, 5.2
        titles = TITLE_4x1
        cbar_kw = dict(location="bottom", shrink=0.85, pad=0.02, aspect=45)
    else:
        raise ValueError(f"unknown layout {layout!r}")

    im = None
    for ax, (key, _) in zip(axlist, PANELS):
        im = ax.imshow(fields[key], origin="lower", extent=extent, vmin=0, vmax=full_vmax,
                       cmap=CMAP, interpolation="bilinear")
        ax.add_patch(Circle(CA, R_B, fill=False, edgecolor=A_EDGE, lw=1.2, ls=A_LS))
        ax.add_patch(Circle(CB, R_B, fill=False, edgecolor=B_EDGE, lw=1.2, ls=B_LS))
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(titles[key], fontsize=title_fs, pad=3)
        if key == "reference":
            la, lb = r"$B_A$", r"$B_B$"
        else:
            la = rf"$E_A\!=\!{locL2[key]['BA']:.2f}$"
            lb = rf"$E_B\!=\!{locL2[key]['BB']:.2f}$"
        add_magnifier(ax, fields[key], extent, CA, a_vmax, "lower left",  A_EDGE, A_LS,
                      la, frac=frac, label_fs=label_fs)
        add_magnifier(ax, fields[key], extent, CB, b_vmax, "lower right", B_EDGE, B_LS,
                      lb, frac=frac, label_fs=label_fs)
    cb = fig.colorbar(im, ax=axlist.tolist() if hasattr(axlist, "tolist") else list(axlist),
                      **cbar_kw)
    cb.set_label(r"full-domain field $u(T{=}1.2,\cdot)$", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="reference_results/switch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layout", choices=["2x2", "4x1", "both"], default="both")
    ap.add_argument("--no_paper_copy", action="store_true",
                    help="do not write the paper/figure/ candidate copies")
    ap.add_argument("--write_main", choices=["2x2", "4x1"], default=None,
                    help="also overwrite paper/figure/switch_snapshots.pdf with this layout")
    args = ap.parse_args()
    repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    RD = args.results_dir if os.path.isabs(args.results_dir) else os.path.join(repo, args.results_dir)

    d = np.load(os.path.join(RD, f"fields_seed{args.seed}.npz"))
    XX, YY = d["XX"], d["YY"]
    extent = [-np.pi, np.pi, -np.pi, np.pi]
    locL2 = final_local_L2(os.path.join(RD, "metrics.csv"))
    fields = {k: d[k] for k, _ in PANELS}

    def region_mask(cx):
        return (np.abs(XX - cx) <= R_ZOOM) & (np.abs(YY) <= R_ZOOM)
    mA, mB = region_mask(CA[0]), region_mask(CB[0])
    full_vmax = float(np.max(fields["reference"]))
    a_vmax = float(np.max(fields["reference"][mA]))
    b_vmax = float(np.max(fields["reference"][mB]))
    clip_policy = ("full=ref global peak; insets=ref peak per region; vmin=0; "
                   "over-concentration saturates (not clipped away)")

    layouts = ["2x2", "4x1"] if args.layout == "both" else [args.layout]
    pd = os.path.join(RD, "plot_data"); os.makedirs(pd, exist_ok=True)
    paper_dir = os.path.join(repo, "paper", "figure")

    for layout in layouts:
        fig = build_figure(layout, fields, extent, locL2, full_vmax, a_vmax, b_vmax)
        stem = os.path.join(RD, f"switch_snapshots_magnifier_{layout}")
        savefig_multi(fig, stem, close=False)
        if not args.no_paper_copy:
            os.makedirs(paper_dir, exist_ok=True)
            fig.savefig(os.path.join(paper_dir, f"switch_snapshots_magnifier_{layout}.pdf"))
        if args.write_main == layout:
            fig.savefig(os.path.join(paper_dir, "switch_snapshots.pdf"))
            print(f"  [write_main] overwrote paper/figure/switch_snapshots.pdf with {layout}")
        plt.close(fig)
        print(f"wrote {stem}.pdf/.png")

    # ONE shared plot-data file (the field data is layout-independent; both candidates
    # regenerate from it -- avoids duplicating the ~12 MB fields per layout).
    np.savez(os.path.join(pd, "switch_snapshots_magnifier.npz"),
             XX=XX, YY=YY, extent=np.array(extent), layouts=np.array(layouts),
             reference=fields["reference"], weighted=fields["weighted"],
             weighted_ess=fields["weighted_ess"], minvar=fields["minvar"],
             cA=np.array(CA), cB=np.array(CB), sigma=SIGMA, eta=ETA, R_B=R_B, R_zoom=R_ZOOM,
             full_vmax=full_vmax, a_vmax=a_vmax, b_vmax=b_vmax, clip_policy=clip_policy,
             locL2_weighted_BA=locL2["weighted"]["BA"], locL2_weighted_BB=locL2["weighted"]["BB"],
             locL2_ess_BA=locL2["weighted_ess"]["BA"], locL2_ess_BB=locL2["weighted_ess"]["BB"],
             locL2_minvar_BA=locL2["minvar"]["BA"], locL2_minvar_BB=locL2["minvar"]["BB"])
    print("wrote plot_data/switch_snapshots_magnifier.npz (shared by both layouts)")

    print("local L2 (final, mean over seeds):",
          {k: {kk: round(vv, 3) for kk, vv in v.items() if kk != 'n'} for k, v in locL2.items()})
    if "4x1" in layouts:
        print("NOTE: the 4x1 layout is cramped at \\linewidth; insets are smaller and "
              "labels are at 5.2pt. The 2x2 layout is more readable. Both are provided "
              "for comparison; switch_snapshots.pdf is NOT changed unless --write_main is set.")


if __name__ == "__main__":
    main()
