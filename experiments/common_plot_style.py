"""common_plot_style.py — publication-figure utilities for the revision experiments.
=====================================================================================

This is the single plotting utility requested in the revision plan (CLAUDE.md §5).
It is a thin SUPERSET of the existing `paper_style.py`: it reuses the exact
physical-width convention (`TEXTWIDTH_IN`, `fig_size`) and the per-method color
map (`METHOD_COLORS`, `METHOD_LABELS`) so that the new figures match the existing
manuscript figures, and adds:

  * `apply_style()`   — the publication rcParams (serif + CM math, in-ticks,
                        vector PDF fonts, 600-dpi PNG export);
  * `savefig_multi()` — save the SAME figure to .pdf (+ .png, optional .svg) so
                        every figure ships in both a vector and a raster form;
  * `add_zoom_inset()`— a magnifying inset on an imshow panel, with the source
                        rectangle drawn on the full-domain panel (CLAUDE.md §5);
  * `ERR_CMAP`, `err_norm()` — fixed 0–50% scale for per-island mass-error
                        heatmaps with a visible marker at the 20% threshold.

Nothing here changes the dynamics or the data; it only standardizes figures.
"""
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Reuse the manuscript-wide conventions (same dir).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from paper_style import (  # noqa: E402
    TEXTWIDTH_IN, fig_size, METHOD_COLORS, METHOD_LABELS,
)


def apply_style():
    """Publication rcParams (CLAUDE.md §5, harmonized with paper_style serif/CM)."""
    mpl.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
        "mathtext.fontset":  "cm",
        "font.size":         8,
        "axes.labelsize":    8,
        "axes.titlesize":    8,
        "legend.fontsize":   7,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "figure.dpi":        150,
        "savefig.dpi":       600,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "axes.linewidth":    0.6,
        "lines.linewidth":   1.2,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon":    False,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.02,
    })


def savefig_multi(fig, stem, formats=("pdf", "png"), close=True):
    """Save `fig` to `<stem>.<fmt>` for each fmt. Returns the list of paths.

    `stem` may include directories (created if missing).  PNG uses the
    600-dpi savefig default; PDF is vector.  Pass close=False to keep the figure.
    """
    d = os.path.dirname(stem)
    if d:
        os.makedirs(d, exist_ok=True)
    paths = []
    for fmt in formats:
        p = f"{stem}.{fmt}"
        fig.savefig(p)
        paths.append(p)
    if close:
        plt.close(fig)
    return paths


def add_zoom_inset(ax, field, extent, zoom_box, *, loc="upper right",
                   zoom_frac=0.42, vmin=None, vmax=None, cmap="viridis",
                   edge="white", mark_on_parent=True):
    """Add a magnifying inset of `field` over `zoom_box` to imshow panel `ax`.

    field    : 2D array shown with origin='lower' on `extent`=[x0,x1,y0,y1].
    zoom_box : (xa, xb, ya, yb) physical sub-rectangle to magnify.
    Draws the source rectangle on the parent panel and (optionally) connector
    lines.  Returns the inset Axes.  The inset shares the parent color scale, so
    peak loss is visible rather than auto-rescaled away.
    """
    xa, xb, ya, yb = zoom_box
    axins = inset_axes(ax, width=f"{int(zoom_frac*100)}%",
                       height=f"{int(zoom_frac*100)}%", loc=loc, borderpad=0.4)
    axins.imshow(field, origin="lower", extent=extent, vmin=vmin, vmax=vmax,
                 cmap=cmap, interpolation="nearest")
    axins.set_xlim(xa, xb)
    axins.set_ylim(ya, yb)
    axins.set_xticks([]); axins.set_yticks([])
    for s in axins.spines.values():
        s.set_color(edge); s.set_linewidth(0.8)
    if mark_on_parent:
        rect = Rectangle((xa, ya), xb - xa, yb - ya, fill=False,
                         edgecolor=edge, linewidth=0.8)
        ax.add_patch(rect)
        try:
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none",
                       ec=edge, lw=0.5, alpha=0.7)
        except Exception:
            pass
    return axins


# Per-island mass-error heatmaps: fixed 0–50% scale with a 20% marker.
ERR_CMAP = plt.get_cmap("inferno")
ERR_VMIN, ERR_VMAX, ERR_THRESH = 0.0, 0.50, 0.20


def err_norm():
    """Normalizer for E_m heatmaps on the fixed [0, 50%] scale."""
    return mpl.colors.Normalize(vmin=ERR_VMIN, vmax=ERR_VMAX)


def annotate_threshold_contour(ax, Em_grid, extent, thresh=ERR_THRESH,
                               color="cyan"):
    """Overlay the E_m = `thresh` contour (the table's 20% line) on a heatmap."""
    try:
        ax.contour(np.asarray(Em_grid), levels=[thresh], colors=color,
                   linewidths=0.8, extent=extent, origin="upper")
    except Exception:
        pass
