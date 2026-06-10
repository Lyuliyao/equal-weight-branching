"""
paper_style.py — unified matplotlib style for ALL figures of the manuscript.
=============================================================================
The document is elsarticle [review] (12pt), \textwidth = 345pt = 4.773 in.
RULE: generate every figure at the EXACT physical width it occupies on the
page (fraction-of-linewidth x 4.773 in) and include it UNSCALED
(width=f\linewidth with the same f). Then the font sizes below are the true
printed sizes, identical across all figures.

Usage:
    from paper_style import apply_style, fig_size, TEXTWIDTH_IN, METHOD_COLORS
    apply_style()
    fig, ax = plt.subplots(figsize=fig_size(1.0, aspect=0.55))  # full width
"""
import matplotlib as mpl

TEXTWIDTH_IN = 345.0 / 72.27          # 4.773 in (elsarticle review, a4)

# one color per method, EVERYWHERE
METHOD_COLORS = {
    "reference": "k",
    "weighted":  "#1f77b4",   # C0 blue
    "resampled": "#9467bd",   # C4 purple
    "poisson":   "#ff7f0e",   # C1 orange
    "minvar":    "#2ca02c",   # C2 green
}
METHOD_LABELS = {
    "reference": "reference",
    "weighted":  "weighted",
    "resampled": "weighted + resample",
    "poisson":   "Poisson branching",
    "minvar":    "min.-variance branching",
}


def apply_style():
    mpl.rcParams.update({
        # fonts: serif + Computer Modern math to match the LaTeX body
        "font.family":       "serif",
        "font.serif":        ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
        "mathtext.fontset":  "cm",
        "font.size":         9,     # base (axis labels, titles)
        "axes.titlesize":    9,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        # lines / axes
        "lines.linewidth":   1.2,
        "lines.markersize":  3.5,
        "axes.linewidth":    0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "axes.grid":         True,
        "grid.linewidth":    0.4,
        "grid.alpha":        0.35,
        "legend.frameon":    False,
        # export
        "figure.dpi":        200,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype":      42,
    })


def fig_size(frac=1.0, aspect=0.62):
    """(width, height) in inches for a figure occupying frac*\\linewidth."""
    w = frac * TEXTWIDTH_IN
    return (w, w * aspect)
