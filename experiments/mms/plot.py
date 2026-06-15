r"""Publication figure for the MMS verification (paper Sec. 5.1, Fig. 1).

TWO square panels, PDF sized to physical width 0.8*TEXTWIDTH_IN to match the manuscript
inclusion \includegraphics[width=0.8\linewidth]{...} (so LaTeX inserts it ~1:1 and does
not scale the fonts).
  Left  : relative L2 error vs particle number N, with an OFFSET O(N^{-1/2}) guide.
  Right : time-step refinement -- the direct error vs the exact solution (blue, at the
          Monte-Carlo floor) and the lineage-coupled projected splitting difference
          (orange, first-order), with an OFFSET O(tau) guide; compact in-axes legend.

Data (existing saved results; no reruns):
  reference_results/mms/errors_vs_N.csv            -- left panel.
  reference_results/mms/errors_vs_tau.csv          -- right panel, blue "total".
  reference_results/mms/errors_vs_tau_branching.csv-- right panel, orange "branching"
      = P(tau)=rho_bar*||phi(tau)|| from the Harris-coupled (tau,tau/2) experiment of
      Appendix B (slope 0.97); the three points were recovered from the manuscript
      figure + Appendix B and stored in that CSV so this figure regenerates w/o a rerun.

Fonts kept small for a 0.8\linewidth figure (labels 8.5pt, ticks 8pt, legend/annot 8pt;
no global paper_style edits, no hard-coded tiny font sizes).  Writes the PDF to
reference_results/mms/ AND paper/figure/mms_convergence.pdf.
"""
import os
import sys
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments"))
from paper_style import apply_style, TEXTWIDTH_IN  # noqa: E402

apply_style()
# small final sizes for a 0.8\linewidth figure (slightly below the ~10pt body text)
mpl.rcParams.update({"axes.labelsize": 8.5, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8})
ANN = 8                                            # guide-label size
DATA = os.path.join(REPO, "reference_results", "mms")
BLUE, ORANGE, GUIDE = "#1f77b4", "#ff7f0e", "0.35"
FIGW = 0.8 * TEXTWIDTH_IN


def load(name):
    return list(csv.DictReader(open(os.path.join(DATA, name))))


def col(rows, c):
    return np.array([float(r[c]) for r in rows])


# Square axes boxes (set_box_aspect) + manual margins (robust with box_aspect; avoids
# constrained_layout fighting the fixed aspect).  Figure height leaves room for the
# square boxes + the x-labels/ticks.
fig, (axL, axR) = plt.subplots(1, 2, figsize=(FIGW, 0.60 * FIGW))
for ax in (axL, axR):
    ax.set_box_aspect(1)
fig.subplots_adjust(left=0.115, right=0.985, bottom=0.165, top=0.97, wspace=0.42)

# --------------------------------------------------------------------- left: vs N
rN = load("errors_vs_N.csv")
N, muN, sdN = col(rN, "N"), col(rN, "mean_L2_rel"), col(rN, "std_L2_rel")
axL.set_xscale("log"); axL.set_yscale("log")
axL.errorbar(N, muN, yerr=sdN, color=BLUE, marker="o", capsize=2, lw=1.2, zorder=4)
# OFFSET N^{-1/2} guide, below the data; small label in the lower-left empty space
c_g = 0.55 * muN[2] * np.sqrt(N[2])
ng = np.array([3000.0, 13000.0])
axL.plot(ng, c_g * ng ** -0.5, ls="--", color=GUIDE, lw=1.0, zorder=2)
axL.text(3150, 0.066, r"$\mathcal{O}(N^{-1/2})$", fontsize=ANN, color=GUIDE,
         ha="left", va="center")
axL.set_xlabel(r"$N$"); axL.set_ylabel(r"relative $L^2$ error")
axL.set_xlim(9e2, 1.1e5)
axL.set_ylim(0.05, 0.38)
axL.set_yticks([0.06, 0.1, 0.2, 0.3])
axL.set_yticklabels([r"$0.06$", r"$0.1$", r"$0.2$", r"$0.3$"])
axL.yaxis.set_minor_formatter(NullFormatter())

# ------------------------------------------------------------------- right: vs tau
rT = load("errors_vs_tau.csv")
tau, muT, sdT = col(rT, "tau"), col(rT, "mean_L2_rel"), col(rT, "std_L2_rel")
rB = load("errors_vs_tau_branching.csv")
bt, bP, bsd = col(rB, "tau"), col(rB, "P"), col(rB, "std_P")
axR.set_xscale("log"); axR.set_yscale("log")
axR.errorbar(tau, muT, yerr=sdT, color=BLUE, marker="o", capsize=2, lw=1.2,
             zorder=4, label="total")
axR.errorbar(bt, bP, yerr=bsd, color=ORANGE, marker="s", capsize=2, lw=1.2,
             zorder=4, label="branching")
# OFFSET O(tau) guide in the empty band above the orange curve; small inside label
slope_o = bP[-1] / bt[-1]
c_t = 2.6 * slope_o
tg = np.array([0.02, 0.12])
axR.plot(tg, c_t * tg, ls="--", color=GUIDE, lw=1.0, zorder=2)
axR.text(0.038, 1.7 * c_t * 0.038, r"$\mathcal{O}(\tau)$", fontsize=ANN, color=GUIDE,
         ha="left", va="bottom")
axR.set_xlabel(r"$\tau$"); axR.set_ylabel(r"relative $L^2$ error")
axR.set_xlim(9e-4, 1.5e-1)
axR.set_ylim(8e-5, 1.3e-1)
for axis in (axR.xaxis, axR.yaxis):
    axis.set_major_locator(LogLocator(base=10.0))
    axis.set_major_formatter(LogFormatterMathtext())
    axis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    axis.set_minor_formatter(NullFormatter())
# compact legend fully inside the axes (lower-left empty region; clear of orange/guide)
axR.legend(loc="lower left", fontsize=8, frameon=False, handlelength=1.4,
           borderaxespad=0.5, labelspacing=0.3)

out_data = os.path.join(DATA, "mms_convergence.pdf")
out_paper = os.path.join(REPO, "paper", "figure", "mms_convergence.pdf")
# save at the exact figure size (no tight crop) so the PDF width is 0.8*TEXTWIDTH
fig.savefig(out_data, bbox_inches=None)
fig.savefig(out_paper, bbox_inches=None)
print(f"wrote {out_data}  (physical width {FIGW:.3f} in = 0.8*TEXTWIDTH)")
print(f"wrote {out_paper}")
