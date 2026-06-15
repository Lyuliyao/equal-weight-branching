r"""Publication figure for the MMS verification (paper Sec. 5.1, Fig. 1).

TWO-panel PDF generated at physical width 0.8*TEXTWIDTH_IN, because the manuscript
includes it at \includegraphics[width=0.8\linewidth]{...}.  Sizing the PDF to 0.8 the
text width means LaTeX inserts it 1:1 and does NOT scale the fonts down, so the labels
stay readable.
  Left  : relative L2 error vs particle number N, with an OFFSET O(N^{-1/2}) guide.
  Right : time-step refinement -- the direct error vs the exact solution (blue, at the
          Monte-Carlo floor) and the lineage-coupled projected splitting difference
          (orange, first-order), with an OFFSET O(tau) guide.

Data (existing saved results; no reruns):
  reference_results/mms/errors_vs_N.csv            -- left panel.
  reference_results/mms/errors_vs_tau.csv          -- right panel, blue "total".
  reference_results/mms/errors_vs_tau_branching.csv-- right panel, orange "branching"
      = P(tau)=rho_bar*||phi(tau)|| from the Harris-coupled (tau,tau/2) experiment of
      Appendix B (slope 0.97); the three points were recovered from the manuscript
      figure + Appendix B and stored in that CSV so this figure regenerates w/o a rerun.

Fonts: paper_style.py defaults, locally bumped here (labels 10pt, ticks 9pt) for
readability at 0.8 width; no global style edits, no hard-coded tiny sizes.
Writes the PDF to reference_results/mms/ AND paper/figure/mms_convergence.pdf.
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
# local readability bump (printed at 0.8 linewidth, 1:1, so these are the final sizes)
mpl.rcParams.update({"axes.labelsize": 10, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "legend.fontsize": 9})
ANN = 9                                            # annotation / guide-label size
DATA = os.path.join(REPO, "reference_results", "mms")
BLUE, ORANGE, GUIDE = "#1f77b4", "#ff7f0e", "0.35"
FIGW = 0.8 * TEXTWIDTH_IN


def load(name):
    return list(csv.DictReader(open(os.path.join(DATA, name))))


def col(rows, c):
    return np.array([float(r[c]) for r in rows])


fig, (axL, axR) = plt.subplots(1, 2, figsize=(FIGW, 0.56 * FIGW),
                               constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.06, wspace=0.04)

# --------------------------------------------------------------------- left: vs N
rN = load("errors_vs_N.csv")
N, muN, sdN = col(rN, "N"), col(rN, "mean_L2_rel"), col(rN, "std_L2_rel")
axL.set_xscale("log"); axL.set_yscale("log")
axL.errorbar(N, muN, yerr=sdN, color=BLUE, marker="o", capsize=2, lw=1.3, zorder=4)
# OFFSET N^{-1/2} reference guide, placed below the data (not on top of it)
c_g = 0.55 * muN[2] * np.sqrt(N[2])             # anchor 45% below the N=8000 datum
ng = np.array([3000.0, 13000.0])
axL.plot(ng, c_g * ng ** -0.5, ls="--", color=GUIDE, lw=1.1, zorder=2)
axL.text(3150, 0.066, r"$\mathcal{O}(N^{-1/2})$",          # lower-left empty space
         fontsize=ANN, color=GUIDE, ha="left", va="center")
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
axR.errorbar(tau, muT, yerr=sdT, color=BLUE, marker="o", capsize=2, lw=1.3, zorder=4)
axR.errorbar(bt, bP, yerr=bsd, color=ORANGE, marker="s", capsize=2, lw=1.3, zorder=4)
# OFFSET O(tau) reference guide, in the empty band above the orange curve
slope_o = bP[-1] / bt[-1]                        # orange ~ slope_o * tau
c_t = 2.6 * slope_o
tg = np.array([0.018, 0.13])
axR.plot(tg, c_t * tg, ls="--", color=GUIDE, lw=1.1, zorder=2)
axR.text(0.05, 1.55 * c_t * 0.05, r"$\mathcal{O}(\tau)$",
         fontsize=ANN, color=GUIDE, ha="left", va="bottom")
# inline curve labels (no occluding legend box)
axR.text(2.3e-3, 0.082, "total", color=BLUE, fontsize=ANN, va="bottom")
axR.text(0.052, 2.6e-4, "branching", color=ORANGE, fontsize=ANN, va="top")
axR.set_xlabel(r"$\tau$"); axR.set_ylabel(r"relative $L^2$ error")
axR.set_xlim(9e-4, 1.5e-1)
axR.set_ylim(8e-5, 1.3e-1)
for axis in (axR.xaxis, axR.yaxis):
    axis.set_major_locator(LogLocator(base=10.0))
    axis.set_major_formatter(LogFormatterMathtext())
    axis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    axis.set_minor_formatter(NullFormatter())

out_data = os.path.join(DATA, "mms_convergence.pdf")
out_paper = os.path.join(REPO, "paper", "figure", "mms_convergence.pdf")
fig.savefig(out_data)
fig.savefig(out_paper)
print(f"wrote {out_data}  (physical width {FIGW:.3f} in = 0.8*TEXTWIDTH)")
print(f"wrote {out_paper}")
