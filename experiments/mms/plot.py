r"""Publication figure for the MMS verification (paper Sec. 5.1, Fig. 1).

Full-width (\linewidth) TWO-panel PDF:
  Left  : relative L2 error vs particle number N, with an OFFSET O(N^{-1/2}) guide.
  Right : time-step refinement -- the direct error vs the exact solution (blue, at the
          Monte-Carlo floor) and the lineage-coupled projected splitting difference
          (orange, first-order), with an OFFSET O(tau) guide.

Data (existing saved results; no reruns):
  reference_results/mms/errors_vs_N.csv            -- left panel.
  reference_results/mms/errors_vs_tau.csv          -- right panel, blue "total".
  reference_results/mms/errors_vs_tau_branching.csv-- right panel, orange "branching"
      = P(tau)=rho_bar*||phi(tau)|| from the Harris-coupled (tau,tau/2) experiment of
      Appendix B (slope 0.97).  That experiment is not in this repo as a standalone
      script; the three points were recovered from the manuscript figure + Appendix B
      and stored in the CSV above so this figure regenerates without a rerun.

Fonts come from paper_style.py (labels 9pt, ticks 8pt); no hard-coded tiny sizes.
Writes the PDF to reference_results/mms/ AND paper/figure/mms_convergence.pdf.
"""
import os
import sys
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments"))
from paper_style import apply_style, TEXTWIDTH_IN  # noqa: E402

apply_style()
DATA = os.path.join(REPO, "reference_results", "mms")
BLUE, ORANGE, GUIDE = "#1f77b4", "#ff7f0e", "0.35"


def load(name):
    return list(csv.DictReader(open(os.path.join(DATA, name))))


def col(rows, c):
    return np.array([float(r[c]) for r in rows])


fig, (axL, axR) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 0.5 * TEXTWIDTH_IN),
                               constrained_layout=True)

# --------------------------------------------------------------------- left: vs N
rN = load("errors_vs_N.csv")
N, muN, sdN = col(rN, "N"), col(rN, "mean_L2_rel"), col(rN, "std_L2_rel")
axL.set_xscale("log"); axL.set_yscale("log")
axL.errorbar(N, muN, yerr=sdN, color=BLUE, marker="o", capsize=2, lw=1.3, zorder=4)
# OFFSET N^{-1/2} reference guide, placed below the data (not on top of it)
c_g = 0.55 * muN[2] * np.sqrt(N[2])             # anchor 45% below the N=8000 datum
ng = np.array([3000.0, 13000.0])
axL.plot(ng, c_g * ng ** -0.5, ls="--", color=GUIDE, lw=1.1, zorder=2)
axL.text(4200, 0.78 * c_g * 4200 ** -0.5, r"$\mathcal{O}(N^{-1/2})$",
         fontsize=8.5, color=GUIDE, ha="left", va="top")
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
         fontsize=8.5, color=GUIDE, ha="left", va="bottom")
# inline curve labels (no occluding legend box)
axR.text(2.3e-3, 0.082, "total", color=BLUE, fontsize=8, va="bottom")
axR.text(0.052, 2.6e-4, "branching", color=ORANGE, fontsize=8, va="top")
axR.set_xlabel(r"$\tau$"); axR.set_ylabel(r"relative $L^2$ error")
axR.set_xlim(9e-4, 1.5e-1)
axR.set_ylim(8e-5, 1.3e-1)
for ax_, axis in ((axR, axR.xaxis), (axR, axR.yaxis)):
    axis.set_major_locator(LogLocator(base=10.0))
    axis.set_major_formatter(LogFormatterMathtext())
    axis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    axis.set_minor_formatter(NullFormatter())

out_data = os.path.join(DATA, "mms_convergence.pdf")
out_paper = os.path.join(REPO, "paper", "figure", "mms_convergence.pdf")
fig.savefig(out_data)
fig.savefig(out_paper)
print("wrote", out_data)
print("wrote", out_paper)
