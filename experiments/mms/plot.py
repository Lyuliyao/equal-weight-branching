"""Publication figure for the MMS verification (paper Sec. 5.1).

Single full-width PDF with three panels (error vs N, tau, K), physical width
4.773 in (include at width=\linewidth)."""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paper_style import apply_style, TEXTWIDTH_IN

apply_style()
RD = "results/mms"
C = "#1f77b4"


def load(name, xcol):
    rows = list(csv.DictReader(open(os.path.join(RD, name))))
    x = np.array([float(r[xcol]) for r in rows])
    mu = np.array([float(r["mean_L2_rel"]) for r in rows])
    sd = np.array([float(r["std_L2_rel"]) for r in rows])
    return x, mu, sd


def panel(ax, x, mu, sd, xlabel, logx=True, fit_slope=False, ref_slope=None):
    if logx:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.errorbar(x, mu, yerr=sd, color=C, marker="o", capsize=2, lw=1.2)
    if fit_slope:
        p = np.polyfit(np.log(x), np.log(mu), 1)
        ax.plot(x, np.exp(np.polyval(p, np.log(x))), "k--", lw=0.8,
                label=rf"slope $={p[0]:.2f}$")
    if ref_slope is not None:
        s, lab = ref_slope
        yref = mu[0] * (x / x[0]) ** s
        ax.plot(x, yref, color="0.45", ls=":", lw=0.9, label=lab)
    ax.set_xlabel(xlabel)
    ax.legend()


fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 0.34 * TEXTWIDTH_IN),
                         constrained_layout=True)
x, mu, sd = load("errors_vs_N.csv", "N")
panel(axes[0], x, mu, sd, r"$N$", fit_slope=True, ref_slope=(-0.5, r"$N^{-1/2}$"))
axes[0].set_ylabel(r"relative $L^2$ error")
x, mu, sd = load("errors_vs_tau.csv", "tau")
panel(axes[1], x, mu, sd, r"$\tau$")
x, mu, sd = load("errors_vs_K.csv", "K")
panel(axes[2], x, mu, sd, r"$K$", logx=False)
fig.savefig(os.path.join(RD, "mms_convergence.pdf"))
print("wrote", os.path.join(RD, "mms_convergence.pdf"))
