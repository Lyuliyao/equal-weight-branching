r"""Publication figure for the MMS verification (paper Sec. 5.1).

Single full-width PDF, three square panels (error vs N, tau, K),
physical width 4.773 in (include at width=\linewidth)."""
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


fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 0.42 * TEXTWIDTH_IN),
                         constrained_layout=True)
for a in axes:
    a.set_box_aspect(1)

# (a) error vs N (log-log) + fitted slope
x, mu, sd = load("errors_vs_N.csv", "N")
axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[0].errorbar(x, mu, yerr=sd, color=C, marker="o", capsize=2, lw=1.2)
p = np.polyfit(np.log(x), np.log(mu), 1)
axes[0].plot(x, np.exp(np.polyval(p, np.log(x))), "k--", lw=0.8)
axes[0].text(0.06, 0.06, rf"slope $={p[0]:.2f}$", transform=axes[0].transAxes,
             fontsize=7, va="bottom")
axes[0].set_xlabel(r"$N$"); axes[0].set_ylabel(r"relative $L^2$ error")
axes[0].set_yticks([0.06, 0.1, 0.2, 0.3])
axes[0].set_yticklabels(["0.06", "0.1", "0.2", "0.3"], fontsize=7)


# (b) error vs tau (log x)
x, mu, sd = load("errors_vs_tau.csv", "tau")
axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[1].errorbar(x, mu, yerr=sd, color=C, marker="o", capsize=2, lw=1.2)
axes[1].set_xlabel(r"$\tau$")
axes[1].set_yticks([0.055, 0.06, 0.065])
axes[1].set_yticklabels(["0.055", "0.060", "0.065"], fontsize=7)


# (c) error vs K (linear x, log y)
x, mu, sd = load("errors_vs_K.csv", "K")
axes[2].set_yscale("log")
axes[2].errorbar(x, mu, yerr=sd, color=C, marker="o", capsize=2, lw=1.2)
axes[2].set_xlabel(r"$K$")
axes[2].set_yticks([0.01, 0.03, 0.09])
axes[2].set_yticklabels(["0.01", "0.03", "0.09"], fontsize=7)
axes[2].minorticks_off()

fig.savefig(os.path.join(RD, "mms_convergence.pdf"))
print("wrote", os.path.join(RD, "mms_convergence.pdf"))
