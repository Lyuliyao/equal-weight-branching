"""
Plotting for Experiment 2 (MMS verification).
Reads results/mms/{errors_vs_N.csv, errors_vs_tau.csv, errors_vs_K.csv} and
writes publication-style PDFs into the SAME results dir:
  - log-log error vs N with fitted slope
  - error vs tau (log-log)
  - error vs K (semilog-y)

Run:  python plot.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = "results/mms"


def load(name):
    rows = list(csv.DictReader(open(os.path.join(RD, name))))
    for r in rows:
        for k in ["value", "N", "tau", "K", "mean_L2_rel", "std_L2_rel"]:
            r[k] = float(r[k])
    return rows


def fit_slope(x, y):
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    sol, *_ = np.linalg.lstsq(A, ly, rcond=None)
    return float(sol[0]), float(sol[1])


def plot_vs_N():
    rows = load("errors_vs_N.csv")
    N = np.array([r["N"] for r in rows])
    y = np.array([r["mean_L2_rel"] for r in rows])
    yerr = np.array([r["std_L2_rel"] for r in rows])
    slope, inter = fit_slope(N, y)
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.errorbar(N, y, yerr=yerr, fmt="o", color="tab:blue", capsize=3, label="measured")
    Nfit = np.array([N.min(), N.max()])
    ax.plot(Nfit, np.exp(inter) * Nfit ** slope, "--", color="k",
            label=f"fit slope = {slope:.3f}")
    ax.plot(Nfit, y[0] * (Nfit / N[0]) ** (-0.5), ":", color="tab:gray",
            label="reference -1/2")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N (particles)"); ax.set_ylabel("mean L2 relative error")
    ax.set_title("MMS error vs N")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.savefig(os.path.join(RD, "errors_vs_N.pdf"))
    plt.close(fig)


def plot_vs_tau():
    rows = load("errors_vs_tau.csv")
    tau = np.array([r["tau"] for r in rows])
    y = np.array([r["mean_L2_rel"] for r in rows])
    yerr = np.array([r["std_L2_rel"] for r in rows])
    slope, inter = fit_slope(tau, y)
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.errorbar(tau, y, yerr=yerr, fmt="s", color="tab:green", capsize=3, label="measured")
    tf = np.array([tau.min(), tau.max()])
    ax.plot(tf, np.exp(inter) * tf ** slope, "--", color="k",
            label=f"fit slope = {slope:.3f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("tau (time step)"); ax.set_ylabel("mean L2 relative error")
    ax.set_title("MMS error vs tau")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.savefig(os.path.join(RD, "errors_vs_tau.pdf"))
    plt.close(fig)


def plot_vs_K():
    rows = load("errors_vs_K.csv")
    K = np.array([r["K"] for r in rows])
    y = np.array([r["mean_L2_rel"] for r in rows])
    yerr = np.array([r["std_L2_rel"] for r in rows])
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.errorbar(K, y, yerr=yerr, fmt="^", color="tab:purple", capsize=3)
    ax.set_yscale("log")
    ax.set_xlabel("K (Fourier modes per direction)"); ax.set_ylabel("mean L2 relative error")
    ax.set_title("MMS error vs K")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(os.path.join(RD, "errors_vs_K.pdf"))
    plt.close(fig)


def main():
    plot_vs_N()
    plot_vs_tau()
    plot_vs_K()
    print("wrote PDFs to", RD)


if __name__ == "__main__":
    main()
