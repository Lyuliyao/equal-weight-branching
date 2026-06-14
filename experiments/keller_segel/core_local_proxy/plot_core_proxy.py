"""
§5.5 figure: core-local / reconstruction-free blow-up-proxy diagnostics.
========================================================================

Reads ONLY saved data written by analyze_core_proxy.py
(reference_results/keller_segel_ldg_pp/core_proxy_<run_id>/plot_data/radii.npz,
radius_fit.csv) plus the baseline S_curves.csv (for S_L2 vs S_core).

Panels:
  (a) reconstruction-free radius collapse R_q(t)^2 with the linear fits
      R_q^2 ~ C_q (T_* - t); the candidate concentration time T_* is the
      t-intercept (markers).  Baseline (grid) and particle (reconstruction).
  (b) candidate T_* fit-window range per series (min-max bar): O(1e-4) but
      window-SENSITIVE (1.4x-4.5x spread), so we do NOT quote a continuum
      blow-up time; LDG numerical blow-up t ~ 1.21e-4 marked (dashed line).
  (c) baseline global S_L2(t) vs core-local S_core(t): they coincide, so for the
      resolving grid the core-localized blow-up proxy equals the global one
      (core-localization is redundant once the grid resolves the core; the
      genuinely reconstruction-free signal is the radius, panel a).

Usage:
  python plot_core_proxy.py --core_dir <core_proxy_run> --baseline_dir <baseline_run>
"""
import os
import sys
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common_plot_style as cps                       # noqa: E402
from common_plot_style import TEXTWIDTH_IN, savefig_multi  # noqa: E402

cps.apply_style()
LDG_TB = 1.21e-4


def fit_line(t, R2, tlo, thi):
    m = (t >= tlo) & (t <= thi) & np.isfinite(R2) & (R2 > 0)
    slope, inter = np.polyfit(t[m], R2[m], 1)
    Tstar = -inter / slope
    return slope, inter, Tstar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core_dir", required=True)
    ap.add_argument("--baseline_dir", required=True)
    args = ap.parse_args()
    d = np.load(os.path.join(args.core_dir, "plot_data", "radii.npz"))
    fit_rows = list(csv.DictReader(open(os.path.join(args.core_dir, "radius_fit.csv"))))
    sens_rows = list(csv.DictReader(open(os.path.join(args.core_dir, "sensitivity.csv"))))

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 0.34 * TEXTWIDTH_IN))

    # ---- (a) radius collapse + fits ----
    ax = axes[0]
    series = [("FVM $n{=}512$", d["t_b512"], d["R05_b512"], d["R08_b512"], "#08519c"),
              ("particle ref.", d["t_pref"], d["R05_pref"], d["R08_pref"], "tab:red")]
    for name, t, R05, R08, c in series:
        for R, ls, q in [(R05, "-", "0.5"), (R08, "--", "0.8")]:
            ax.plot(t, R ** 2, ls, color=c, lw=1.0, ms=2,
                    label=f"{name} $R_{{{q}}}$")
            sl, ic, Ts = fit_line(t, R ** 2, 3e-5, 2e-4)
            tt = np.array([3e-5, Ts])
            ax.plot(tt, sl * tt + ic, ":", color=c, lw=0.7)
            ax.plot([Ts], [0], "v", color=c, ms=4)
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$R_q(t)^2$")
    ax.set_title(r"radius collapse", fontsize=6.8)
    ax.legend(fontsize=4.4, loc="upper right")

    # ---- (b) candidate T*: window-SENSITIVITY (honest negative) ----
    # T* is fit over several windows; the min-max bar shows it is NOT a stable
    # estimate (1.4x-4.5x spread), so we do not quote a continuum blow-up time.
    ax = axes[1]
    cmap = {"FVM n=256": "#4292c6", "FVM n=512": "#08519c",
            "particle base": "tab:orange", "particle refined": "tab:red"}
    order = ["FVM n=256", "FVM n=512", "particle base", "particle refined"]
    xtick_lab = {"FVM n=256": "FVM\n256", "FVM n=512": "FVM\n512",
                 "particle base": "part.\nbase", "particle refined": "part.\nref."}
    from collections import defaultdict
    spread = defaultdict(list)
    for r in sens_rows:
        try:
            v = float(r["Tstar"])
        except ValueError:
            continue
        if v == v and v > 0:                       # drop NaN / non-positive
            spread[r["series"]].append(v)
    for xi, s in enumerate(order):
        vs = spread.get(s, [])
        if not vs:
            continue
        ax.vlines(xi, min(vs), max(vs), color=cmap[s], lw=4, alpha=0.35)
        ax.scatter([xi] * len(vs), vs, c=cmap[s], s=10, zorder=3,
                   edgecolor="k", linewidth=0.2)
    ax.axhline(LDG_TB, color="0.4", ls="--", lw=0.8)
    ax.text(0.02, LDG_TB * 1.04, r"LDG $t\!\approx\!1.21{\times}10^{-4}$",
            transform=ax.get_yaxis_transform(), fontsize=4.6, color="0.4", va="bottom")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([xtick_lab[s] for s in order], fontsize=4.8)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylabel(r"candidate $T_*$")
    ax.set_title(r"$T_*$ window-sensitive", fontsize=6.6)
    ax.set_ylim(0, 3.0e-4)

    # ---- (c) baseline global S_L2 vs core S_core ----
    ax = axes[2]
    rows = list(csv.DictReader(open(os.path.join(args.baseline_dir, "n512", "S_curves.csv"))))
    t = np.array([float(r["t"]) for r in rows])
    SL2 = np.array([float(r["S_L2"]) for r in rows])
    Sco = np.array([float(r["S_core"]) for r in rows])
    ax.plot(t, SL2, "-", color="#08519c", lw=1.4, label=r"global $S_{L^2}$")
    ax.plot(t, Sco, "--", color="tab:green", lw=1.0, label=r"core $S_{\rm core}$")
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$S(t)$ (FVM $n{=}512$)")
    ax.set_yscale("log"); ax.legend(fontsize=5.0, loc="lower right")
    ax.set_title(r"core $\approx$ global", fontsize=6.8)

    fig.tight_layout(pad=0.4)
    out = os.path.join(args.core_dir, "figures", "ks_core_proxy")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    savefig_multi(fig, out, close=True)
    print("wrote", out + ".pdf/.png")


if __name__ == "__main__":
    main()
