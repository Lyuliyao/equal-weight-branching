"""
Figure for the q_window sensitivity sweep (Experiment B).
=========================================================
Reads qwindow_summary.csv; two panels:
  (a) particle T_core (seed-bootstrap median + CI) vs q_window for N=8e4,3.2e5, with the
      LDG reference line -> does a more core-local window move T_core toward LDG?
  (b) outside_v_frac and R_0.2/(L/K) vs q_window (N=3.2e5) -> the cost/benefit: a smaller
      window resolves the inner core (R_0.2/h_eff up) but may shed chemical mass (out_v up).

Usage:  python plot_qwindow_sensitivity.py --sweep_dir <qwindow_run> --ldg_T 1.215e-4
"""
import os
import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt

import sys
_EXP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from common_plot_style import apply_style, savefig_multi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--ldg_T", type=float, default=1.215e-4)
    args = ap.parse_args()
    apply_style()
    rows = list(csv.DictReader(open(os.path.join(args.sweep_dir, "qwindow_summary.csv"))))

    def series(N, col):
        out = []
        for r in rows:
            if r["N"] == str(N):
                try:
                    out.append((float(r["q_window"]), float(r[col])))
                except ValueError:
                    out.append((float(r["q_window"]), np.nan))
        return sorted(out)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for N, col in (("80000", "#1f77b4"), ("320000", "#2ca02c")):
        s = series(N, "T_core_boot"); lo = series(N, "boot_p5"); hi = series(N, "boot_p95")
        if not s:
            continue
        x = [a for a, _ in s]; y = np.array([b for _, b in s]) * 1e4
        ylo = np.array([b for _, b in lo]) * 1e4; yhi = np.array([b for _, b in hi]) * 1e4
        axa.errorbar(x, y, yerr=[y - ylo, yhi - y], marker="o", color=col, capsize=2,
                     lw=1.1, label=f"N={N}")
    axa.axhline(args.ldg_T * 1e4, color="0.5", ls="-.", lw=0.9)
    axa.text(0.5, args.ldg_T * 1e4 + 0.02, "LDG", fontsize=6, color="0.5")
    axa.set_xlabel(r"$q_{\rm window}$"); axa.set_ylabel(r"$T_{core}$ (boot) $[\times10^{-4}]$")
    axa.set_title("(a) particle $T_{core}$ vs window"); axa.legend(fontsize=6)

    s_ov = series("320000", "outside_v_frac"); s_rr = series("320000", "R02_over_heff")
    x = [a for a, _ in s_ov]
    axb.plot(x, [b for _, b in s_ov], marker="s", color="#d62728", lw=1.1, label="outside_v_frac")
    axb.set_xlabel(r"$q_{\rm window}$"); axb.set_ylabel("outside_v_frac", color="#d62728")
    axb.tick_params(axis="y", labelcolor="#d62728")
    axc = axb.twinx()
    axc.plot(x, [b for _, b in s_rr], marker="^", color="#1f77b4", lw=1.1, ls=":")
    axc.set_ylabel(r"$R_{0.2}/(L/K)$", color="#1f77b4"); axc.tick_params(axis="y", labelcolor="#1f77b4")
    axb.set_title("(b) cost/benefit (N=3.2e5)")

    fig.tight_layout()
    figdir = os.path.join(args.sweep_dir, "figures"); os.makedirs(figdir, exist_ok=True)
    savefig_multi(fig, os.path.join(figdir, "qwindow_sensitivity"))
    print(f"wrote {figdir}/qwindow_sensitivity.pdf/.png")


if __name__ == "__main__":
    main()
