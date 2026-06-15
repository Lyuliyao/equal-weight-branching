"""
Plot the blob-residual solver-field sweep (corrected dual-CFL).
==============================================================

Two panels (reads diag_*.csv + blob_sweep_compare.json only; no solver rerun):

  (a) Stability tradeoff: blob abort time (mean+/-std) and mean REAL solver-CFL
      (drift_cfl_solver_field) vs the blob bandwidth c_h, with current_fourier and
      spectral-Kl=24 shown as reference bands.  Shows the smoother blob (larger c_h)
      recovers/exceeds the global-K abort time with a smaller real solver drift.
  (b) Solver-vs-Fourier CFL per config: grouped bars of max drift_cfl_solver_field
      (the field that drives/aborts) vs max drift_cfl_fourier_diag (the single-K
      diagnostic).  current_fourier: equal (same field); spectral: solver > diag;
      blob: solver < diag (tighter core, smoother drift) -- this is exactly why the
      dual-CFL logging is needed and why the prior single-column claim was wrong.

Saves plot_data/figure_blob_sweep.npz + figures/figure_blob_sweep.{pdf,png}.

Usage:  python plot_blob_sweep.py --sdir <sf_blob_run>
"""
import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

import sys
_EXP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from common_plot_style import apply_style, savefig_multi  # noqa: E402

BLOBS = [("blob_fracL_ch0.04", 0.04), ("blob_fracL_ch0.06", 0.06),
         ("blob_fracL_ch0.09", 0.09)]
BARS = [("current_fourier", "global $K$=10"), ("spectral_taper0.25", "spectral $K_l$=24"),
        ("blob_fracL_ch0.04", "blob 0.04"), ("blob_fracL_ch0.06", "blob 0.06"),
        ("blob_fracL_ch0.09", "blob 0.09")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdir", required=True)
    args = ap.parse_args()
    apply_style()
    summ = json.load(open(os.path.join(args.sdir, "blob_sweep_compare.json")))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # ---- panel (a): blob stability tradeoff vs c_h ----
    ch = np.array([c for _, c in BLOBS])
    ab = np.array([summ[n]["final_t_mean"] for n, _ in BLOBS]) * 1e4
    abs_ = np.array([summ[n]["final_t_std"] for n, _ in BLOBS]) * 1e4
    cS = np.array([summ[n]["cflS_max_mean"] for n, _ in BLOBS])
    c_ab = "#1f77b4"; c_cfl = "#d62728"
    axa.errorbar(ch, ab, yerr=abs_, marker="o", color=c_ab, capsize=2, lw=1.3,
                 label="blob abort $t$")
    cf, sp = summ["current_fourier"], summ["spectral_taper0.25"]
    axa.axhline(cf["final_t_mean"] * 1e4, color="k", ls="--", lw=1.0)
    axa.text(0.045, cf["final_t_mean"] * 1e4 + 0.01, "global-$K$", fontsize=6, color="k")
    axa.axhline(sp["final_t_mean"] * 1e4, color="0.45", ls=":", lw=1.0)
    axa.text(0.045, sp["final_t_mean"] * 1e4 - 0.05, "spectral $K_l$=24", fontsize=6, color="0.45")
    axa.set_xlabel(r"blob bandwidth $c_h$ (=$h/L$; larger = smoother)")
    axa.set_ylabel(r"abort time $t\,[\times 10^{-4}]$", color=c_ab)
    axa.tick_params(axis="y", labelcolor=c_ab)
    axa.set_title("(a) blob stability vs bandwidth")
    axc = axa.twinx()
    axc.plot(ch, cS, marker="s", color=c_cfl, lw=1.2, ls=":")
    axc.axhline(cf["cflS_max_mean"], color=c_cfl, ls="--", lw=0.9, alpha=0.5)
    axc.set_ylabel(r"mean max solver CFL", color=c_cfl)
    axc.tick_params(axis="y", labelcolor=c_cfl)

    # ---- panel (b): real solver CFL vs Fourier-diag CFL per config ----
    labels = [lab for _, lab in BARS]
    cflS = [summ[n]["cflS_max_mean"] for n, _ in BARS]
    cflF = [summ[n]["cflF_max_mean"] for n, _ in BARS]
    x = np.arange(len(BARS)); w = 0.38
    axb.bar(x - w / 2, cflS, w, label="solver field (drives/aborts)", color="#2ca02c")
    axb.bar(x + w / 2, cflF, w, label="Fourier diag (single-$K$)", color="#9467bd")
    axb.axhline(5.0, color="k", ls="--", lw=0.8)
    axb.text(-0.4, 5.08, "cfl_abort=5", fontsize=6, ha="left", va="bottom")
    axb.set_xticks(x); axb.set_xticklabels(labels, rotation=30, ha="right", fontsize=6)
    axb.set_ylabel("max drift CFL"); axb.set_ylim(0, 6.2)
    axb.set_title("(b) real solver CFL vs Fourier diagnostic")
    axb.legend(fontsize=5.5, loc="upper center", ncol=1, framealpha=0.9)

    fig.tight_layout()
    figdir = os.path.join(args.sdir, "figures"); pdir = os.path.join(args.sdir, "plot_data")
    os.makedirs(figdir, exist_ok=True); os.makedirs(pdir, exist_ok=True)
    savefig_multi(fig, os.path.join(figdir, "figure_blob_sweep"))
    np.savez(os.path.join(pdir, "figure_blob_sweep.npz"),
             ch=ch, abort_mean=ab, abort_std=abs_, cflS_blob=cS,
             cf_abort=cf["final_t_mean"] * 1e4, sp_abort=sp["final_t_mean"] * 1e4,
             cf_cflS=cf["cflS_max_mean"],
             bar_labels=np.array(labels), bar_cflS=np.array(cflS), bar_cflF=np.array(cflF))
    print(f"wrote {figdir}/figure_blob_sweep.pdf/.png + plot_data/figure_blob_sweep.npz")


if __name__ == "__main__":
    main()
