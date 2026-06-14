"""
Plot the v_hi-taper Scenario-C tradeoff for the two-level spectral-residual drift.
=================================================================================

Two panels (reads diag_*.csv + taper_sweep_compare.json only; no solver rerun):

  (a) Q1 tradeoff curve: mean drift-CFL abort time and mean max drift_cfl vs the
      local-operator smoothness (taper_hi).  current_fourier shown as a reference
      band.  Shows that smoothing the high-Kl core residual (taper_hi 0.5->0.25)
      recovers the abort time of the plain global-K drift while keeping a smoother
      drift.
  (b) drift_cfl(t) traces for one representative seed per config, making the
      "high-mode noise spike -> early abort" mechanism visible.

Saves plot_data/figure_taper_sweep.npz + figures/figure_taper_sweep.{pdf,png}.

Usage:  python plot_taper_sweep.py --sdir <sf_taper_run> [--seed 1]
"""
import os
import csv
import glob
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

import sys
_EXP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from common_plot_style import apply_style, savefig_multi  # noqa: E402

CONFIGS = [
    ("current_fourier",     "current_fourier",      None),
    ("two_level_taper0.5",  "two_level_taper0.5",   0.50),
    ("two_level_taper0.35", "two_level_taper0.35",  0.35),
    ("two_level_taper0.25", "two_level_taper0.25",  0.25),
]


def load_diag(d):
    cs = glob.glob(os.path.join(d, "diag_*.csv"))
    if not cs:
        return None
    rows = list(csv.DictReader(open(cs[0])))
    t = np.array([float(r["t"]) for r in rows])
    cfl = np.array([float(r["drift_cfl"]) for r in rows])
    sl2 = np.array([float(r["S_L2_u"]) for r in rows])
    return t, cfl, sl2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdir", required=True)
    ap.add_argument("--seed", type=int, default=1,
                    help="representative seed for the drift_cfl(t) traces")
    args = ap.parse_args()
    apply_style()

    summ = json.load(open(os.path.join(args.sdir, "taper_sweep_compare.json")))

    # ---- panel (a) data: taper_hi axis (current_fourier plotted as a band) ----
    taper_x, abort_m, abort_s, cfl_m, cfl_s = [], [], [], [], []
    for name, _, th in CONFIGS:
        if th is None:
            continue
        s = summ[name]
        taper_x.append(th)
        abort_m.append(s["final_t_mean"]); abort_s.append(s["final_t_std"])
        cfl_m.append(s["cfl_max_mean"]); cfl_s.append(s["cfl_max_std"])
    taper_x = np.array(taper_x)
    order = np.argsort(taper_x)
    taper_x = taper_x[order]
    abort_m = np.array(abort_m)[order]; abort_s = np.array(abort_s)[order]
    cfl_m = np.array(cfl_m)[order]; cfl_s = np.array(cfl_s)[order]
    cf = summ["current_fourier"]

    # ---- panel (b) data: drift_cfl traces for the chosen seed ----
    traces = {}
    for name, prefix, _ in CONFIGS:
        d = os.path.join(args.sdir, f"{prefix}_seed{args.seed}")
        r = load_diag(d)
        if r is not None:
            traces[name] = r

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # panel (a)
    c_abort = "#1f77b4"; c_cfl = "#d62728"
    axa.errorbar(taper_x, np.array(abort_m) * 1e4, yerr=np.array(abort_s) * 1e4,
                 marker="o", color=c_abort, capsize=2, lw=1.2, label="hybrid abort $t$")
    axa.axhline(cf["final_t_mean"] * 1e4, color=c_abort, ls="--", lw=1.0)
    axa.axhspan((cf["final_t_mean"] - cf["final_t_std"]) * 1e4,
                (cf["final_t_mean"] + cf["final_t_std"]) * 1e4,
                color=c_abort, alpha=0.12)
    axa.text(0.50, cf["final_t_mean"] * 1e4 + 0.03, "global-$K$ drift",
             color=c_abort, fontsize=6, va="bottom")
    axa.set_xlabel(r"local-operator width $h_{\mathrm{hi}}$ (smaller = smoother)")
    axa.set_ylabel(r"drift-CFL abort time  $t\,[\times 10^{-4}]$", color=c_abort)
    axa.tick_params(axis="y", labelcolor=c_abort)
    axa.invert_xaxis()  # smoother (small taper) on the right
    axa.set_title("(a) abort time vs local-operator smoothness")

    axc = axa.twinx()
    axc.errorbar(taper_x, cfl_m, yerr=cfl_s, marker="s", color=c_cfl,
                 capsize=2, lw=1.2, ls=":")
    axc.axhline(cf["cfl_max_mean"], color=c_cfl, ls="--", lw=1.0, alpha=0.6)
    axc.set_ylabel(r"max drift CFL", color=c_cfl)
    axc.tick_params(axis="y", labelcolor=c_cfl)

    # panel (b)
    colors = {"current_fourier": "k", "two_level_taper0.5": "#d62728",
              "two_level_taper0.35": "#ff7f0e", "two_level_taper0.25": "#2ca02c"}
    labels = {"current_fourier": "global $K$=10", "two_level_taper0.5": r"$h_{hi}$=0.50",
              "two_level_taper0.35": r"$h_{hi}$=0.35", "two_level_taper0.25": r"$h_{hi}$=0.25"}
    for name, _, _ in CONFIGS:
        if name not in traces:
            continue
        t, cfl, _ = traces[name]
        axb.plot(t * 1e4, cfl, color=colors[name], lw=1.1, label=labels[name])
    axb.set_xlabel(r"$t\,[\times 10^{-4}]$")
    axb.set_ylabel("drift CFL")
    axb.set_title(f"(b) drift CFL vs $t$ (seed {args.seed})")
    axb.legend(fontsize=6, ncol=2, loc="upper left")

    fig.tight_layout()
    figdir = os.path.join(args.sdir, "figures")
    pdir = os.path.join(args.sdir, "plot_data")
    os.makedirs(figdir, exist_ok=True); os.makedirs(pdir, exist_ok=True)
    savefig_multi(fig, os.path.join(figdir, "figure_taper_sweep"))

    np.savez(os.path.join(pdir, "figure_taper_sweep.npz"),
             taper_x=taper_x, abort_mean=abort_m, abort_std=abort_s,
             cfl_mean=cfl_m, cfl_std=cfl_s,
             cf_abort_mean=cf["final_t_mean"], cf_abort_std=cf["final_t_std"],
             cf_cfl_mean=cf["cfl_max_mean"], cf_cfl_std=cf["cfl_max_std"],
             trace_seed=args.seed,
             **{f"trace_{k}_t": v[0] for k, v in traces.items()},
             **{f"trace_{k}_cfl": v[1] for k, v in traces.items()})
    print(f"wrote {figdir}/figure_taper_sweep.pdf/.png and plot_data/figure_taper_sweep.npz")


if __name__ == "__main__":
    main()
