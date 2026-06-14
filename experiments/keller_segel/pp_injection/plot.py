"""plot.py — publication figure for the parabolic-parabolic KS injection run.

Reads the saved time series (plot_data/figure_mass_balance.npz, with a CSV
fallback) and reproduces the chemical-mass-balance figure WITHOUT rerunning the
solver.  Two panels:

  (left)  M_v(t) from the injection particle cloud vs. the exact analytic law
          M_u + (M_v0 - M_u) e^{-t}, with the conserved cell mass M_u(t) shown;
  (right) the relative error of M_v(t) vs. the exact law (injection unbiasedness).

Uses the shared style experiments/common_plot_style.py and saves both .pdf and
.png via cps.savefig_multi.

Run:  python plot.py --results_dir results_smoke
      python plot.py --results_dir <reference_results/pp_injection/<run_id>>
"""
import os
import sys
import csv
import argparse

import numpy as np

# Shared publication style lives in experiments/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
import common_plot_style as cps  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


def load_series(results_dir):
    """Load the saved time series from the npz (preferred) or the CSV fallback."""
    npz = os.path.join(results_dir, "plot_data", "figure_mass_balance.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        return {k: d[k] for k in d.files}
    # CSV fallback (mass_balance.csv) — never reruns the solver.
    csv_path = os.path.join(results_dir, "mass_balance.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No plot_data/figure_mass_balance.npz or mass_balance.csv in {results_dir}")
    cols = {}
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    out = {k: np.array(v) for k, v in cols.items()}
    out["M_v_exact"] = out.get("M_v_exact", out["M_v"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True,
                    help="directory holding plot_data/figure_mass_balance.npz "
                         "(or mass_balance.csv)")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="where to write figures/ (default: <results_dir>/figures)")
    ap.add_argument("--stem", type=str, default="figure_mass_balance")
    args = ap.parse_args()

    d = load_series(args.results_dir)
    t = d["t"]
    M_v = d["M_v"]
    M_v_exact = d["M_v_exact"]
    M_u = d["M_u"]
    rel_err = d["rel_err"]
    Mu0 = float(d["Mu0"]) if "Mu0" in d else float(M_u[0])
    Mv0 = float(d["Mv0"]) if "Mv0" in d else float(M_v[0])
    max_relerr = float(d["max_relerr_Mv_law"]) if "max_relerr_Mv_law" in d \
        else float(np.max(rel_err))

    cps.apply_style()
    fig, ax = plt.subplots(1, 2, figsize=cps.fig_size(1.0, 0.40))

    # ---- panel 1: chemical mass balance ----
    ax[0].plot(t, M_v_exact, "-", color="C0", zorder=2,
               label=r"exact $M_u+(M_{v0}-M_u)\,e^{-t}$")
    ax[0].plot(t, M_v, "o", ms=3.0, color="C1", mfc="none", mew=0.9, zorder=3,
               label=r"$M_v$ (injection particles)")
    ax[0].axhline(Mu0, ls="--", color="0.55", lw=0.9, zorder=1,
                  label=r"$M_u$ (conserved)")
    ax[0].set_xlabel(r"$t$")
    ax[0].set_ylabel("mass")
    ax[0].set_title("Chemical mass balance")
    ax[0].set_xlim(t.min(), t.max())
    ax[0].set_ylim(top=1.06)
    ax[0].legend(loc="lower right", borderaxespad=0.5)
    ax[0].text(0.04, 0.95,
               rf"$M_{{v0}}={Mv0:.2f},\ M_u={Mu0:.2f}$",
               transform=ax[0].transAxes, fontsize=7, va="top")

    # ---- panel 2: relative error of the mass law ----
    ax[1].plot(t, rel_err, "-", color="C3", lw=1.2)
    ax[1].plot(t, rel_err, ".", ms=3.0, color="C3")
    ax[1].set_xlabel(r"$t$")
    ax[1].set_ylabel(r"rel. error of $M_v(t)$")
    ax[1].set_title("Injection-reaction unbiasedness")
    ax[1].set_xlim(t.min(), t.max())
    ax[1].set_ylim(bottom=0.0)
    ax[1].text(0.96, 0.92, rf"$\max_t\,\mathrm{{relerr}}={max_relerr:.2e}$",
               transform=ax[1].transAxes, fontsize=7, ha="right", va="top")

    fig.tight_layout()

    out_dir = args.out_dir or os.path.join(args.results_dir, "figures")
    stem = os.path.join(out_dir, args.stem)
    paths = cps.savefig_multi(fig, stem)
    for p in paths:
        print("wrote", os.path.abspath(p))


if __name__ == "__main__":
    main()
