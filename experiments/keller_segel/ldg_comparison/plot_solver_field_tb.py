"""
Figures for the solver-field LDG-style t_b comparison (next_stage.md §9).
========================================================================
Reads plot_data/tb_ratio_<cfg>.npz (written by analyze_solver_field_tb.py) and the
diag_*.csv seed curves; produces:
  solver_field_tb_ratio.{pdf,png}  -- DG resolution-gap ratio R(t)=Sbar_high/Sbar_low
                                       per config, theta line, t_b markers + CI, LDG ref.
  solver_field_S_curves.{pdf,png}  -- Sbar_low and Sbar_high per config.
  solver_field_core_radii.{pdf,png}-- seed-mean R_0.2, R_0.5 (N=3.2e5) per config.
  solver_field_dual_cfl.{pdf,png}  -- seed-mean drift_cfl_solver vs _fourier (N=8e4).

Usage:  python plot_solver_field_tb.py --sdir <solver_field_tb_run>
"""
import os
import csv
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt

import sys
_EXP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from common_plot_style import apply_style, savefig_multi  # noqa: E402

CONFIGS = ["current_fourier", "blob_ch006", "blob_ch009", "spectral_taper025"]
COLORS = {"current_fourier": "k", "blob_ch006": "#2ca02c",
          "blob_ch009": "#1f77b4", "spectral_taper025": "#d62728"}
LAB = {"current_fourier": "global $K$=10", "blob_ch006": "blob $c_h$=0.06",
       "blob_ch009": "blob $c_h$=0.09", "spectral_taper025": "spectral $K_l$=24"}
LDG_REF = (5.95e-5, 8.43e-5)


def seedmean_curve(sdir, cfg, N, col, dt=1e-6, tmax=2e-4):
    grid = np.arange(0.0, tmax + 0.5 * dt, dt)
    curves = []
    for d in sorted(glob.glob(os.path.join(sdir, f"{cfg}_N{N}_seed*"))):
        cs = glob.glob(os.path.join(d, "diag_*.csv"))
        if not cs:
            continue
        rows = list(csv.DictReader(open(cs[0])))
        if not rows or col not in rows[0]:
            continue
        t = np.array([float(r["t"]) for r in rows])
        try:
            y = np.array([float(r[col]) for r in rows])
        except (ValueError, TypeError):
            continue
        yi = np.interp(grid, t, y, left=y[0], right=np.nan)
        yi[grid > t[-1] + 1e-12] = np.nan
        curves.append(yi)
    if not curves:
        return grid, None
    return grid, np.nanmean(np.vstack(curves), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdir", required=True)
    args = ap.parse_args()
    apply_style()
    figdir = os.path.join(args.sdir, "figures"); os.makedirs(figdir, exist_ok=True)
    present = [c for c in CONFIGS
               if os.path.exists(os.path.join(args.sdir, "plot_data", f"tb_ratio_{c}.npz"))]

    # ---- Fig 1: ratio R(t) ----
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for c in present:
        d = np.load(os.path.join(args.sdir, "plot_data", f"tb_ratio_{c}.npz"))
        g, R, tb = d["grid"], d["R"], float(d["tb"])
        ax.plot(g * 1e4, R, color=COLORS[c], lw=1.2, label=LAB[c])
        if np.isfinite(tb):
            ax.axvline(tb * 1e4, color=COLORS[c], ls=":", lw=0.9)
            cl, ch = float(d["ci_low"]), float(d["ci_high"])
            if np.isfinite(cl) and np.isfinite(ch):
                ax.axvspan(cl * 1e4, ch * 1e4, color=COLORS[c], alpha=0.10)
    ax.axhline(1.05, color="0.4", ls="--", lw=0.8)
    ax.text(0.2, 1.052, r"$\theta=1.05$", fontsize=6, color="0.4")
    for ref in LDG_REF:
        ax.axvline(ref * 1e4, color="0.6", ls="-.", lw=0.7)
    ax.text(LDG_REF[0] * 1e4, 0.965, "LDG", fontsize=6, color="0.5", ha="center")
    ax.set_xlabel(r"$t\,[\times10^{-4}]$"); ax.set_ylabel(r"$\bar S_{high}/\bar S_{low}$")
    ax.set_title("DG resolution-gap ratio")
    ax.set_ylim(0.95, max(1.2, ax.get_ylim()[1])); ax.legend(fontsize=6, loc="upper left")
    fig.tight_layout(); savefig_multi(fig, os.path.join(figdir, "solver_field_tb_ratio"))
    plt.close(fig)

    # ---- Fig 2: S curves (low + high) ----
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for c in present:
        d = np.load(os.path.join(args.sdir, "plot_data", f"tb_ratio_{c}.npz"))
        g = d["grid"]
        ax.plot(g * 1e4, d["Lo"], color=COLORS[c], lw=1.0, ls="--", alpha=0.8)
        ax.plot(g * 1e4, d["Hi"], color=COLORS[c], lw=1.3, label=LAB[c])
    ax.set_xlabel(r"$t\,[\times10^{-4}]$"); ax.set_ylabel(r"$S^{DG}_{cross}$")
    ax.set_title("DG $L^2$ (dashed=low 8e4/80, solid=high 3.2e5/160)")
    ax.legend(fontsize=6, loc="upper left")
    fig.tight_layout(); savefig_multi(fig, os.path.join(figdir, "solver_field_S_curves"))
    plt.close(fig)

    # ---- Fig 3: core radii (N=3.2e5) ----
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for c in present:
        g, r02 = seedmean_curve(args.sdir, c, 320000, "R_0.2")
        g, r05 = seedmean_curve(args.sdir, c, 320000, "R_0.5")
        if r02 is not None:
            ax.plot(g * 1e4, r02, color=COLORS[c], lw=1.3, label=LAB[c])
        if r05 is not None:
            ax.plot(g * 1e4, r05, color=COLORS[c], lw=0.9, ls=":")
    ax.set_xlabel(r"$t\,[\times10^{-4}]$"); ax.set_ylabel(r"core radius")
    ax.set_title("core radii (solid $R_{0.2}$, dotted $R_{0.5}$; N=3.2e5)")
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout(); savefig_multi(fig, os.path.join(figdir, "solver_field_core_radii"))
    plt.close(fig)

    # ---- Fig 4: dual CFL (N=8e4) ----
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for c in present:
        g, cs = seedmean_curve(args.sdir, c, 80000, "drift_cfl_solver_field")
        g, cf = seedmean_curve(args.sdir, c, 80000, "drift_cfl_fourier_diag")
        if cs is not None:
            ax.plot(g * 1e4, cs, color=COLORS[c], lw=1.3, label=LAB[c])
        if cf is not None:
            ax.plot(g * 1e4, cf, color=COLORS[c], lw=0.9, ls=":")
    ax.axhline(5.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel(r"$t\,[\times10^{-4}]$"); ax.set_ylabel("drift CFL")
    ax.set_title("solver (solid) vs Fourier-diag (dotted) CFL; N=8e4")
    ax.legend(fontsize=6, loc="upper left")
    fig.tight_layout(); savefig_multi(fig, os.path.join(figdir, "solver_field_dual_cfl"))
    plt.close(fig)
    print(f"wrote 4 figures to {figdir}")


if __name__ == "__main__":
    main()
