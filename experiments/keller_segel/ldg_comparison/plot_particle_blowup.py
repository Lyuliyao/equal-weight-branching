"""Figure for the LDG-style particle blow-up proxy sweep (reads diag CSVs only)."""
import os, sys, csv, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common_plot_style as cps
from common_plot_style import TEXTWIDTH_IN, savefig_multi
cps.apply_style()
LDG = (5.953e-5, 8.428e-5)


def ens(rdir, Np, col, grid):
    M = []
    for d in sorted(glob.glob(os.path.join(rdir, f"Np{Np}_seed*"))):
        cs = glob.glob(os.path.join(d, "diag_*.csv"))
        if not cs:
            continue
        r = list(csv.DictReader(open(cs[0])))
        if col not in r[0]:
            continue
        t = np.array([float(x["t"]) for x in r]); s = np.array([float(x[col]) for x in r])
        o = np.argsort(t); t, s = t[o], s[o]
        M.append(np.interp(grid, t, s, right=np.nan))
    return np.nanmean(M, 0) if M else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rdir", required=True)
    args = ap.parse_args()
    grid = np.arange(0, 1.45e-4, 1e-6)
    fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 0.36 * TEXTWIDTH_IN))
    # (a) S_dg_cross curves
    series = [("8e4, n=80", 80000, "S_dg_cross_80", "#4292c6"),
              ("3.2e5, n=160", 320000, "S_dg_cross_160", "#08519c"),
              ("8e4, n=160 (recon)", 80000, "S_dg_cross_160", "tab:orange"),
              ("2e4, n=40", 20000, "S_dg_cross_40", "#9ecae1")]
    for lab, Np, col, c in series:
        s = ens(args.rdir, Np, col, grid)
        if s is not None:
            ax[0].plot(grid, s, "-", color=c, lw=1.1, label=lab)
    ax[0].set_yscale("log")
    ax[0].axvspan(LDG[0], LDG[1], color="0.7", alpha=0.3, label="LDG $t_b$")
    ax[0].set_xlabel("$t$"); ax[0].set_ylabel(r"$S^{\rm DG}_{\rm cross}(t)$")
    ax[0].legend(fontsize=4.6, loc="lower right"); ax[0].set_title("LDG-matched DG readout", fontsize=7)
    # (b) gap ratios
    pairs = [("main (8e4,80)->(3.2e5,160)", 80000, "S_dg_cross_80", 320000, "S_dg_cross_160", "#08519c", 9.2e-5),
             ("recon (8e4,80)->(8e4,160)", 80000, "S_dg_cross_80", 80000, "S_dg_cross_160", "tab:orange", 4.8e-5),
             ("main (2e4,40)->(8e4,80)", 20000, "S_dg_cross_40", 80000, "S_dg_cross_80", "#9ecae1", 7e-6)]
    for lab, Nl, cl, Nh, ch, c, tb in pairs:
        sl = ens(args.rdir, Nl, cl, grid); sh = ens(args.rdir, Nh, ch, grid)
        if sl is None or sh is None:
            continue
        ax[1].plot(grid, sh / sl, "-", color=c, lw=1.1, label=lab)
        ax[1].plot([tb], [1.05], "v", color=c, ms=4)
    ax[1].axhline(1.05, color="0.5", ls=":", lw=0.8)
    ax[1].axvspan(LDG[0], LDG[1], color="0.7", alpha=0.3)
    ax[1].set_xlabel("$t$"); ax[1].set_ylabel(r"$S_{\rm high}/S_{\rm low}$")
    ax[1].set_ylim(0.9, 3.0); ax[1].legend(fontsize=4.4, loc="upper left")
    ax[1].set_title(r"resolution-gap ratio ($\theta=1.05$)", fontsize=6.6)
    fig.suptitle("Particle blow-up proxy vs fixed-flux LDG (grey band = LDG $t_b$)", fontsize=7)
    fig.tight_layout(pad=0.4, rect=[0, 0, 1, 0.94])
    out = os.path.join(args.rdir, "figures", "particle_blowup_metric")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    savefig_multi(fig, out, close=True)
    print("wrote", out + ".pdf/.png")


if __name__ == "__main__":
    main()
