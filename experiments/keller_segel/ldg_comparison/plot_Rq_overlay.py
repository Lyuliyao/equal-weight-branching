"""plot_Rq_overlay.py -- overlay R_q(t) for several quantiles q AND several K on a
single panel, to reveal crossings between the radii / bandwidths.

color = K (reconstruction bandwidth), linestyle = quantile q.
Optionally overlay an adaptive-window reference run (black).
Reads diag_*.csv only (no solver).
"""
import os, glob, argparse
import csv
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LIT = 1.21e-4
QSTYLE = {0.8: "-", 0.9: "--", 0.99: ":"}


def seedmean(csvs, key):
    per = []
    for f in csvs:
        R = list(csv.DictReader(open(f)))
        if not R or key not in R[0]:
            return None, None
        t = np.array([float(r["t"]) for r in R])
        y = np.array([float(r[key]) for r in R])
        per.append((t, y))
    tg = np.linspace(max(t.min() for t, _ in per), min(t.max() for t, _ in per), 700)
    return tg, np.mean([np.interp(tg, t, y) for t, y in per], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--Ks", type=int, nargs="+", default=[5, 8, 10, 16])
    ap.add_argument("--qs", type=float, nargs="+", default=[0.8, 0.9, 0.99])
    ap.add_argument("--tau", default="2e-7")
    ap.add_argument("--N", default="6400000")
    ap.add_argument("--adaptive_dir", default=None,
                    help="optional adaptive-window run dir to overlay (black ref)")
    ap.add_argument("--adaptive_K", type=int, default=5)
    ap.add_argument("--out_prefix", default="Rq_overlay")
    args = ap.parse_args()

    cmap = plt.cm.viridis(np.linspace(0.12, 0.85, len(args.Ks)))
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for K, col in zip(args.Ks, cmap):
        cs = glob.glob(f"{args.run_dir}/sens_K{K}_tau{args.tau}_q0.8_N{args.N}_seed*/diag_*.csv")
        if not cs:
            continue
        for q in args.qs:
            t, y = seedmean(cs, f"R_{q:g}")
            if t is None:
                continue
            ax.semilogy(t * 1e4, y, QSTYLE.get(q, "-"), color=col, lw=1.6)

    # optional adaptive-window reference (black)
    if args.adaptive_dir:
        cs = glob.glob(f"{args.adaptive_dir}/sens_K{args.adaptive_K}_tau{args.tau}_q0.8_N{args.N}_seed*/diag_*.csv")
        for q in args.qs:
            t, y = seedmean(cs, f"R_{q:g}")
            if t is not None:
                ax.semilogy(t * 1e4, y, QSTYLE.get(q, "-"), color="k", lw=1.4, alpha=0.8)

    ax.axvline(LIT * 1e4, color="r", ls=":", lw=0.9)
    ax.set_xlabel(r"$t\ (\times10^{-4})$")
    ax.set_ylabel(r"$R_q(t)$  (radius of $q$-fraction of particles)")
    ax.set_title("Mass-quantile radii vs bandwidth $K$ (fixed domain $L$=0.5)\n"
                 "color = $K$, linestyle = $q$", fontsize=10)
    ax.grid(alpha=0.3, which="both")

    # two legends: color=K, linestyle=q
    kleg = [Line2D([0], [0], color=c, lw=2, label=f"K={K}")
            for K, c in zip(args.Ks, cmap)]
    if args.adaptive_dir:
        kleg.append(Line2D([0], [0], color="k", lw=2, label=f"adaptive K={args.adaptive_K}"))
    qleg = [Line2D([0], [0], color="0.4", lw=2, ls=QSTYLE.get(q, "-"),
                   label=f"q={q:g} ({int(q*100)}%)") for q in args.qs]
    l1 = ax.legend(handles=kleg, loc="upper right", fontsize=8, title="bandwidth")
    ax.add_artist(l1)
    ax.legend(handles=qleg, loc="lower left", fontsize=8, title="quantile")

    fd = os.path.join(args.run_dir, "figures")
    os.makedirs(fd, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"{args.out_prefix}.{ext}"),
                    dpi=200, bbox_inches="tight")
    print(f"wrote {fd}/{args.out_prefix}.pdf/.png")


if __name__ == "__main__":
    main()
