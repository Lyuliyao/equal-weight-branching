"""plot_S_curves_tbK32.py -- the S(t) curves underlying the tb_K32_converge gap.

(a) the resolution LADDER used in the gap: S_dg_cross_{n} from the N_p run, for
    (80k,80) (320k,160) (1.28M,320) (5.12M,640).  If consecutive curves overlap,
    the gap ratio ~ 1 (gap closed) -- which is exactly what t_b showed.
(b) readout-grid sweep at the finest run (5.12M): S_dg_cross at 80/160/320/640,
    showing that beyond a grid the readout has converged (core fully resolved),
    plus the Fourier S_L2 for contrast.
Seed-averaged, reads diag_*.csv only.
"""
import os, glob, csv, argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

LIT = 1.21e-4
LADDER = [(80000, 80), (320000, 160), (1280000, 320), (5120000, 640)]


def seedmean(rdir, Np, col):
    ts, ys = [], []
    for d in sorted(glob.glob(os.path.join(rdir, f"cf_N{Np}_seed*"))):
        cs = glob.glob(os.path.join(d, "diag_*.csv"))
        if not cs:
            continue
        R = list(csv.DictReader(open(cs[0])))
        if not R or col not in R[0]:
            continue
        ts.append(np.array([float(r["t"]) for r in R]))
        ys.append(np.array([float(r[col]) for r in R]))
    if not ts:
        return None, None
    tg = np.linspace(max(t.min() for t in ts), min(t.max() for t in ts), 500)
    return tg, np.mean([np.interp(tg, t, y) for t, y in zip(ts, ys)], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_prefix", default="S_curves_tbK32")
    args = ap.parse_args()

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.6))
    cmap = plt.cm.viridis(np.linspace(0.12, 0.85, len(LADDER)))

    # (a) the matched-resolution ladder (what the gap compares)
    for (Np, n), c in zip(LADDER, cmap):
        t, S = seedmean(args.run_dir, Np, f"S_dg_cross_{n}")
        if t is None:
            continue
        axa.semilogy(t * 1e4, S, color=c, lw=1.9,
                     label=f"$N_p$={Np//1000}k, grid {n}")
    axa.axvline(LIT * 1e4, color="r", ls=":", lw=1.0, label=f"target {LIT:.2e}")
    axa.set_xlabel(r"$t\ (\times10^{-4})$")
    axa.set_ylabel(r"$S_{\rm dg}(t)=\|P^{\rm DG}_n\mu_{N_p}\|_{L^2}$")
    axa.set_title("(a) resolution ladder (matched $N_p$,grid): consecutive\n"
                  "curves overlap $\\Rightarrow$ gap ratio $\\to1$", fontsize=9)
    axa.set_xlim(0, 1.6); axa.legend(fontsize=7); axa.grid(alpha=0.3, which="both")

    # (b) readout-grid sweep at the finest run (5.12M) + Fourier S_L2
    Np_fine = 5120000
    grids = [80, 160, 320, 640]
    gcols = plt.cm.plasma(np.linspace(0.1, 0.8, len(grids)))
    for n, c in zip(grids, gcols):
        t, S = seedmean(args.run_dir, Np_fine, f"S_dg_cross_{n}")
        if t is not None:
            axb.semilogy(t * 1e4, S, color=c, lw=1.7, label=f"DG grid {n}")
    t, S = seedmean(args.run_dir, Np_fine, "S_L2_u")
    if t is not None:
        axb.semilogy(t * 1e4, S, color="k", ls="--", lw=1.6, label="Fourier $S_{L^2}$ (K=32)")
    axb.axvline(LIT * 1e4, color="r", ls=":", lw=1.0)
    axb.set_xlabel(r"$t\ (\times10^{-4})$"); axb.set_ylabel(r"$S(t)$")
    axb.set_title(f"(b) readout sweep at $N_p$={Np_fine//1000000}M: DG grids\n"
                  "converge; Fourier $S_{L^2}$ band-limited", fontsize=9)
    axb.set_xlim(0, 1.6); axb.legend(fontsize=7); axb.grid(alpha=0.3, which="both")

    fig.suptitle("S(t) behind the K=32 fixed-domain $t_b$ test "
                 "(DG-projection $L^2$, seed-mean)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fd = os.path.join(args.run_dir, "figures"); os.makedirs(fd, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"{args.out_prefix}.{ext}"), dpi=200, bbox_inches="tight")
    print(f"wrote {fd}/{args.out_prefix}.pdf/.png")
    # quick numeric: late-time values of the ladder (do they converge?)
    print("late-time S_dg (t~1.2e-4) along the ladder:")
    for Np, n in LADDER:
        t, S = seedmean(args.run_dir, Np, f"S_dg_cross_{n}")
        if t is not None:
            print(f"  ({Np//1000}k, grid {n}): S={S[np.argmin(abs(t-1.2e-4))]:.1f}")


if __name__ == "__main__":
    main()
