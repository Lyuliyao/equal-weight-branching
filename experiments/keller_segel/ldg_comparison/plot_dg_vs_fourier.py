"""plot_dg_vs_fourier.py -- compare the chemotactic-drift reconstruction operator
INSIDE the solver: strong-P1-DG drift vs single-K Fourier drift (matched setup).

For each method (dg, current_fourier) seed-mean the diag curves and overlay:
  (a) core radii R_0.5, R_0.8(t)      [reconstruction-free]
  (b) global sigma(t)=sqrt(S_u/2)     [reconstruction-free]
  (c) reconstructed peak / S_L2(t)    [K=32 diagnostic, same for both]
  (d) drift CFL(t)                    [stability]
Reads diag_*.csv only.
"""
import os, glob, csv, argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

LIT = 1.21e-4
METHODS = [("dg", "C0", "DG drift (P1)"), ("current_fourier", "C3", "Fourier drift (K=32)")]


def seedmean(rdir, method, col):
    ts, ys = [], []
    for d in sorted(glob.glob(os.path.join(rdir, f"{method}_seed*"))):
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
    ap.add_argument("--out_prefix", default="dg_vs_fourier")
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    (axr, axs), (axp, axc) = axes

    for method, col, lab in METHODS:
        # (a) core radii
        for q, ls in [(0.5, "-"), (0.8, "--")]:
            t, y = seedmean(args.run_dir, method, f"R_{q:g}")
            if t is not None:
                axr.semilogy(t * 1e4, y, ls, color=col, lw=1.7,
                             label=f"{lab} R_{q:g}")
        # (b) sigma
        t, Su = seedmean(args.run_dir, method, "S_u")
        if t is not None:
            axs.plot(t * 1e4, np.sqrt(np.maximum(Su, 0) / 2), color=col, lw=1.8, label=lab)
        # (c) peak + S_L2
        t, pk = seedmean(args.run_dir, method, "peak_PK_u")
        if t is not None:
            axp.semilogy(t * 1e4, pk, color=col, lw=1.8, label=f"{lab} peak")
        t, sl = seedmean(args.run_dir, method, "S_L2_u")
        if t is not None:
            axp.semilogy(t * 1e4, sl, color=col, ls=":", lw=1.5, label=f"{lab} $S_{{L^2}}$")
        # (d) drift CFL
        t, cf = seedmean(args.run_dir, method, "drift_cfl")
        if t is not None:
            axc.plot(t * 1e4, cf, color=col, lw=1.8, label=lab)

    for ax in (axr, axs, axp, axc):
        ax.axvline(LIT * 1e4, color="r", ls=":", lw=0.9)
        ax.set_xlabel(r"$t\ (\times10^{-4})$"); ax.grid(alpha=0.3)
    axr.set_ylabel("$R_q(t)$"); axr.set_title("(a) core radii (recon-free)"); axr.legend(fontsize=7)
    axs.set_ylabel(r"$\sigma=\sqrt{S_u/2}$"); axs.set_title("(b) global width (recon-free)"); axs.legend(fontsize=8)
    axp.set_ylabel("peak / $S_{L^2}$"); axp.set_title("(c) K=32 reconstructed peak & $S_{L^2}$"); axp.legend(fontsize=7)
    axc.axhline(8.0, color="0.5", ls="--", lw=0.8, label="cfl_abort=8")
    axc.set_ylabel("drift CFL"); axc.set_title("(d) drift CFL (stability)"); axc.legend(fontsize=8)

    fig.suptitle("Solver drift reconstruction: strong-P1-DG vs Fourier "
                 "(fixed domain L=0.5, K=32 diag, $\\tau$=5e-8, N=1.28M)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fd = os.path.join(args.run_dir, "figures"); os.makedirs(fd, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"{args.out_prefix}.{ext}"), dpi=200, bbox_inches="tight")
    # numeric summary at t~1.2e-4
    print("at t~1.2e-4 (seed-mean):")
    for method, _, lab in METHODS:
        out = {}
        for col in ["R_0.5", "R_0.8", "S_u", "peak_PK_u", "S_L2_u", "drift_cfl"]:
            t, y = seedmean(args.run_dir, method, col)
            out[col] = y[np.argmin(np.abs(t - 1.2e-4))] if t is not None else np.nan
        sig = np.sqrt(out["S_u"] / 2)
        print(f"  {lab:24}: R0.5={out['R_0.5']:.4f} R0.8={out['R_0.8']:.4f} "
              f"sigma={sig:.4f} peak={out['peak_PK_u']:.0f} S_L2={out['S_L2_u']:.1f} "
              f"maxCFL≈{out['drift_cfl']:.2f}")
    print(f"wrote {fd}/{args.out_prefix}.pdf/.png")


if __name__ == "__main__":
    main()
