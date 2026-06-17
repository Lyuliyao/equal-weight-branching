"""plot_Rq.py -- mass-quantile core radii R_q(t) and the R_q^2 collapse fit.

(a) R_q(t) for several q (the reconstruction-free core collapse).
(b) R_{q0}^2(t) with the linear fits R^2=alpha-beta t on the four windows
    {[4,9],[5,10],[6,11],[7,12]}x1e-5, each extrapolated to R^2=0 at T=alpha/beta.
    The spread of those T's is the "fit-window range" of the collapse time T_core.

Reads the dense R_q(t) from a run's diag_*.csv (no solver run).
"""
import os, csv, glob, argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

WIN = [(4e-5, 9e-5), (5e-5, 1.0e-4), (6e-5, 1.1e-4), (7e-5, 1.2e-4)]
LIT = 1.21e-4


def lf(t, y):
    A = np.vstack([np.ones_like(t), t]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = a + b * t; ssr = np.sum((y - yh) ** 2); sst = np.sum((y - np.mean(y)) ** 2)
    return a, b, (1 - ssr / sst if sst > 0 else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True, help="path to a diag_*.csv (or run subdir)")
    ap.add_argument("--qs", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5, 0.8])
    ap.add_argument("--q0", type=float, default=0.2, help="quantile for the R^2 fit panel")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    path = args.diag
    if os.path.isdir(path):
        path = glob.glob(os.path.join(path, "diag_*.csv"))[0]
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r["t"]) for r in rows])

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) R_q(t)
    for q in args.qs:
        key = f"R_{q:g}"
        if key not in rows[0]:
            continue
        R = np.array([float(r[key]) for r in rows])
        axa.semilogy(t * 1e4, R, lw=1.4, label=rf"$R_{{{q:g}}}$")
    axa.set_xlabel(r"$t\ (\times 10^{-4})$"); axa.set_ylabel(r"$R_q(t)$")
    axa.set_title("(a) mass-quantile core radii")
    axa.legend(fontsize=8, ncol=2); axa.grid(alpha=0.3, which="both")

    # (b) R_q0^2(t) + window fits extrapolated to R^2=0
    Rq0 = np.array([float(r[f"R_{args.q0:g}"]) for r in rows])
    y = Rq0 ** 2
    axb.plot(t * 1e4, y, "k.-", ms=3, lw=1.0, label=rf"$R_{{{args.q0:g}}}^2(t)$ (data)")
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(WIN)))
    Ts = []
    for (w0, w1), c in zip(WIN, colors):
        m = (t >= w0) & (t <= w1)
        if m.sum() < 3:
            continue
        a, b, r2 = lf(t[m], y[m])
        if b >= 0:
            continue
        T = a / (-b)
        Ts.append(T)
        tt = np.linspace(w0, T, 50)
        axb.plot(tt * 1e4, a + b * tt, "--", color=c, lw=1.2,
                 label=rf"[{w0*1e5:.0f},{w1*1e5:.0f}]: $T$={T*1e4:.2f}")
        axb.plot([T * 1e4], [0], "v", color=c, ms=7)
    axb.axhline(0, color="gray", lw=0.6)
    axb.axvline(LIT * 1e4, color="r", ls=":", lw=1.0, label=r"LDG/lit $1.21$")
    if Ts:
        axb.axvspan(min(Ts) * 1e4, max(Ts) * 1e4, color="orange", alpha=0.12,
                    label=f"window range\n[{min(Ts)*1e4:.2f},{max(Ts)*1e4:.2f}]")
    axb.set_xlabel(r"$t\ (\times 10^{-4})$"); axb.set_ylabel(rf"$R_{{{args.q0:g}}}^2$")
    axb.set_title(r"(b) $R^2$ collapse fit $\Rightarrow$ window-dependent $T$")
    axb.set_xlim(0, max(3.0, (max(Ts) * 1e4 + 0.3) if Ts else 3.0))
    axb.set_ylim(-0.0002, float(np.nanmax(y)) * 1.05)
    axb.legend(fontsize=7.5, loc="upper right")
    if args.title:
        fig.suptitle(args.title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96] if args.title else None)
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    fig.savefig(args.out + ".png", dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}.pdf/.png ; window-T range = [{min(Ts):.3e},{max(Ts):.3e}]" if Ts else "no fits")


if __name__ == "__main__":
    main()
