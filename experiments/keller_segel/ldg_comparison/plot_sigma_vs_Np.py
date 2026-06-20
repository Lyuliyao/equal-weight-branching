"""plot_sigma_vs_Np.py -- global second-moment width sigma(t)=sqrt(S_u/2) (straight
from particle positions) as the particle count N_p increases.

sigma is reconstruction-free (just the cloud second moment); the question is the
Monte-Carlo convergence in N_p. Shows seed-mean sigma(t) with a +/-1 std band over
seeds for each N_p, on the K=32 fixed-domain ladder 80k/320k/1.28M/5.12M.
"""
import os, glob, csv, argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

LIT = 1.21e-4
NPS = [80000, 320000, 1280000, 5120000]


def seed_sigma(rdir, Np):
    ts, ss = [], []
    for d in sorted(glob.glob(os.path.join(rdir, f"cf_N{Np}_seed*"))):
        cs = glob.glob(os.path.join(d, "diag_*.csv"))
        if not cs:
            continue
        R = list(csv.DictReader(open(cs[0])))
        t = np.array([float(r["t"]) for r in R])
        s = np.sqrt(np.maximum(np.array([float(r["S_u"]) for r in R]), 0) / 2)
        ts.append(t); ss.append(s)
    if not ts:
        return None, None, None
    tg = np.linspace(max(t.min() for t in ts), min(t.max() for t in ts), 500)
    S = np.array([np.interp(tg, t, s) for t, s in zip(ts, ss)])
    return tg, S.mean(0), (S.std(0) if len(S) > 1 else np.zeros_like(tg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_prefix", default="sigma_vs_Np")
    args = ap.parse_args()

    cmap = plt.cm.viridis(np.linspace(0.12, 0.85, len(NPS)))
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.6))

    rows = []
    for Np, c in zip(NPS, cmap):
        t, m, sd = seed_sigma(args.run_dir, Np)
        if t is None:
            continue
        lab = f"$N_p$={Np/1e6:g}M" if Np >= 1e6 else f"$N_p$={Np//1000}k"
        axa.plot(t * 1e4, m, color=c, lw=1.9, label=lab)
        axa.fill_between(t * 1e4, m - sd, m + sd, color=c, alpha=0.2, lw=0)
        # late-time zoom (panel b)
        axb.plot(t * 1e4, m, color=c, lw=1.9, label=lab)
        axb.fill_between(t * 1e4, m - sd, m + sd, color=c, alpha=0.2, lw=0)
        i = np.argmin(np.abs(t - 1.2e-4))
        rows.append((Np, m[i], sd[i]))

    for ax in (axa, axb):
        ax.axvline(LIT * 1e4, color="r", ls=":", lw=1.0)
        ax.set_xlabel(r"$t\ (\times10^{-4})$")
        ax.grid(alpha=0.3)
    axa.set_ylabel(r"global $\sigma(t)=\sqrt{S_u/2}$  (from particle 2nd moment)")
    axa.set_title("(a) $\\sigma(t)$ vs particle count $N_p$ (K=32 fixed domain)")
    axa.legend(fontsize=8)
    axb.set_title("(b) late-time zoom: $\\sigma$ converges as $N_p\\uparrow$")
    axb.set_xlim(0.8, 1.45); axb.set_ylim(0.010, 0.022); axb.legend(fontsize=8)

    fig.suptitle("Global second-moment width $\\sigma(t)$ vs particle count "
                 "(reconstruction-free; band = $\\pm1$ std over seeds)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fd = os.path.join(args.run_dir, "figures"); os.makedirs(fd, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"{args.out_prefix}.{ext}"), dpi=200, bbox_inches="tight")
    # plot data
    with open(os.path.join(args.run_dir, f"{args.out_prefix}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["N_p", "sigma_at_1.2e-4", "seed_std"])
        for Np, m, sd in rows:
            w.writerow([Np, f"{m:.6e}", f"{sd:.6e}"])
    print("sigma at t~1.2e-4 vs N_p (seed-mean +/- std):")
    for Np, m, sd in rows:
        print(f"  N_p={Np:>8}:  sigma={m:.5f} +/- {sd:.5f}")
    print(f"wrote {fd}/{args.out_prefix}.pdf/.png")


if __name__ == "__main__":
    main()
