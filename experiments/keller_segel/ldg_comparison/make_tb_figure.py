"""make_tb_figure.py -- LDG-style resolution-gap numerical blow-up time t_b:
ratio S_fine/S_coarse(t) with the theta=1.05 crossing, and t_b convergence vs
refinement for LDG and the particle method.

particle: seed-mean S_dg_cross from cf_N{Np}_seed* runs (readout grids 80/160/320),
pairs (Np,n)->(4Np,2n): (8e4,80)->(3.2e5,160) and (3.2e5,160)->(1.28e6,320).
LDG: S_L2 from <ldg_dir>/N{n}/S_curves.csv, pairs (80,160) and (160,320).
"""
import os, csv, glob, argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

LIT = 1.21e-4
THETA = 1.05


def seedmean(run_dir, Npat, col):
    ts, ys = [], []
    for f in sorted(glob.glob(os.path.join(run_dir, Npat, "diag_*.csv"))):
        rows = list(csv.DictReader(open(f)))
        if col not in rows[0]:
            continue
        ts.append(np.array([float(r["t"]) for r in rows]))
        ys.append(np.array([float(r[col]) for r in rows]))
    if not ts:
        return None, None
    tg = np.linspace(max(t.min() for t in ts), min(t.max() for t in ts), 500)
    return tg, np.mean([np.interp(tg, t, y) for t, y in zip(ts, ys)], axis=0)


def ldg_S(ldg_dir, n):
    f = os.path.join(ldg_dir, f"N{n}", "S_curves.csv")
    if not os.path.exists(f):
        return None, None
    rows = list(csv.DictReader(open(f)))
    sc = "S_L2" if "S_L2" in rows[0] else [c for c in rows[0] if c.lower().startswith("s")][0]
    return (np.array([float(r["t"]) for r in rows]),
            np.array([float(r[sc]) for r in rows]))


def cross_tb(tg, ratio, theta=THETA, hold=5e-6):
    """first t where ratio>=theta and stays >=theta for >=hold."""
    above = ratio >= theta
    for i in range(len(tg)):
        if above[i]:
            j = i
            while j < len(tg) and tg[j] - tg[i] < hold:
                j += 1
            if np.all(above[i:j]):
                return tg[i]
    return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_dir", required=True)
    ap.add_argument("--ldg_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_ladder", type=int, nargs=3, default=[80000, 320000, 1280000],
                    help="three particle counts N0<N1<N2 forming (N0,80)->(N1,160)->(N2,320)")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    N0, N1, N2 = args.n_ladder
    def lab(n):
        return f"{n/1e6:g}M" if n >= 1e6 else f"{n/1e3:g}k"
    # particle pairs (Np_low,col_low) -> (Np_high,col_high)
    p_pairs = [(f"{lab(N0)}->{lab(N1)} (n80->160)", f"cf_N{N0}_seed*", "S_dg_cross_80",
                f"cf_N{N1}_seed*", "S_dg_cross_160"),
               (f"{lab(N1)}->{lab(N2)} (n160->320)", f"cf_N{N1}_seed*", "S_dg_cross_160",
                f"cf_N{N2}_seed*", "S_dg_cross_320")]
    fig, (axr, axc) = plt.subplots(1, 2, figsize=(11, 4.2))
    p_tb = []
    for lab, lo, clo, hi, chi in p_pairs:
        tl, Sl = seedmean(args.tb_dir, lo, clo); th, Sh = seedmean(args.tb_dir, hi, chi)
        if tl is None or th is None:
            print(f"[tb] missing {lab}"); continue
        tg = np.linspace(max(tl.min(), th.min()), min(tl.max(), th.max()), 500)
        r = np.interp(tg, th, Sh) / np.maximum(np.interp(tg, tl, Sl), 1e-30)
        tb = cross_tb(tg, r)
        p_tb.append((lab, tb))
        axr.plot(tg * 1e4, r, lw=1.6, label=f"particle {lab}: $t_b$={tb*1e4:.2f}")
        if np.isfinite(tb): axr.axvline(tb * 1e4, ls=":", lw=0.8, color=axr.lines[-1].get_color())

    l_tb = []
    for (na, nb, lab) in [(80, 160, "LDG n80->160"), (160, 320, "LDG n160->320")]:
        ta, Sa = ldg_S(args.ldg_dir, na); tb_, Sb = ldg_S(args.ldg_dir, nb)
        if ta is None or tb_ is None:
            continue
        tg = np.linspace(max(ta.min(), tb_.min()), min(ta.max(), tb_.max()), 500)
        r = np.interp(tg, tb_, Sb) / np.maximum(np.interp(tg, ta, Sa), 1e-30)
        tb = cross_tb(tg, r); l_tb.append((lab, tb))
        axr.plot(tg * 1e4, r, "--", lw=1.3, label=f"{lab}: $t_b$={tb*1e4:.2f}")

    axr.axhline(THETA, color="k", ls="-", lw=0.7, label=rf"$\theta$={THETA}")
    axr.set_xlabel(r"$t\ (\times 10^{-4})$"); axr.set_ylabel(r"$S_{\rm fine}/S_{\rm coarse}$")
    axr.set_title("(a) resolution-gap ratio + $t_b$ crossings"); axr.set_xlim(0, 2.0)
    axr.legend(fontsize=7); axr.grid(alpha=0.3)

    # convergence: t_b vs refinement level
    lvl_p = list(range(len(p_tb))); lvl_l = list(range(len(l_tb)))
    if p_tb:
        axc.plot(lvl_p, [tb * 1e4 for _, tb in p_tb], "o-", label="particle")
        axc.set_xticks(range(max(len(p_tb), len(l_tb))))
        axc.set_xticklabels([lab.split(" ")[0] for lab, _ in p_tb], fontsize=7, rotation=15)
    if l_tb:
        axc.plot(lvl_l, [tb * 1e4 for _, tb in l_tb], "s--", label="LDG")
    axc.axhline(LIT * 1e4, color="r", ls=":", label=r"literature $1.21$")
    axc.set_ylabel(r"$t_b\ (\times 10^{-4})$"); axc.set_xlabel("refinement pair")
    axc.set_title(r"(b) $t_b$ convergence $\to$ literature"); axc.legend(fontsize=8); axc.grid(alpha=0.3)
    fig.suptitle(r"Numerical blow-up time $t_b=\inf\{t:\ S_{\rm fine}/S_{\rm coarse}\geq\theta\}$", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out + ".pdf", bbox_inches="tight"); fig.savefig(args.out + ".png", dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}.pdf/.png")
    print("particle t_b:", [(l, f"{tb:.3e}") for l, tb in p_tb])
    print("LDG t_b:", [(l, f"{tb:.3e}") for l, tb in l_tb])


if __name__ == "__main__":
    main()
