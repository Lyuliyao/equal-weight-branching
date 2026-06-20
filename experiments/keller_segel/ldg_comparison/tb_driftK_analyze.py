"""tb_driftK_analyze.py -- does t_b converge to ~1.21e-4 as we refine the DRIFT
bandwidth K (fixed domain, fixed N_p)?

Refining the drift bandwidth refines the dynamics (the field the particles feel),
so the solution can keep concentrating -- the particle analogue of refining the LDG
PDE grid.  Gap pairs (K)->(2K): 8->16, 16->32, 32->64.

  primary  readout = bandwidth-matched Fourier L2: ratio = S_L2_u(2K)/S_L2_u(K)
  xcheck   fixed DG readout (grid 320): ratio = S_dg_cross_320(2K)/S_dg_cross_320(K)
           -- isolates the pure drift-dynamics effect at a common readout.

t_b = inf{t: ratio>=theta, persist >= 5e-6}; seed-mean + bootstrap CI.
"""
import os, glob, csv, argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

LIT = 1.21e-4
PERSIST = 5e-6
PAIRS = [(8, 16), (16, 32), (32, 64)]
N = 1280000


def seed_curves(rdir, K, col):
    out = []
    for d in sorted(glob.glob(os.path.join(rdir, f"driftK{K}_N{N}_seed*"))):
        cs = glob.glob(os.path.join(d, "diag_*.csv"))
        if not cs:
            continue
        R = list(csv.DictReader(open(cs[0])))
        if not R or col not in R[0]:
            continue
        t = np.array([float(r["t"]) for r in R]); y = np.array([float(r[col]) for r in R])
        o = np.argsort(t); keep = np.concatenate([[True], np.diff(t[o]) > 0])
        out.append((t[o][keep], y[o][keep]))
    return out


def gap_time(grid, lo, hi, theta):
    ratio = hi / np.maximum(lo, 1e-300)
    above = ratio >= theta
    for i in range(len(grid)):
        if above[i]:
            j = i
            while j < len(grid) and grid[j] - grid[i] < PERSIST:
                j += 1
            if np.all(above[i:j]):
                return grid[i]
    return np.nan


def run_one(rdir, col, theta, nboot=400):
    res = []
    for Klo, Khi in PAIRS:
        lo = seed_curves(rdir, Klo, col); hi = seed_curves(rdir, Khi, col)
        if not lo or not hi:
            res.append((f"{Klo}→{Khi}", np.nan, np.nan, np.nan, np.nan, len(lo), len(hi)))
            continue
        t0 = max(max(t.min() for t, _ in lo), max(t.min() for t, _ in hi))
        t1 = min(min(t.max() for t, _ in lo), min(t.max() for t, _ in hi))
        grid = np.linspace(t0, t1, 600)
        LO = np.array([np.interp(grid, t, y) for t, y in lo])
        HI = np.array([np.interp(grid, t, y) for t, y in hi])
        tb = gap_time(grid, LO.mean(0), HI.mean(0), theta)
        rng = np.random.default_rng(0); boot = []
        for _ in range(nboot):
            il = rng.integers(0, len(LO), len(LO)); ih = rng.integers(0, len(HI), len(HI))
            boot.append(gap_time(grid, LO[il].mean(0), HI[ih].mean(0), theta))
        boot = np.array(boot); fin = boot[np.isfinite(boot)]
        clo, chi = (np.percentile(fin, [2.5, 97.5]) if len(fin) else (np.nan, np.nan))
        rmax = float(np.nanmax(HI.mean(0) / np.maximum(LO.mean(0), 1e-300)))
        res.append((f"{Klo}→{Khi}", tb, clo, chi, rmax, len(lo), len(hi),
                    grid, LO.mean(0), HI.mean(0)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--theta", type=float, default=1.05)
    args = ap.parse_args()

    primary = run_one(args.run_dir, "S_L2_u", args.theta)
    xcheck = run_one(args.run_dir, "S_dg_cross_320", args.theta)

    fig, (axr, axc) = plt.subplots(1, 2, figsize=(12, 4.4))
    for pi, r in enumerate(primary):
        if len(r) < 8:
            continue
        lab, tb, clo, chi, rmax, nlo, nhi, grid, LO, HI = r
        c = plt.cm.viridis(pi / max(len(PAIRS) - 1, 1))
        axr.plot(grid * 1e4, HI / np.maximum(LO, 1e-300), color=c, lw=1.8,
                 label=f"K {lab}: $t_b$={tb*1e4:.2f}e-4, max={rmax:.2f}"
                 if np.isfinite(tb) else f"K {lab}: none (max={rmax:.2f})")
        if np.isfinite(tb):
            axr.plot([tb * 1e4], [args.theta], "o", color=c, ms=7)
    axr.axhline(args.theta, color="k", ls="--", lw=0.8)
    axr.axvline(LIT * 1e4, color="r", ls=":", lw=1.0, label=f"target {LIT:.2e}")
    axr.set_xlabel(r"$t\ (\times10^{-4})$"); axr.set_ylabel(r"$S_{L^2}(2K)/S_{L^2}(K)$")
    axr.set_title("(a) drift-bandwidth gap ratio (matched Fourier readout)")
    axr.set_xlim(0, 1.6); axr.legend(fontsize=7); axr.grid(alpha=0.3)

    labs = [r[0] for r in primary]
    x = np.arange(len(labs))
    for res, mk, cc, nm in [(primary, "o-", "C0", "matched $S_{L^2}$"),
                            (xcheck, "s--", "C1", "fixed DG readout (320)")]:
        tb = np.array([r[1] for r in res]); lo = np.array([r[2] for r in res])
        hi = np.array([r[3] for r in res])
        axc.errorbar(x, tb * 1e4, yerr=[(tb - lo) * 1e4, (hi - tb) * 1e4],
                     fmt=mk, color=cc, capsize=4, lw=1.6, label=nm)
    axc.axhline(LIT * 1e4, color="r", ls=":", lw=1.2, label=f"target {LIT:.2e}")
    axc.set_xticks(x); axc.set_xticklabels(labs); axc.set_xlabel("drift-K pair")
    axc.set_ylabel(r"$t_b\ (\times10^{-4})$")
    axc.set_title(f"(b) $t_b$ vs drift bandwidth ($\\theta$={args.theta})")
    axc.legend(fontsize=8); axc.grid(alpha=0.3)

    fig.suptitle("t_b vs DRIFT bandwidth K (fixed domain L=0.5, N_p=1.28M): "
                 "does refining the dynamics converge to the LDG target?", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fd = os.path.join(args.run_dir, "figures"); os.makedirs(fd, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fd, "tb_driftK.{}".format(ext)), dpi=200, bbox_inches="tight")

    with open(os.path.join(args.run_dir, "tb_driftK.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["readout", "pair", "t_b", "ci_lo", "ci_hi", "ratio_max"])
        for nm, res in [("matched_S_L2", primary), ("fixed_dg_320", xcheck)]:
            for r in res:
                w.writerow([nm, r[0], f"{r[1]:.4e}", f"{r[2]:.4e}", f"{r[3]:.4e}", f"{r[4]:.3f}"])
    print(f"target ~ {LIT:.2e}")
    print("matched Fourier S_L2:")
    for r in primary:
        print(f"  K {r[0]}: t_b={r[1]:.3e} CI[{r[2]:.2e},{r[3]:.2e}] ratio_max={r[4]:.2f} (n {r[5]}/{r[6]})")
    print("fixed DG readout (320):")
    for r in xcheck:
        print(f"  K {r[0]}: t_b={r[1]:.3e} ratio_max={r[4]:.2f}")
    print(f"wrote {fd}/tb_driftK.pdf/.png and tb_driftK.csv")


if __name__ == "__main__":
    main()
