"""tb_K32_converge_analyze.py -- does the LDG-style t_b converge to ~1.21e-4 on the
NEW setup (K=32, fixed domain) as we refine the LDG-matched main gap?

pairs (N_p, n) -> (4N_p, 2n), ppc=12.5 fixed:
  (80k,80)->(320k,160) ; (320k,160)->(1.28M,320) ; (1.28M,320)->(5.12M,640)
ratio(t) = S_dg_cross_{2n}(4N_p) / S_dg_cross_{n}(N_p);  readout = unbiased
(cross) P1-DG L2 (NOT Fourier).  t_b = inf{t: ratio>=theta, persist >= 5e-6}.
Seed-mean + bootstrap CI over seeds. Target = LDG fixed-flux ~1.21e-4.
"""
import os, glob, csv, argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

LIT = 1.21e-4
PERSIST = 5e-6
PAIRS = [(80000, 80, 320000, 160), (320000, 160, 1280000, 320),
         (1280000, 320, 5120000, 640)]


def seed_curves(rdir, Np, col):
    out = []
    for d in sorted(glob.glob(os.path.join(rdir, f"cf_N{Np}_seed*"))):
        cs = glob.glob(os.path.join(d, "diag_*.csv"))
        if not cs:
            continue
        R = list(csv.DictReader(open(cs[0])))
        if not R or col not in R[0]:
            continue
        t = np.array([float(r["t"]) for r in R])
        y = np.array([float(r[col]) for r in R])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--theta", type=float, default=1.05)
    ap.add_argument("--nboot", type=int, default=400)
    args = ap.parse_args()

    fig, (axr, axc) = plt.subplots(1, 2, figsize=(12, 4.4))
    rows = []
    tb_pts, tb_lo, tb_hi, labels = [], [], [], []
    for pi, (Nlo, nlo, Nhi, nhi) in enumerate(PAIRS):
        lo = seed_curves(args.run_dir, Nlo, f"S_dg_cross_{nlo}")
        hi = seed_curves(args.run_dir, Nhi, f"S_dg_cross_{nhi}")
        if not lo or not hi:
            print(f"[skip] pair {Nlo}->{Nhi}: missing data (lo={len(lo)},hi={len(hi)})")
            continue
        t0 = max(max(t.min() for t, _ in lo), max(t.min() for t, _ in hi))
        t1 = min(min(t.max() for t, _ in lo), min(t.max() for t, _ in hi))
        grid = np.linspace(t0, t1, 600)
        LO = np.array([np.interp(grid, t, y) for t, y in lo])
        HI = np.array([np.interp(grid, t, y) for t, y in hi])
        tb = gap_time(grid, LO.mean(0), HI.mean(0), args.theta)
        # bootstrap over seed pairs
        boot = []
        rng = np.random.default_rng(0)
        for _ in range(args.nboot):
            il = rng.integers(0, len(LO), len(LO)); ih = rng.integers(0, len(HI), len(HI))
            boot.append(gap_time(grid, LO[il].mean(0), HI[ih].mean(0), args.theta))
        boot = np.array(boot); finite = boot[np.isfinite(boot)]
        clo, chi = (np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan))
        lab = f"({Nlo//1000}k,{nlo})→({Nhi//1000}k,{nhi})"
        labels.append(lab); tb_pts.append(tb); tb_lo.append(clo); tb_hi.append(chi)
        rows.append((lab, Nlo, nlo, Nhi, nhi, tb, clo, chi,
                     float(np.nanmax(HI.mean(0) / np.maximum(LO.mean(0), 1e-300)))))
        c = plt.cm.viridis(pi / max(len(PAIRS) - 1, 1))
        axr.plot(grid * 1e4, HI.mean(0) / np.maximum(LO.mean(0), 1e-300), color=c, lw=1.8,
                 label=f"{lab}: $t_b$={tb*1e4:.2f}e-4" if np.isfinite(tb) else f"{lab}: none")
        if np.isfinite(tb):
            axr.plot([tb * 1e4], [args.theta], "o", color=c, ms=7)

    axr.axhline(args.theta, color="k", ls="--", lw=0.8, label=f"$\\theta$={args.theta}")
    axr.axvline(LIT * 1e4, color="r", ls=":", lw=1.0, label=f"LDG target {LIT:.2e}")
    axr.set_xlabel(r"$t\ (\times10^{-4})$"); axr.set_ylabel(r"$S_{2n,4N_p}/S_{n,N_p}$")
    axr.set_title("(a) resolution-gap ratio (K=32 fixed domain, DG readout)")
    axr.set_xlim(0, 1.6); axr.legend(fontsize=7); axr.grid(alpha=0.3)

    x = np.arange(len(labels))
    yerr = [np.array(tb_pts) - np.array(tb_lo), np.array(tb_hi) - np.array(tb_pts)]
    axc.errorbar(x, np.array(tb_pts) * 1e4, yerr=np.array(yerr) * 1e4, fmt="o-",
                 color="C0", capsize=4, lw=1.6, label="$t_b$ (K=32 fixed)")
    axc.axhline(LIT * 1e4, color="r", ls=":", lw=1.2, label=f"LDG target {LIT:.2e}")
    axc.set_xticks(x); axc.set_xticklabels(labels, rotation=20, fontsize=7)
    axc.set_ylabel(r"$t_b\ (\times10^{-4})$")
    axc.set_title(f"(b) $t_b$ convergence ($\\theta$={args.theta})")
    axc.legend(fontsize=8); axc.grid(alpha=0.3)

    fig.suptitle("LDG-style $t_b$ convergence on K=32 fixed-domain setup "
                 "(N_p ladder 80k→5.12M, readout 80→640)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fd = os.path.join(args.run_dir, "figures"); os.makedirs(fd, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fd, "tb_K32_converge.{}".format(ext)), dpi=200, bbox_inches="tight")

    with open(os.path.join(args.run_dir, "tb_K32_converge.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pair", "N_lo", "n_lo", "N_hi", "n_hi",
                                       "t_b", "ci_lo", "ci_hi", "ratio_max"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], r[4], f"{r[5]:.4e}",
                        f"{r[6]:.4e}", f"{r[7]:.4e}", f"{r[8]:.3f}"])
    print(f"target LDG t_b ~ {LIT:.2e}")
    print("| pair | t_b | CI | ratio_max |")
    for r in rows:
        print(f"| {r[0]} | {r[5]:.3e} | [{r[6]:.2e},{r[7]:.2e}] | {r[8]:.2f} |")
    print(f"wrote {fd}/tb_K32_converge.pdf/.png and tb_K32_converge.csv")


if __name__ == "__main__":
    main()
