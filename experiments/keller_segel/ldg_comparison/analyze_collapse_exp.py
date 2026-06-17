"""analyze_collapse_exp.py -- EXPONENTIAL core-collapse rate lambda and its
sensitivity to (K, tau, q_window, N_p).

The mass-quantile radius collapses as R_q(t) ~ exp(-lambda t) (log-linear), NOT
as a finite-time R_q^2 = alpha - beta t.  We therefore fit  log R_q = c - lambda t
on a PRE-blow-up window [t_lo, t_hi] (default [2e-5, 1.2e-4]; data past the ~1.2e-4
numerical blow-up is excluded), report lambda (median over q in {0.1,0.2,0.3} and
seeds) and the fit R^2, and compare to the finite-time R^2 as a model check.

Outputs a lambda-sensitivity table (CSV + stdout) and a baseline log-R_q figure.
"""
import os, re, csv, glob, argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

QSET = [0.1, 0.2, 0.3]
BASE = (5, "2e-7", 0.8, 6400000)
T_LO, T_HI = 2e-5, 1.2e-4          # pre-blow-up fit window


def lf(t, y):
    A = np.vstack([np.ones_like(t), t]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = a + b * t; ssr = np.sum((y - yh) ** 2); sst = np.sum((y - np.mean(y)) ** 2)
    return a, b, (1 - ssr / sst if sst > 0 else np.nan)


def parse(n):
    m = re.search(r"sens_K(\d+)_tau([0-9.e-]+)_q([0-9.]+)_N(\d+)_seed(\d+)", n)
    return (int(m[1]), m[2], float(m[3]), int(m[4]), int(m[5])) if m else None


def fit_lambda(csvs, t_lo=T_LO, t_hi=T_HI):
    lams, r2e, r2p = [], [], []
    for f in csvs:
        rows = list(csv.DictReader(open(f))); t = np.array([float(r["t"]) for r in rows])
        for q in QSET:
            R = np.array([float(r[f"R_{q}"]) for r in rows])
            m = (t >= t_lo) & (t <= t_hi) & (R > 0)
            if m.sum() < 4:
                continue
            _, b, r2 = lf(t[m], np.log(R[m]))           # exponential
            if b < 0:
                lams.append(-b); r2e.append(r2)
            _, bp, r2pw = lf(t[m], R[m] ** 2)           # finite-time (model check)
            r2p.append(r2pw)
    med = lambda x: float(np.median(x)) if len(x) else np.nan
    return dict(lam=med(lams), r2_exp=med(r2e), r2_pow=med(r2p), n=len(csvs))


def axis_of(k):
    K, tau, q, N = k
    if k == BASE: return "baseline"
    if (tau, q, N) == (BASE[1], BASE[2], BASE[3]): return "K"
    if (K, q, N) == (BASE[0], BASE[2], BASE[3]): return "tau"
    if (K, tau, N) == (BASE[0], BASE[1], BASE[3]): return "q"
    if (K, tau, q) == (BASE[0], BASE[1], BASE[2]): return "N"
    return "?"


def baseline_figure(base_csv, out):
    rows = list(csv.DictReader(open(base_csv))); t = np.array([float(r["t"]) for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    qs = [0.1, 0.2, 0.3, 0.5, 0.8]
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(qs)))
    for q, c in zip(qs, cols):
        R = np.array([float(r[f"R_{q}"]) for r in rows])
        ax.semilogy(t * 1e4, R, ".", color=c, ms=3, alpha=0.6)
        m = (t >= T_LO) & (t <= T_HI) & (R > 0)
        a, b, r2 = lf(t[m], np.log(R[m]))
        tt = np.linspace(T_LO, T_HI, 50)
        ax.semilogy(tt * 1e4, np.exp(a + b * tt), "-", color=c, lw=1.6,
                    label=rf"$R_{{{q:g}}}$: $\lambda$={-b:.2e}, $R^2$={r2:.2f}")
    ax.axvspan(T_HI * 1e4, t.max() * 1e4, color="gray", alpha=0.15)
    ax.text((T_HI + (t.max() - T_HI) * 0.5) * 1e4, ax.get_ylim()[1] * 0.5,
            "post-blow-up\n(excluded)", ha="center", va="top", fontsize=8, color="dimgray")
    ax.set_xlabel(r"$t\ (\times 10^{-4})$"); ax.set_ylabel(r"$R_q(t)$ (log)")
    ax.set_title(r"Exponential core collapse $R_q\sim e^{-\lambda t}$ (baseline K=5,$\tau$=2e-7,q=0.8,N=6.4M)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    out_dir = os.path.join(args.run_dir, "figures"); os.makedirs(out_dir, exist_ok=True)

    groups = {}
    for d in sorted(glob.glob(os.path.join(args.run_dir, "sens_K*_seed*"))):
        c = parse(os.path.basename(d))
        if not c: continue
        cs = glob.glob(os.path.join(d, "diag_*.csv"))
        if cs: groups.setdefault(c[:4], []).append(cs[0])

    rows = {k: fit_lambda(v) for k, v in groups.items()}
    csv_out = os.path.join(args.run_dir, "collapse_lambda.csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["axis", "K", "tau", "q", "N", "lambda", "R2_exp", "R2_finiteT", "n_seed"])
        for k in sorted(rows, key=lambda k: ({"baseline":0,"K":1,"tau":2,"q":3,"N":4}.get(axis_of(k),9), k[0], k[3])):
            a = rows[k]
            w.writerow([axis_of(k), k[0], k[1], k[2], k[3], f"{a['lam']:.4e}", f"{a['r2_exp']:.3f}", f"{a['r2_pow']:.3f}", a["n"]])

    print(f"\n=== exponential collapse rate lambda (fit log R_q = c - lambda t on [{T_LO:.0e},{T_HI:.0e}]) ===")
    print("| axis | K | tau | q | N | lambda | R2(exp) | R2(finite-T) |")
    print("|---|---|---|---|---|---|---|---|")
    for k in sorted(rows, key=lambda k: ({"baseline":0,"K":1,"tau":2,"q":3,"N":4}.get(axis_of(k),9), k[0], k[3])):
        a = rows[k]
        print(f"| {axis_of(k)} | {k[0]} | {k[1]} | {k[2]} | {k[3]/1e6:.1f}M | "
              f"{a['lam']:.3e} | {a['r2_exp']:.2f} | {a['r2_pow']:.2f} |")

    bk = os.path.join(args.run_dir, "sens_K5_tau2e-7_q0.8_N6400000_seed0")
    bcsv = glob.glob(os.path.join(bk, "diag_*.csv"))
    if bcsv:
        baseline_figure(bcsv[0], os.path.join(out_dir, "Rq_exp_baseline"))
        print(f"\nwrote {csv_out}\nwrote {out_dir}/Rq_exp_baseline.pdf/.png")


if __name__ == "__main__":
    main()
