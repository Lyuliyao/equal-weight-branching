"""sens_finite_time.py -- finite-time core-collapse time (T_par, T_core) sensitivity
to (K, tau, q_window, N_p), LINEAR fit R_q^2 = alpha - beta t, T = alpha/beta.

IMPORTANT: the fit uses ONLY t <= t_cap (default 1.2e-4, the numerical blow-up);
post-blow-up data (where R_q hits the reconstruction floor and rebounds) is excluded.

  T_par : one linear fit over [t0, t_cap]; T = alpha/beta. (complete for every config)
  T_core: median over the four windows {[4,9],[5,10],[6,11],[7,12]}x1e-5 (all <= t_cap)
          of T_q passing the quality gate (beta>0, R^2>=0.9, window_end<T<=3 window_end).

R_q is reconstruction-free (ordered particle distances about the centroid), so this is
translation-invariant -- unaffected by any core wander.
"""
import os, re, csv, glob, argparse
import numpy as np

QSET = [0.1, 0.2, 0.3]
WIN = [(4e-5, 9e-5), (5e-5, 1.0e-4), (6e-5, 1.1e-4), (7e-5, 1.2e-4)]
BASE = (5, "2e-7", 0.8, 6400000)


def lf(t, y):
    A = np.vstack([np.ones_like(t), t]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = a + b * t; ssr = np.sum((y - yh) ** 2); sst = np.sum((y - np.mean(y)) ** 2)
    return a, b, (1 - ssr / sst if sst > 0 else np.nan)


def parse(n):
    m = re.search(r"sens_K(\d+)_tau([0-9.e-]+)_q([0-9.]+)_N(\d+)_seed(\d+)", n)
    return (int(m[1]), m[2], float(m[3]), int(m[4])) if m else None


def analyze(csvs, t0, tcap):
    Tpar, R2par, Tcore, ngate = [], [], [], 0
    for f in csvs:
        rows = list(csv.DictReader(open(f))); t = np.array([float(r["t"]) for r in rows])
        for q in QSET:
            y = np.array([float(r[f"R_{q}"]) for r in rows]) ** 2
            m = (t >= t0) & (t <= tcap)
            if m.sum() >= 4:
                a, b, r2 = lf(t[m], y[m])
                if b < 0:
                    Tpar.append(a / (-b)); R2par.append(r2)
            for (w0, w1) in WIN:
                if w1 > tcap + 1e-12:
                    continue
                mm = (t >= w0) & (t <= w1)
                if mm.sum() < 3:
                    continue
                a, b, r2 = lf(t[mm], y[mm])
                if b < 0:
                    T = a / (-b)
                    if r2 >= 0.9 and w1 < T <= 3 * w1:
                        Tcore.append(T); ngate += 1
    med = lambda x: float(np.median(x)) if len(x) else np.nan
    return med(Tpar), med(R2par), med(Tcore), ngate, len(csvs) * 12


def axis_of(k):
    K, tau, q, N = k
    if k == BASE: return "baseline"
    if (tau, q, N) == (BASE[1], BASE[2], BASE[3]): return "K"
    if (K, q, N) == (BASE[0], BASE[2], BASE[3]): return "tau"
    if (K, tau, N) == (BASE[0], BASE[1], BASE[3]): return "q"
    return "N"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--t0", type=float, default=3e-5)
    ap.add_argument("--t_cap", type=float, default=1.2e-4, help="exclude t > t_cap (blow-up)")
    args = ap.parse_args()

    groups = {}
    for d in sorted(glob.glob(os.path.join(args.run_dir, "sens_K*_seed*"))):
        c = parse(os.path.basename(d))
        if c:
            cs = glob.glob(os.path.join(d, "diag_*.csv"))
            if cs: groups.setdefault(c, []).append(cs[0])

    out = os.path.join(args.run_dir, "sens_finite_time.csv")
    rows = []
    for k in sorted(groups, key=lambda k: ({"baseline": 0, "K": 1, "tau": 2, "q": 3, "N": 4}[axis_of(k)], k[0], k[3])):
        Tp, R2p, Tc, ng, nt = analyze(groups[k], args.t0, args.t_cap)
        rows.append((axis_of(k), k[0], k[1], k[2], k[3], Tp, R2p, Tc, ng, nt))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["axis", "K", "tau", "q", "N", "T_par", "R2_finiteT", "T_core_windowed", "gate_pass", "n_fit"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], r[4], f"{r[5]:.4e}", f"{r[6]:.3f}", f"{r[7]:.4e}", r[8], r[9]])

    print(f"finite-time collapse, linear R_q^2=alpha-beta t, t in [{args.t0:.0e},{args.t_cap:.1e}] (no t>blow-up)")
    print("| axis | K | tau | q | N | T_par | R2(finiteT) | T_core(win) | gate |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        tc = f"{r[7]:.3e}" if np.isfinite(r[7]) else "--"
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]/1e6:.1f}M | {r[5]:.3e} | {r[6]:.2f} | {tc} | {r[8]}/{r[9]} |")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
