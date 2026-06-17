"""sens_Tcore_analyze.py -- particle core-collapse time T_core^part and its
sensitivity to (K, tau, q_window, N_p).

For each run, read the dense reconstruction-free R_q(t) from its diag CSV, fit
R_q(t)^2 = alpha_q - beta_q t on late windows, T_q = alpha_q/beta_q, and aggregate
(median, p10-p90 spread) over q in {0.1,0.2,0.3}, the 4 fit windows, and seeds.
This is the same definition as core_collapse_time/fit_core_collapse.py.

Emits a one-at-a-time sensitivity table (CSV + markdown) around the baseline.
"""
import os
import re
import csv
import glob
import argparse

import numpy as np

WINDOWS = [(4e-5, 9e-5), (5e-5, 1.0e-4), (6e-5, 1.1e-4), (7e-5, 1.2e-4)]
QSET = [0.1, 0.2, 0.3]
R2_MIN = 0.9
FAR = 3.0
BASE = dict(K=5, tau=2e-7, q=0.8, N=6400000)


def lin_fit(t, y):
    t, y = np.asarray(t, float), np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y) & (y > 0)
    t, y = t[ok], y[ok]
    if len(t) < 3:
        return np.nan, np.nan, np.nan
    A = np.vstack([np.ones_like(t), t]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = a + b * t
    ssr = float(np.sum((y - yhat) ** 2)); sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ssr / sst if sst > 0 else np.nan
    return float(a), float(b), float(r2)


def read_diag(path):
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r["t"]) for r in rows])
    Rq = {q: np.array([float(r[f"R_{q}"]) for r in rows]) for q in QSET
          if f"R_{q}" in rows[0]}
    return t, Rq


def valid_T_values(diag_csvs):
    """All valid T_q = alpha/beta over seeds x q x windows."""
    Ts = []
    for path in diag_csvs:
        t, Rq = read_diag(path)
        for q in QSET:
            if q not in Rq:
                continue
            y = Rq[q] ** 2
            for (w0, w1) in WINDOWS:
                m = (t >= w0) & (t <= w1)
                if m.sum() < 3:
                    continue
                a, b, r2 = lin_fit(t[m], y[m])
                if not np.isfinite(b) or b >= 0:        # need beta>0 (collapse): y=a-beta t
                    continue
                T = a / (-b)                            # T = alpha/beta, alpha=a, beta=-b>0
                if r2 >= R2_MIN and w1 < T <= FAR * w1:
                    Ts.append(T)
    return np.array(Ts)


def summarize(diag_csvs):
    Ts = valid_T_values(diag_csvs)
    if len(Ts) == 0:
        return dict(T_core=np.nan, p10=np.nan, p90=np.nan, n=0, rel_spread=np.nan)
    med = float(np.median(Ts)); p10 = float(np.percentile(Ts, 10)); p90 = float(np.percentile(Ts, 90))
    rs = (p90 - p10) / med if med > 0 else np.nan
    return dict(T_core=med, p10=p10, p90=p90, n=len(Ts), rel_spread=rs)


def parse_cfg(dirname):
    m = re.search(r"sens_K(\d+)_tau([0-9.e-]+)_q([0-9.]+)_N(\d+)_seed(\d+)", dirname)
    if not m:
        return None
    return dict(K=int(m.group(1)), tau=float(m.group(2)), q=float(m.group(3)),
                N=int(m.group(4)), seed=int(m.group(5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # group diag csvs by config (K,tau,q,N), collecting seeds
    groups = {}
    for d in sorted(glob.glob(os.path.join(args.run_dir, "sens_K*_seed*"))):
        cfg = parse_cfg(os.path.basename(d))
        if cfg is None:
            continue
        csvs = glob.glob(os.path.join(d, "diag_*.csv"))
        if not csvs:
            continue
        key = (cfg["K"], cfg["tau"], cfg["q"], cfg["N"])
        groups.setdefault(key, []).append(csvs[0])

    rows = []
    for key, csvs in sorted(groups.items()):
        K, tau, q, N = key
        s = summarize(csvs)
        axis = ("K" if (tau, q, N) == (BASE["tau"], BASE["q"], BASE["N"]) and K != BASE["K"] else
                "tau" if (K, q, N) == (BASE["K"], BASE["q"], BASE["N"]) and tau != BASE["tau"] else
                "q" if (K, tau, N) == (BASE["K"], BASE["tau"], BASE["N"]) and q != BASE["q"] else
                "N" if (K, tau, q) == (BASE["K"], BASE["tau"], BASE["q"]) and N != BASE["N"] else
                "baseline")
        rows.append(dict(axis=axis, K=K, tau=tau, q=q, N=N, n_seed=len(csvs), **s))

    out = args.out or os.path.join(args.run_dir, "sens_Tcore_table.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["axis", "K", "tau", "q", "N", "n_seed",
                                          "T_core", "p10", "p90", "rel_spread", "n"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # markdown to stdout
    base = next((r for r in rows if r["axis"] == "baseline"), None)
    print(f"\n=== T_core^part sensitivity (baseline K=5, tau=2e-7, q=0.8, N=6.4M) ===")
    if base:
        print(f"baseline T_core = {base['T_core']:.3e}  spread[{base['p10']:.3e},{base['p90']:.3e}]"
              f"  rel={base['rel_spread']:.2f}  (n={base['n']} fits, {base['n_seed']} seeds)")
    print(f"\n| axis | K | tau | q | N | T_core | [p10,p90] | rel_spread | n_fit |")
    print(f"|---|---|---|---|---|---|---|---|---|")
    order = {"baseline": 0, "K": 1, "tau": 2, "q": 3, "N": 4}
    for r in sorted(rows, key=lambda r: (order.get(r["axis"], 9), r["K"], -r["tau"], r["q"], -r["N"])):
        print(f"| {r['axis']} | {r['K']} | {r['tau']:.0e} | {r['q']} | {r['N']/1e6:.1f}M | "
              f"{r['T_core']:.3e} | [{r['p10']:.2e},{r['p90']:.2e}] | {r['rel_spread']:.2f} | {r['n']} |")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
