"""sens_Tcore_report.py -- T_core^part sensitivity report (table CSV + 4-panel figure).

Per config (K, tau, q_window, N_p), from the dense reconstruction-free R_q(t):
  - T_late: q-median (q in {0.1,0.2,0.3}) of the late-window [7e-5,1.2e-4]
    extrapolated collapse time T=alpha/beta (R_q^2=alpha-beta t); the physically
    admissible window (T > window end).
  - T window-range: min..max of the per-window T (all 4 windows, q in QSET) -- the
    convergence indicator (large range => window-sensitive => not converged).
  - collapse rate: median early-window [4e-5,9e-5] slope of R_0.2^2.
  - gate-pass: # fits passing the strict fit_core_collapse gate (r2>=0.9, T>w_end).

4-panel figure: T_late (+ window-range bar) vs each of K, tau, q, N, one-at-a-time
around the baseline, with the LDG/literature value ~1.21e-4 marked.
"""
import os, re, csv, glob, argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

WIN = [(4e-5, 9e-5), (5e-5, 1.0e-4), (6e-5, 1.1e-4), (7e-5, 1.2e-4)]
LATE = (7e-5, 1.2e-4)
QSET = [0.1, 0.2, 0.3]
BASE = (5, "2e-7", 0.8, 6400000)
LIT = 1.21e-4


def lf(t, y):
    A = np.vstack([np.ones_like(t), t]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = a + b * t; ssr = np.sum((y - yh) ** 2); sst = np.sum((y - np.mean(y)) ** 2)
    return a, b, (1 - ssr / sst if sst > 0 else np.nan)


def parse(n):
    m = re.search(r"sens_K(\d+)_tau([0-9.e-]+)_q([0-9.]+)_N(\d+)_seed(\d+)", n)
    return (int(m[1]), m[2], float(m[3]), int(m[4]), int(m[5])) if m else None


def analyze(csvs):
    T_late, T_all, rate = [], [], []
    ngate = 0
    for f in csvs:
        rows = list(csv.DictReader(open(f))); t = np.array([float(r["t"]) for r in rows])
        for q in QSET:
            y = np.array([float(r[f"R_{q}"]) for r in rows]) ** 2
            for (w0, w1) in WIN:
                m = (t >= w0) & (t <= w1)
                if m.sum() < 3:
                    continue
                a, b, r2 = lf(t[m], y[m])
                if b < 0:
                    T = a / (-b)
                    if 0 < T < 4e-4:
                        T_all.append(T)
                        if (w0, w1) == LATE:
                            T_late.append(T)
                    if r2 >= 0.9 and w1 < T <= 3 * w1:
                        ngate += 1
        y2 = np.array([float(r["R_0.2"]) for r in rows]) ** 2
        m = (t >= 4e-5) & (t <= 9e-5)
        _, b, _ = lf(t[m], y2[m]); rate.append(b)
    med = lambda x: float(np.median(x)) if len(x) else np.nan
    return dict(T_late=med(T_late),
                Tmin=float(np.min(T_all)) if T_all else np.nan,
                Tmax=float(np.max(T_all)) if T_all else np.nan,
                rate=med(rate), ngate=ngate, nfit=len(csvs) * 12)


def axis_of(k):
    K, tau, q, N = k
    if k == BASE: return "baseline"
    if (tau, q, N) == (BASE[1], BASE[2], BASE[3]): return "K"
    if (K, q, N) == (BASE[0], BASE[2], BASE[3]): return "tau"
    if (K, tau, N) == (BASE[0], BASE[1], BASE[3]): return "q"
    if (K, tau, q) == (BASE[0], BASE[1], BASE[2]): return "N"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.run_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    groups = {}
    for d in sorted(glob.glob(os.path.join(args.run_dir, "sens_K*_seed*"))):
        c = parse(os.path.basename(d))
        if not c: continue
        cs = glob.glob(os.path.join(d, "diag_*.csv"))
        if cs: groups.setdefault(c[:4], []).append(cs[0])

    rows = {k: analyze(v) for k, v in groups.items()}
    base = rows.get(BASE)

    # CSV
    csv_out = os.path.join(args.run_dir, "sens_Tcore_report.csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["axis", "K", "tau", "q", "N", "T_late", "Twin_min", "Twin_max", "collapse_rate_R02sq", "gate_pass", "n_fit"])
        for k in sorted(rows, key=lambda k: ({"baseline":0,"K":1,"tau":2,"q":3,"N":4}.get(axis_of(k),9), k[0], k[3])):
            a = rows[k]
            w.writerow([axis_of(k), k[0], k[1], k[2], k[3], f"{a['T_late']:.4e}", f"{a['Tmin']:.4e}", f"{a['Tmax']:.4e}", f"{a['rate']:.3f}", a["ngate"], a["nfit"]])

    # figure: 4 panels (vs K, tau, q, N)
    plt.rcParams.update({"font.size": 9})
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.2))
    specs = [("K", "K", lambda k: k[0], [5, 8, 10]),
             ("tau", r"$\tau$", lambda k: float(k[1]), [2e-7, 1e-7, 5e-8]),
             ("q", r"$q_{\rm window}$", lambda k: k[2], [0.5, 0.6, 0.7, 0.8]),
             ("N", r"$N_p$", lambda k: k[3], [1.6e6, 3.2e6, 6.4e6])]
    for ax, (axname, xlabel, getx, xs) in zip(axs, specs):
        pts = []
        for k, a in rows.items():
            if axis_of(k) in (axname, "baseline"):
                # include baseline in every panel at its x
                if axis_of(k) == "baseline" or axis_of(k) == axname:
                    pts.append((getx(k), a))
        pts = sorted(set((p[0], id(p[1])) for p in pts))  # de-dup baseline
        xv, tl, lo, hi = [], [], [], []
        seen = {}
        for k, a in rows.items():
            if axis_of(k) == axname or k == BASE:
                x = getx(k); seen[x] = a
        for x in sorted(seen):
            a = seen[x]; xv.append(x); tl.append(a["T_late"]); lo.append(a["Tmin"]); hi.append(a["Tmax"])
        xv = np.array(xv); tl = np.array(tl)
        ax.fill_between(range(len(xv)), lo, hi, color="tab:blue", alpha=0.15, label="window range")
        ax.plot(range(len(xv)), tl, "o-", color="tab:blue", label=r"$T_{\rm core}$ (late win)")
        ax.axhline(LIT, color="k", ls="--", lw=0.8, label="LDG/lit $1.21\\times10^{-4}$")
        ax.set_xticks(range(len(xv)))
        ax.set_xticklabels([f"{v:g}" if axname not in ("tau", "N") else (f"{v:.0e}" if axname == "tau" else f"{v/1e6:g}M") for v in xv])
        ax.set_xlabel(xlabel); ax.set_ylim(0.6e-4, 3.0e-4)
        if axname == "K": ax.set_ylabel(r"$T_{\rm core}^{\rm part}$")
        ax.set_title(f"vs {xlabel}", fontsize=9)
    axs[-1].legend(fontsize=6, loc="upper right")
    fig.suptitle(r"$T_{\rm core}^{\rm part}$ sensitivity (baseline K=5, $\tau$=2e-7, $q$=0.8, $N_p$=6.4M); bar = fit-window range",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    stem = os.path.join(out_dir, "sens_Tcore")
    fig.savefig(stem + ".pdf", bbox_inches="tight"); fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {csv_out}\nwrote {stem}.pdf/.png")
    if base:
        print(f"baseline T_core(late)={base['T_late']:.3e}, window-range[{base['Tmin']:.2e},{base['Tmax']:.2e}], rate={base['rate']:.2f}")


if __name__ == "__main__":
    main()
