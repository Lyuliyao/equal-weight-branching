"""
Core-collapse-time fit T_core from mass-quantile radii (core_collapse plan §3,7,8).
==================================================================================

Fit  R_q(t)^2 = alpha_q - beta_q t  on late-time windows; T_q = alpha_q/beta_q is
the radius-extrapolated collapse time.  Aggregate over q in {0.1,0.2,0.3} and over
fitting windows:  T_core = median,  spread = [p10,p90],  relative_spread.
Secondary: T_L2 from S^{-2} = a - b t, T_peak from peak^{-1} = a - b t.

Stability gates (§7) decide `valid_quote`:
  * each fit: beta>0, R^2>=R2_min, T_est>window_end, T_est<=far*window_end;
  * quantile+window consistency: relative_spread = (p90-p10)/median <= spread_max.

Applied FIRST to the fixed-flux LDG reference (Step 1).  Particle runs optional.

Usage:
  python fit_core_collapse.py --ldg_dir <core_collapse_run>/ldg \
     [--particle_root <core_collapse_run>/particle] \
     [--mass raw] [--outdir <core_collapse_run>]
"""
import os
import csv
import glob
import json
import argparse

import numpy as np

DEF_WINDOWS = [(4e-5, 9e-5), (5e-5, 1.0e-4), (6e-5, 1.1e-4), (7e-5, 1.2e-4)]
Q_CORE = [0.1, 0.2, 0.3]


def lin_fit(t, y):
    """Fit y = a + b t (least squares); return a, b, R^2 (NaN if <2 finite pts)."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y) & (y > 0)
    t, y = t[ok], y[ok]
    if len(t) < 3:
        return np.nan, np.nan, np.nan, len(t)
    A = np.vstack([np.ones_like(t), t]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    yhat = a + b * t
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return a, b, r2, len(t)


def fit_quantity(t, y, windows, r2_min=0.9, far=10.0):
    """Fit y(t)=a-b t (linear) per window; T = a/(-b) = -a/b.  Returns list of dicts."""
    rows = []
    for (w0, w1) in windows:
        sel = (t >= w0) & (t <= w1)
        a, b, r2, npts = lin_fit(t[sel], y[sel])
        beta = -b if np.isfinite(b) else np.nan        # y decreasing -> b<0 -> beta>0
        T = (-a / b) if (np.isfinite(b) and b != 0) else np.nan
        valid = bool(np.isfinite(T) and beta > 0 and np.isfinite(r2) and r2 >= r2_min
                     and T > w1 and T <= far * w1 and npts >= 5)
        reason = ""
        if not valid:
            if not (np.isfinite(beta) and beta > 0):
                reason = "beta<=0 (not collapsing)"
            elif not (np.isfinite(r2) and r2 >= r2_min):
                reason = f"R2<{r2_min} (curved)"
            elif not (np.isfinite(T) and T > w1):
                reason = "T<=window_end"
            elif T > far * w1:
                reason = "T absurdly far"
            elif npts < 5:
                reason = "too few points"
        rows.append(dict(window_start=w0, window_end=w1, alpha=a, beta=beta,
                         T_est=T, R2_fit=r2, n_points=npts, valid_fit=valid,
                         invalid_reason=reason))
    return rows


def aggregate(T_list, spread_max=0.25):
    Ts = np.array([t for t in T_list if np.isfinite(t)], float)
    if len(Ts) == 0:
        return dict(T_median=np.nan, T_p10=np.nan, T_p90=np.nan, T_min=np.nan,
                    T_max=np.nan, relative_spread=np.nan, n=0)
    med = float(np.median(Ts))
    p10, p90 = float(np.percentile(Ts, 10)), float(np.percentile(Ts, 90))
    rel = (p90 - p10) / med if med != 0 else np.nan
    return dict(T_median=med, T_p10=p10, T_p90=p90, T_min=float(Ts.min()),
                T_max=float(Ts.max()), relative_spread=rel, n=len(Ts))


def load_ldg(ldg_dir, mass):
    """Return {N: dict(t, R{q}, S_L2, peak)} from ldg_core_radii_N<N>.csv."""
    out = {}
    for f in sorted(glob.glob(os.path.join(ldg_dir, "N*", "ldg_core_radii_N*.csv"))):
        r = list(csv.DictReader(open(f)))
        if not r:
            continue
        N = int(float(r[0]["N"]))
        t = np.array([float(x["t"]) for x in r])
        d = dict(t=t, S_L2=np.array([float(x["S_L2"]) for x in r]),
                 peak=np.array([float(x["peak"]) for x in r]), R={})
        for q in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]:
            col = f"R_{q}_{mass}"
            if col in r[0]:
                d["R"][q] = np.array([float(x[col]) for x in r])
        out[N] = d
    return out


def load_particle(part_root, qs):
    """Return {(N): list of seed dicts} from particle diag_*.csv grouped by N."""
    out = {}
    for d in sorted(glob.glob(os.path.join(part_root, "*N*seed*"))):
        cs = glob.glob(os.path.join(d, "diag_*.csv"))
        if not cs:
            continue
        rows = list(csv.DictReader(open(cs[0])))
        if not rows:
            continue
        base = os.path.basename(d)
        try:
            N = int(base.split("_N")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        t = np.array([float(x["t"]) for x in rows])
        rec = dict(t=t, R={})
        for q in qs:
            col = f"R_{q}"
            if col in rows[0]:
                rec["R"][q] = np.array([float(x[col]) for x in rows])
        rec["S_dg"] = (np.array([float(x.get("S_dg_cross_160", "nan")) for x in rows])
                       if "S_dg_cross_160" in rows[0] else None)
        rec["peak"] = (np.array([float(x.get("peak_PK_u", "nan")) for x in rows])
                       if "peak_PK_u" in rows[0] else None)
        out.setdefault(N, []).append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ldg_dir", required=True)
    ap.add_argument("--particle_root", default=None)
    ap.add_argument("--mass", default="raw", choices=["raw", "clip"])
    ap.add_argument("--r2_min", type=float, default=0.9)
    ap.add_argument("--spread_max", type=float, default=0.25)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.ldg_dir))
    windows = DEF_WINDOWS

    all_rows = []
    summary_rows = []

    def add_method(method, resolution, t, Rdict, S, peak):
        # primary: T_core from R_q^2, q in Q_CORE
        T_all = []
        for q in Q_CORE:
            if q not in Rdict:
                continue
            y = Rdict[q] ** 2
            for fr in fit_quantity(t, y, windows, args.r2_min):
                fr.update(method=method, resolution=resolution, seed_group="ens",
                          q=q, quantity="Rq2")
                all_rows.append(fr)
                if fr["valid_fit"]:
                    T_all.append(fr["T_est"])
        agg = aggregate(T_all, args.spread_max)
        valid_quote = bool(agg["n"] >= 3 and np.isfinite(agg["relative_spread"])
                           and agg["relative_spread"] <= args.spread_max)
        summary_rows.append(dict(method=method, resolution=resolution, quantity="T_core",
                                 q_set="0.1,0.2,0.3", window_set=str(windows), **agg,
                                 valid_quote=valid_quote,
                                 decision=("quotable" if valid_quote else
                                           "unstable (spread>%.2f or <3 valid fits)" % args.spread_max)))
        # secondary: T_L2 from S^{-2}
        if S is not None:
            TL = []
            for fr in fit_quantity(t, 1.0 / np.maximum(S, 1e-300) ** 2, windows, args.r2_min):
                fr.update(method=method, resolution=resolution, seed_group="ens",
                          q=-1, quantity="Sminus2")
                all_rows.append(fr)
                if fr["valid_fit"]:
                    TL.append(fr["T_est"])
            aL = aggregate(TL, args.spread_max)
            summary_rows.append(dict(method=method, resolution=resolution, quantity="T_L2",
                                     q_set="-", window_set=str(windows), **aL,
                                     valid_quote=bool(aL["n"] >= 2), decision="secondary"))
        # secondary: T_peak from peak^{-1}
        if peak is not None:
            TP = []
            for fr in fit_quantity(t, 1.0 / np.maximum(peak, 1e-300), windows, args.r2_min):
                fr.update(method=method, resolution=resolution, seed_group="ens",
                          q=-2, quantity="peakminus1")
                all_rows.append(fr)
                if fr["valid_fit"]:
                    TP.append(fr["T_est"])
            aP = aggregate(TP, args.spread_max)
            summary_rows.append(dict(method=method, resolution=resolution, quantity="T_peak",
                                     q_set="-", window_set=str(windows), **aP,
                                     valid_quote=bool(aP["n"] >= 2), decision="secondary"))

    # ---- LDG ----
    ldg = load_ldg(args.ldg_dir, args.mass)
    for N in sorted(ldg):
        d = ldg[N]
        add_method("LDG", N, d["t"], d["R"], d["S_L2"], d["peak"])

    # ---- particle (optional; seed-mean radii) ----
    if args.particle_root and os.path.isdir(args.particle_root):
        part = load_particle(args.particle_root, Q_CORE)
        for N in sorted(part):
            seeds = part[N]
            t0 = seeds[0]["t"]
            Rdict = {}
            for q in Q_CORE:
                arrs = [s["R"][q] for s in seeds if q in s["R"] and len(s["R"][q]) == len(t0)]
                if arrs:
                    Rdict[q] = np.nanmean(np.vstack(arrs), axis=0)
            S = None; pk = None
            sdgs = [s["S_dg"] for s in seeds if s.get("S_dg") is not None and len(s["S_dg"]) == len(t0)]
            if sdgs:
                S = np.nanmean(np.vstack(sdgs), axis=0)
            pks = [s["peak"] for s in seeds if s.get("peak") is not None and len(s["peak"]) == len(t0)]
            if pks:
                pk = np.nanmean(np.vstack(pks), axis=0)
            add_method("particle", N, t0, Rdict, S, pk)

    # ---- write ----
    os.makedirs(outdir, exist_ok=True)
    acols = ["method", "resolution", "seed_group", "q", "window_start", "window_end",
             "quantity", "alpha", "beta", "T_est", "R2_fit", "n_points", "valid_fit",
             "invalid_reason"]
    with open(os.path.join(outdir, "core_fit_all.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=acols); w.writeheader()
        for r in all_rows:
            w.writerow({c: r.get(c) for c in acols})
    scols = ["method", "resolution", "quantity", "q_set", "window_set", "T_median",
             "T_p10", "T_p90", "T_min", "T_max", "relative_spread", "n", "valid_quote",
             "decision"]
    with open(os.path.join(outdir, "core_fit_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scols); w.writeheader()
        for r in summary_rows:
            w.writerow({c: r.get(c) for c in scols})
    json.dump(summary_rows, open(os.path.join(outdir, "core_fit_summary.json"), "w"),
              indent=2, default=str)

    # ---- console ----
    print(f"\n=== Core-collapse time T_core (mass={args.mass}, windows={windows}) ===")
    print(f"{'method':<9} {'N':>7} {'quantity':<11} {'T_median':>10} {'[p10':>10} "
          f"{'p90]':>10} {'rel_spread':>10} {'nfit':>5} {'quote?':>7}")
    for r in summary_rows:
        print(f"{r['method']:<9} {str(r['resolution']):>7} {r['quantity']:<11} "
              f"{r['T_median']:>10.3e} {r['T_p10']:>10.3e} {r['T_p90']:>10.3e} "
              f"{r['relative_spread']:>10.3f} {r['n']:>5} {str(r['valid_quote']):>7}")
    print(f"\nwrote core_fit_all.csv + core_fit_summary.csv/.json to {outdir}")
    return summary_rows


if __name__ == "__main__":
    main()
