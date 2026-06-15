"""
Core-collapse-time fit T_core from mass-quantile radii (core_collapse plan §3,7,8;
repaired per "Next experiments" Experiment A: seed common-grid, seed bootstrap, q-sets).
==================================================================================

Fit  R_q(t)^2 = alpha_q - beta_q t  on late-time windows; T_q = alpha_q/beta_q.
Aggregate over q in a q-set and over fitting windows: T_core = median, [p10,p90].
Secondary: T_L2 from S^{-2}, T_peak from peak^{-1}.

Particle seed handling (A1/A2): each seed's R_q(t) is interpolated to a COMMON grid
(dt) within its own time coverage; the seed-mean uses only times where at least
`min_seed_coverage` seeds are present, and a fitting window is valid for the particle
ENSEMBLE only if every sampled time in it has n_seed_eff >= min_seed_coverage.  The
quoted particle uncertainty is a SEED BOOTSTRAP (resample seeds with replacement),
not only the q/window spread.

q-set sensitivity (A3): pass several --q_sets; each gets its own summary so a single
particle T_core is quoted only if it is q-set-stable.

Stability gates (§7): per fit beta>0, R^2>=r2_min, T>window_end, T<=far*window_end;
quotable only if n>=3 valid fits AND relative_spread<=spread_max.

Usage:
  python fit_core_collapse.py --ldg_dir <run>/ldg [--particle_root <run>/particle] \
     [--mass raw] [--q_sets "0.1,0.2,0.3" "0.2,0.3" "0.2" "0.3"] \
     [--bootstrap_seeds 1000] [--min_seed_coverage 3] [--common_grid_dt 1e-6] \
     [--outdir <run>]
"""
import os
import csv
import glob
import json
import argparse

import numpy as np

DEF_WINDOWS = [(4e-5, 9e-5), (5e-5, 1.0e-4), (6e-5, 1.1e-4), (7e-5, 1.2e-4)]
DEF_QSETS = [[0.1, 0.2, 0.3], [0.2, 0.3], [0.2], [0.3]]
ALL_Q = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]


# --------------------------------------------------------------------------- fits
def lin_fit(t, y):
    t = np.asarray(t, float); y = np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y) & (y > 0)
    t, y = t[ok], y[ok]
    if len(t) < 3:
        return np.nan, np.nan, np.nan, len(t)
    A = np.vstack([np.ones_like(t), t]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    yhat = a + b * t
    ss_res = float(np.sum((y - yhat) ** 2)); ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return a, b, r2, len(t)


def fit_quantity(t, y, windows, r2_min=0.9, far=10.0, coverage=None, min_cov=1):
    """Fit y(t)=a-b t per window; T=-a/b.  If `coverage` (n_seed_eff on the same grid
    as t) is given, a window is skipped unless coverage>=min_cov over all its points."""
    rows = []
    for (w0, w1) in windows:
        sel = (t >= w0) & (t <= w1)
        if coverage is not None:
            covsel = coverage[sel]
            if covsel.size == 0 or np.any(covsel < min_cov):
                rows.append(dict(window_start=w0, window_end=w1, alpha=np.nan, beta=np.nan,
                                 T_est=np.nan, R2_fit=np.nan, n_points=int(sel.sum()),
                                 valid_fit=False, invalid_reason="seed coverage<min"))
                continue
        a, b, r2, npts = lin_fit(t[sel], y[sel])
        beta = -b if np.isfinite(b) else np.nan
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
        rows.append(dict(window_start=w0, window_end=w1, alpha=a, beta=beta, T_est=T,
                         R2_fit=r2, n_points=npts, valid_fit=valid, invalid_reason=reason))
    return rows


def aggregate(T_list):
    Ts = np.array([t for t in T_list if np.isfinite(t)], float)
    if len(Ts) == 0:
        return dict(T_median=np.nan, T_p10=np.nan, T_p90=np.nan, T_min=np.nan,
                    T_max=np.nan, relative_spread=np.nan, n=0)
    med = float(np.median(Ts)); p10, p90 = float(np.percentile(Ts, 10)), float(np.percentile(Ts, 90))
    return dict(T_median=med, T_p10=p10, T_p90=p90, T_min=float(Ts.min()),
                T_max=float(Ts.max()), relative_spread=(p90 - p10) / med if med else np.nan, n=len(Ts))


def core_T_from_curves(t, Rdict, qset, windows, r2_min, coverage=None, min_cov=1):
    """Return (list of valid T_est, all fit rows) for a q-set given R_q(t) curves."""
    T_all, rows = [], []
    for q in qset:
        if q not in Rdict:
            continue
        for fr in fit_quantity(t, Rdict[q] ** 2, windows, r2_min, coverage=coverage, min_cov=min_cov):
            fr = dict(fr, q=q); rows.append(fr)
            if fr["valid_fit"]:
                T_all.append(fr["T_est"])
    return T_all, rows


# --------------------------------------------------------------------------- IO
def load_ldg(ldg_dir, mass):
    out = {}
    for f in sorted(glob.glob(os.path.join(ldg_dir, "N*", "ldg_core_radii_N*.csv"))):
        r = list(csv.DictReader(open(f)))
        if not r:
            continue
        N = int(float(r[0]["N"]))
        t = np.array([float(x["t"]) for x in r])
        d = dict(t=t, S_L2=np.array([float(x["S_L2"]) for x in r]),
                 peak=np.array([float(x["peak"]) for x in r]), R={})
        for q in ALL_Q:
            c = f"R_{q}_{mass}"
            if c in r[0]:
                d["R"][q] = np.array([float(x[c]) for x in r])
        out[N] = d
    return out


def load_particle_seeds(part_root, qs, config="current_fourier"):
    """{N: [seed dicts]} with per-seed t, R_q, S_dg, peak (NO length filtering)."""
    out = {}
    for d in sorted(glob.glob(os.path.join(part_root, f"{config}_N*seed*"))):
        cs = glob.glob(os.path.join(d, "diag_*.csv"))
        if not cs:
            continue
        rows = list(csv.DictReader(open(cs[0])))
        if not rows:
            continue
        try:
            N = int(os.path.basename(d).split("_N")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        t = np.array([float(x["t"]) for x in rows])
        rec = dict(t=t, R={})
        for q in qs:
            c = f"R_{q}"
            if c in rows[0]:
                rec["R"][q] = np.array([float(x[c]) for x in rows])
        rec["S_dg"] = (np.array([float(x.get("S_dg_cross_160", "nan")) for x in rows])
                       if "S_dg_cross_160" in rows[0] else None)
        rec["peak"] = (np.array([float(x.get("peak_PK_u", "nan")) for x in rows])
                       if "peak_PK_u" in rows[0] else None)
        out.setdefault(N, []).append(rec)
    return out


def seed_mean_common_grid(seeds, qs, dt, min_cov):
    """Interpolate each seed to a common grid (within its own coverage), return
    t_grid, {q: seed-mean R}, {q: (n_seed,n_grid) array for bootstrap}, n_seed_eff(t)."""
    t_end = max(s["t"][-1] for s in seeds)
    grid = np.arange(0.0, t_end + 0.5 * dt, dt)
    per_q = {}
    for q in qs:
        mats = []
        for s in seeds:
            if q not in s["R"]:
                continue
            y = np.interp(grid, s["t"], s["R"][q], left=s["R"][q][0], right=np.nan)
            y[grid > s["t"][-1] + 1e-12] = np.nan
            mats.append(y)
        per_q[q] = np.vstack(mats) if mats else None
    # n_seed_eff(t): seeds present (use q=min available as proxy; all q share seed times)
    ref = next((per_q[q] for q in qs if per_q.get(q) is not None), None)
    n_eff = np.sum(np.isfinite(ref), axis=0) if ref is not None else np.zeros(len(grid))
    Rmean = {q: (np.nanmean(per_q[q], axis=0) if per_q.get(q) is not None else None) for q in qs}
    return grid, Rmean, per_q, n_eff


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ldg_dir", required=True)
    ap.add_argument("--particle_root", default=None)
    ap.add_argument("--particle_config", default="current_fourier")
    ap.add_argument("--mass", default="raw", choices=["raw", "clip"])
    ap.add_argument("--q_sets", nargs="+", default=None,
                    help='e.g. "0.1,0.2,0.3" "0.2,0.3" "0.2" "0.3"')
    ap.add_argument("--r2_min", type=float, default=0.9)
    ap.add_argument("--spread_max", type=float, default=0.25)
    ap.add_argument("--common_grid_dt", type=float, default=1.0e-6)
    ap.add_argument("--min_seed_coverage", type=int, default=3)
    ap.add_argument("--bootstrap_seeds", type=int, default=1000)
    ap.add_argument("--boot_rng", type=int, default=12345)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--windows", default=None,
                    help='override fit windows, e.g. "7e-5:1.15e-4,8e-5:1.18e-4,9e-5:1.2e-4"')
    args = ap.parse_args()
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.ldg_dir))
    os.makedirs(outdir, exist_ok=True)
    windows = (DEF_WINDOWS if not args.windows
               else [tuple(float(x) for x in w.split(":")) for w in args.windows.split(",")])
    qsets = ([[float(x) for x in s.split(",")] for s in args.q_sets] if args.q_sets else DEF_QSETS)
    rng = np.random.default_rng(args.boot_rng)

    all_rows, summary_rows, boot_rows = [], [], []

    def secondary(method, res, t, S, peak, cov=None):
        for qty, y in (("T_L2", (1.0 / np.maximum(S, 1e-300) ** 2) if S is not None else None),
                       ("T_peak", (1.0 / np.maximum(peak, 1e-300)) if peak is not None else None)):
            if y is None:
                continue
            Ts = []
            for fr in fit_quantity(t, y, windows, args.r2_min, coverage=cov, min_cov=args.min_seed_coverage):
                all_rows.append(dict(fr, method=method, resolution=res, seed_group="ens",
                                     q=(-1 if qty == "T_L2" else -2), quantity=qty.replace("T_", "")))
                if fr["valid_fit"]:
                    Ts.append(fr["T_est"])
            ag = aggregate(Ts)
            summary_rows.append(dict(method=method, resolution=res, quantity=qty, q_set="-",
                                     window_set="def", **ag, valid_quote=bool(ag["n"] >= 2),
                                     decision="secondary"))

    # ---------- LDG (no seeds; coverage None) ----------
    ldg = load_ldg(args.ldg_dir, args.mass)
    for N in sorted(ldg):
        d = ldg[N]
        for qset in qsets:
            T_all, rows = core_T_from_curves(d["t"], d["R"], qset, windows, args.r2_min)
            for fr in rows:
                all_rows.append(dict(fr, method="LDG", resolution=N, seed_group="-",
                                     quantity="Rq2", q_set=",".join(map(str, qset))))
            ag = aggregate(T_all)
            n_tot = len(qset) * len(windows)
            vq = bool(ag["n"] >= 3 and np.isfinite(ag["relative_spread"]) and ag["relative_spread"] <= args.spread_max)
            summary_rows.append(dict(method="LDG", resolution=N, quantity="T_core",
                                     q_set=",".join(map(str, qset)), window_set=f"{ag['n']}/{n_tot} valid",
                                     **ag, valid_quote=vq,
                                     decision=("quotable" if vq else "unstable/insufficient")))
        secondary("LDG", N, d["t"], d["S_L2"], d["peak"])

    # ---------- particle (common grid + seed bootstrap) ----------
    if args.particle_root and os.path.isdir(args.particle_root):
        part = load_particle_seeds(args.particle_root, ALL_Q, args.particle_config)
        for N in sorted(part):
            seeds = part[N]
            grid, Rmean, per_q, n_eff = seed_mean_common_grid(seeds, ALL_Q, args.common_grid_dt,
                                                              args.min_seed_coverage)
            # secondary on seed-mean DG (interpolate each seed to the common grid first)
            Smean = None
            sdg_seeds = [s for s in seeds if s.get("S_dg") is not None]
            if sdg_seeds:
                mats = []
                for s in sdg_seeds:
                    y = np.interp(grid, s["t"], s["S_dg"], left=s["S_dg"][0], right=np.nan)
                    y[grid > s["t"][-1] + 1e-12] = np.nan
                    mats.append(y)
                Smean = np.nanmean(np.vstack(mats), axis=0)
            for qset in qsets:
                T_all, rows = core_T_from_curves(grid, Rmean, qset, windows, args.r2_min,
                                                 coverage=n_eff, min_cov=args.min_seed_coverage)
                for fr in rows:
                    all_rows.append(dict(fr, method="particle", resolution=N, seed_group="mean",
                                         quantity="Rq2", q_set=",".join(map(str, qset))))
                ag = aggregate(T_all)
                # SEED BOOTSTRAP: resample seeds, rebuild mean, refit T_core median
                nb = args.bootstrap_seeds
                boots = []
                nseed = len(seeds)
                min_cov_eff = []
                for _ in range(nb):
                    idx = rng.integers(0, nseed, nseed)
                    bseeds = [seeds[k] for k in idx]
                    g2, Rm2, _, ne2 = seed_mean_common_grid(bseeds, qset, args.common_grid_dt,
                                                            args.min_seed_coverage)
                    Tb, _ = core_T_from_curves(g2, Rm2, qset, windows, args.r2_min,
                                               coverage=ne2, min_cov=args.min_seed_coverage)
                    if Tb:
                        boots.append(float(np.median(Tb)))
                boots = np.array(boots, float)
                bmed = float(np.median(boots)) if boots.size else np.nan
                bp10 = float(np.percentile(boots, 5)) if boots.size else np.nan
                bp90 = float(np.percentile(boots, 95)) if boots.size else np.nan
                # min n_seed_eff over the fitted windows
                wmask = (grid >= windows[0][0]) & (grid <= windows[-1][1])
                nse_min = int(np.min(n_eff[wmask])) if np.any(wmask) else 0
                vq = bool(ag["n"] >= 3 and np.isfinite(ag["relative_spread"]) and ag["relative_spread"] <= args.spread_max)
                summary_rows.append(dict(method="particle", resolution=N, quantity="T_core",
                                         q_set=",".join(map(str, qset)),
                                         window_set=f"{ag['n']}/{len(qset)*len(windows)} valid",
                                         **ag, valid_quote=vq,
                                         decision=("quotable" if vq else "unstable/insufficient")))
                boot_rows.append(dict(method="particle", resolution=N, q_set=",".join(map(str, qset)),
                                      n_seed=nseed, n_boot_valid=int(boots.size),
                                      T_qwindow_median=ag["T_median"], T_qwindow_p10=ag["T_p10"],
                                      T_qwindow_p90=ag["T_p90"], T_boot_median=bmed,
                                      T_boot_p10=bp10, T_boot_p90=bp90,
                                      n_seed_eff_min_in_windows=nse_min))
            secondary("particle", N, grid, Smean, None, cov=n_eff)

    # ---------- write ----------
    acols = ["method", "resolution", "seed_group", "q", "q_set", "window_start", "window_end",
             "quantity", "alpha", "beta", "T_est", "R2_fit", "n_points", "valid_fit", "invalid_reason"]
    with open(os.path.join(outdir, "core_fit_all.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=acols); w.writeheader()
        for r in all_rows:
            w.writerow({c: r.get(c) for c in acols})
    scols = ["method", "resolution", "quantity", "q_set", "window_set", "T_median", "T_p10",
             "T_p90", "T_min", "T_max", "relative_spread", "n", "valid_quote", "decision"]
    with open(os.path.join(outdir, "core_fit_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scols); w.writeheader()
        for r in summary_rows:
            w.writerow({c: r.get(c) for c in scols})
    json.dump(summary_rows, open(os.path.join(outdir, "core_fit_summary.json"), "w"),
              indent=2, default=str)
    bcols = ["method", "resolution", "q_set", "n_seed", "n_boot_valid", "T_qwindow_median",
             "T_qwindow_p10", "T_qwindow_p90", "T_boot_median", "T_boot_p10", "T_boot_p90",
             "n_seed_eff_min_in_windows"]
    with open(os.path.join(outdir, "core_fit_bootstrap.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bcols); w.writeheader()
        for r in boot_rows:
            w.writerow({c: r.get(c) for c in bcols})
    json.dump(boot_rows, open(os.path.join(outdir, "core_fit_bootstrap.json"), "w"),
              indent=2, default=str)

    # ---------- console ----------
    print(f"\n=== T_core (mass={args.mass}, min_seed_cov={args.min_seed_coverage}, "
          f"bootstrap={args.bootstrap_seeds}) ===")
    print(f"{'method':<9} {'N':>7} {'q_set':<13} {'T_median':>10} {'rel_spr':>8} "
          f"{'valid':>7} {'quote?':>6}")
    for r in summary_rows:
        if r["quantity"] != "T_core":
            continue
        print(f"{r['method']:<9} {str(r['resolution']):>7} {r['q_set']:<13} "
              f"{r['T_median']:>10.3e} {r['relative_spread']:>8.3f} {r['window_set']:>7} "
              f"{str(r['valid_quote']):>6}")
    if boot_rows:
        print("\n=== particle SEED BOOTSTRAP T_core ===")
        print(f"{'N':>7} {'q_set':<13} {'T_boot_med':>11} {'[p5':>10} {'p95]':>10} {'nse_min':>7}")
        for r in boot_rows:
            print(f"{str(r['resolution']):>7} {r['q_set']:<13} {r['T_boot_median']:>11.3e} "
                  f"{r['T_boot_p10']:>10.3e} {r['T_boot_p90']:>10.3e} "
                  f"{r['n_seed_eff_min_in_windows']:>7}")
    print(f"\nwrote core_fit_all/summary/bootstrap to {outdir}")
    return summary_rows, boot_rows


if __name__ == "__main__":
    main()
