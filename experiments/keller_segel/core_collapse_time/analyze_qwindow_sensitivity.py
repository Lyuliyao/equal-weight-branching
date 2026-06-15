"""
q_window dynamics sensitivity analysis ("Next experiments" Experiment B3/B4).
=============================================================================
For each q_window in the sweep, compute (reusing fit_core_collapse machinery):
  * particle T_core (common-grid seed mean + seed bootstrap CI), q-set {0.1,0.2,0.3};
  * window-resolution diagnostics from the diag CSVs (seed-mean at t_match):
    outside_v_frac, R_0.2/(L/K), R_0.1/(L/K), drift_cfl_solver_field, L(t);
  * abort/final time + fraction reaching T.
Compares each q_window's T_core to the LDG reference (1.215e-4) to see whether a more
core-local window reduces the offset while keeping outside_v_frac small + stable (B4).

Usage:
  python analyze_qwindow_sensitivity.py --sweep_dir <qwindow_run> \
      --ldg_T 1.215e-4 [--t_match 1.0e-4] [--bootstrap_seeds 1000]
"""
import os
import csv
import glob
import json
import argparse

import numpy as np

import fit_core_collapse as F   # reuse seed common-grid + fits + bootstrap

QSET = [0.1, 0.2, 0.3]


def diag_seedmean_at(dirs, col, t_match, dt=1e-6):
    """seed-mean of a diag column at t_match via common-grid interpolation."""
    vals = []
    for d in dirs:
        cs = glob.glob(os.path.join(d, "diag_*.csv"))
        if not cs:
            continue
        rows = list(csv.DictReader(open(cs[0])))
        if not rows or col not in rows[0]:
            continue
        t = np.array([float(x["t"]) for x in rows])
        try:
            y = np.array([float(x[col]) for x in rows])
        except (ValueError, TypeError):
            continue
        idx = np.searchsorted(t, t_match)
        if idx < len(t):
            vals.append(y[idx])
    return (float(np.nanmean(vals)), float(np.nanstd(vals))) if vals else (np.nan, np.nan)


def final_times(dirs):
    out = []
    for d in dirs:
        cs = glob.glob(os.path.join(d, "diag_*.csv"))
        if not cs:
            continue
        rows = list(csv.DictReader(open(cs[0])))
        if rows:
            out.append(float(rows[-1]["t"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--ldg_T", type=float, default=1.215e-4)
    ap.add_argument("--t_match", type=float, default=1.0e-4)
    ap.add_argument("--bootstrap_seeds", type=int, default=1000)
    ap.add_argument("--min_seed_coverage", type=int, default=3)
    ap.add_argument("--K", type=int, default=10)
    args = ap.parse_args()
    rng = np.random.default_rng(2024)

    # discover (qwin, N) groups
    groups = {}
    for d in sorted(glob.glob(os.path.join(args.sweep_dir, "qwin*_N*_seed*"))):
        base = os.path.basename(d)
        qw = base.split("qwin")[1].split("_N")[0]
        N = base.split("_N")[1].split("_seed")[0]
        groups.setdefault((qw, N), []).append(d)

    rows = []
    for (qw, N), dirs in sorted(groups.items(), key=lambda kv: (float(kv[0][0]), int(kv[0][1]))):
        # particle T_core via fit_core_collapse seed machinery
        seeds = []
        for d in dirs:
            cs = glob.glob(os.path.join(d, "diag_*.csv"))
            if not cs:
                continue
            drows = list(csv.DictReader(open(cs[0])))
            if not drows:
                continue
            t = np.array([float(x["t"]) for x in drows])
            rec = dict(t=t, R={})
            for q in F.ALL_Q:
                c = f"R_{q}"
                if c in drows[0]:
                    rec["R"][q] = np.array([float(x[c]) for x in drows])
            seeds.append(rec)
        if not seeds:
            continue
        grid, Rmean, per_q, n_eff = F.seed_mean_common_grid(seeds, F.ALL_Q, 1e-6, args.min_seed_coverage)
        T_all, _ = F.core_T_from_curves(grid, Rmean, QSET, F.DEF_WINDOWS, 0.9,
                                        coverage=n_eff, min_cov=args.min_seed_coverage)
        ag = F.aggregate(T_all)
        # seed bootstrap
        boots = []
        for _ in range(args.bootstrap_seeds):
            idx = rng.integers(0, len(seeds), len(seeds))
            g2, Rm2, _, ne2 = F.seed_mean_common_grid([seeds[k] for k in idx], QSET, 1e-6,
                                                      args.min_seed_coverage)
            Tb, _ = F.core_T_from_curves(g2, Rm2, QSET, F.DEF_WINDOWS, 0.9,
                                         coverage=ne2, min_cov=args.min_seed_coverage)
            if Tb:
                boots.append(float(np.median(Tb)))
        boots = np.array(boots, float)
        tb_med = float(np.median(boots)) if boots.size else np.nan
        tb_lo = float(np.percentile(boots, 5)) if boots.size else np.nan
        tb_hi = float(np.percentile(boots, 95)) if boots.size else np.nan
        # window-resolution diagnostics
        ovf = diag_seedmean_at(dirs, "outside_v_frac", args.t_match)
        L_tm = diag_seedmean_at(dirs, "L", args.t_match)
        r02 = diag_seedmean_at(dirs, "R_0.2", args.t_match)
        r01 = diag_seedmean_at(dirs, "R_0.1", args.t_match)
        cfl = diag_seedmean_at(dirs, "drift_cfl_solver_field", args.t_match)
        heff = (L_tm[0] / args.K) if np.isfinite(L_tm[0]) else np.nan
        fts = final_times(dirs)
        frac_T = float(np.mean([t >= 1.99e-4 for t in fts])) if fts else np.nan
        contains_ldg = bool(np.isfinite(tb_lo) and np.isfinite(tb_hi) and tb_lo <= args.ldg_T <= tb_hi)
        rows.append(dict(q_window=qw, N=N, T_core_qwin=ag["T_median"],
                         T_core_boot=tb_med, boot_p5=tb_lo, boot_p95=tb_hi,
                         offset_vs_ldg=(tb_med / args.ldg_T - 1.0) if np.isfinite(tb_med) else np.nan,
                         ci_contains_ldg=contains_ldg,
                         outside_v_frac=ovf[0], R02_over_heff=(r02[0] / heff if heff else np.nan),
                         R01_over_heff=(r01[0] / heff if heff else np.nan),
                         L_at_tm=L_tm[0], drift_cfl=cfl[0], frac_reached_T=frac_T,
                         t_end_mean=float(np.mean(fts)) if fts else np.nan, n_valid_fits=ag["n"]))

    cols = ["q_window", "N", "T_core_qwin", "T_core_boot", "boot_p5", "boot_p95",
            "offset_vs_ldg", "ci_contains_ldg", "outside_v_frac", "R02_over_heff",
            "R01_over_heff", "L_at_tm", "drift_cfl", "frac_reached_T", "t_end_mean", "n_valid_fits"]
    with open(os.path.join(args.sweep_dir, "qwindow_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    json.dump(rows, open(os.path.join(args.sweep_dir, "qwindow_summary.json"), "w"),
              indent=2, default=str)

    print(f"\n=== q_window sensitivity (LDG T_core ref = {args.ldg_T:.3e}) ===")
    print(f"{'qwin':>5} {'N':>7} {'T_boot':>10} {'[p5':>10} {'p95]':>10} {'off%':>6} "
          f"{'ldg?':>5} {'out_v':>7} {'R0.2/h':>7} {'cfl':>6} {'fracT':>6}")
    for r in rows:
        print(f"{r['q_window']:>5} {r['N']:>7} {r['T_core_boot']:>10.3e} {r['boot_p5']:>10.3e} "
              f"{r['boot_p95']:>10.3e} {100*r['offset_vs_ldg']:>5.1f} {str(r['ci_contains_ldg']):>5} "
              f"{r['outside_v_frac']:>7.3f} {r['R02_over_heff']:>7.2f} {r['drift_cfl']:>6.2f} "
              f"{r['frac_reached_T']:>6.2f}")
    print(f"\nwrote qwindow_summary.csv/.json to {args.sweep_dir}")


if __name__ == "__main__":
    main()
