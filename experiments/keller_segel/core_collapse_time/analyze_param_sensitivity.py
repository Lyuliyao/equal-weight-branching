"""
Generic parameter-sensitivity analysis for particle T_core (Experiments B/C/D).
===============================================================================
Sweep dir holds runs named <prefix><val>_N<N>_seed<seed> (e.g. K12_N80000_seed0,
tau1e-07_N320000_seed2, qwin0.65_N80000_seed1).  For each (val, N) group, compute the
particle core-collapse time T_core (common-grid seed mean + 1000x seed bootstrap, reusing
fit_core_collapse) and window-resolution diagnostics; compare each to the LDG reference.

Usage:
  python analyze_param_sensitivity.py --sweep_dir <run> --prefix K --param K \
      --ldg_T 1.215e-4 [--t_match 1e-4] [--bootstrap_seeds 1000]
"""
import os
import csv
import glob
import json
import argparse

import numpy as np
import fit_core_collapse as F

QSET = [0.1, 0.2, 0.3]


def diag_at(dirs, col, t_match):
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
    return float(np.nanmean(vals)) if vals else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--prefix", required=True, help="dir name prefix before the value, e.g. K, tau, qwin")
    ap.add_argument("--param", default="param", help="column name for the swept value")
    ap.add_argument("--ldg_T", type=float, default=1.215e-4)
    ap.add_argument("--t_match", type=float, default=1.0e-4)
    ap.add_argument("--bootstrap_seeds", type=int, default=1000)
    ap.add_argument("--min_seed_coverage", type=int, default=3)
    ap.add_argument("--K_for_heff", type=int, default=10, help="K used for L/K (override per K sweep)")
    args = ap.parse_args()
    rng = np.random.default_rng(2025)

    groups = {}
    for d in sorted(glob.glob(os.path.join(args.sweep_dir, f"{args.prefix}*_N*_seed*"))):
        base = os.path.basename(d)
        val = base.split(args.prefix, 1)[1].split("_N")[0]
        N = base.split("_N")[1].split("_seed")[0]
        groups.setdefault((val, N), []).append(d)

    rows = []
    for (val, N), dirs in sorted(groups.items(), key=lambda kv: (kv[0][0], int(kv[0][1]))):
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
        grid, Rmean, _, n_eff = F.seed_mean_common_grid(seeds, F.ALL_Q, 1e-6, args.min_seed_coverage)
        T_all, _ = F.core_T_from_curves(grid, Rmean, QSET, F.DEF_WINDOWS, 0.9, coverage=n_eff,
                                        min_cov=args.min_seed_coverage)
        ag = F.aggregate(T_all)
        boots = []
        for _ in range(args.bootstrap_seeds):
            idx = rng.integers(0, len(seeds), len(seeds))
            g2, Rm2, _, ne2 = F.seed_mean_common_grid([seeds[k] for k in idx], QSET, 1e-6,
                                                      args.min_seed_coverage)
            Tb, _ = F.core_T_from_curves(g2, Rm2, QSET, F.DEF_WINDOWS, 0.9, coverage=ne2,
                                         min_cov=args.min_seed_coverage)
            if Tb:
                boots.append(float(np.median(Tb)))
        boots = np.array(boots, float)
        tbm = float(np.median(boots)) if boots.size else np.nan
        tlo = float(np.percentile(boots, 5)) if boots.size else np.nan
        thi = float(np.percentile(boots, 95)) if boots.size else np.nan
        # K for h_eff: if sweeping K, val is the K; else use --K_for_heff
        Kh = int(float(val)) if args.prefix == "K" else args.K_for_heff
        L_tm = diag_at(dirs, "L", args.t_match); r02 = diag_at(dirs, "R_0.2", args.t_match)
        heff = (L_tm / Kh) if np.isfinite(L_tm) and Kh else np.nan
        fts = []
        for d in dirs:
            cs = glob.glob(os.path.join(d, "diag_*.csv"))
            if cs:
                rr = list(csv.DictReader(open(cs[0])))
                if rr:
                    fts.append(float(rr[-1]["t"]))
        rows.append(dict(**{args.param: val}, N=N, T_core_qwin=ag["T_median"], T_core_boot=tbm,
                         boot_p5=tlo, boot_p95=thi,
                         offset_vs_ldg=(tbm / args.ldg_T - 1.0) if np.isfinite(tbm) else np.nan,
                         ci_contains_ldg=bool(np.isfinite(tlo) and tlo <= args.ldg_T <= thi),
                         outside_v_frac=diag_at(dirs, "outside_v_frac", args.t_match),
                         R02_over_heff=(r02 / heff if heff else np.nan),
                         drift_cfl=diag_at(dirs, "drift_cfl_solver_field", args.t_match),
                         frac_reached_T=float(np.mean([t >= 1.99e-4 for t in fts])) if fts else np.nan,
                         t_end_mean=float(np.mean(fts)) if fts else np.nan, n_valid_fits=ag["n"]))

    cols = [args.param, "N", "T_core_qwin", "T_core_boot", "boot_p5", "boot_p95",
            "offset_vs_ldg", "ci_contains_ldg", "outside_v_frac", "R02_over_heff",
            "drift_cfl", "frac_reached_T", "t_end_mean", "n_valid_fits"]
    with open(os.path.join(args.sweep_dir, f"{args.param}_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    json.dump(rows, open(os.path.join(args.sweep_dir, f"{args.param}_summary.json"), "w"),
              indent=2, default=str)

    print(f"\n=== {args.param} sensitivity (LDG T_core ref = {args.ldg_T:.3e}) ===")
    print(f"{args.param:>8} {'N':>7} {'T_boot':>10} {'[p5':>10} {'p95]':>10} {'off%':>6} "
          f"{'ldg?':>5} {'out_v':>6} {'R0.2/h':>7} {'cfl':>5} {'fracT':>6} {'nfit':>4}")
    for r in rows:
        print(f"{str(r[args.param]):>8} {r['N']:>7} {r['T_core_boot']:>10.3e} {r['boot_p5']:>10.3e} "
              f"{r['boot_p95']:>10.3e} {100*r['offset_vs_ldg']:>5.1f} {str(r['ci_contains_ldg']):>5} "
              f"{r['outside_v_frac']:>6.2f} {r['R02_over_heff']:>7.2f} {r['drift_cfl']:>5.2f} "
              f"{r['frac_reached_T']:>6.2f} {r['n_valid_fits']:>4}")
    print(f"\nwrote {args.param}_summary.csv/.json to {args.sweep_dir}")


if __name__ == "__main__":
    main()
