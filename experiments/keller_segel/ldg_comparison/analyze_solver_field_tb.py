"""
Solver-field comparison via the SAME LDG-style particle resolution-gap time t_b.
================================================================================
(next_stage.md §8.)  Compares solver-field reconstructions (current_fourier,
blob_ch006, blob_ch009, optionally spectral) NOT by abort/final time but by the
LDG-matched P1-DG resolution-gap time:

    S^DG_{Np,n}(t) = || Pi_n^{P1DG} mu^u_{Np}(t) ||_{L2}     (cross/split estimator),
    t_b^m(theta) = inf{ t : Sbar_high(t)/Sbar_low(t) >= theta, held for >= 5e-6 },

main pair (Np,n): low=(8e4,80) via S_dg_cross_80, high=(3.2e5,160) via S_dg_cross_160,
theta=1.05.  Seed-mean ratio, then a 1000x seed bootstrap CI.  Fixed-flux LDG ref:
t_b(80->160)=5.95e-5, t_b(160->320)=8.43e-5.  This is a numerical resolution-gap
indicator, NOT a continuum blow-up time.

Usage:  python analyze_solver_field_tb.py --sdir <solver_field_tb_run> \
            [--theta 1.05] [--persist 5e-6] [--nboot 1000] [--t_match 1e-4]
"""
import os
import csv
import glob
import json
import argparse

import numpy as np

LDG_REF_80_160 = 5.95e-5
LDG_REF_160_320 = 8.43e-5
# config display order; low group uses S_dg_cross_80 at N=80000, high uses
# S_dg_cross_160 at N=320000.
CONFIGS = ["current_fourier", "blob_ch006", "blob_ch009", "spectral_taper025"]
DT_OUT = 1.0e-6
SEC_COLS = ("R_0.1", "R_0.2", "R_0.5", "R_0.8", "drift_cfl_solver_field",
            "drift_cfl_fourier_diag", "solver_field_residual_E")


def load_curve(d, col):
    cs = glob.glob(os.path.join(d, "diag_*.csv"))
    if not cs:
        return None
    rows = list(csv.DictReader(open(cs[0])))
    if not rows or col not in rows[0]:
        return None
    t = np.array([float(r["t"]) for r in rows])
    y = np.array([float(r[col]) for r in rows])
    sec = {}
    for c in SEC_COLS:
        if c in rows[0]:
            try:
                sec[c] = np.array([float(r[c]) for r in rows])
            except (ValueError, TypeError):
                sec[c] = np.full(len(rows), np.nan)
    return t, y, float(t[-1]), sec


def seed_dirs(sdir, cfg, N):
    return sorted(glob.glob(os.path.join(sdir, f"{cfg}_N{N}_seed*")))


def interp_to(grid, t, y):
    """Interpolate y(t) onto grid; NaN past the curve's last time (no extrapolation)."""
    out = np.interp(grid, t, y, left=y[0], right=np.nan)
    out[grid > t[-1] + 1e-12] = np.nan
    return out


def crossing(grid, ratio, theta, persist):
    """First t where ratio>=theta and stays >=theta over [t, t+persist], fully within
    available finite data.  Returns np.nan if no such confirmed crossing exists."""
    n = len(grid)
    for i in range(n):
        if not (np.isfinite(ratio[i]) and ratio[i] >= theta):
            continue
        win = (grid >= grid[i]) & (grid <= grid[i] + persist + 1e-15)
        if grid[win][-1] < grid[i] + persist - 1e-12:
            continue                      # window runs past available data; cannot confirm
        rw = ratio[win]
        if np.all(np.isfinite(rw) & (rw >= theta)):
            return float(grid[i])
    return np.nan


def mean_ratio(grid, lows, highs):
    """Seed-mean low and high curves -> ratio (NaN where either mean is NaN/0)."""
    Lo = np.nanmean(np.vstack(lows), axis=0) if lows else np.full(len(grid), np.nan)
    Hi = np.nanmean(np.vstack(highs), axis=0) if highs else np.full(len(grid), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = Hi / Lo
    R[~np.isfinite(Lo) | ~np.isfinite(Hi) | (Lo <= 0)] = np.nan
    return Lo, Hi, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdir", required=True)
    ap.add_argument("--theta", type=float, default=1.05)
    ap.add_argument("--persist", type=float, default=5.0e-6)
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--t_match", type=float, default=1.0e-4)
    ap.add_argument("--seed_bootstrap", type=int, default=12345)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed_bootstrap)

    summary = []
    plotdata = {}
    for cfg in CONFIGS:
        low_dirs = seed_dirs(args.sdir, cfg, 80000)
        high_dirs = seed_dirs(args.sdir, cfg, 320000)
        if not low_dirs or not high_dirs:
            continue
        lows, highs, t_ends_lo, t_ends_hi, sec_lo, sec_hi = [], [], [], [], [], []
        for d in low_dirs:
            r = load_curve(d, "S_dg_cross_80")
            if r is None:
                continue
            t, y, te, sec = r
            lows.append((t, y)); t_ends_lo.append(te); sec_lo.append((t, sec))
        for d in high_dirs:
            r = load_curve(d, "S_dg_cross_160")
            if r is None:
                continue
            t, y, te, sec = r
            highs.append((t, y)); t_ends_hi.append(te); sec_hi.append((t, sec))
        if not lows or not highs:
            continue

        tmax = min(min(t_ends_lo), min(t_ends_hi))
        grid = np.arange(0.0, tmax + 0.5 * DT_OUT, DT_OUT)
        lo_i = [interp_to(grid, t, y) for t, y in lows]
        hi_i = [interp_to(grid, t, y) for t, y in highs]
        Lo, Hi, R = mean_ratio(grid, lo_i, hi_i)
        valid = tmax >= 5.0e-5
        tb = crossing(grid, R, args.theta, args.persist) if valid else np.nan

        # bootstrap over seeds (independent low/high resampling)
        boot = []
        nlo, nhi = len(lo_i), len(hi_i)
        for _ in range(args.nboot if valid else 0):
            il = rng.integers(0, nlo, nlo); ih = rng.integers(0, nhi, nhi)
            _, _, Rb = mean_ratio(grid, [lo_i[k] for k in il], [hi_i[k] for k in ih])
            boot.append(crossing(grid, Rb, args.theta, args.persist))
        boot = np.array(boot, dtype=float)
        ci_lo = float(np.nanpercentile(boot, 5)) if np.isfinite(boot).any() else np.nan
        ci_hi = float(np.nanpercentile(boot, 95)) if np.isfinite(boot).any() else np.nan

        # secondary diagnostics at t_match
        def sec_at(sec_list, col):
            vals = []
            for t, sd in sec_list:
                if col in sd:
                    idx = np.searchsorted(t, args.t_match)
                    if idx < len(t):
                        vals.append(sd[col][idx])
            return (float(np.nanmean(vals)), float(np.nanstd(vals))) if vals else (np.nan, np.nan)

        def cflmax(sec_list, col):
            vals = [np.nanmax(sd[col]) for t, sd in sec_list if col in sd]
            return (float(np.nanmean(vals)), float(np.nanstd(vals))) if vals else (np.nan, np.nan)

        R02_lo = sec_at(sec_lo, "R_0.2"); R02_hi = sec_at(sec_hi, "R_0.2")
        cflS_lo = cflmax(sec_lo, "drift_cfl_solver_field")
        cflS_hi = cflmax(sec_hi, "drift_cfl_solver_field")
        cflF_lo = cflmax(sec_lo, "drift_cfl_fourier_diag")
        cflF_hi = cflmax(sec_hi, "drift_cfl_fourier_diag")
        resE_lo = sec_at(sec_lo, "solver_field_residual_E")
        resE_hi = sec_at(sec_hi, "solver_field_residual_E")
        T = grid[-1]  # horizon proxy from data
        frac_T_lo = float(np.mean([te >= 1.99e-4 for te in t_ends_lo]))
        frac_T_hi = float(np.mean([te >= 1.99e-4 for te in t_ends_hi]))
        on_ldg = bool(valid and np.isfinite(tb)
                      and (LDG_REF_80_160 * 0.5 <= tb <= LDG_REF_160_320 * 2.0))

        row = dict(
            config=cfg, theta=args.theta, low_group="N80000_dg80",
            high_group="N320000_dg160", n_low=80000, n_high=320000,
            n_seed_low=nlo, n_seed_high=nhi, tmax_complete=tmax,
            tb=tb, ci_low=ci_lo, ci_high=ci_hi,
            ratio_max=float(np.nanmax(R)) if np.isfinite(R).any() else np.nan,
            ldg_ref_80_160=LDG_REF_80_160, ldg_ref_160_320=LDG_REF_160_320,
            on_ldg_scale=on_ldg,
            fraction_reached_T_low=frac_T_lo, fraction_reached_T_high=frac_T_hi,
            t_end_mean_low=float(np.mean(t_ends_lo)), t_end_mean_high=float(np.mean(t_ends_hi)),
            cfl_solver_max_mean_low=cflS_lo[0], cfl_solver_max_mean_high=cflS_hi[0],
            cfl_fourier_max_mean_low=cflF_lo[0], cfl_fourier_max_mean_high=cflF_hi[0],
            R02_tm_mean_low=R02_lo[0], R02_tm_mean_high=R02_hi[0],
            residual_E_tm_mean_low=resE_lo[0], residual_E_tm_mean_high=resE_hi[0],
            valid_main_tb=valid, invalid_reason=("" if valid else "tmax<5e-5 (aborts)"),
        )
        summary.append(row)
        plotdata[cfg] = dict(grid=grid, Lo=Lo, Hi=Hi, R=R, tb=tb, ci=(ci_lo, ci_hi))

    # ---- write outputs ----
    os.makedirs(os.path.join(args.sdir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.sdir, "plot_data"), exist_ok=True)
    cols = ["config", "theta", "low_group", "high_group", "n_low", "n_high",
            "n_seed_low", "n_seed_high", "tmax_complete", "tb", "ci_low", "ci_high",
            "ratio_max", "ldg_ref_80_160", "ldg_ref_160_320", "on_ldg_scale",
            "fraction_reached_T_low", "fraction_reached_T_high", "t_end_mean_low",
            "t_end_mean_high", "cfl_solver_max_mean_low", "cfl_solver_max_mean_high",
            "cfl_fourier_max_mean_low", "cfl_fourier_max_mean_high", "R02_tm_mean_low",
            "R02_tm_mean_high", "residual_E_tm_mean_low", "residual_E_tm_mean_high",
            "valid_main_tb", "invalid_reason"]
    with open(os.path.join(args.sdir, "solver_field_tb_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in summary:
            w.writerow(r)
    json.dump(summary, open(os.path.join(args.sdir, "solver_field_tb_summary.json"), "w"),
              indent=2, default=str)
    for cfg, pd in plotdata.items():
        np.savez(os.path.join(args.sdir, "plot_data", f"tb_ratio_{cfg}.npz"),
                 grid=pd["grid"], Lo=pd["Lo"], Hi=pd["Hi"], R=pd["R"],
                 tb=pd["tb"], ci_low=pd["ci"][0], ci_high=pd["ci"][1])

    # ---- console report ----
    print(f"\n=== LDG-style particle t_b (theta={args.theta}, persist={args.persist:.0e}) ===")
    print(f"LDG ref: 80->160 = {LDG_REF_80_160:.2e}, 160->320 = {LDG_REF_160_320:.2e}\n")
    print(f"{'config':<18} {'tb':>10} {'CI low':>10} {'CI high':>10} {'on_LDG':>7} "
          f"{'tmax':>9} {'cflS_lo':>8} {'cflS_hi':>8} {'fracT_hi':>8} {'valid':>6}")
    for r in summary:
        print(f"{r['config']:<18} {r['tb']:>10.3e} {r['ci_low']:>10.3e} {r['ci_high']:>10.3e} "
              f"{str(r['on_ldg_scale']):>7} {r['tmax_complete']:>9.2e} "
              f"{r['cfl_solver_max_mean_low']:>8.2f} {r['cfl_solver_max_mean_high']:>8.2f} "
              f"{r['fraction_reached_T_high']:>8.2f} {str(r['valid_main_tb']):>6}")
    print(f"\nwrote solver_field_tb_summary.csv/.json + plot_data/ to {args.sdir}")
    return summary, plotdata


if __name__ == "__main__":
    main()
