"""
Analyze the v_hi-taper sweep for the two-level spectral-residual solver field.
=============================================================================

The solver-hybrid design note found a Scenario-C tradeoff for Form I:

  * current_fourier (global K=10 drift): the inner core is UNDER-resolved, but the
    drift is smooth, so the run survives far in time (~1.9e-4) before the drift-CFL
    abort.
  * two_level (Kg=8 + Kl=24 core residual, sharp): the inner core is resolved, but
    the high-Kl part injects Monte-Carlo high-mode NOISE into the drift, so the run
    aborts EARLY (~1.4e-4) and with large seed-to-seed variance.

The fix tested here is a Gaussian taper of width taper_hi on the high-Kl part: this
is exactly an eta_h Gaussian-blob residual (the blob's FT is exp(-h^2 k^2/2)), i.e.
a SMOOTHER local operator.  Smaller taper_hi = smoother = less noise but less core
resolution.  This script maps that tradeoff across:

    current_fourier  |  two_level_taper0.5  |  two_level_taper0.35  |  two_level_taper0.25

For each config (4 seeds) it reports:
  Q1 / noise      : mean +/- std abort time (final t), max drift_cfl mean +/- std.
  Q3 / resolution : reconstruction-free inner-core radii R_0.1, R_0.2 at a matched
                    reference time, and R_0.2 / h_eff_core (h_eff_core = L/K_core,
                    K_core=10 for current_fourier, Kl=24 for the hybrids).

Decision read-off: does a smaller taper_hi recover the abort time toward
current_fourier (drift stability) WHILE keeping the hybrid inner-core gain?

Usage:
    python analyze_taper_sweep.py --sdir <sf_taper_run> [--t_match 1.0e-4]
"""
import os
import csv
import glob
import json
import argparse

import numpy as np

# config name -> (subdir prefix, K_core for h_eff_core)
CONFIGS = [
    ("current_fourier",     "current_fourier",      10),
    ("two_level_taper0.5",  "two_level_taper0.5",   24),
    ("two_level_taper0.35", "two_level_taper0.35",  24),
    ("two_level_taper0.25", "two_level_taper0.25",  24),
]
COLS = ("R_0.1", "R_0.2", "R_0.5", "R_0.8", "L", "h_eff",
        "drift_cfl", "peak_PK_u", "S_L2_u")


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    t = np.array([float(r["t"]) for r in rows])
    out = {c: np.array([float(r[c]) for r in rows]) for c in COLS if c in rows[0]}
    return t, out


def at_time(t, y, t_match):
    """Value of y at the first time >= t_match; NaN if the run aborted earlier."""
    idx = np.searchsorted(t, t_match)
    if idx >= len(t):
        return np.nan
    return float(y[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdir", required=True)
    ap.add_argument("--t_match", type=float, default=1.0e-4,
                    help="reference time for the matched-time resolution comparison")
    args = ap.parse_args()

    summary = {}
    hdr = (f"{'config':<22} {'seed':>4} {'final_t':>10} {'cfl_max':>8} "
           f"{'R0.2@tm':>9} {'R0.1@tm':>9} {'R0.2/hc@tm':>11}")
    print(hdr)
    print("-" * len(hdr))
    for name, prefix, Kcore in CONFIGS:
        dirs = sorted(glob.glob(os.path.join(args.sdir, f"{prefix}_seed*")))
        finals, cflmax, r02_tm, r01_tm, r02h_tm = [], [], [], [], []
        for d in dirs:
            cs = glob.glob(os.path.join(d, "diag_*.csv"))
            if not cs:
                continue
            seed = os.path.basename(d).split("seed")[-1]
            t, c = load(cs[0])
            ft = float(t[-1])
            cm = float(np.nanmax(c["drift_cfl"])) if "drift_cfl" in c else np.nan
            r2 = at_time(t, c["R_0.2"], args.t_match) if "R_0.2" in c else np.nan
            r1 = at_time(t, c["R_0.1"], args.t_match) if "R_0.1" in c else np.nan
            Ltm = at_time(t, c["L"], args.t_match) if "L" in c else np.nan
            heff_core = (Ltm / Kcore) if np.isfinite(Ltm) else np.nan
            r2h = (r2 / heff_core) if (np.isfinite(r2) and np.isfinite(heff_core)) else np.nan
            finals.append(ft); cflmax.append(cm)
            r02_tm.append(r2); r01_tm.append(r1); r02h_tm.append(r2h)
            print(f"{name:<22} {seed:>4} {ft:>10.3e} {cm:>8.2f} "
                  f"{r2:>9.4f} {r1:>9.4f} {r2h:>11.2f}")
        summary[name] = dict(
            Kcore=Kcore, n_seeds=len(finals),
            final_t_mean=float(np.nanmean(finals)) if finals else np.nan,
            final_t_std=float(np.nanstd(finals)) if finals else np.nan,
            final_t_seeds=finals,
            cfl_max_mean=float(np.nanmean(cflmax)) if cflmax else np.nan,
            cfl_max_std=float(np.nanstd(cflmax)) if cflmax else np.nan,
            R02_at_tmatch_mean=float(np.nanmean(r02_tm)) if r02_tm else np.nan,
            R01_at_tmatch_mean=float(np.nanmean(r01_tm)) if r01_tm else np.nan,
            R02_over_heff_at_tmatch_mean=float(np.nanmean(r02h_tm)) if r02h_tm else np.nan,
        )

    print(f"\n=== Q1 / NOISE: abort time and drift-CFL spikes "
          f"(later abort + smaller spread = more stable) ===")
    print(f"{'config':<22} {'final_t mean':>14} {'final_t std':>12} "
          f"{'cfl_max mean':>13} {'cfl_max std':>12}")
    for name, _, _ in CONFIGS:
        s = summary[name]
        print(f"{name:<22} {s['final_t_mean']:>14.3e} {s['final_t_std']:>12.2e} "
              f"{s['cfl_max_mean']:>13.2f} {s['cfl_max_std']:>12.2f}")

    print(f"\n=== Q3 / RESOLUTION at t_match={args.t_match:.2e} "
          f"(reconstruction-free inner-core radii; smaller R = tighter core) ===")
    print(f"{'config':<22} {'R_0.2':>10} {'R_0.1':>10} {'R0.2/h_core':>12}")
    for name, _, _ in CONFIGS:
        s = summary[name]
        print(f"{name:<22} {s['R02_at_tmatch_mean']:>10.4f} "
              f"{s['R01_at_tmatch_mean']:>10.4f} {s['R02_over_heff_at_tmatch_mean']:>12.2f}")

    # decision read-off relative to current_fourier
    cur = summary["current_fourier"]
    print("\n=== DECISION read-off (vs current_fourier) ===")
    print(f"current_fourier abort = {cur['final_t_mean']:.3e} +/- {cur['final_t_std']:.1e}, "
          f"R_0.2@tm = {cur['R02_at_tmatch_mean']:.4f}")
    for name, _, _ in CONFIGS[1:]:
        s = summary[name]
        d_abort = (s['final_t_mean'] / cur['final_t_mean']) if cur['final_t_mean'] else np.nan
        tighter = (s['R02_at_tmatch_mean'] < cur['R02_at_tmatch_mean'])
        print(f"  {name:<22}: abort {d_abort:.2f}x current  | "
              f"core {'TIGHTER' if tighter else 'not tighter'} "
              f"(R_0.2 {s['R02_at_tmatch_mean']:.4f} vs {cur['R02_at_tmatch_mean']:.4f})  | "
              f"cfl_max {s['cfl_max_mean']:.1f}")

    outp = os.path.join(args.sdir, "taper_sweep_compare.json")
    json.dump(summary, open(outp, "w"), indent=2, default=str)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
