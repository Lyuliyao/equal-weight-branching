"""
Compare solver fields (current_fourier vs two_level_spectral_residual) for the
parabolic-parabolic KS particle drift.
==========================================================================

Reads the diag_*.csv of the lean comparison runs and answers the solver-hybrid
design note's main questions:

  Q1  Does the residual solver field DELAY/eliminate the drift-CFL abort seen with
      the single-K Fourier drift?  -> compare final/abort times.
  Q3  Do inner-core diagnostics show the single-K Fourier under-resolves the core
      before the abort?  -> R_0.2 / h_eff and R_0.1 / h_eff over time, where
      h_eff = L / K_core (K=10 for current_fourier, Kl=24 in the hybrid core).
  (Q2 DG-matched tb is computed by analyze_particle_blowup_metric.py when the DG
   readout is on; this lean run focuses on Q1/Q3.)

Usage:  python analyze_solver_field.py --sdir <solver_field_sweep_run> --Kl 24
"""
import os
import csv
import glob
import json
import argparse

import numpy as np


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    t = np.array([float(r["t"]) for r in rows])
    cols = ("R_0.1", "R_0.2", "R_0.5", "R_0.8", "L", "h_eff", "drift_cfl",
            "peak_PK_u", "S_L2_u")
    out = {c: np.array([float(r[c]) for r in rows]) for c in cols if c in rows[0]}
    return t, out


def collect(sdir, mode):
    runs = []
    for d in sorted(glob.glob(os.path.join(sdir, f"lean_{mode}_seed*"))):
        cs = glob.glob(os.path.join(d, "diag_*.csv"))
        if cs:
            runs.append((d, *load(cs[0])))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdir", required=True)
    ap.add_argument("--Kcur", type=int, default=10)
    ap.add_argument("--Kl", type=int, default=24)
    args = ap.parse_args()
    modes = {"current_fourier": args.Kcur, "two_level_spectral_residual": args.Kl}

    summary = {}
    print(f"{'mode':<32} {'seed':>4} {'final_t':>10} {'R0.2/heff@end':>14} "
          f"{'R0.1/heff@end':>14} {'cfl_max':>9}")
    for mode, Kcore in modes.items():
        rows = collect(args.sdir, mode)
        finals, r02h, r01h = [], [], []
        for i, (d, t, c) in enumerate(rows):
            L = c.get("L");
            heff_core = L / Kcore if L is not None else None      # effective core resolution
            r2 = c.get("R_0.2"); r1 = c.get("R_0.1")
            ft = float(t[-1])
            v02 = float(r2[-1] / heff_core[-1]) if (r2 is not None and heff_core is not None) else np.nan
            v01 = float(r1[-1] / heff_core[-1]) if (r1 is not None and heff_core is not None) else np.nan
            cflmax = float(np.nanmax(c["drift_cfl"])) if "drift_cfl" in c else np.nan
            finals.append(ft); r02h.append(v02); r01h.append(v01)
            print(f"{mode:<32} {i:>4} {ft:>10.3e} {v02:>14.2f} {v01:>14.2f} {cflmax:>9.2f}")
        summary[mode] = dict(final_t=float(np.mean(finals)) if finals else np.nan,
                             final_t_seeds=finals,
                             R02_heff_end=float(np.nanmean(r02h)) if r02h else np.nan,
                             R01_heff_end=float(np.nanmean(r01h)) if r01h else np.nan,
                             Kcore=Kcore)

    cur = summary.get("current_fourier", {})
    hyb = summary.get("two_level_spectral_residual", {})
    print("\n=== Q1: drift-CFL abort time (later = more stable) ===")
    print(f"  current_fourier : mean final_t = {cur.get('final_t', np.nan):.3e}")
    print(f"  two_level hybrid: mean final_t = {hyb.get('final_t', np.nan):.3e}")
    if cur.get("final_t") and hyb.get("final_t"):
        delay = hyb["final_t"] / cur["final_t"]
        print(f"  hybrid runs {delay:.2f}x as far in time "
              + ("(DELAYS the abort)" if delay > 1.05 else "(no clear delay)"))
    print("\n=== Q3: inner-core resolution at end (R_q / h_eff_core; >~2-3 = resolved) ===")
    print(f"  current_fourier (K={cur.get('Kcore')}): R0.2/heff={cur.get('R02_heff_end', np.nan):.2f} "
          f"R0.1/heff={cur.get('R01_heff_end', np.nan):.2f}")
    print(f"  two_level (Kl={hyb.get('Kcore')}):     R0.2/heff={hyb.get('R02_heff_end', np.nan):.2f} "
          f"R0.1/heff={hyb.get('R01_heff_end', np.nan):.2f}")
    json.dump(summary, open(os.path.join(args.sdir, "solver_field_compare.json"), "w"),
              indent=2, default=str)
    print(f"\nwrote solver_field_compare.json to {args.sdir}")


if __name__ == "__main__":
    main()
