"""
Analyze the blob-residual solver-field sweep (solver-hybrid blob plan §7.2 / §8).
================================================================================

Reads the CORRECTED diagnostics: drift_cfl_solver_field is now the ACTUAL selected
solver field's CFL (not the single-K Fourier diagnostic, which is in
drift_cfl_fourier_diag).  Compares, per config (seeds pooled):

  Q1 / stability : abort time (final t) mean+/-std; max drift_cfl_solver_field
                   mean+/-std (the REAL solver drift); and the solver-vs-fourier CFL
                   ratio (how much the local operator amplifies the drift).
  Q3 / resolution: reconstruction-free inner-core radii R_0.1,R_0.2 at a matched
                   time; R_0.2 / h_core (h_core = local operator scale: L/K for
                   fourier, L/Kl for spectral, the blob bandwidth h for blob);
                   solver_field_residual_E (high-freq energy the local op adds).

Decision (plan §8): does the BLOB recover the abort time toward current_fourier
(stability) with a genuinely SMALLER solver-CFL than the spectral Form I, while
keeping a non-trivial residual (resolution)?

Usage:  python analyze_blob_sweep.py --sdir <sf_blob_run> [--t_match 1.0e-4]
"""
import os
import csv
import glob
import json
import argparse

import numpy as np

# (display name, subdir prefix, local-operator scale spec)
#   ('fourier_K', K)         -> h_core = L / K
#   ('spectral_Kl', Kl)      -> h_core = L / Kl
#   ('blob', None)           -> h_core = solver_field_h (read from CSV)
CONFIGS = [
    ("current_fourier",       "current_fourier",       ("fourier_K", 10)),
    ("spectral_taper0.25",    "spectral_taper0.25",    ("spectral_Kl", 24)),
    ("blob_fracL_ch0.04",     "blob_fracL_ch0.04",     ("blob", None)),
    ("blob_fracL_ch0.06",     "blob_fracL_ch0.06",     ("blob", None)),
    ("blob_fracL_ch0.09",     "blob_fracL_ch0.09",     ("blob", None)),
    ("blob_corespacing_ch1.2", "blob_corespacing_ch1.2", ("blob", None)),
]
NUM = ("R_0.1", "R_0.2", "L", "drift_cfl_solver_field", "drift_cfl_fourier_diag",
       "max_grad_solver_field", "max_grad_fourier_diag", "solver_field_h",
       "solver_field_residual_E", "peak_PK_u")


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    t = np.array([float(r["t"]) for r in rows])
    out = {}
    for c in NUM:
        if c in rows[0]:
            vals = []
            for r in rows:
                try:
                    vals.append(float(r[c]))
                except (ValueError, TypeError):
                    vals.append(np.nan)
            out[c] = np.array(vals)
    return t, out


def at_time(t, y, tm):
    idx = np.searchsorted(t, tm)
    return float(y[idx]) if idx < len(t) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdir", required=True)
    ap.add_argument("--t_match", type=float, default=1.0e-4)
    args = ap.parse_args()

    summary = {}
    hdr = (f"{'config':<24} {'seed':>4} {'final_t':>10} {'cflS_max':>9} "
           f"{'cflF_max':>9} {'R0.2@tm':>9} {'R0.2/hc':>8} {'resE@tm':>8}")
    print(hdr); print("-" * len(hdr))
    for name, prefix, hspec in CONFIGS:
        dirs = sorted(glob.glob(os.path.join(args.sdir, f"{prefix}_seed*")))
        finals, cflS, cflF, r02, r01, r02h, resE = [], [], [], [], [], [], []
        for d in dirs:
            cs = glob.glob(os.path.join(d, "diag_*.csv"))
            if not cs:
                continue
            seed = os.path.basename(d).split("seed")[-1]
            t, c = load(cs[0])
            ft = float(t[-1])
            cS = float(np.nanmax(c["drift_cfl_solver_field"])) if "drift_cfl_solver_field" in c else np.nan
            cF = float(np.nanmax(c["drift_cfl_fourier_diag"])) if "drift_cfl_fourier_diag" in c else np.nan
            R2 = at_time(t, c["R_0.2"], args.t_match) if "R_0.2" in c else np.nan
            R1 = at_time(t, c["R_0.1"], args.t_match) if "R_0.1" in c else np.nan
            Ltm = at_time(t, c["L"], args.t_match) if "L" in c else np.nan
            # local-operator scale h_core
            if hspec[0] in ("fourier_K", "spectral_Kl"):
                hc = (Ltm / hspec[1]) if np.isfinite(Ltm) else np.nan
            else:
                hc = at_time(t, c["solver_field_h"], args.t_match) if "solver_field_h" in c else np.nan
            R2h = (R2 / hc) if (np.isfinite(R2) and np.isfinite(hc) and hc > 0) else np.nan
            rE = at_time(t, c["solver_field_residual_E"], args.t_match) if "solver_field_residual_E" in c else np.nan
            finals.append(ft); cflS.append(cS); cflF.append(cF)
            r02.append(R2); r01.append(R1); r02h.append(R2h); resE.append(rE)
            print(f"{name:<24} {seed:>4} {ft:>10.3e} {cS:>9.2f} {cF:>9.2f} "
                  f"{R2:>9.4f} {R2h:>8.2f} {rE:>8.3f}")
        if not finals:
            continue
        summary[name] = dict(
            final_t_mean=float(np.nanmean(finals)), final_t_std=float(np.nanstd(finals)),
            final_t_seeds=finals,
            cflS_max_mean=float(np.nanmean(cflS)), cflS_max_std=float(np.nanstd(cflS)),
            cflF_max_mean=float(np.nanmean(cflF)),
            R02_tm=float(np.nanmean(r02)), R01_tm=float(np.nanmean(r01)),
            R02_over_hcore_tm=float(np.nanmean(r02h)),
            residual_E_tm=float(np.nanmean(resE)),
        )

    print(f"\n=== Q1 / STABILITY (later abort + smaller, less-spiky solver CFL = better) ===")
    print(f"{'config':<24} {'final_t mean':>13} {'final_t std':>12} "
          f"{'cflS_max mean':>13} {'cflS_max std':>12} {'cflF_max mean':>13}")
    for name, _, _ in CONFIGS:
        if name not in summary:
            continue
        s = summary[name]
        print(f"{name:<24} {s['final_t_mean']:>13.3e} {s['final_t_std']:>12.2e} "
              f"{s['cflS_max_mean']:>13.2f} {s['cflS_max_std']:>12.2f} {s['cflF_max_mean']:>13.2f}")

    print(f"\n=== Q3 / RESOLUTION at t_match={args.t_match:.2e} ===")
    print(f"{'config':<24} {'R_0.2':>10} {'R_0.1':>10} {'R0.2/h_core':>12} {'residual_E':>11}")
    for name, _, _ in CONFIGS:
        if name not in summary:
            continue
        s = summary[name]
        print(f"{name:<24} {s['R02_tm']:>10.4f} {s['R01_tm']:>10.4f} "
              f"{s['R02_over_hcore_tm']:>12.2f} {s['residual_E_tm']:>11.3f}")

    cur = summary.get("current_fourier")
    spec = summary.get("spectral_taper0.25")
    print("\n=== DECISION read-off ===")
    if cur:
        print(f"current_fourier: abort={cur['final_t_mean']:.3e}+/-{cur['final_t_std']:.1e} "
              f"cflS_max={cur['cflS_max_mean']:.1f}")
    if spec:
        print(f"spectral Kl=24:  abort={spec['final_t_mean']:.3e}+/-{spec['final_t_std']:.1e} "
              f"cflS_max={spec['cflS_max_mean']:.1f}")
    for name, _, _ in CONFIGS:
        if not name.startswith("blob") or name not in summary:
            continue
        s = summary[name]
        da = (s['final_t_mean'] / cur['final_t_mean']) if cur else np.nan
        vs_spec = ("smaller-CFL-than-spectral" if (spec and s['cflS_max_mean'] < spec['cflS_max_mean'])
                   else "not-smaller-than-spectral")
        print(f"  {name:<24}: abort {da:.2f}x current | cflS_max {s['cflS_max_mean']:.1f} ({vs_spec}) "
              f"| residual_E {s['residual_E_tm']:.3f}")

    outp = os.path.join(args.sdir, "blob_sweep_compare.json")
    json.dump(summary, open(outp, "w"), indent=2, default=str)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
