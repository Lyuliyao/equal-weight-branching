"""
Core-local / reconstruction-free blow-up-proxy analysis (CLAUDE.md §7.3 / §5.5).
================================================================================

Pure post-processing of saved diagnostics.  No solver run.  Consumes:
  * FVM baseline S_curves.csv at grids n, 2n, 4n  (ldg_pp_baseline), columns
    t, S_L2 (global), S_core (core-local), peak, R_0_5, R_0_8;
  * particle pp diag CSVs (ldg_comparison base/refined), columns
    t, S_L2, peak_PK_u, R_0.5, R_0.8.

It answers the §5.5 questions:

  (A) Resolution-gap times.  Global   t_b^{glob}(n;θ)=inf{t: S_{2n}(t)≥θ S_n(t)}
      and core-local t_b^{core}(n;θ) with S replaced by S_core.  For the grid
      baseline these coincide because the concentrating field carries essentially
      all of its L2 mass inside the core (S_core≈S_L2), i.e. core-localization is
      redundant once the grid resolves the core.

  (B) Reconstruction-free candidate concentration time.  Fit the half-/80%-mass
      radii to R_q(t)^2 ≈ C_q (T_* - t) over a fit window and report the candidate
      T_*; sweep q, resolution, and fit window to expose its (in)stability.  These
      radii are reconstruction-free (grid cell masses / particle quantiles), in
      contrast to the bandwidth-sensitive reconstructed peak and global L2.

Outputs:  global_core_tb.csv, radius_fit.csv, sensitivity.csv, plot_data/*.npz.
"""
import os
import sys
import csv
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def load_csv(path, tcol="t", cols=()):
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r[tcol]) for r in rows])
    out = {c: np.array([float(r[c]) for r in rows]) for c in cols}
    o = np.argsort(t)
    keep = np.concatenate([[True], np.diff(t[o]) > 0])
    idx = o[keep]
    return t[idx], {c: out[c][idx] for c in cols}


def tb_from_series(tb, Sb, tr, Sr, threshes=(1.05, 1.10)):
    lo, hi = max(tb.min(), tr.min()), min(tb.max(), tr.max())
    grid = tb[(tb >= lo) & (tb <= hi)]
    if grid.size < 2:
        grid = np.linspace(lo, hi, 128)
    ratio = np.interp(grid, tr, Sr) / np.interp(grid, tb, Sb)
    out = {"ratio_max": float(np.nanmax(ratio))}
    for thr in threshes:
        hit = np.where(np.isfinite(ratio) & (ratio >= thr))[0]
        out[f"t_b_{thr:.2f}"] = float(grid[hit[0]]) if hit.size else float("nan")
    return out


def fit_Tstar(t, R, tlo, thi):
    """R^2 ~ C (T*-t): linear fit of R^2 vs t over [tlo,thi]; return (T*, C, npts)."""
    m = (t >= tlo) & (t <= thi) & (R > 0)
    if m.sum() < 3:
        return None
    slope, inter = np.polyfit(t[m], R[m] ** 2, 1)
    if slope >= 0:
        return None
    return (-inter / slope, -slope, int(m.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", required=True,
                    help="reference_results/keller_segel_ldg_pp/baseline_<run_id>")
    ap.add_argument("--particle_base", required=True)
    ap.add_argument("--particle_refined", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--report_times", type=float, nargs="+",
                    default=[6e-5, 1.2e-4, 2.0e-4])
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, "plot_data"), exist_ok=True)

    bcols = ("S_L2", "S_core", "peak", "R_0_5", "R_0_8")
    base = {}
    for n in (128, 256, 512):
        p = os.path.join(args.baseline_dir, f"n{n}", "S_curves.csv")
        base[n] = load_csv(p, cols=bcols)

    pcols = ("S_L2", "peak_PK_u", "R_0.5", "R_0.8")
    tpb, pb = load_csv(args.particle_base, cols=pcols)
    tpr, pr = load_csv(args.particle_refined, cols=pcols)

    # ---- (A) global vs core-local t_b for the grid baseline pairs ----
    tb_rows = []
    for (nb, nr) in [(128, 256), (256, 512)]:
        tbb, db = base[nb]
        tbr, dr = base[nr]
        g = tb_from_series(tbb, db["S_L2"], tbr, dr["S_L2"])
        c = tb_from_series(tbb, db["S_core"], tbr, dr["S_core"])
        tb_rows.append(dict(method="FVM baseline", n_base=nb, n_ref=nr,
                            tb_global_105=g["t_b_1.05"], tb_global_110=g["t_b_1.10"],
                            tb_core_105=c["t_b_1.05"], tb_core_110=c["t_b_1.10"],
                            ratio_max_global=g["ratio_max"], ratio_max_core=c["ratio_max"]))
    # particle (N,K)->(4N,2K) global gap (core-local not archived: window already core-centred)
    gp = tb_from_series(tpb, pb["S_L2"], tpr, pr["S_L2"])
    tb_rows.append(dict(method="particle", n_base=20000, n_ref=80000,
                        tb_global_105=gp["t_b_1.05"], tb_global_110=gp["t_b_1.10"],
                        tb_core_105="", tb_core_110="",
                        ratio_max_global=gp["ratio_max"], ratio_max_core=""))
    _write_csv(os.path.join(args.out_dir, "global_core_tb.csv"), tb_rows)

    # ---- (B) reconstruction-free candidate T* + sensitivity ----
    fit_rows = []
    sens_rows = []
    windows = [(3e-5, 2e-4), (6e-5, 2e-4), (3e-5, 1.5e-4)]
    series = {
        "FVM n=256": (base[256][0], base[256][1]["R_0_5"], base[256][1]["R_0_8"]),
        "FVM n=512": (base[512][0], base[512][1]["R_0_5"], base[512][1]["R_0_8"]),
        "particle base": (tpb, pb["R_0.5"], pb["R_0.8"]),
        "particle refined": (tpr, pr["R_0.5"], pr["R_0.8"]),
    }
    for name, (t, R05, R08) in series.items():
        for qname, R in [("0.5", R05), ("0.8", R08)]:
            f = fit_Tstar(t, R, 3e-5, 2e-4)
            if f:
                fit_rows.append(dict(series=name, q=qname, Tstar=f[0], C=f[1], npts=f[2],
                                     fit_window="3e-5..2e-4"))
            for (wlo, whi) in windows:
                fw = fit_Tstar(t, R, wlo, whi)
                sens_rows.append(dict(series=name, q=qname,
                                      fit_window=f"{wlo:.0e}..{whi:.0e}",
                                      Tstar=(fw[0] if fw else float("nan"))))
    _write_csv(os.path.join(args.out_dir, "radius_fit.csv"), fit_rows)
    _write_csv(os.path.join(args.out_dir, "sensitivity.csv"), sens_rows)

    # ---- plot_data ----
    pdir = os.path.join(args.out_dir, "plot_data")
    np.savez(os.path.join(pdir, "radii.npz"),
             t_b256=base[256][0], R05_b256=base[256][1]["R_0_5"], R08_b256=base[256][1]["R_0_8"],
             t_b512=base[512][0], R05_b512=base[512][1]["R_0_5"], R08_b512=base[512][1]["R_0_8"],
             t_pbase=tpb, R05_pbase=pb["R_0.5"], R08_pbase=pb["R_0.8"],
             t_pref=tpr, R05_pref=pr["R_0.5"], R08_pref=pr["R_0.8"],
             peak_b512=base[512][1]["peak"], t_peakb512=base[512][0],
             peak_pref=pr["peak_PK_u"], t_peakpref=tpr,
             report_times=np.array(args.report_times))

    # ---- console summary ----
    print("=== global vs core-local t_b ===")
    for r in tb_rows:
        print(f"  {r['method']:14s} ({r['n_base']}->{r['n_ref']}): "
              f"global t_b(1.05)={_fmt(r['tb_global_105'])} "
              f"core t_b(1.05)={_fmt(r['tb_core_105'])}")
    print("=== candidate T* (reconstruction-free radius fit) -- WINDOW-SENSITIVE ===")
    from collections import defaultdict
    sp = defaultdict(list)
    for r in sens_rows:
        v = r["Tstar"]
        if isinstance(v, float) and v == v and v > 0:
            sp[r["series"]].append(v)
    for s, vs in sp.items():
        print(f"  {s:18s}: T* in [{min(vs):.2e}, {max(vs):.2e}]  "
              f"spread={max(vs)/min(vs):.2f}x")
    print("  => T* is O(1e-4) (cf. LDG numerical blow-up ~1.21e-4) but NOT stable "
          "across fit windows; we do NOT quote a continuum blow-up time.")
    print(f"wrote global_core_tb.csv, radius_fit.csv, sensitivity.csv, plot_data to {args.out_dir}")


def _fmt(x):
    return f"{x:.3e}" if isinstance(x, float) and not (x != x) else str(x)


def _write_csv(path, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
