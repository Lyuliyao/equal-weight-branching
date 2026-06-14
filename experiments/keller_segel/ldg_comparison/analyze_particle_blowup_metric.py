"""
LDG-style particle blow-up proxy: reconstruction-operator sweep + decision report.
==================================================================================

Reads the diag_*.csv produced by simulation.py --dg_readout_n (multiple Np, seeds)
and forms the particle analogue of the LDG L2-resolution-gap numerical blow-up
indicator for three readout operators:

  A  LDG-matched P1 DG projection (S_dg_cross primary, S_dg_raw as noise diag)
  B  global/core-window Fourier reconstruction (S_L2_u, the current solver field)
  (C  particle-adaptive residual is covered by the separate adaptive_recon audit)

and three gaps (theta in {1.05,1.10}):

  main      (N_p, n) -> (4 N_p, 2 n)   -- the LDG n->2n analogue
  sampling  (N_p, n) -> (4 N_p,   n)   -- particle-noise sensitivity at fixed readout
  recon     (N_p, n) -> (  N_p, 2 n)   -- same cloud, finer readout

Ensemble-averaged over seeds; bootstrap CI over seeds; a persistence rule
(ratio must stay >= theta for at least dt_persist) avoids one-time noisy crossings.
Compared against the FIXED-FLUX LDG tb (80->160=5.953e-5, 160->320=8.428e-5).

Usage:
  python analyze_particle_blowup_metric.py --rdir <particle_blowup_run> --out_dir <same>
"""
import os
import csv
import glob
import json
import argparse

import numpy as np

LDG_TB = {"80->160": 5.953e-5, "160->320": 8.428e-5}      # fixed-flux LDG, theta=1.05
DT_PERSIST = 5e-6
THETAS = (1.05, 1.10)


def load_seed_curves(csv_path, cols):
    rows = list(csv.DictReader(open(csv_path)))
    t = np.array([float(r["t"]) for r in rows])
    out = {}
    for c in cols:
        if c in rows[0]:
            out[c] = np.array([float(r[c]) for r in rows])
    o = np.argsort(t)
    keep = np.concatenate([[True], np.diff(t[o]) > 0])
    idx = o[keep]
    return t[idx], {c: v[idx] for c, v in out.items()}


def collect(rdir, Np, cols):
    """Per-seed (t, curves) for all seeds of a given Np."""
    runs = []
    for d in sorted(glob.glob(os.path.join(rdir, f"Np{Np}_seed*"))):
        cs = glob.glob(os.path.join(d, "diag_*.csv"))
        if cs:
            runs.append(load_seed_curves(cs[0], cols))
    return runs


def ensemble_on_grid(runs, col, grid):
    """Mean over seeds of `col` interpolated to `grid`; returns (mean, per-seed array)."""
    M = []
    for t, c in runs:
        if col in c:
            M.append(np.interp(grid, t, c[col]))
    if not M:
        return None, None
    M = np.array(M)
    return M.mean(0), M


def gap_time(grid, S_low, S_high, theta, persist=DT_PERSIST):
    """inf{t: ratio>=theta and stays >=theta for >= persist}."""
    ratio = S_high / np.maximum(S_low, 1e-300)
    above = ratio >= theta
    dt = np.median(np.diff(grid)) if len(grid) > 1 else 1e-6
    npers = max(1, int(round(persist / dt)))
    for i in range(len(grid)):
        if above[i] and np.all(above[i:i + npers]):
            return float(grid[i])
    return float("nan")


def bootstrap_gap(grid, low_seeds, high_seeds, theta, nb=400, seed=0):
    """Bootstrap CI of the gap time over seed resampling (paired low/high means)."""
    rng = np.random.default_rng(seed)
    nlo, nhi = low_seeds.shape[0], high_seeds.shape[0]
    vals = []
    for _ in range(nb):
        il = rng.integers(0, nlo, nlo); ih = rng.integers(0, nhi, nhi)
        v = gap_time(grid, low_seeds[il].mean(0), high_seeds[ih].mean(0), theta)
        if v == v:
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 5)), float(np.percentile(vals, 95)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdir", required=True)
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()
    out_dir = args.out_dir or args.rdir
    cols = ([f"S_dg_raw_{n}" for n in (40, 80, 160, 320)]
            + [f"S_dg_cross_{n}" for n in (40, 80, 160, 320)]
            + [f"ppc_{n}" for n in (40, 80, 160, 320)]
            + ["S_L2_u", "R_0.5", "R_0.8", "peak_PK_u"])
    # which Np runs are present
    Nps = sorted({int(os.path.basename(d).split("_")[0][2:])
                  for d in glob.glob(os.path.join(args.rdir, "Np*_seed*"))})
    print("Np present:", Nps)
    data = {Np: collect(args.rdir, Np, cols) for Np in Nps}
    # common time grid = overlap of all runs, 1e-6 spacing
    tmax = min(min(t.max() for t, _ in data[Np]) for Np in Nps if data[Np])
    grid = np.arange(0, tmax + 1e-12, 1e-6)

    # ----- gaps -----
    rows = []

    def add_gap(version, kind, lo_Np, lo_col, hi_Np, hi_col, ppc_col=None):
        if lo_Np not in data or hi_Np not in data or not data[lo_Np] or not data[hi_Np]:
            return
        _, lo_s = ensemble_on_grid(data[lo_Np], lo_col, grid)
        _, hi_s = ensemble_on_grid(data[hi_Np], hi_col, grid)
        if lo_s is None or hi_s is None:
            return
        ppc = ""
        if ppc_col:
            pm, _ = ensemble_on_grid(data[lo_Np], ppc_col, grid)
            ppc = round(float(np.nanmedian(pm)), 1) if pm is not None else ""
        for th in THETAS:
            tb = gap_time(grid, lo_s.mean(0), hi_s.mean(0), th)
            lo_ci, hi_ci = bootstrap_gap(grid, lo_s, hi_s, th)
            rows.append(dict(version=version, kind=kind, theta=th,
                             low=f"({lo_Np},{lo_col})", high=f"({hi_Np},{hi_col})",
                             ppc_low=ppc, tb=tb, ci_low=lo_ci, ci_high=hi_ci,
                             ratio_max=float(np.nanmax(hi_s.mean(0) / np.maximum(lo_s.mean(0), 1e-300)))))

    # Version A (DG cross) -- main matched-ppc pairs + sampling + recon
    add_gap("A_dg_cross", "main",     20000, "S_dg_cross_40", 80000, "S_dg_cross_80", "ppc_40")
    add_gap("A_dg_cross", "main",     80000, "S_dg_cross_80", 320000, "S_dg_cross_160", "ppc_80")
    add_gap("A_dg_cross", "sampling", 20000, "S_dg_cross_80", 80000, "S_dg_cross_80", "ppc_80")
    add_gap("A_dg_cross", "recon",    80000, "S_dg_cross_80", 80000, "S_dg_cross_160", "ppc_80")
    # Version A raw (noise diagnostic)
    add_gap("A_dg_raw", "main",       20000, "S_dg_raw_40", 80000, "S_dg_raw_80", "ppc_40")
    # Version B (Fourier S_L2_u): main pair (2e4,K5)->(8e4,K10)
    add_gap("B_fourier", "main",      20000, "S_L2_u", 80000, "S_L2_u")
    add_gap("B_fourier", "main",      80000, "S_L2_u", 320000, "S_L2_u")

    # ----- write summary CSV -----
    schema = ["version", "kind", "theta", "low", "high", "ppc_low", "tb",
              "ci_low", "ci_high", "ratio_max"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "particle_tb_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=schema, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ----- console + decision -----
    print(f"\nLDG fixed-flux reference: tb(1.05) 80->160={LDG_TB['80->160']:.3e}, "
          f"160->320={LDG_TB['160->320']:.3e}")
    print("\n=== particle blow-up gap times (theta=1.05) ===")
    for r in [x for x in rows if x["theta"] == 1.05]:
        print(f"  {r['version']:12s} {r['kind']:9s} {r['low']:>22s}->{r['high']:<22s} "
              f"ppc~{r['ppc_low']}: tb={_f(r['tb'])}  CI[{_f(r['ci_low'])},{_f(r['ci_high'])}]  "
              f"ratio_max={r['ratio_max']:.2f}")
    _decision(rows)
    print(f"\nwrote particle_tb_summary.csv to {out_dir}")
    json.dump({"ldg_ref": LDG_TB, "rows": [{k: r[k] for k in schema} for r in rows]},
              open(os.path.join(out_dir, "particle_tb_summary.json"), "w"), indent=2, default=str)


def _f(x):
    try:
        return f"{float(x):.3e}"
    except (TypeError, ValueError):
        return str(x)


def _decision(rows):
    def highest_main(v):
        # main gaps for version v, theta=1.05, with a finite tb; pick the one whose
        # low side has the most particles (least shot noise) = highest low_Np.
        m = [r for r in rows if r["version"] == v and r["kind"] == "main"
             and r["theta"] == 1.05 and r["tb"] == r["tb"]]
        if not m:
            return None, None
        m.sort(key=lambda r: int(r["low"].strip("(").split(",")[0]))
        return m[-1]["tb"], m[-1]["low"]
    A, A_pair = highest_main("A_dg_cross")
    B, B_pair = highest_main("B_fourier")
    recon = [r for r in rows if r["version"] == "A_dg_cross" and r["kind"] == "recon"
             and r["theta"] == 1.05 and r["tb"] == r["tb"]]
    R = recon[-1]["tb"] if recon else None
    low_A = [r for r in rows if r["version"] == "A_dg_cross" and r["kind"] == "main"
             and r["theta"] == 1.05 and r["tb"] == r["tb"]]
    onscale = lambda x: x is not None and 3e-5 <= x <= 2e-4
    print("\n=== DECISION (vs LDG fixed-flux tb(1.05) = 5.95e-5 .. 8.43e-5) ===")
    if A is not None:
        print(f"  Version A (LDG-matched DG) main, highest-Np pair {A_pair}: tb={A:.2e} -- "
              + ("ON the LDG scale" if onscale(A) else "OFF the LDG scale"))
    if R is not None:
        print(f"  Version A same-cloud recon gap: tb={R:.2e} -- "
              + ("ON the LDG scale" if onscale(R) else "OFF the LDG scale"))
    if len(low_A) > 1:
        lows = sorted(low_A, key=lambda r: int(r["low"].strip("(").split(",")[0]))
        print(f"  (low-Np pair {lows[0]['low']} tb={float(lows[0]['tb']):.2e}: shot-noise "
              "limited -- the metric needs adequate particles-per-cell.)")
    if B is not None:
        print(f"  Version B (Fourier) main, highest-Np pair {B_pair}: tb={B:.2e}")
    if onscale(A) or onscale(R):
        print("  => SCENARIO 1/2: at adequate particle counts the LDG-matched DG gap is on the "
              "LDG scale. Use Version A (DG) as the LDG-comparable metric; report the low-Np "
              "shot-noise limitation; Fourier as a reconstruction-sensitivity diagnostic.")
    else:
        print("  => SCENARIO 4: no on-scale particle gap; report concentration + radii only.")


if __name__ == "__main__":
    main()
