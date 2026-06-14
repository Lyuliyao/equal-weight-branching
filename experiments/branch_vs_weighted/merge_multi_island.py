"""Merge per-seed multi_island.py output directories into one combined run dir.

The SLURM array runs one seed per task into seed_<i>/ subdirs; this concatenates
their CSVs, recomputes metrics_summary (mean +/- std over seeds), and copies the
field npz + clouds (from the lowest seed) into the combined dir.

Usage:  python merge_multi_island.py --in_dir <run_dir> --seeds 0 1 2 3 4 5 6 7
"""
import os
import sys
import csv
import glob
import json
import shutil
import argparse

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from multi_island import _write_csv  # noqa: E402


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="combined run dir containing seed_*/")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    args = ap.parse_args()
    out = args.in_dir
    os.makedirs(out, exist_ok=True)

    seed_dirs = [os.path.join(out, f"seed_{s}") for s in args.seeds]
    seed_dirs = [d for d in seed_dirs if os.path.isdir(d)]
    if not seed_dirs:
        print("no seed_* dirs found in", out); return

    ts, isl, isle, per = [], [], [], []
    for d in seed_dirs:
        ts += read_rows(os.path.join(d, "time_series.csv"))
        isl += read_rows(os.path.join(d, "island_masses.csv"))
        isle += read_rows(os.path.join(d, "island_local_ess.csv"))
        per += read_rows(os.path.join(d, "per_seed_metrics.csv"))

    _write_csv(os.path.join(out, "time_series.csv"), ts,
               list(ts[0]) if ts else [])
    _write_csv(os.path.join(out, "island_masses.csv"), isl, list(isl[0]) if isl else [])
    _write_csv(os.path.join(out, "island_local_ess.csv"), isle, list(isle[0]) if isle else [])
    _write_csv(os.path.join(out, "per_seed_metrics.csv"), per, list(per[0]) if per else [])

    # metrics_summary over all merged seeds
    methods = []
    for r in per:
        if r["method"] not in methods:
            methods.append(r["method"])
    fnum = lambda r, c: float(r[c]) if r.get(c) not in (None, "", "nan") else np.nan
    summ = []
    for method in methods:
        sub = [r for r in per if r["method"] == method]
        def ms(c):
            v = [fnum(r, c) for r in sub]
            return float(np.nanmean(v)), float(np.nanstd(v))
        row = dict(method=method, n_seeds=len(sub),
                   N0=int(float(sub[0]["N0"])))
        for c in ["Nact_final", "particle_steps", "global_rel_L2", "global_nESS",
                  "max_mean_weight", "mean_Em", "median_Em", "max_Em",
                  "num_Em_gt_20pct", "min_local_eff", "median_local_eff", "n_resamples"]:
            m, s = ms(c)
            row[c] = m
            if c in ("mean_Em", "max_Em"):
                row[c + "_std"] = s
        summ.append(row)
    _write_csv(os.path.join(out, "metrics_summary.csv"), summ,
               ["method", "n_seeds", "N0", "Nact_final", "particle_steps",
                "global_rel_L2", "global_nESS", "max_mean_weight", "mean_Em",
                "mean_Em_std", "median_Em", "max_Em", "max_Em_std",
                "num_Em_gt_20pct", "min_local_eff", "median_local_eff", "n_resamples"])

    # copy field npz + clouds + config/manifest from the lowest seed dir
    d0 = seed_dirs[0]
    for f in ["fields_ref.npz", "config.json", "manifest.json"]:
        if os.path.exists(os.path.join(d0, f)):
            shutil.copy(os.path.join(d0, f), os.path.join(out, f))
    for f in glob.glob(os.path.join(d0, "fields_seed*.npz")):
        shutil.copy(f, out)
    # also bring forward per-seed field npz from every seed dir
    for d in seed_dirs:
        for f in glob.glob(os.path.join(d, "fields_seed*.npz")):
            shutil.copy(f, out)
    if os.path.isdir(os.path.join(d0, "clouds")):
        os.makedirs(os.path.join(out, "clouds"), exist_ok=True)
        for f in glob.glob(os.path.join(d0, "clouds", "*.npz")):
            shutil.copy(f, os.path.join(out, "clouds"))

    print(f"merged {len(seed_dirs)} seeds into {out}")
    for r in summ:
        print(f"{r['method']:24s}: mean E_m={r['mean_Em']:.3f}+/-{r.get('mean_Em_std',0):.3f}  "
              f"max E_m={r['max_Em']:.3f}  #(>20%)={r['num_Em_gt_20pct']:.1f}  "
              f"nESS={r['global_nESS']:.3f}  L2={r['global_rel_L2']:.3f}  "
              f"minLocEff={r['min_local_eff']:.0f}")


if __name__ == "__main__":
    main()
