"""Merge per-seed staged_multi_island.py output dirs into one combined run dir.

The SLURM array runs one seed per task into seed_<i>/; this concatenates their
CSVs, recomputes metrics_summary + late_group_metrics (mean +/- std over seeds),
and copies field npz + clouds (from the lowest seed) into the combined dir.

Usage:  python merge_staged_multi_island.py --in_dir <run_dir> --seeds 0 1 2 ...
"""
import os
import sys
import csv
import glob
import shutil
import argparse

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from staged_multi_island import _write_csv  # noqa: E402


def read_rows(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def fnum(r, c):
    v = r.get(c)
    return float(v) if v not in (None, "", "nan") else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    args = ap.parse_args()
    out = args.in_dir
    seed_dirs = [os.path.join(out, f"seed_{s}") for s in args.seeds if os.path.isdir(os.path.join(out, f"seed_{s}"))]
    if not seed_dirs:
        print("no seed_* dirs in", out); return

    ts, isl, per = [], [], []
    for d in seed_dirs:
        ts += read_rows(os.path.join(d, "time_series.csv"))
        isl += read_rows(os.path.join(d, "island_masses.csv"))
        per += read_rows(os.path.join(d, "per_seed_metrics.csv"))
    _write_csv(os.path.join(out, "time_series.csv"), ts, list(ts[0]) if ts else [])
    _write_csv(os.path.join(out, "island_masses.csv"), isl, list(isl[0]) if isl else [])
    _write_csv(os.path.join(out, "island_local_ess.csv"), isl,
               ["seed", "method", "m", "group", "is_late", "amplitude", "local_eff", "E_m", "local_L2"])
    _write_csv(os.path.join(out, "per_seed_metrics.csv"), per, list(per[0]) if per else [])

    methods = []
    for r in per:
        if r["method"] not in methods:
            methods.append(r["method"])
    cols = ["method", "n_seeds", "N0", "particle_steps", "Nact_final", "global_rel_L2",
            "global_nESS", "mean_Em", "max_Em", "num_Em_gt_20pct", "mean_late_Em",
            "mean_late_Em_std", "max_late_Em", "max_late_localL2", "num_late_gt_20pct",
            "min_local_eff", "min_late_local_eff"]
    summ = []
    for method in methods:
        sub = [r for r in per if r["method"] == method]
        mean = lambda c: float(np.nanmean([fnum(r, c) for r in sub]))
        std = lambda c: float(np.nanstd([fnum(r, c) for r in sub]))
        row = dict(method=method, n_seeds=len(sub), N0=int(mean("N0")))
        for c in cols[3:]:
            if c == "mean_late_Em_std":
                row[c] = std("mean_late_Em")
            else:
                row[c] = mean(c)
        summ.append(row)
    _write_csv(os.path.join(out, "metrics_summary.csv"), summ, cols)
    _write_csv(os.path.join(out, "late_group_metrics.csv"), summ, cols)

    d0 = seed_dirs[0]
    for f in ["fields_ref.npz", "config.json", "manifest.json"]:
        if os.path.exists(os.path.join(d0, f)):
            shutil.copy(os.path.join(d0, f), os.path.join(out, f))
    for d in seed_dirs:
        for f in glob.glob(os.path.join(d, "fields_seed*.npz")):
            shutil.copy(f, out)
    if os.path.isdir(os.path.join(d0, "clouds")):
        os.makedirs(os.path.join(out, "clouds"), exist_ok=True)
        for f in glob.glob(os.path.join(d0, "clouds", "*.npz")):
            shutil.copy(f, os.path.join(out, "clouds"))

    print(f"merged {len(seed_dirs)} seeds into {out}")
    late_n = sum(1 for r in isl if r["method"] == methods[0] and int(r["is_late"]) == 1 and int(r["seed"]) == int(per[0]["seed"]))
    for r in summ:
        print(f"{r['method']:34s}: late mean E_m={r['mean_late_Em']:.3f}+/-{r['mean_late_Em_std']:.3f} "
              f"max={r['max_late_Em']:.3f} lateLocL2={r['max_late_localL2']:.3f} "
              f"#late>20%={r['num_late_gt_20pct']:.1f} nESS={r['global_nESS']:.3f} "
              f"minLateEff={r['min_late_local_eff']:.0f} Nact={r['Nact_final']/r['N0']:.1f}x ps={r['particle_steps']:.2e}")


if __name__ == "__main__":
    main()
