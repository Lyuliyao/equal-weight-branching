"""Resolution-gap focusing indicator t_gap from PAIRED LDG runs.

PURE POST-PROCESSING.  No simulation.  Reads two diag_*.csv produced by
simulation_ldg.py:
  * a BASE run at resolution (N, K),
  * a REFINED run at (4N, 2K),
both started from the same IC and run with the same dt over a common time span.

Idea (LDG focusing test).  Let S_{K,N}(t) = ||P_K mu||_{L2}(t) (the `S_L2`
column).  While the solution is well resolved by both meshes the two
reconstructed L2 norms agree; once focusing/concentration outruns the coarse
resolution, the refined run resolves MORE L2 mass and the ratio

    ratio(t) = S_{2K,4N}(t) / S_{K,N}(t)

grows past 1.  The resolution-gap focusing time is

    t_gap = inf{ t : ratio(t) >= thresh }

for thresh in {1.05, 1.10}.  A small/early t_gap (and a t_gap that DECREASES as
the base resolution increases) is evidence that under-resolution sets in before
the nominal final time -- the LDG "resolution gap opens" signature.

Both series are linearly interpolated to a common time grid (the intersection of
their time spans) before forming the ratio.  Output: a small JSON + CSV table
t_gap(N,K) over the supplied pairs.
"""
import os
import sys
import csv
import json
import argparse

import numpy as np


def load_diag(path):
    with open(path) as f:
        r = list(csv.DictReader(f))
    # skip non-float columns (e.g. the string `solver_field_mode` added for the
    # solver-field diagnostics); this loader only uses numeric columns.
    out = {}
    for k in r[0]:
        try:
            out[k] = np.array([float(x[k]) for x in r])
        except (ValueError, TypeError):
            out[k] = np.array([x[k] for x in r])
    return out


def common_grid(t_base, t_ref, n=None):
    """Common time grid = overlap of the two spans, sampled at the union of the
    base times that fall inside the overlap (or n uniform points if n given)."""
    lo = max(t_base.min(), t_ref.min())
    hi = min(t_base.max(), t_ref.max())
    if hi <= lo:
        return None
    if n is not None:
        return np.linspace(lo, hi, n)
    g = t_base[(t_base >= lo) & (t_base <= hi)]
    if g.size < 2:
        g = np.linspace(lo, hi, 64)
    return g


def t_gap_from_pair(base_csv, ref_csv, threshes=(1.05, 1.10), col="S_L2",
                    n_grid=None):
    """Return dict with t_gap for each thresh, plus the (N,K) metadata of the
    base run and the peak ratio attained."""
    cb = load_diag(base_csv)
    cr = load_diag(ref_csv)

    # validate the (N,K) -> (4N,2K) pairing so an accidental wrong pair fails loudly
    N_base, K_base = int(cb["N"][0]), int(cb["K"][0])
    N_ref, K_ref = int(cr["N"][0]), int(cr["K"][0])
    if N_ref != 4 * N_base or K_ref != 2 * K_base:
        raise ValueError(
            f"t_gap pairing must be (N,K)->(4N,2K): got base=({N_base},{K_base}), "
            f"ref=({N_ref},{K_ref}). Check that base_csv/ref_csv are the right pair.")

    tb, sb = cb["t"], cb[col]
    tr, sr = cr["t"], cr[col]

    # sort + dedup for safe interpolation
    def _clean(t, s):
        o = np.argsort(t)
        t, s = t[o], s[o]
        keep = np.concatenate([[True], np.diff(t) > 0])
        return t[keep], s[keep]
    tb, sb = _clean(tb, sb)
    tr, sr = _clean(tr, sr)

    grid = common_grid(tb, tr, n=n_grid)
    if grid is None:
        return {"error": "no time overlap between base and refined runs"}

    Sb = np.interp(grid, tb, sb)
    Sr = np.interp(grid, tr, sr)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = Sr / Sb

    out = {
        "base_csv": os.path.basename(base_csv),
        "ref_csv": os.path.basename(ref_csv),
        "N_base": int(cb["N"][0]),
        "K_base": int(cb["K"][0]),
        "N_ref": int(cr["N"][0]),
        "K_ref": int(cr["K"][0]),
        "t_overlap": [float(grid[0]), float(grid[-1])],
        "ratio_max": float(np.nanmax(ratio)),
        "ratio_final": float(ratio[-1]),
        "column": col,
    }
    for thr in threshes:
        hit = np.where(np.isfinite(ratio) & (ratio >= thr))[0]
        out[f"t_gap_{thr:.2f}"] = (float(grid[hit[0]]) if hit.size
                                   else float("nan"))
    return out


def main():
    p = argparse.ArgumentParser(description="t_gap(N,K) from paired LDG runs")
    p.add_argument("--pairs", nargs="+", required=True,
                   help="base.csv:refined.csv pairs (colon-separated)")
    p.add_argument("--threshes", type=float, nargs="+",
                   default=[1.05, 1.10])
    p.add_argument("--col", type=str, default="S_L2")
    p.add_argument("--n_grid", type=int, default=None,
                   help="optional #points for a uniform common grid")
    p.add_argument("--out", type=str, default="results/tgap_table")
    args = p.parse_args()

    table = []
    for pr in args.pairs:
        base, ref = pr.split(":")
        rec = t_gap_from_pair(base, ref, threshes=tuple(args.threshes),
                              col=args.col, n_grid=args.n_grid)
        table.append(rec)
        print(json.dumps(rec, indent=2))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(table, f, indent=2)
    # flat CSV
    cols = ["N_base", "K_base", "N_ref", "K_ref"] \
        + [f"t_gap_{t:.2f}" for t in args.threshes] \
        + ["ratio_max", "ratio_final", "base_csv", "ref_csv"]
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for rec in table:
            if "error" in rec:
                continue
            w.writerow([rec.get(c, "") for c in cols])
    print(f"\nwrote {args.out}.json and {args.out}.csv")


if __name__ == "__main__":
    main()
