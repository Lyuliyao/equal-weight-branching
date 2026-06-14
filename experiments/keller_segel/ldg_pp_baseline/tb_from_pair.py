"""
LDG-style numerical blow-up proxy t_b from a grid-refinement pair (CLAUDE.md §7.1.3).
=====================================================================================

Reads two S_curves.csv produced by fvm_baseline.py at resolutions n and 2n
(same IC, same physical time span) and forms the resolution-gap time

    t_b(n; theta) = inf{ t : S_{2n}(t) >= theta S_n(t) },   S_n(t) = ||u_n(t)||_L2.

While the solution is resolved by both grids the two L2 norms agree; once
concentration outruns the coarse grid the refined grid resolves more L2 mass and
the ratio grows past theta.  A finite, O(1e-4) t_b is the LDG-style numerical
blow-up indicator (NOT the continuum blow-up time).

Usage:
    python tb_from_pair.py --pairs results/n128/S_curves.csv:results/n256/S_curves.csv \
                                   results/n256/S_curves.csv:results/n512/S_curves.csv \
           --out results/tb_table
"""
import os
import csv
import json
import argparse

import numpy as np


def load(path, col="S_L2"):
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r["t"]) for r in rows])
    S = np.array([float(r[col]) for r in rows])
    n = int(float(rows[0]["n"]))
    o = np.argsort(t)
    t, S = t[o], S[o]
    keep = np.concatenate([[True], np.diff(t) > 0])
    return t[keep], S[keep], n


def tb_pair(base_csv, ref_csv, threshes=(1.05, 1.10), col="S_L2"):
    tb, Sb, nb = load(base_csv, col)
    tr, Sr, nr = load(ref_csv, col)
    if nr != 2 * nb:
        raise ValueError(f"pair must be (n)->(2n): got base n={nb}, ref n={nr}")
    lo = max(tb.min(), tr.min())
    hi = min(tb.max(), tr.max())
    grid = tb[(tb >= lo) & (tb <= hi)]
    if grid.size < 2:
        grid = np.linspace(lo, hi, 128)
    Sb_i = np.interp(grid, tb, Sb)
    Sr_i = np.interp(grid, tr, Sr)
    ratio = Sr_i / Sb_i
    out = dict(n_base=nb, n_ref=nr, t_overlap=[float(grid[0]), float(grid[-1])],
               ratio_max=float(np.nanmax(ratio)), ratio_final=float(ratio[-1]))
    for thr in threshes:
        hit = np.where(np.isfinite(ratio) & (ratio >= thr))[0]
        out[f"t_b_{thr:.2f}"] = float(grid[hit[0]]) if hit.size else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="base.csv:ref.csv pairs (colon-separated, n->2n)")
    ap.add_argument("--threshes", type=float, nargs="+", default=[1.05, 1.10])
    ap.add_argument("--col", default="S_L2", help="S_L2 (global) or S_core")
    ap.add_argument("--out", default="results/tb_table")
    args = ap.parse_args()
    table = []
    for pr in args.pairs:
        base, ref = pr.split(":")
        rec = tb_pair(base, ref, threshes=tuple(args.threshes), col=args.col)
        rec["col"] = args.col
        table.append(rec)
        print(json.dumps(rec, indent=2))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(table, f, indent=2)
    cols = ["n_base", "n_ref"] + [f"t_b_{t:.2f}" for t in args.threshes] \
        + ["ratio_max", "ratio_final"]
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for rec in table:
            w.writerow([rec.get(c, "") for c in cols])
    print(f"\nwrote {args.out}.json/.csv")


if __name__ == "__main__":
    main()
