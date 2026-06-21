"""vc_load.py -- shared, config-safe loader for validation-closure analysis/plot
scripts.  Parses run directories <label>_chi<>_N<>_K<>_tau<>_seed<>/diagnostics.csv
and exposes config-tuple-grouped seed means.  NEVER pools runs whose (M,N,K,tau)
differ: every selector must be matched explicitly (floats by tolerance).
"""
import os, csv, glob, re, json
import numpy as np

_PAT = r"(\w+?)_chi([0-9.]+)_N(\d+)_K(\d+)_tau([0-9.e+-]+)_seed(\d+)"


def load_runs(run_dir):
    runs = []
    for d in sorted(glob.glob(os.path.join(run_dir, "*_chi*_N*_K*_tau*_seed*"))):
        m = re.search(_PAT, os.path.basename(d))
        cs = os.path.join(d, "diagnostics.csv")
        if not m or not os.path.exists(cs):
            continue
        R = list(csv.DictReader(open(cs)))
        if not R:
            continue
        cols = {c: np.array([float(r[c]) for r in R]) for c in R[0]}
        man = {}
        mf = os.path.join(d, "manifest.json")
        if os.path.exists(mf):
            try:
                man = json.load(open(mf))
            except Exception:
                pass
        runs.append(dict(label=m[1], arm=m[1], chi=float(m[2]), N=int(m[3]), K=int(m[4]),
                         tau=float(m[5]), seed=int(m[6]),
                         M=float(man.get("M", np.nan)), a=float(man.get("a", np.nan)),
                         cols=cols, dir=d))
    return runs


def match(r, sel):
    for k, v in sel.items():
        rv = r.get(k)
        if isinstance(v, float):
            if not np.isclose(rv, v, rtol=1e-6, atol=1e-12, equal_nan=True):
                return False
        elif rv != v:
            return False
    return True


def select(runs, sel):
    return [r for r in runs if match(r, sel)]


def seedmean(runs, sel, col, tgrid=None):
    """Seed-mean curve of `col` over runs matching the FULL config `sel`.
    Returns (t, mean, std, nseed); (None,None,None,0) if no run matches."""
    grp = [r for r in runs if match(r, sel) and col in r["cols"]]
    if not grp:
        return None, None, None, 0
    tg = grp[0]["cols"]["t"] if tgrid is None else np.asarray(tgrid)
    Y = np.array([np.interp(tg, r["cols"]["t"], r["cols"][col]) for r in grp])
    return tg, Y.mean(0), (Y.std(0) if len(Y) > 1 else np.zeros_like(tg)), len(grp)


def t_turn(t, R):
    """Turnover time = argmax of R_0.5 (peak-expansion turnover, the existing rule)."""
    return float(t[int(np.argmax(R))])


def t_focus(t, R, frac=0.9):
    """First time R drops to <= frac*R(0) (10%-focusing time at frac=0.9)."""
    below = t[R <= frac * R[0]]
    return float(below[0]) if below.size else float("nan")
