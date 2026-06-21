"""plot_tetra_control.py -- Figure C: tetrahedral active vs diffusion-control (plan 11).
Reads tetra diagnostics.csv (never runs the solver).  Applies the section 7.3
interpretation rule and prints the verdict (attraction vs merging).  Panels:
 (a) 3D cluster-centroid trajectories (active, seed 0)
 (b) d_min(t): active vs control (seed-mean +/- band)
 (c) m_center(t): active vs control
 (d) overlap O(t) and symmetry residual E_sym(t)

CONFIG-SAFE GROUPING (validation-closure section 7.2): seed means are grouped by the
FULL config tuple (arm, M, a, N, K_dyn, tau); the baseline (N,K,tau,M,a) is taken from
EXPLICIT CLI selectors.  Baseline and refinement runs are NEVER averaged together.
"""
import os, csv, glob, re, argparse, json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load(run_dir):
    runs = []
    for d in sorted(glob.glob(os.path.join(run_dir, "*_chi*_N*_K*_tau*_seed*"))):
        m = re.search(r"(active|control)_chi([0-9.]+)_N(\d+)_K(\d+)_tau([0-9.e-]+)_seed(\d+)",
                      os.path.basename(d))
        cs = os.path.join(d, "diagnostics.csv")
        if not m or not os.path.exists(cs):
            continue
        R = list(csv.DictReader(open(cs)))
        if not R:
            continue
        cols = {c: np.array([float(r[c]) for r in R]) for c in R[0]}
        for q in ("R05", "R09"):
            keys = [f"{q}_c{m4}" for m4 in range(4) if f"{q}_c{m4}" in cols]
            if keys:
                cols[f"{q}_mean"] = np.mean([cols[k] for k in keys], axis=0)
        Mval, aval = np.nan, np.nan
        mf = os.path.join(d, "manifest.json")
        if os.path.exists(mf):
            try:
                mj = json.load(open(mf)); Mval = float(mj.get("M", np.nan)); aval = float(mj.get("a", np.nan))
            except Exception:
                pass
        runs.append(dict(arm=m[1], chi=float(m[2]), N=int(m[3]), K=int(m[4]),
                         tau=float(m[5]), seed=int(m[6]), M=Mval, a=aval, cols=cols))
    return runs


def _match(r, sel):
    for k, v in sel.items():
        rv = r[k]
        if isinstance(v, float):
            if not np.isclose(rv, v, rtol=1e-6, atol=1e-12, equal_nan=True):
                return False
        elif rv != v:
            return False
    return True


def seedmean(runs, sel, col):
    grp = [r for r in runs if _match(r, sel) and col in r["cols"]]
    if not grp:
        return None, None, None
    tg = grp[0]["cols"]["t"]
    Y = np.array([np.interp(tg, r["cols"]["t"], r["cols"][col]) for r in grp])
    return tg, Y.mean(0), (Y.std(0) if len(Y) > 1 else np.zeros_like(tg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--base_N", type=int, default=80000)
    ap.add_argument("--base_K", type=int, default=12)
    ap.add_argument("--base_tau", type=float, default=1e-3)
    ap.add_argument("--M", type=float, default=240.0)
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--out_prefix", default="figureC_tetra_control")
    args = ap.parse_args()
    runs = load(args.run_dir)
    if not runs:
        raise SystemExit("no tetra runs found")
    base = dict(N=args.base_N, K=args.base_K, tau=args.base_tau, M=args.M, a=args.a)
    sa = dict(arm="active", **base); sc = dict(arm="control", **base)

    fig = plt.figure(figsize=(16, 4))
    ax0 = fig.add_subplot(1, 4, 1, projection="3d")
    act0 = next((r for r in runs if _match(r, sa) and r["seed"] == 0), None)
    if act0:
        for m in range(4):
            c = act0["cols"]
            ax0.plot(c[f"c{m}_x"], c[f"c{m}_y"], c[f"c{m}_z"], "-", lw=1.4)
            ax0.scatter(c[f"c{m}_x"][0], c[f"c{m}_y"][0], c[f"c{m}_z"][0], s=20)
    ax0.set_title("(a) centroid paths (active)"); ax0.set_xlabel("x"); ax0.set_ylabel("y")
    axb = fig.add_subplot(1, 4, 2)
    for sel, lab, c in [(sc, "control", "C0"), (sa, "active", "C3")]:
        t, m, s = seedmean(runs, sel, "d_min")
        if t is not None:
            axb.plot(t, m, c, label=lab); axb.fill_between(t, m - s, m + s, color=c, alpha=0.2)
    axb.set_title("(b) $d_{min}(t)$"); axb.set_xlabel("t"); axb.set_ylabel("$d_{min}$")
    axb.legend(fontsize=8)
    axc = fig.add_subplot(1, 4, 3)
    for sel, lab, c in [(sc, "control", "C0"), (sa, "active", "C3")]:
        t, m, s = seedmean(runs, sel, "R05_mean")
        if t is not None:
            axc.plot(t, m, c, label=lab); axc.fill_between(t, m - s, m + s, color=c, alpha=0.2)
    axc.set_title("(c) mean per-cluster $R_{0.5}(t)$"); axc.set_xlabel("t")
    axc.set_ylabel("$\\overline{R_{0.5}}$ (cluster)"); axc.legend(fontsize=8)
    axd = fig.add_subplot(1, 4, 4)
    for sel, lab, c in [(sc, "control", "C0"), (sa, "active", "C3")]:
        t, o, _ = seedmean(runs, sel, "overlap")
        if t is not None:
            axd.plot(t, o, c, label=f"{lab} overlap")
    axd.axhline(1.0, color="0.5", ls=":", lw=1, label="overlap=1")
    axd.set_title("(d) overlap $\\mathcal{O}(t)$"); axd.set_xlabel("t")
    axd.set_ylabel("$\\mathcal{O}$"); axd.legend(fontsize=7)

    fig.suptitle(f"Figure C: 3D tetrahedral chemotactic aggregation vs diffusion control "
                 f"(M={args.M:g}, a={args.a:g}, N={args.base_N}, K={args.base_K}, "
                 f"tau={args.base_tau:g})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fd = os.path.join(args.run_dir, "figures"); os.makedirs(fd, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"{args.out_prefix}.{ext}"), dpi=200, bbox_inches="tight")
    pdd = os.path.join(args.run_dir, "plot_data"); os.makedirs(pdd, exist_ok=True)
    pdata = dict(base_N=args.base_N, base_K=args.base_K, base_tau=args.base_tau,
                 M=args.M, a=args.a)
    for sel, arm in [(sa, "active"), (sc, "control")]:
        for col in ("d_min", "R05_mean", "R09_mean", "m_center", "overlap", "E_sym"):
            pdata[f"{arm}_{col}"] = np.array(seedmean(runs, sel, col)[:2], dtype=object)
    if act0:
        for m4 in range(4):
            for ax_ in ("x", "y", "z"):
                pdata[f"traj_active_c{m4}_{ax_}"] = act0["cols"][f"c{m4}_{ax_}"]
    np.savez(os.path.join(pdd, f"{args.out_prefix}.npz"), **pdata)

    ta, da, _ = seedmean(runs, sa, "d_min"); tc, dc, _ = seedmean(runs, sc, "d_min")
    _, ra, _ = seedmean(runs, sa, "R05_mean"); _, rc, _ = seedmean(runs, sc, "R05_mean")
    _, oa, _ = seedmean(runs, sa, "overlap")
    verdict = "inconclusive"
    if da is not None and dc is not None:
        dmin_drop = da[-1] < 0.9 * dc[-1]
        focus = (ra is not None and rc is not None and ra[-1] < 0.7 * rc[-1])
        merged = oa is not None and oa[-1] <= 1.2
        if dmin_drop and focus and merged:
            verdict = "cluster MERGING (d_min<<control, clusters collapse, cores overlap)"
        elif dmin_drop and focus:
            verdict = ("mutual chemotactic ATTRACTION + individual COLLAPSE "
                       "(d_min<control, per-cluster R_0.5 collapses vs spreading control)")
        elif dmin_drop:
            verdict = "mutual chemotactic ATTRACTION (d_min<control)"
        else:
            verdict = "no aggregation distinguishable from diffusion control"
        print(f"active d_min {da[0]:.3f}->{da[-1]:.3f}; control {dc[0]:.3f}->{dc[-1]:.3f}")
        if ra is not None and rc is not None:
            print(f"active mean-cluster R0.5(T) {ra[-1]:.3f} vs control {rc[-1]:.3f}; "
                  f"overlap(T) {oa[-1]:.2f}")
    print(f"VERDICT (plan 7.3): {verdict}")
    print(f"wrote {fd}/{args.out_prefix}.pdf/.png")


if __name__ == "__main__":
    main()
