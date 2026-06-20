"""plot_tetra_control.py -- Figure C: tetrahedral active vs diffusion-control (plan 11).
Reads tetra diagnostics.csv (never runs the solver).  Applies the section 7.3
interpretation rule and prints the verdict (attraction vs merging).  Panels:
 (a) 3D cluster-centroid trajectories (active, seed 0)
 (b) d_min(t): active vs control (seed-mean +/- band)
 (c) m_center(t): active vs control
 (d) overlap O(t) and symmetry residual E_sym(t)
"""
import os, csv, glob, re, argparse
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
        # derived: mean per-cluster quantile radius over the 4 clusters (focusing signal)
        for q in ("R05", "R09"):
            keys = [f"{q}_c{m4}" for m4 in range(4) if f"{q}_c{m4}" in cols]
            if keys:
                cols[f"{q}_mean"] = np.mean([cols[k] for k in keys], axis=0)
        runs.append(dict(arm=m[1], chi=float(m[2]), N=int(m[3]), seed=int(m[6]), cols=cols))
    return runs


def seedmean(runs, arm, col):
    grp = [r for r in runs if r["arm"] == arm]
    if not grp:
        return None, None, None
    tg = grp[0]["cols"]["t"]
    Y = np.array([np.interp(tg, r["cols"]["t"], r["cols"][col]) for r in grp])
    return tg, Y.mean(0), (Y.std(0) if len(Y) > 1 else np.zeros_like(tg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_prefix", default="figureC_tetra_control")
    args = ap.parse_args()
    runs = load(args.run_dir)
    if not runs:
        raise SystemExit("no tetra runs found")

    fig = plt.figure(figsize=(16, 4))
    # (a) 3D centroid trajectories, active seed 0
    ax0 = fig.add_subplot(1, 4, 1, projection="3d")
    act0 = next((r for r in runs if r["arm"] == "active" and r["seed"] == 0), None)
    if act0:
        for m in range(4):
            c = act0["cols"]
            ax0.plot(c[f"c{m}_x"], c[f"c{m}_y"], c[f"c{m}_z"], "-", lw=1.4)
            ax0.scatter(c[f"c{m}_x"][0], c[f"c{m}_y"][0], c[f"c{m}_z"][0], s=20)
    ax0.set_title("(a) centroid paths (active)"); ax0.set_xlabel("x"); ax0.set_ylabel("y")
    # (b) d_min active vs control
    axb = fig.add_subplot(1, 4, 2)
    for arm, c in [("control", "C0"), ("active", "C3")]:
        t, m, s = seedmean(runs, arm, "d_min")
        if t is not None:
            axb.plot(t, m, c, label=arm); axb.fill_between(t, m - s, m + s, color=c, alpha=0.2)
    axb.set_title("(b) $d_{min}(t)$"); axb.set_xlabel("t"); axb.set_ylabel("$d_{min}$")
    axb.legend(fontsize=8)
    # (c) mean per-cluster core radius active vs control (focusing vs spreading)
    axc = fig.add_subplot(1, 4, 3)
    for arm, c in [("control", "C0"), ("active", "C3")]:
        t, m, s = seedmean(runs, arm, "R05_mean")
        if t is not None:
            axc.plot(t, m, c, label=arm); axc.fill_between(t, m - s, m + s, color=c, alpha=0.2)
    axc.set_title("(c) mean per-cluster $R_{0.5}(t)$"); axc.set_xlabel("t")
    axc.set_ylabel("$\\overline{R_{0.5}}$ (cluster)"); axc.legend(fontsize=8)
    # (d) overlap + E_sym
    axd = fig.add_subplot(1, 4, 4)
    for arm, c in [("control", "C0"), ("active", "C3")]:
        t, o, _ = seedmean(runs, arm, "overlap")
        if t is not None:
            axd.plot(t, o, c, label=f"{arm} overlap")
    axd.axhline(1.0, color="0.5", ls=":", lw=1, label="overlap=1")
    axd.set_title("(d) overlap $\\mathcal{O}(t)$"); axd.set_xlabel("t")
    axd.set_ylabel("$\\mathcal{O}$"); axd.legend(fontsize=7)

    fig.suptitle("Figure C: 3D tetrahedral chemotactic aggregation vs diffusion control",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fd = os.path.join(args.run_dir, "figures"); os.makedirs(fd, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"{args.out_prefix}.{ext}"), dpi=200, bbox_inches="tight")
    # save plot data for solver-free regeneration
    pdd = os.path.join(args.run_dir, "plot_data"); os.makedirs(pdd, exist_ok=True)
    pdata = {}
    for arm in ("active", "control"):
        for col in ("d_min", "R05_mean", "R09_mean", "m_center", "overlap", "E_sym"):
            pdata[f"{arm}_{col}"] = np.array(seedmean(runs, arm, col)[:2], dtype=object)
    if act0:
        for m4 in range(4):
            for ax_ in ("x", "y", "z"):
                pdata[f"traj_active_c{m4}_{ax_}"] = act0["cols"][f"c{m4}_{ax_}"]
    np.savez(os.path.join(pdd, f"{args.out_prefix}.npz"), **pdata)

    # ---- section 7.3 interpretation verdict ----
    ta, da, _ = seedmean(runs, "active", "d_min"); tc, dc, _ = seedmean(runs, "control", "d_min")
    _, ra, _ = seedmean(runs, "active", "R05_mean"); _, rc, _ = seedmean(runs, "control", "R05_mean")
    _, oa, _ = seedmean(runs, "active", "overlap")
    verdict = "inconclusive"
    if da is not None and dc is not None:
        dmin_drop = da[-1] < 0.9 * dc[-1]                 # centroids closer than control => attraction
        focus = (ra is not None and rc is not None and ra[-1] < 0.7 * rc[-1])  # clusters hold/collapse vs spread
        merged = oa is not None and oa[-1] <= 1.2         # collapsed cores actually overlap
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
