"""analyze_tetra_validation.py -- validation-closure Task D: tetra one-factor refinement
+ centroid-reliability closure.  Reads diagnostics only (never runs the solver).

Baseline M=240, a=1.0, sigma_c=0.25, N=80000, K=12, tau=1e-3, active(chi1)/control(chi0),
seeds 0-3; one-factor refinements N=160000, K=16, tau=5e-4 at seed 0.  For each arm and
config it reports d_min, the six pairwise centroid distances, mean per-cluster R_0.5/R_0.9,
overlap, central mass, symmetry residual, the conservative centroid-reliability score
A_min(t)=min_m min_j A_{m,j}, drift-resolution, and M_v error.

Centroid-reliable interval: while the diffusion control's A_min stays above a threshold
(default 0.2).  Beyond it the broad control centroid (hence d_min) is unreliable.

Writes tetra_refinement_metrics.csv + figure_tetra_resolution_validation.
"""
import os, csv, argparse, sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import vc_load as V

L = 12.0


def hK(K):
    return L / (2 * K + 1)


def add_derived(runs):
    for r in runs:
        c = r["cols"]
        for q in ("R05", "R09"):
            keys = [f"{q}_c{m}" for m in range(4) if f"{q}_c{m}" in c]
            if keys:
                c[f"{q}_mean"] = np.mean([c[k] for k in keys], axis=0)
        amk = [f"A_c{m}" for m in range(4) if f"A_c{m}" in c]
        if amk:
            c["A_min"] = np.min([c[k] for k in amk], axis=0)
    return runs


def reliable_tmax(runs, sel_control, thr):
    t, A, _, n = V.seedmean(runs, sel_control, "A_min")
    if t is None:
        return np.nan
    ok = t[A >= thr]
    return float(ok[-1]) if ok.size else float(t[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refine_dir", required=True, help="tetra_refinement run dir")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--rel_thr", type=float, default=0.2, help="centroid-reliability threshold")
    ap.add_argument("--prefix", default="figure_tetra_resolution_validation")
    args = ap.parse_args()
    runs = add_derived(V.load_runs(args.refine_dir))
    if not runs:
        raise SystemExit("no tetra refinement runs found")

    M, a = 240.0, 1.0
    BASE = dict(N=80000, K=12, tau=1e-3, M=M, a=a)
    configs = [("baseline", BASE, "-"),
               ("N=160k", dict(N=160000, K=12, tau=1e-3, M=M, a=a), "--"),
               ("K=16", dict(N=80000, K=16, tau=1e-3, M=M, a=a), ":"),
               ("tau=5e-4", dict(N=80000, K=12, tau=5e-4, M=M, a=a), "-.")]

    t_reliable = reliable_tmax(runs, dict(arm="control", **BASE), args.rel_thr)

    # ---- metrics CSV ----
    rows = []
    for name, cfg, _ in configs:
        for arm in ("active", "control"):
            sel = dict(arm=arm, **cfg)
            grp = V.select(runs, sel)
            if not grp:
                continue
            t, dmin, dmins, n = V.seedmean(runs, sel, "d_min")
            _, r05, _, _ = V.seedmean(runs, sel, "R05_mean")
            _, ov, _, _ = V.seedmean(runs, sel, "overlap")
            _, mc, _, _ = V.seedmean(runs, sel, "m_center")
            _, es, _, _ = V.seedmean(runs, sel, "E_sym")
            _, amin, _, _ = V.seedmean(runs, sel, "A_min")
            _, dr, _, _ = V.seedmean(runs, sel, "drift_resolution_number")
            # M_v rel error (max over t where exact significant), worst over seeds
            mverr = 0.0
            for r in grp:
                mvx, mv = r["cols"]["M_v_exact"], r["cols"]["M_v"]
                mask = mvx > 1e-3 * M
                if mask.any():
                    mverr = max(mverr, float(np.max(np.abs(mv[mask] - mvx[mask]) / mvx[mask])))
            # value at reliable tmax
            ir = int(np.argmin(np.abs(t - t_reliable))) if np.isfinite(t_reliable) else -1
            rows.append(dict(config=name, arm=arm, N=cfg["N"], K=cfg["K"], tau=cfg["tau"],
                             nseed=n, dmin_0=float(dmin[0]), dmin_T=float(dmin[-1]),
                             dmin_relT=float(dmin[ir]), R05mean_T=float(r05[-1]),
                             overlap_T=float(ov[-1]), m_center_T=float(mc[-1]),
                             E_sym_T=float(es[-1]), A_min_T=float(amin[-1]),
                             A_min_relT=float(amin[ir]), drift_res_max=float(dr.max()),
                             Mv_relerr_max=mverr, hK=hK(cfg["K"])))
    out_dir = args.refine_dir
    with open(os.path.join(out_dir, "tetra_refinement_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)

    # active-control d_min gap over the reliable interval (baseline vs refinements)
    gaps = {}
    for name, cfg, _ in configs:
        ta, da, _, na = V.seedmean(runs, dict(arm="active", **cfg), "d_min")
        tc, dc, _, nc = V.seedmean(runs, dict(arm="control", **cfg), "d_min")
        if ta is None or tc is None:
            continue
        dci = np.interp(ta, tc, dc)
        gap = dci - da                                   # control - active (attraction => >0)
        mm = ta <= (t_reliable if np.isfinite(t_reliable) else ta[-1])
        gaps[name] = (ta, gap, float(np.mean(gap[mm])), float(da[-1]), float(dci[-1]))
    base_gap = gaps.get("baseline", (None, None, np.nan))[2]

    # traceability CSV for the active-control d_min gap numbers quoted in the report
    with open(os.path.join(args.refine_dir, "tetra_gap_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "t_reliable", "mean_gap_reliable", "pct_vs_baseline",
                    "active_dmin_T", "control_dmin_T", "final_instant_gap",
                    "attraction_mean_positive"])
        for name, _, _ in configs:
            if name not in gaps:
                continue
            _, _, gm, daT, dcT = gaps[name]
            pct = (gm - base_gap) / base_gap * 100 if (base_gap and np.isfinite(base_gap)) else np.nan
            w.writerow([name, round(t_reliable, 4), round(gm, 4), round(pct, 1),
                        round(daT, 4), round(dcT, 4), round(dcT - daT, 4), bool(gm > 0)])

    # ---- figure ----
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    A = ax[0, 0]
    for name, cfg, ls in configs:
        ta, da, das, na = V.seedmean(runs, dict(arm="active", **cfg), "d_min")
        tc, dc, dcs, nc = V.seedmean(runs, dict(arm="control", **cfg), "d_min")
        if ta is not None:
            A.plot(ta, da, ls, color="C3", lw=1.4, label=f"active {name}")
        if tc is not None:
            A.plot(tc, dc, ls, color="C0", lw=1.0, label=f"control {name}")
    if np.isfinite(t_reliable):
        A.axvline(t_reliable, color="0.4", lw=1); A.text(t_reliable, A.get_ylim()[1]*0.9,
                                                         "reliable\nlimit", fontsize=6)
    A.set_title("(a) $d_{min}(t)$ active (red) vs control (blue)"); A.set_xlabel("t")
    A.set_ylabel("$d_{min}$"); A.legend(fontsize=6, ncol=2)
    B = ax[0, 1]
    for name, cfg, ls in configs:
        if name in gaps:
            ta, gap, gm = gaps[name][0], gaps[name][1], gaps[name][2]
            B.plot(ta, gap, ls, lw=1.3, label=f"{name} (mean gap {gm:.2f})")
    if np.isfinite(t_reliable):
        B.axvspan(0, t_reliable, color="0.9", alpha=0.6, lw=0)
    B.axhline(0, color="0.5", lw=0.8)
    B.set_title("(b) active$-$control $d_{min}$ gap (shaded=reliable interval)")
    B.set_xlabel("t"); B.set_ylabel("$d_{min}^{ctrl}-d_{min}^{act}$"); B.legend(fontsize=6)
    C = ax[1, 0]
    for name, cfg, ls in configs:
        ta, ra, _, _ = V.seedmean(runs, dict(arm="active", **cfg), "R05_mean")
        tc, rc, _, _ = V.seedmean(runs, dict(arm="control", **cfg), "R05_mean")
        if ta is not None:
            C.plot(ta, ra, ls, color="C3", lw=1.3, label=f"active {name}")
        if tc is not None:
            C.plot(tc, rc, ls, color="C0", lw=1.0, label=f"control {name}")
    for k, c in [(12, "0.5"), (16, "0.7")]:
        C.axhline(hK(k), color=c, ls="--", lw=0.8)
        C.text(C.get_xlim()[1]*0.7, hK(k)*1.02, f"$h_K$(K={k})={hK(k):.2f}", fontsize=6, color=c)
    C.set_title("(c) mean per-cluster $R_{0.5}(t)$ (field scale $h_K$ marked)")
    C.set_xlabel("t"); C.set_ylabel("$\\overline{R_{0.5}}$"); C.legend(fontsize=6, ncol=2)
    D = ax[1, 1]
    for name, cfg, ls in configs:
        ta, aa, _, _ = V.seedmean(runs, dict(arm="active", **cfg), "A_min")
        tc, ac, _, _ = V.seedmean(runs, dict(arm="control", **cfg), "A_min")
        if ta is not None:
            D.plot(ta, aa, ls, color="C3", lw=1.3, label=f"active {name}")
        if tc is not None:
            D.plot(tc, ac, ls, color="C0", lw=1.0, label=f"control {name}")
    for thr in (0.1, 0.2, 0.3):
        D.axhline(thr, color="0.6", ls=":", lw=0.7)
    D.set_title("(d) centroid reliability $A_{min}(t)=\\min_m\\min_j A_{m,j}$")
    D.set_xlabel("t"); D.set_ylabel("$A_{min}$"); D.legend(fontsize=6, ncol=2)

    fig.suptitle("Tetra resolution / centroid-reliability validation "
                 "(M=240, a=1.0; one-factor N/K/tau refinement)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fd = os.path.join(args.out_root, "figures"); os.makedirs(fd, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"{args.prefix}.{ext}"), dpi=200, bbox_inches="tight")
    pdd = os.path.join(args.out_root, "plot_data"); os.makedirs(pdd, exist_ok=True)
    save = dict(t_reliable=t_reliable, rel_thr=args.rel_thr, base_gap=base_gap)
    for name, cfg, _ in configs:
        for arm in ("active", "control"):
            for col in ("d_min", "R05_mean", "A_min", "overlap"):
                t, m, s, n = V.seedmean(runs, dict(arm=arm, **cfg), col)
                if t is not None:
                    save[f"{name}_{arm}_{col}_t"] = t; save[f"{name}_{arm}_{col}"] = m
        if name in gaps:
            save[f"{name}_gap_t"] = gaps[name][0]; save[f"{name}_gap"] = gaps[name][1]
    np.savez(os.path.join(pdd, f"{args.prefix}.npz"), **save)

    print(f"centroid-reliable interval: t <= {t_reliable:.3f} (control A_min >= {args.rel_thr})")
    print("active-control d_min gap over reliable interval:")
    for name in [n for n, _, _ in configs]:
        if name in gaps:
            gm = gaps[name][2]
            chg = (gm - base_gap) / base_gap * 100 if (base_gap and np.isfinite(base_gap)) else np.nan
            flag = " <<< >20% change, expand seeds" if np.isfinite(chg) and abs(chg) > 20 else ""
            print(f"  {name}: mean gap {gm:.3f} (vs baseline {chg:+.1f}%){flag}")
    print(f"wrote {fd}/{args.prefix}.pdf/.png and tetra_refinement_metrics.csv")


if __name__ == "__main__":
    main()
