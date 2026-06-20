"""analyze_radial_validation.py -- validation-closure B.1 (tau -> tau/2) + B.2 (multi-seed
K_dyn).  Reads diagnostics only (never runs the solver).  Produces per-config scalar
metrics (t_turn, t_focus10, R_0.5(T)/R_0.5(0), G_v(T), M_v rel error, drift-resolution)
with seed spread, the ensemble-mean R_0.5 curve differences across tau, and the combined
figure_radial_tau_K_validation.

The delayed-turnover conclusion is robust only if the t_turn distributions OVERLAP
across tau and across K_dyn; this script reports the spreads so that can be judged.
"""
import os, csv, argparse, sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import vc_load as V

L = 12.0


def seed_scalars(runs, sel):
    rows = []
    for r in V.select(runs, sel):
        c = r["cols"]; t = c["t"]; R = c["R_0_5"]
        mvx, mv, M = c["M_v_exact"], c["M_v"], r["M"]
        mask = mvx > 1e-3 * M
        rel = np.abs(mv[mask] - mvx[mask]) / mvx[mask] if mask.any() else np.array([0.0])
        rows.append(dict(seed=r["seed"], t_turn=V.t_turn(t, R), t_focus=V.t_focus(t, R),
                         R05_0=float(R[0]), R05_T=float(R[-1]), R05_ratio=float(R[-1] / R[0]),
                         Gv_T=float(c["G_v"][-1]), Mv_relerr=float(rel.max()),
                         drift_max=float(c["drift_resolution_number"].max())))
    return rows


def agg(rows):
    def ms(key):
        v = np.array([r[key] for r in rows], float)
        v = v[np.isfinite(v)]
        return (float(np.mean(v)), float(np.std(v))) if v.size else (np.nan, np.nan)
    n = len(rows)
    tt = ms("t_turn"); tf = ms("t_focus"); rr = ms("R05_ratio"); gv = ms("Gv_T")
    return dict(nseed=n, t_turn_mean=tt[0], t_turn_std=tt[1], t_focus_mean=tf[0],
                t_focus_std=tf[1], R05_ratio_mean=rr[0], R05_ratio_std=rr[1],
                Gv_T_mean=gv[0], Mv_relerr_max=float(np.nanmax([r["Mv_relerr"] for r in rows])),
                drift_res_max=float(np.nanmax([r["drift_max"] for r in rows])))


def write_csv(path, configs, metric):
    with open(path, "w", newline="") as f:
        cols = ["config", "nseed", "t_turn_mean", "t_turn_std", "t_focus_mean", "t_focus_std",
                "R05_ratio_mean", "R05_ratio_std", "Gv_T_mean", "Mv_relerr_max", "drift_res_max"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for name, m in configs:
            w.writerow(dict(config=name, **{k: m[k] for k in cols[1:]}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K_dir", required=True, help="radial_K (Kref runs, K=8/12/16 seeds 0-3)")
    ap.add_argument("--tau_fine_dir", required=True, help="radial_tau (tauref, tau=5e-4 seeds 0-7)")
    ap.add_argument("--tau_base_dir", required=True, help="existing radial production (delayed tau=1e-3 8 seeds)")
    ap.add_argument("--out_root", required=True, help="validation_closure dir (figures/, plot_data/)")
    ap.add_argument("--prefix", default="figure_radial_tau_K_validation")
    args = ap.parse_args()

    Kruns = V.load_runs(args.K_dir)
    finer = V.load_runs(args.tau_fine_dir)
    baser = V.load_runs(args.tau_base_dir)

    base = dict(M=96.0, N=100000)
    sel_tau1 = dict(label="delayed", K=12, tau=1e-3, **base)
    sel_tau2 = dict(label="tauref", K=12, tau=5e-4, **base)
    selK = {k: dict(label="Kref", K=k, tau=1e-3, **base) for k in (8, 12, 16)}

    tau_cfg = [("tau=1e-3", agg(seed_scalars(baser, sel_tau1))),
               ("tau=5e-4", agg(seed_scalars(finer, sel_tau2)))]
    K_cfg = [(f"K={k}", agg(seed_scalars(Kruns, selK[k]))) for k in (8, 12, 16)]

    fd = os.path.join(args.out_root, "figures"); os.makedirs(fd, exist_ok=True)
    pdd = os.path.join(args.out_root, "plot_data"); os.makedirs(pdd, exist_ok=True)
    write_csv(os.path.join(args.tau_fine_dir, "radial_tau_metrics.csv"), tau_cfg, "tau")
    write_csv(os.path.join(args.K_dir, "radial_K_metrics.csv"), K_cfg, "K")

    # ---- curves ----
    t1, R1, S1, n1 = V.seedmean(baser, sel_tau1, "R_0_5")
    t2, R2, S2, n2 = V.seedmean(finer, sel_tau2, "R_0_5")
    Kcurves = {k: V.seedmean(Kruns, selK[k], "R_0_5") for k in (8, 12, 16)}

    # tau curve-difference over full / through t_turn / through t_focus10 (common grid t1)
    diffs = {}
    if t1 is not None and t2 is not None:
        R2i = np.interp(t1, t2, R2)
        tt = t1[np.argmax(R1)]; tf = V.t_focus(t1, R1)
        for nm, tmax in [("full", t1[-1]), ("through_tturn", tt),
                         ("through_tfocus10", tf if np.isfinite(tf) else t1[-1])]:
            mm = t1 <= tmax
            diffs[nm] = dict(maxabs=float(np.max(np.abs(R1[mm] - R2i[mm]))),
                             rms=float(np.sqrt(np.mean((R1[mm] - R2i[mm]) ** 2))))

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    # (a) tau comparison
    a = ax[0, 0]
    for t, R, Sd, lab, c in [(t1, R1, S1, f"tau=1e-3 (n={n1})", "k"),
                             (t2, R2, S2, f"tau=5e-4 (n={n2})", "C3")]:
        if t is not None:
            a.plot(t, R, c, label=lab); a.fill_between(t, R - Sd, R + Sd, color=c, alpha=0.2, lw=0)
            a.axvline(t[np.argmax(R)], color=c, ls=":", lw=0.8)
    a.set_title("(a) $R_{0.5}(t)$: tau refinement (M=96, K=12)"); a.set_xlabel("t")
    a.set_ylabel("$R_{0.5}^u$"); a.legend(fontsize=8)
    # (b) K comparison
    b = ax[0, 1]
    for k, c in [(8, "C0"), (12, "C1"), (16, "C3")]:
        t, R, Sd, n = Kcurves[k]
        if t is not None:
            b.plot(t, R, c, label=f"K={k} (n={n})"); b.fill_between(t, R - Sd, R + Sd, color=c, alpha=0.2, lw=0)
    b.set_title("(b) $R_{0.5}(t)$: K refinement (M=96, tau=1e-3)"); b.set_xlabel("t")
    b.set_ylabel("$R_{0.5}^u$"); b.legend(fontsize=8)
    # (c) t_turn / t_focus10 with seed spread
    cax = ax[1, 0]
    names = [n for n, _ in tau_cfg] + [n for n, _ in K_cfg]
    mets = [m for _, m in tau_cfg] + [m for _, m in K_cfg]
    x = np.arange(len(names))
    cax.errorbar(x - 0.1, [m["t_turn_mean"] for m in mets], yerr=[m["t_turn_std"] for m in mets],
                 fmt="o", color="C0", capsize=3, label="$t_{turn}$")
    cax.errorbar(x + 0.1, [m["t_focus_mean"] for m in mets], yerr=[m["t_focus_std"] for m in mets],
                 fmt="s", color="C3", capsize=3, label="$t_{focus10}$")
    cax.set_xticks(x); cax.set_xticklabels(names, rotation=30, fontsize=7)
    cax.set_title("(c) $t_{turn}$, $t_{focus10}$ (mean $\\pm$ seed std)"); cax.set_ylabel("t")
    cax.legend(fontsize=8)
    # (d) R05 ratio with seed spread
    dax = ax[1, 1]
    dax.bar(x, [m["R05_ratio_mean"] for m in mets], yerr=[m["R05_ratio_std"] for m in mets],
            color="0.6", capsize=3)
    dax.axhline(1.0, color="0.4", ls=":", lw=1)
    dax.set_xticks(x); dax.set_xticklabels(names, rotation=30, fontsize=7)
    dax.set_title("(d) $R_{0.5}(T)/R_{0.5}(0)$ (mean $\\pm$ seed std)"); dax.set_ylabel("ratio")

    fig.suptitle("Radial delayed-focusing validation: tau and K_dyn refinement "
                 "(reconstruction-free $R_{0.5}$)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"{args.prefix}.{ext}"), dpi=200, bbox_inches="tight")

    save = dict(names=np.array(names, dtype=object),
                t_turn_mean=np.array([m["t_turn_mean"] for m in mets]),
                t_turn_std=np.array([m["t_turn_std"] for m in mets]),
                t_focus_mean=np.array([m["t_focus_mean"] for m in mets]),
                t_focus_std=np.array([m["t_focus_std"] for m in mets]),
                R05_ratio_mean=np.array([m["R05_ratio_mean"] for m in mets]),
                R05_ratio_std=np.array([m["R05_ratio_std"] for m in mets]),
                tau_diffs=np.array(list(diffs.items()), dtype=object))
    for k in (8, 12, 16):
        t, R, Sd, n = Kcurves[k]
        if t is not None:
            save[f"K{k}_t"] = t; save[f"K{k}_R05"] = R; save[f"K{k}_R05_std"] = Sd
    if t1 is not None:
        save.update(tau1_t=t1, tau1_R05=R1, tau1_R05_std=S1)
    if t2 is not None:
        save.update(tau2_t=t2, tau2_R05=R2, tau2_R05_std=S2)
    np.savez(os.path.join(pdd, f"{args.prefix}.npz"), **save)

    print("=== tau metrics ===")
    for n, m in tau_cfg:
        print(f"  {n}: t_turn {m['t_turn_mean']:.3f}+-{m['t_turn_std']:.3f}, "
              f"t_focus10 {m['t_focus_mean']:.3f}+-{m['t_focus_std']:.3f}, "
              f"R05 ratio {m['R05_ratio_mean']:.3f}+-{m['R05_ratio_std']:.3f}, "
              f"Mv_relerr_max {m['Mv_relerr_max']:.3f}, drift_max {m['drift_res_max']:.3f}")
    print("=== K metrics ===")
    for n, m in K_cfg:
        print(f"  {n}: t_turn {m['t_turn_mean']:.3f}+-{m['t_turn_std']:.3f}, "
              f"t_focus10 {m['t_focus_mean']:.3f}+-{m['t_focus_std']:.3f}, "
              f"R05 ratio {m['R05_ratio_mean']:.3f}+-{m['R05_ratio_std']:.3f}, "
              f"drift_max {m['drift_res_max']:.3f}")
    print("=== tau R_0.5 curve difference (1e-3 vs 5e-4) ===")
    for nm, dv in diffs.items():
        print(f"  {nm}: max|dR05| {dv['maxabs']:.4f}, rms {dv['rms']:.4f}")
    print(f"wrote {fd}/{args.prefix}.pdf/.png and metrics CSVs")


if __name__ == "__main__":
    main()
