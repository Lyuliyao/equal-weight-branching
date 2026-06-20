"""analyze_resolution_audit.py -- validation-closure B.3 (audit the invalid Q_0.2>=3
gate) + B.4 (same-cloud chemical-drift resolution diagnostic).

Q_{0.2}(t) = R_{0.2}(t) / h_K,  h_K = L/(2 K_dyn + 1).  The old plan proposed the gate
Q_{0.2} >= 3; this script shows the complete Q_{0.2}(t) history for K=8/12/16 and that
for the chosen Gaussian (sigma=0.45) it fails already at t=0, so it cannot be cited as a
passed resolution criterion.  In its place it reports the more direct SAME-CLOUD drift
discrepancy delta_{8,12}, delta_{12,16}: grad v reconstructed at K=8/12/16 on the SAME
current u/v clouds (from the K=12 production runs' drift_probe columns), with relative
values marked inactive below a documented G_v floor.

Reads diagnostics only.  Writes radial_resolution_audit.csv + figure_radial_resolution_audit.
"""
import os, csv, argparse, sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import vc_load as V

L = 12.0
GV_FLOOR_FRAC = 0.05   # relative delta is inactive while Gv_K12 < this fraction of its peak


def hK(K):
    return L / (2 * K + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K_dir", required=True, help="radial_K (Kref K=8/12/16 seeds 0-3, drift_probe)")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--prefix", default="figure_radial_resolution_audit")
    args = ap.parse_args()
    runs = V.load_runs(args.K_dir)
    base = dict(label="Kref", M=96.0, N=100000, tau=1e-3)
    selK = {k: dict(K=k, **base) for k in (8, 12, 16)}

    # ---- Q_0.2(t) for each K ----
    Q = {}
    for k in (8, 12, 16):
        t, R02, _, n = V.seedmean(runs, selK[k], "R_0_2")
        if t is not None:
            Q[k] = (t, R02 / hK(k), n)

    # ---- same-cloud drift from the K=12 runs ----
    t12, R05, R05s, n12 = V.seedmean(runs, selK[12], "R_0_5")
    _, Gv12, _, _ = V.seedmean(runs, selK[12], "Gv_K12")
    _, Gv8, _, _ = V.seedmean(runs, selK[12], "Gv_K8")
    _, Gv16, _, _ = V.seedmean(runs, selK[12], "Gv_K16")
    _, drel_812, _, _ = V.seedmean(runs, selK[12], "drel_8_12")
    _, drel_1216, _, _ = V.seedmean(runs, selK[12], "drel_12_16")
    _, dabs_812, _, _ = V.seedmean(runs, selK[12], "dabs_8_12")
    _, dabs_1216, _, _ = V.seedmean(runs, selK[12], "dabs_12_16")
    tturn = float(t12[np.argmax(R05)]) if t12 is not None else np.nan
    gv_floor = GV_FLOOR_FRAC * float(np.max(Gv12)) if Gv12 is not None else 0.0
    active = Gv12 >= gv_floor if Gv12 is not None else None

    def cross(t, d, thr):
        if t is None:
            return np.nan
        m = active & (d >= thr)
        return float(t[m][0]) if m.any() else np.nan

    crossings = {}
    for thr in (0.05, 0.10, 0.20):
        crossings[thr] = dict(d812=cross(t12, drel_812, thr), d1216=cross(t12, drel_1216, thr))

    # ---- CSV ----
    out_dir = os.path.join(args.out_root, "radial_resolution_audit"); os.makedirs(out_dir, exist_ok=True)
    if t12 is not None:
        with open(os.path.join(out_dir, "radial_resolution_audit.csv"), "w", newline="") as f:
            cols = ["t", "Q02_K8", "Q02_K12", "Q02_K16", "R_0_5", "Gv_K8", "Gv_K12", "Gv_K16",
                    "dabs_8_12", "drel_8_12", "dabs_12_16", "drel_12_16", "gv_active"]
            w = csv.writer(f); w.writerow(cols)
            for i, t in enumerate(t12):
                def qv(k):
                    return Q[k][1][i] if k in Q and len(Q[k][1]) == len(t12) else np.interp(t, Q[k][0], Q[k][1]) if k in Q else np.nan
                w.writerow([t, qv(8), qv(12), qv(16), R05[i], Gv8[i], Gv12[i], Gv16[i],
                            dabs_812[i], drel_812[i], dabs_1216[i], drel_1216[i], int(active[i])])

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    a = ax[0]
    for k, c in [(8, "C0"), (12, "C1"), (16, "C3")]:
        if k in Q:
            a.plot(Q[k][0], Q[k][1], c, label=f"K={k} (h_K={hK(k):.3f})")
    a.axhline(3.0, color="r", ls="--", lw=1, label="proposed gate $Q_{0.2}\\geq 3$")
    a.set_title("(a) $Q_{0.2}(t)=R_{0.2}/h_K$ (proposed gate fails at $t=0$)")
    a.set_xlabel("t"); a.set_ylabel("$Q_{0.2}$"); a.legend(fontsize=7)
    b = ax[1]
    if t12 is not None:
        b.plot(t12, drel_812, "C0", label="$\\delta_{8,12}^{rel}$")
        b.plot(t12, drel_1216, "C3", label="$\\delta_{12,16}^{rel}$")
        for thr, ls in [(0.05, ":"), (0.10, "--"), (0.20, "-.")]:
            b.axhline(thr, color="0.5", ls=ls, lw=0.8)
        if np.isfinite(tturn):
            b.axvline(tturn, color="0.3", lw=1); b.text(tturn, b.get_ylim()[1]*0.9, "$t_{turn}$", fontsize=7)
        # shade Gv-inactive region
        inact = t12[~active]
        if inact.size:
            b.axvspan(t12[0], inact[-1], color="0.85", alpha=0.5, lw=0,
                      label=f"$G_v$ below floor ({GV_FLOOR_FRAC:.0%} peak)")
    b.set_title("(b) same-cloud grad-$v$ discrepancy (K=12 dynamics)")
    b.set_xlabel("t"); b.set_ylabel("relative discrepancy"); b.legend(fontsize=7)
    c = ax[2]
    if t12 is not None:
        for g, lab, col in [(Gv8, "$G_v^{K=8}$", "C0"), (Gv12, "$G_v^{K=12}$", "C1"),
                            (Gv16, "$G_v^{K=16}$", "C3")]:
            c.plot(t12, g, col, label=lab)
        c.set_ylabel("$G_v$ (rms $|\\nabla v|$)"); c.set_xlabel("t")
        c2 = c.twinx(); c2.plot(t12, R05, "k--", lw=1, label="$R_{0.5}$"); c2.set_ylabel("$R_{0.5}$")
    c.set_title("(c) $G_v(t)$ by bandwidth and $R_{0.5}(t)$"); c.legend(fontsize=7, loc="upper left")

    fig.suptitle("Radial resolution audit: invalid $Q_{0.2}$ gate vs same-cloud drift "
                 "discrepancy (M=96, N=1e5, tau=1e-3)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fd = os.path.join(args.out_root, "figures"); os.makedirs(fd, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"{args.prefix}.{ext}"), dpi=200, bbox_inches="tight")
    pdd = os.path.join(args.out_root, "plot_data"); os.makedirs(pdd, exist_ok=True)
    save = dict(tturn=tturn, gv_floor=gv_floor, gv_floor_frac=GV_FLOOR_FRAC)
    if t12 is not None:
        save.update(t=t12, R_0_5=R05, Gv_K8=Gv8, Gv_K12=Gv12, Gv_K16=Gv16,
                    drel_8_12=drel_812, drel_12_16=drel_1216, dabs_8_12=dabs_812,
                    dabs_12_16=dabs_1216, gv_active=active.astype(int))
    for k in (8, 12, 16):
        if k in Q:
            save[f"Q02_K{k}_t"] = Q[k][0]; save[f"Q02_K{k}"] = Q[k][1]
    np.savez(os.path.join(pdd, f"{args.prefix}.npz"), **save)

    print(f"Q_0.2(t=0): " + ", ".join(f"K{k}={Q[k][1][0]:.2f}" for k in (8, 12, 16) if k in Q)
          + "  (gate requires >=3)")
    for k in (8, 12, 16):
        if k in Q:
            print(f"  K={k}: max Q_0.2={Q[k][1].max():.2f} (h_K={hK(k):.3f}); "
                  f"ever>=3: {bool((Q[k][1]>=3).any())}")
    print(f"G_v floor = {gv_floor:.3f} (={GV_FLOOR_FRAC:.0%} of peak Gv_K12); t_turn={tturn:.3f}")
    print("same-cloud relative drift crossing times (active window):")
    for thr in (0.05, 0.10, 0.20):
        print(f"  thr={thr:.0%}: delta_8,12 at t={crossings[thr]['d812']}, "
              f"delta_12,16 at t={crossings[thr]['d1216']}")
    if t12 is not None:
        print(f"delta_12,16(T)={drel_1216[-1]:.3f}, delta_8,12(T)={drel_812[-1]:.3f}")
    print(f"wrote {fd}/{args.prefix}.pdf/.png and radial_resolution_audit.csv")


if __name__ == "__main__":
    main()
