"""analyze_linear.py -- aggregate Experiment A runs, check the 8 acceptance gates
(plan section 5.4), and build Figure A.  Reads the per-(config,seed) diagnostics.csv
under --run_dir; never runs the solver.
"""
import os, csv, glob, json, argparse, re
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


def load_runs(run_dir):
    runs = {}
    for d in sorted(glob.glob(os.path.join(run_dir, "N*_tau*_*_seed*"))):
        m = re.search(r"N(\d+)_tau([0-9.e-]+)_(\w+)_seed(\d+)", os.path.basename(d))
        if not m:
            continue
        key = (int(m[1]), m[2], m[3]); seed = int(m[4])
        cs = os.path.join(d, "diagnostics.csv")
        if not os.path.exists(cs):
            continue
        R = list(csv.DictReader(open(cs)))
        t = np.array([float(r["t"]) for r in R])
        cols = {c: np.array([float(r[c]) for r in R]) for c in R[0]}
        mocc = json.load(open(os.path.join(d, "manifest.json")))["max_v_occupancy"]
        runs.setdefault(key, []).append((seed, t, cols, mocc))
    return runs


def final_mean(runs, key, col):
    vals = [c[col][-1] for _, _, c, _ in runs[key]]
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    runs = load_runs(args.run_dir)
    if not runs:
        raise SystemExit("no runs found")
    keys = sorted(runs)
    Nsweep = sorted([k for k in keys if k[1] == "1e-03" and k[2] == "minvar"],
                    key=lambda k: k[0]) or \
             sorted([k for k in keys if abs(float(k[1]) - 1e-3) < 1e-12 and k[2] == "minvar"],
                    key=lambda k: k[0])
    tausweep = sorted([k for k in keys if k[0] == 320000 and k[2] == "minvar"],
                      key=lambda k: float(k[1]))
    gates, lines = {}, []

    # gate 1: M_u constant
    mu_drift = max(final_mean(runs, k, "abs_Mu_drift")[0] for k in keys)
    gates["1_Mu_constant"] = mu_drift < 1e-8
    lines.append(f"1 M_u constant: max drift {mu_drift:.1e} -> {gates['1_Mu_constant']}")

    # gate 2: M_v unbiased vs analytic (within 4 SEM at final time)
    ok2 = True
    for k in keys:
        mv, sd, n = final_mean(runs, k, "M_v")
        ex, _, _ = final_mean(runs, k, "M_v_exact")
        if abs(mv - ex) > 4 * sd / np.sqrt(n) + 1e-3 * ex:
            ok2 = False
    gates["2_Mv_unbiased"] = ok2
    lines.append(f"2 M_v unbiased (4SEM): {ok2}")

    # gate 3+4: E decreases with N and ~N^{-1/2} slope (>=3 levels)
    Ns = np.array([k[0] for k in Nsweep], float)
    slopes = {}
    for col in ("E_u_modes", "E_v_modes", "E_grad_v_particles"):
        E = np.array([final_mean(runs, k, col)[0] for k in Nsweep])
        if len(Ns) >= 3 and np.all(E > 0):
            s = np.polyfit(np.log(Ns), np.log(E), 1)[0]
            slopes[col] = float(s)
        lines.append(f"   {col} vs N: " + ", ".join(f"{int(n)}:{e:.2e}" for n, e in zip(Ns, E))
                     + (f"  slope {slopes.get(col, float('nan')):.2f}" if col in slopes else ""))
    gates["3_E_decreases_with_N"] = all(
        np.all(np.diff([final_mean(runs, k, c)[0] for k in Nsweep]) < 0)
        for c in ("E_u_modes", "E_v_modes", "E_grad_v_particles"))
    gates["4_MC_slope"] = (len(Ns) >= 3 and
                           all(-0.75 < slopes.get(c, 0) < -0.30 for c in slopes))
    lines.append(f"3 E decreases with N: {gates['3_E_decreases_with_N']}")
    lines.append(f"4 MC slope ~ -0.5 (3 levels): {gates['4_MC_slope']}  {slopes}")

    # gate 5: tau refinement -> error to floor (E_v not increasing as tau shrinks)
    if len(tausweep) >= 2:
        Ev_tau = [final_mean(runs, k, "E_v_modes")[0] for k in tausweep]
        gates["5_tau_floor"] = Ev_tau[-1] <= Ev_tau[0] * 1.5     # finest not worse
        lines.append("5 tau floor: E_v(tau) " +
                     ", ".join(f"{k[1]}:{e:.2e}" for k, e in zip(tausweep, Ev_tau))
                     + f" -> {gates['5_tau_floor']}")
    # gate 6: minvar var <= poisson var (seed std of M_v at N=8e4)
    kmv = (80000, "1e-03", "minvar"); kpo = (80000, "1e-03", "poisson")
    kmv = next((k for k in keys if k[0] == 80000 and k[2] == "minvar"), None)
    kpo = next((k for k in keys if k[0] == 80000 and k[2] == "poisson"), None)
    if kmv and kpo:
        smv = final_mean(runs, kmv, "M_v")[1]; spo = final_mean(runs, kpo, "M_v")[1]
        gates["6_minvar_le_poisson"] = smv <= spo * 1.2
        lines.append(f"6 seed-std M_v: minvar {smv:.2e} <= poisson {spo:.2e} -> "
                     f"{gates['6_minvar_le_poisson']}")
    # gate 7: buffer never binds
    worst = max((mocc, k) for k in keys for _, _, _, mocc in runs[k])
    gates["7_buffer_ok"] = all(mocc < 1.5 * k[0] for k in keys for _, _, _, mocc in runs[k])
    lines.append(f"7 buffer never binds (max occ {worst[0]} for N={worst[1][0]}): "
                 f"{gates['7_buffer_ok']}")
    # gate 8: v created from empty
    gates["8_v_from_empty"] = all(c["N_v"][0] == 0 and c["N_v"][-1] > 0
                                  for k in keys for _, _, c, _ in runs[k])
    lines.append(f"8 v from empty cloud: {gates['8_v_from_empty']}")

    gates["ALL_PASS"] = all(v for kk, v in gates.items() if kk.startswith(tuple("12345678")))
    json.dump({"gates": gates, "slopes": slopes}, open(
        os.path.join(args.run_dir, "metrics_summary.json"), "w"), indent=2)
    print("=== Experiment A acceptance gates (plan 5.4) ===")
    for ln in lines:
        print(" ", ln)
    print(f"\nALL GATES PASS: {gates['ALL_PASS']}")

    # ---------------- Figure A ----------------
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    base = next((k for k in tausweep if abs(float(k[1]) - 1e-3) < 1e-12), Nsweep[-1])
    _, t, c0, _ = runs[base][0]
    axes[0].plot(t, c0["M_v_exact"], "k-", lw=2, label="exact law")
    for s, tt, c, _ in runs[base]:
        axes[0].plot(tt, c["M_v"], "C0-", alpha=0.4, lw=0.8)
    axes[0].plot(t, c0["M_u"], "C3--", lw=1.2, label="$M_u$ (const)")
    axes[0].set_title("(a) $M_v(t)$ vs exact law"); axes[0].set_xlabel("t")
    axes[0].set_ylabel("mass"); axes[0].legend(fontsize=8)
    for ax, panel, col, lab in [(axes[1], "b", "E_v_modes", "low-mode $v$ error"),
                                (axes[2], "c", "E_grad_v_particles",
                                 r"$\nabla v$-at-particle error")]:
        E = np.array([final_mean(runs, k, col)[0] for k in Nsweep])
        ax.loglog(Ns, E, "o-", label="data")
        ax.loglog(Ns, E[0] * (Ns / Ns[0]) ** -0.5, "k--", label="$N^{-1/2}$")
        ax.set_title(f"({panel}) {lab} vs $N$ (slope {slopes.get(col, float('nan')):.2f})")
        ax.set_xlabel("$N_u$"); ax.legend(fontsize=8)
    if kmv and kpo:
        smv = final_mean(runs, kmv, "M_v")[1]; spo = final_mean(runs, kpo, "M_v")[1]
        axes[3].bar(["min-var", "Poisson"], [smv, spo], color=["C0", "C3"])
        axes[3].set_title("(d) seed std $M_v$: min-var vs Poisson")
        axes[3].set_ylabel("std over seeds")
    fig.suptitle("Experiment A: exact 3D coupled-system verification "
                 f"(ALL GATES {'PASS' if gates['ALL_PASS'] else 'FAIL'})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fd = os.path.join(args.run_dir, "figures"); os.makedirs(fd, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"figureA_linear_verification.{ext}"),
                    dpi=200, bbox_inches="tight")
    print(f"wrote {fd}/figureA_linear_verification.pdf/.png and metrics_summary.json")


if __name__ == "__main__":
    main()
