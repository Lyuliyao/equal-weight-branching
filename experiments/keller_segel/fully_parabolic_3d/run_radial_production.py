"""run_radial_production.py -- Experiment B production / resolution checks (plan 6.5).
ONE (chi, N, K_dyn, tau, seed) config per invocation -> diagnostics CSV + repro records.
experiment='radial' (chi != 0, v0=0).

validation-closure additions (all OFF by default, so existing runs are unaffected):
  --drift_probe K1 K2 ...  same-cloud grad-v reconstruction discrepancy at diag times
                           (records Gv_K*, dabs_*, drel_* columns; draws no RNG, does
                           not enter the drift -> trajectory identical to no probe);
  --save_times t1 t2 ...   snapshot the raw (u,v) particle clouds near these times to
                           snapshots/clouds_seed<seed>.npz (for the state-evolution
                           figure; reconstruction-free).
"""
import os, csv, json, argparse, sys, datetime
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import simulation_pp3d as S
import repro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chi", type=float, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--K_dyn", type=int, default=8)
    ap.add_argument("--tau", type=float, default=1e-3)
    ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--L", type=float, default=12.0)
    ap.add_argument("--D", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--M", type=float, default=10.0)
    ap.add_argument("--sigma", type=float, default=0.45)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--label", default="prod", help="weak / delayed / Nref / tauref / Kref")
    ap.add_argument("--buffer_factor", type=float, default=1.6)
    ap.add_argument("--fast", action="store_true", help="JITTED fixed-capacity grad-v buffer")
    ap.add_argument("--drift_probe", type=int, nargs="+", default=None,
                    help="same-cloud grad-v reconstruction-discrepancy bandwidths, e.g. 8 12 16")
    ap.add_argument("--save_times", type=float, nargs="+", default=None,
                    help="times to snapshot raw (u,v) clouds for the state-evolution figure")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    n_steps = int(round(args.T / args.tau)); diag_every = max(1, int(round(0.04 / args.tau)))
    cfg = dict(experiment="radial", ic="radial", L=args.L, Du=args.D, Dv=args.D,
               alpha=args.alpha, beta=args.beta, chi=args.chi, M=args.M, sigma=args.sigma,
               v0=0.0, K_dyn=args.K_dyn, K_test=4, Nu=args.N, tau=args.tau,
               n_steps=n_steps, kernel="minvar", buffer_factor=args.buffer_factor)
    if args.drift_probe:
        cfg["drift_probe_K"] = list(args.drift_probe)
    if args.save_times:
        cfg["save_times"] = list(args.save_times)
    tag = f"{args.label}_chi{args.chi:g}_N{args.N}_K{args.K_dyn}_tau{args.tau:.0e}_seed{args.seed}"
    run_dir = os.path.join(args.out_dir, tag); os.makedirs(run_dir, exist_ok=True)

    t0 = datetime.datetime.now(); aborted = False; err = ""
    try:
        recs, summ, _ = S.simulate(cfg, seed=args.seed, diag_every=diag_every, fast=args.fast)
    except RuntimeError as e:
        aborted = True; err = str(e); recs, summ = [], {"max_v_occupancy": -1, "omega": np.nan,
                                                        "clouds": []}
    dt = (datetime.datetime.now() - t0).total_seconds()

    if recs:
        cols = list(recs[0].keys())
        with open(os.path.join(run_dir, "diagnostics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
            for r in recs:
                w.writerow(r)
    # raw-cloud snapshots for the state-evolution figure (reconstruction-free)
    clouds = summ.get("clouds", [])
    if clouds:
        snap_dir = os.path.join(run_dir, "snapshots"); os.makedirs(snap_dir, exist_ok=True)
        save = {"times": np.array([c[0] for c in clouds]),
                "omega": float(summ.get("omega", cfg["M"] / args.N)),
                "M": float(args.M), "L": float(args.L), "sigma": float(args.sigma),
                "K_dyn": int(args.K_dyn), "seed": int(args.seed)}
        for i, (tt, X, Y) in enumerate(clouds):
            save[f"u_{i}"] = X.astype(np.float32)
            save[f"v_{i}"] = Y.astype(np.float32)
        np.savez_compressed(os.path.join(snap_dir, f"clouds_seed{args.seed}.npz"), **save)

    Ncap = int(np.ceil(args.buffer_factor * args.N))
    manifest = dict(experiment="keller_segel/fully_parabolic_3d/radial",
                    label=args.label, seed=args.seed, N_u=args.N, kernel="minvar",
                    L=args.L, D_u=args.D, D_v=args.D, alpha=args.alpha, beta=args.beta,
                    chi=args.chi, M=args.M, sigma=args.sigma, v0=0.0, tau=args.tau, T=args.T,
                    n_steps=n_steps, diag_every=diag_every, K_dyn=args.K_dyn, K_test=4,
                    drift_probe_K=cfg.get("drift_probe_K"), save_times=cfg.get("save_times"),
                    Nv_cap=summ.get("Nv_cap"), population_control=False, fast=bool(args.fast),
                    buffer_factor=args.buffer_factor, buffer_capacity=(Ncap if args.fast else None),
                    max_v_occupancy=summ["max_v_occupancy"], final_Nv=summ.get("final_Nv"),
                    aborted=aborted, error=err, runtime_s=dt)
    manifest.update(repro.env_record(_HERE))
    repro.dump(run_dir, manifest, cfg)
    if recs:
        fin = recs[-1]
        print(f"[{tag}] {dt:.0f}s  R0.5 {recs[0]['R_0_5']:.4f}->{fin['R_0_5']:.4f} "
              f"(ratio {fin['R_0_5']/recs[0]['R_0_5']:.3f})  Gv(T)={fin['G_v']:.3f} "
              f"Cd_max={max(r['drift_resolution_number'] for r in recs):.2f} "
              f"occ={summ['max_v_occupancy']}/{Ncap}")
    else:
        print(f"[{tag}] ABORTED after {dt:.0f}s: {err}")


if __name__ == "__main__":
    main()
