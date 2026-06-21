"""run_tetra_control.py -- Experiment C: tetrahedral nonradial aggregation with a
mandatory diffusion control (plan section 7).  ONE (chi, seed) per invocation;
chi>0 = active, chi=0 = diffusion control.  Same seed => common initial particles
(sample_tetra) and common early random streams (practical common-randomness).
experiment='tetra', v0=0; cluster labels preserved (u-cloud conservative).
"""
import os, csv, json, argparse, sys, datetime
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import simulation_pp3d as S
import repro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chi", type=float, required=True, help="active chi>0, or 0 (control)")
    ap.add_argument("--N", type=int, default=80000)
    ap.add_argument("--K_dyn", type=int, default=8)
    ap.add_argument("--tau", type=float, default=1e-3)
    ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--L", type=float, default=12.0)
    ap.add_argument("--D", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--M", type=float, default=10.0)
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--sigma_c", type=float, default=0.25)
    ap.add_argument("--r_center", type=float, default=1.0)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--buffer_factor", type=float, default=1.6)
    ap.add_argument("--fast", action="store_true", help="JITTED fixed-capacity grad-v buffer")
    ap.add_argument("--save_times", type=float, nargs="+", default=None,
                    help="times to snapshot raw u-clouds and persistent cluster labels")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    n_steps = int(round(args.T / args.tau)); diag_every = max(1, int(round(0.04 / args.tau)))
    cfg = dict(experiment="tetra", ic="tetra", L=args.L, Du=args.D, Dv=args.D,
               alpha=args.alpha, beta=args.beta, chi=args.chi, M=args.M, a=args.a,
               sigma_c=args.sigma_c, r_center=args.r_center, v0=0.0, K_dyn=args.K_dyn,
               K_test=4, Nu=args.N, tau=args.tau, n_steps=n_steps, kernel="minvar",
               buffer_factor=args.buffer_factor)
    if args.save_times:
        cfg["save_times"] = list(args.save_times)
    arm = "active" if args.chi > 0 else "control"
    tag = f"{arm}_chi{args.chi:g}_N{args.N}_K{args.K_dyn}_tau{args.tau:.0e}_seed{args.seed}"
    run_dir = os.path.join(args.out_dir, tag); os.makedirs(run_dir, exist_ok=True)

    t0 = datetime.datetime.now(); aborted = False; err = ""
    try:
        recs, summ, (_, _, labels) = S.simulate(cfg, seed=args.seed, diag_every=diag_every,
                                                fast=args.fast)
    except RuntimeError as e:
        aborted = True; err = str(e); recs, summ, labels = [], {"max_v_occupancy": -1,
                                                                "clouds": []}, None
    dt = (datetime.datetime.now() - t0).total_seconds()

    if recs:
        cols = list(recs[0].keys())
        with open(os.path.join(run_dir, "diagnostics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
            for r in recs:
                w.writerow(r)
    clouds = summ.get("clouds", [])
    if clouds:
        snap_dir = os.path.join(run_dir, "snapshots"); os.makedirs(snap_dir, exist_ok=True)
        save = {"times": np.array([c[0] for c in clouds]),
                "labels": labels.astype(np.int16),
                "omega": float(summ.get("omega", cfg["M"] / args.N)),
                "M": float(args.M), "L": float(args.L), "a": float(args.a),
                "sigma_c": float(args.sigma_c), "K": int(args.K_dyn),
                "tau": float(args.tau), "T": float(args.T), "chi": float(args.chi),
                "seed": int(args.seed)}
        for i, (_tt, X, _Y) in enumerate(clouds):
            save[f"u_{i}"] = X
        np.savez_compressed(os.path.join(snap_dir, f"u_clouds_seed{args.seed}.npz"), **save)
    Ncap = int(np.ceil(args.buffer_factor * args.N))
    manifest = dict(experiment="keller_segel/fully_parabolic_3d/tetra", arm=arm,
                    seed=args.seed, N_u=args.N, kernel="minvar", L=args.L, D_u=args.D,
                    D_v=args.D, alpha=args.alpha, beta=args.beta, chi=args.chi, M=args.M,
                    a=args.a, sigma_c=args.sigma_c, r_center=args.r_center, v0=0.0,
                    tau=args.tau, T=args.T, n_steps=n_steps, diag_every=diag_every,
                    K_dyn=args.K_dyn, K_test=4, save_times=cfg.get("save_times"),
                    Nv_cap=summ.get("Nv_cap"),
                    population_control=False, fast=bool(args.fast),
                    buffer_factor=args.buffer_factor,
                    buffer_capacity=(Ncap if args.fast else None),
                    max_v_occupancy=summ["max_v_occupancy"], final_Nv=summ.get("final_Nv"),
                    aborted=aborted, error=err, runtime_s=dt)
    manifest.update(repro.env_record(_HERE))
    repro.dump(run_dir, manifest, cfg)
    if recs:
        fin = recs[-1]
        print(f"[{tag}] {dt:.0f}s  d_min {recs[0]['d_min']:.3f}->{fin['d_min']:.3f}  "
              f"m_center {recs[0]['m_center']:.3f}->{fin['m_center']:.3f}  "
              f"overlap {fin['overlap']:.2f}  E_sym {fin['E_sym']:.3f}  "
              f"occ={summ['max_v_occupancy']}/{int(args.buffer_factor*args.N)}")
    else:
        print(f"[{tag}] ABORTED: {err}")


if __name__ == "__main__":
    main()
