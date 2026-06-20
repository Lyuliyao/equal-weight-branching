"""run_linear_verification.py -- Experiment A (exact linear verification) runner.
ONE (N, tau, kernel, seed) config per invocation -> diagnostics CSV + repro records.
A SLURM array enumerates the sweep; analyze_linear.py aggregates and checks gates.

chi=0, Du=Dv=D, v0=0, wrapped-Gaussian u0; the coupled solver must reproduce the
exact mass law and analytic low modes / grad v (plan section 5).
"""
import os, csv, json, argparse, subprocess, sys, datetime
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import simulation_pp3d as S


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", _HERE, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def pkg_versions():
    import jax
    return {"python": sys.version.split()[0], "numpy": np.__version__, "jax": jax.__version__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--K_dyn", type=int, default=8)
    ap.add_argument("--K_test", type=int, default=4)
    ap.add_argument("--L", type=float, default=12.0)
    ap.add_argument("--D", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--M", type=float, default=10.0)
    ap.add_argument("--sigma", type=float, default=0.45)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--kernel", choices=["minvar", "poisson"], default="minvar")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    n_steps = int(round(args.T / args.tau))
    diag_every = max(1, int(round(0.02 / args.tau)))     # ~50 diagnostic times
    cfg = dict(ic="radial", L=args.L, Du=args.D, Dv=args.D, alpha=args.alpha,
               beta=args.beta, chi=0.0, M=args.M, sigma=args.sigma, v0=0.0,
               K_dyn=args.K_dyn, K_test=args.K_test, Nu=args.N, tau=args.tau,
               n_steps=n_steps, kernel=args.kernel)

    tag = f"N{args.N}_tau{args.tau:.0e}_{args.kernel}_seed{args.seed}"
    run_dir = os.path.join(args.out_dir, tag)
    os.makedirs(run_dir, exist_ok=True)

    t0 = datetime.datetime.now()
    records, summary, _ = S.simulate(cfg, seed=args.seed, diag_every=diag_every,
                                     record_linear=True)
    dt = (datetime.datetime.now() - t0).total_seconds()

    # diagnostics.csv
    cols = ["t", "seed", "N_u", "N_v", "M_u", "M_v", "M_v_exact", "abs_Mu_drift",
            "abs_Mv_error", "rel_Mv_error", "E_u_modes", "E_v_modes",
            "E_grad_v_particles", "G_v", "max_v_occupancy"]
    with open(os.path.join(run_dir, "diagnostics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)

    manifest = dict(experiment="keller_segel/fully_parabolic_3d/linear_verification",
                    git=git_hash(), date=str(t0), versions=pkg_versions(),
                    seed=args.seed, N_u=args.N, kernel=args.kernel,
                    L=args.L, D_u=args.D, D_v=args.D, alpha=args.alpha, beta=args.beta,
                    chi=0.0, M=args.M, sigma=args.sigma, v0=0.0,
                    tau=args.tau, T=args.T, n_steps=n_steps, diag_every=diag_every,
                    K_dyn=args.K_dyn, K_test=args.K_test,
                    omega=summary["omega"], max_v_occupancy=summary["max_v_occupancy"],
                    Nv_cap=None, population_control=False, runtime_s=dt)
    json.dump(manifest, open(os.path.join(run_dir, "manifest.json"), "w"), indent=2)
    json.dump(cfg, open(os.path.join(run_dir, "config_used.json"), "w"), indent=2)
    with open(os.path.join(run_dir, "command.txt"), "w") as f:
        f.write(" ".join(["python", os.path.basename(__file__)]
                         + [f"--{k} {v}" for k, v in vars(args).items()]) + "\n")
    fin = records[-1]
    print(f"[{tag}] done {dt:.0f}s  M_v(T)={fin['M_v']:.4f} (exact {fin['M_v_exact']:.4f}, "
          f"rel {fin['rel_Mv_error']:.1e})  E_u={fin['E_u_modes']:.2e} "
          f"E_v={fin['E_v_modes']:.2e} E_gv={fin['E_grad_v_particles']:.2e}  "
          f"maxocc={summary['max_v_occupancy']}/{int(1.5*args.N)}")


if __name__ == "__main__":
    main()
