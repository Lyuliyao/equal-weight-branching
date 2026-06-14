"""Parabolic-parabolic Keller-Segel: cross-species INJECTION chemical reaction.

Conservative 2D Keller-Segel on the torus [0, 2pi]^2 with two equal-weight
particle clouds: cells u (X1) and chemical v (X2).  Per-particle mass is the SAME
for both clouds (= M_u / n_samples), so one v-particle carries the same mass as
one u-particle (omega_u == omega_v).

  cells     u_t = div(grad u - chi u grad v) = 0   (conservative; X1 transports by
            chemotaxis + diffusion, NO reaction -> u-mass exactly conserved)
  chemical  v_t = Delta v + u - v                   (X2 diffuses + REACTS)

The chemical reaction half-step  v_t = u - v  has the EXACT integrator over tau

    v^{n+1} = e^{-tau} v* + (1 - e^{-tau}) u* ,

which this code realizes as a genuine TWO-SPECIES birth-death process (NO u/v
division, NO positivity floor):

    * DECAY : each existing v-particle dies          with prob  p = 1 - e^{-tau}
    * BIRTH : each u-particle spawns a v-particle at  with prob  p = 1 - e^{-tau}
              its own location  (cross-species u -> v source, placed where u is)

When omega_u == omega_v the general birth mean (1 - e^{-tau}) omega_u/omega_v
reduces to the Bernoulli probability p = 1 - e^{-tau}.  Then

    E[N_v^{n+1}] = e^{-tau} N_v + (1 - e^{-tau}) N_u
    => E[M_v^{n+1}] = e^{-tau} M_v + (1 - e^{-tau}) M_u   (unbiased),

so the cloud mass follows the exact conservative balance

    M_u(t) = M_u(0),    M_v(t) = M_u + (M_v(0) - M_u) e^{-t}.

This script verifies that law, reports per-step birth/death counts, and writes a
full reproducibility record (config.json, manifest.json), the mass-balance time
series (mass_balance.csv), a scalar summary (metrics_summary.csv), and a plot-data
file (plot_data/figure_mass_balance.npz) used by plot.py.

The legacy multiplicative (u-v)/v chemical rate is DEPRECATED and is not used in
the paper; --legacy_multiplicative_v_source raises NotImplementedError on purpose.

Run:  python simulation.py                 # default production-ish (n_samples=40000)
      python simulation.py --smoke         # tiny/fast smoke test
"""
import os
import sys
import csv
import json
import argparse
import datetime
import platform
import subprocess

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from density import generate_density_estimation

PERIOD = jnp.array([[0.0, 2 * np.pi], [0.0, 2 * np.pi]])
TWO_PI = 2.0 * np.pi
_HERE = os.path.dirname(os.path.abspath(__file__))


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_samples", type=int, default=40000,
                   help="particles for cell mass M_u = 1")
    p.add_argument("--Mv0", type=float, default=0.3,
                   help="initial chemical mass (with M_u = 1)")
    p.add_argument("--chi", type=float, default=1.0,
                   help="chemotactic sensitivity of u")
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--n_freq", type=int, default=5,
                   help="Fourier modes/axis for the reconstructed field")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--smoke", action="store_true",
                   help="tiny/fast configuration for a quick correctness check")
    p.add_argument("--legacy_multiplicative_v_source", action="store_true",
                   help="DEPRECATED legacy multiplicative (u-v)/v chemical rate; "
                        "raises NotImplementedError. The paper uses the injection "
                        "kernel (the default).")
    return p.parse_args()


def git_commit(repo_dir):
    """Read the current git commit hash (read-only; never mutates the repo)."""
    for args in (["git", "-C", repo_dir, "rev-parse", "HEAD"],):
        try:
            out = subprocess.run(args, capture_output=True, text=True, timeout=10)
            if out.returncode == 0:
                return out.stdout.strip()
        except Exception:
            pass
    return "unknown"


def package_versions():
    vers = {"python": platform.python_version()}
    for name in ("numpy", "jax", "matplotlib"):
        try:
            mod = __import__(name)
            vers[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            vers[name] = "not-installed"
    return vers


def write_repro_records(args, out_dir, n_steps, p_birth_death, Mu, Mv_init):
    """Write config.json and manifest.json reproducibility records."""
    now = datetime.datetime.now().isoformat()
    repo_dir = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
    commit = git_commit(repo_dir)
    versions = package_versions()

    config = {
        "experiment": "keller_segel/pp_injection",
        "model": "2D parabolic-parabolic Keller-Segel on torus [0,2pi]^2",
        "cells_pde": "u_t = div(grad u - chi u grad v) = 0  (conservative, M_u const)",
        "chemical_pde": "v_t = Delta v + u - v  (cross-species injection reaction)",
        "reaction_substep": "v^{n+1} = e^{-tau} v* + (1 - e^{-tau}) u*",
        "reaction_implementation": (
            "two-species birth-death: each v dies w.p. p=1-e^{-tau}; each u "
            "spawns a v at its location w.p. p=1-e^{-tau} (omega_u==omega_v)"
        ),
        "p_birth_death": p_birth_death,
        "exact_mass_law": "M_u(t)=M_u(0); M_v(t)=M_u+(M_v0-M_u)e^{-t}",
        "args": vars(args),
        "n_steps": n_steps,
        "M_u_initial": Mu,
        "M_v_initial": Mv_init,
        "omega_u_equals_omega_v": True,
        "population_control_active": False,
        "deterministic_reference_solver": (
            "none (validation is against the exact analytic mass law, not a grid/"
            "Fourier/FD/LDG solver)"
        ),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    manifest = {
        "git_commit": commit,
        "git_repo": repo_dir,
        "command_line": sys.argv,
        "command_string": " ".join([sys.executable] + sys.argv),
        "python_version": platform.python_version(),
        "package_versions": versions,
        "resolved_args": vars(args),
        "random_seed": args.seed,
        "output_dir": os.path.abspath(out_dir),
        "datetime": now,
        "population_control_active": False,
        "deterministic_reference_used": False,
        "reference_kind": "analytic exact mass law (no grid/Fourier/FD/LDG solver)",
        "outputs": [
            "config.json", "manifest.json", "mass_balance.csv",
            "metrics_summary.csv", "plot_data/figure_mass_balance.npz",
        ],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return commit


def main():
    args = parse()
    if args.legacy_multiplicative_v_source:
        raise NotImplementedError(
            "The legacy multiplicative (u-v)/v chemical rate on existing v "
            "particles is DEPRECATED and is NOT used in the paper. It is fragile "
            "near small v, conceptually wrong for a cross-species source, and "
            "produces no v where v is absent. Use the default cross-species "
            "injection kernel instead: v^{n+1} = e^{-tau} v* + (1-e^{-tau}) u*, "
            "i.e. v decays w.p. 1-e^{-tau} and transported u spawns v w.p. "
            "1-e^{-tau} (omega_u==omega_v)."
        )

    if args.smoke:
        args.n_samples, args.T, args.dt = 8000, 1.0, 2e-3
        if args.out_dir == "results":
            args.out_dir = "results_smoke"
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "plot_data"), exist_ok=True)

    dt, chi, n_samples = args.dt, args.chi, args.n_samples
    p = 1.0 - np.exp(-dt)                       # per-step death / birth probability
    nsteps = int(round(args.T / dt))

    dens_est, dens_eval = generate_density_estimation(
        n_freq=args.n_freq, extend="periodic", period=PERIOD)
    grad_eval = jax.grad(dens_eval, argnums=0)  # grad of the (probability) density

    # ---- initial clouds (equal per-particle mass = M_u/n_samples, M_u := 1) ----
    rng = jax.random.PRNGKey(args.seed)
    rng, k = jax.random.split(rng)

    def sample_u0(key, n):
        # u0 ~ sin^2 x cos^2 y (normalized); rejection sample -> M_u = 1
        out, got = [], 0
        while got < n:
            key, ka, kb, ku = jax.random.split(key, 4)
            xy = jax.random.uniform(ka, (4 * n, 2), minval=0.0, maxval=TWO_PI)
            dens = jnp.sin(xy[:, 0]) ** 2 * jnp.cos(xy[:, 1]) ** 2   # max 1
            keep = jax.random.uniform(kb, (4 * n,)) < dens
            out.append(np.asarray(xy[keep]))
            got += int(jnp.sum(keep))
        return jnp.asarray(np.concatenate(out)[:n])

    X1 = sample_u0(k, n_samples)                 # u cloud, N_u = n_samples, M_u = 1
    # v0: uniform on the torus, mass M_v0 -> N_v0 = round(M_v0 * n_samples)
    rng, k = jax.random.split(rng)
    Nv0 = int(round(args.Mv0 * n_samples))
    X2 = jax.random.uniform(k, (Nv0, 2), minval=0.0, maxval=TWO_PI)

    Mu = X1.shape[0] / n_samples                 # conserved cell mass (== 1)
    Mv_init = X2.shape[0] / n_samples            # actual initial chemical mass (count-aligned)

    commit = write_repro_records(args, args.out_dir, nsteps, float(p), float(Mu),
                                 float(Mv_init))

    # Evaluate the chemotactic gradient eagerly (the density coeff dict carries the
    # static mode count K, which a jit would turn into a tracer). Nothing is jitted,
    # so there is no per-step recompile from the changing v-cloud size.
    def transport(X1, X2, k1, k2):
        coeff_v = dens_est(X2)                                  # probability density of v
        Mv = X2.shape[0] / n_samples
        gradv = Mv * jax.vmap(grad_eval, in_axes=(0, None))(X1, coeff_v)   # grad of physical v field
        X1n = jnp.mod(X1 + chi * gradv * dt + jnp.sqrt(2.0 * dt) * k1, TWO_PI)   # chemotaxis + diffusion
        X2n = jnp.mod(X2 + jnp.sqrt(2.0 * dt) * k2, TWO_PI)                       # diffusion
        return X1n, X2n

    rows = []
    run_max_relerr = 0.0

    def mv_exact(t):
        return Mu + (Mv_init - Mu) * np.exp(-t)

    def record(i, n_birth, n_death):
        t = i * dt
        Mv = X2.shape[0] / n_samples
        Me = mv_exact(t)
        rows.append(dict(step=i, t=t, M_u=X1.shape[0] / n_samples, M_v=Mv,
                         M_v_exact=Me, rel_err=abs(Mv - Me) / Me,
                         n_birth=int(n_birth), n_death=int(n_death),
                         N_u=int(X1.shape[0]), N_v=int(X2.shape[0])))

    record(0, 0, 0)
    for i in range(1, nsteps + 1):
        rng, k1k, k2k, kd, kb = jax.random.split(rng, 5)
        xi1 = jax.random.normal(k1k, X1.shape, dtype=jnp.float64)
        xi2 = jax.random.normal(k2k, X2.shape, dtype=jnp.float64)
        X1, X2 = transport(X1, X2, xi1, xi2)
        # ---- cross-species injection reaction (the multi-species branching) ----
        death_v = np.asarray(jax.random.uniform(kd, (X2.shape[0],)) < p)   # v decays
        birth_u = np.asarray(jax.random.uniform(kb, (X1.shape[0],)) < p)   # u spawns v at its location
        n_death = int(death_v.sum())
        n_birth = int(birth_u.sum())
        X2 = jnp.asarray(np.concatenate([np.asarray(X2)[~death_v],
                                         np.asarray(X1)[birth_u]], axis=0))
        # running max of the mass-law error over EVERY step (not just snapshots)
        run_max_relerr = max(run_max_relerr,
                             abs(X2.shape[0] / n_samples - mv_exact(i * dt)) / mv_exact(i * dt))
        if i % max(1, nsteps // 40) == 0 or i == nsteps:
            record(i, n_birth, n_death)

    # ---- write mass-balance time series CSV ----
    cols = ["step", "t", "M_u", "M_v", "M_v_exact", "rel_err",
            "n_birth", "n_death", "N_u", "N_v"]
    mb_path = os.path.join(args.out_dir, "mass_balance.csv")
    with open(mb_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ---- key scalar metrics ----
    max_relerr = max(run_max_relerr, max(r["rel_err"] for r in rows))   # over EVERY step
    Mu_drift = max(abs(r["M_u"] - Mu) for r in rows)
    tot_b = sum(r["n_birth"] for r in rows)
    tot_d = sum(r["n_death"] for r in rows)
    last = rows[-1]

    metrics = {
        "n_samples": n_samples,
        "chi": chi,
        "T": args.T,
        "dt": dt,
        "n_steps": nsteps,
        "M_u_initial": Mu,
        "M_v_initial": Mv_init,
        "max_abs_Mu_drift": Mu_drift,
        "max_relerr_Mv_law": max_relerr,
        "M_v_final": last["M_v"],
        "M_v_exact_final": last["M_v_exact"],
        "M_v_final_abs_err": abs(last["M_v"] - last["M_v_exact"]),
        "total_births_recorded_snapshots": tot_b,
        "total_deaths_recorded_snapshots": tot_d,
        "p_birth_death": float(p),
        "git_commit": commit,
    }
    ms_path = os.path.join(args.out_dir, "metrics_summary.csv")
    with open(ms_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for kk, vv in metrics.items():
            w.writerow([kk, vv])

    # ---- plot-data npz (consumed by plot.py; NO solver rerun needed) ----
    t_arr = np.array([r["t"] for r in rows])
    npz_path = os.path.join(args.out_dir, "plot_data", "figure_mass_balance.npz")
    np.savez(
        npz_path,
        t=t_arr,
        M_u=np.array([r["M_u"] for r in rows]),
        M_v=np.array([r["M_v"] for r in rows]),
        M_v_exact=np.array([r["M_v_exact"] for r in rows]),
        rel_err=np.array([r["rel_err"] for r in rows]),
        n_birth=np.array([r["n_birth"] for r in rows]),
        n_death=np.array([r["n_death"] for r in rows]),
        N_u=np.array([r["N_u"] for r in rows]),
        N_v=np.array([r["N_v"] for r in rows]),
        Mu0=np.array(Mu),
        Mv0=np.array(Mv_init),
        max_relerr_Mv_law=np.array(max_relerr),
        max_abs_Mu_drift=np.array(Mu_drift),
    )

    print("\n=== cross-species injection reaction: v_(n+1) = e^-tau v* + (1-e^-tau) u* ===")
    print(f"n_samples={n_samples}  chi={chi}  T={args.T}  dt={dt}  steps={nsteps}  p={p:.4e}")
    print(f"M_u (cell mass): conserved at {Mu:.4f}  (max drift {Mu_drift:.2e})")
    print(f"M_v: {Mv_init:.3f} -> {last['M_v']:.4f}   exact law -> {last['M_v_exact']:.4f}")
    print(f"MAX rel. error of M_v(t) vs M_u+(M_v0-M_u)e^-t (every step): {max_relerr:.3e}")
    print(f"recorded births(from u)={tot_b}  deaths(of v)={tot_d}  (sampled snapshots)")
    print(f"wrote {mb_path}")
    print(f"wrote {ms_path}")
    print(f"wrote {npz_path}")
    print(f"wrote {os.path.join(args.out_dir, 'config.json')}")
    print(f"wrote {os.path.join(args.out_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
