"""2D parabolic-elliptic Keller-Segel blow-up simulation with LDG diagnostics.

THIN VARIANT of `../blowup_time/simulation_blowup.py`.  Identical model, identical
particle update, identical core-adaptive Fourier window and screened-Poisson
chemotactic force (all imported from `../blowup_time/adaptive_window.py`).  The
only additions are LDG-style diagnostics:
  * extra per-step CSV columns  S_L2  and  Mcore_0.01 / Mcore_0.02 / Mcore_0.04
    (from ldg_diagnostics.recon_L2_norm and ldg_diagnostics.core_mass), and
  * reconstructed-field SNAPSHOTS (the physical-u field on the adaptive-window
    grid, plus x_c and L) saved to .npz at a configurable list of report times.

Model (unchanged):  u_t = Delta u - chi div(u grad v),  -Delta v + v = u, chi=1.
Particle SDE:  dX = +chi grad v dt + sqrt(2 dt) xi   (inward, blow-up driving;
the drift-sign note lives in ../blowup_time/simulation_blowup.py / README).
Canonical IC: u0 = 840 exp(-84|x|^2), physical mass = 10*pi > 8*pi (supercrit).

REUSE.  sample_u0, quantile_radii, core_counts and the per-step force/window are
imported, not re-implemented.  No file in ../blowup_time is modified.
"""
import os
import sys
import csv
import json
import time
import argparse

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Reuse the validated blow-up modules, vendored into THIS directory so the
# experiment is self-contained (no cross-directory imports).
_BLOWUP_DIR = os.path.dirname(os.path.abspath(__file__))
if _BLOWUP_DIR not in sys.path:
    sys.path.insert(0, _BLOWUP_DIR)

from adaptive_window import (compute_window, density_coeffs_y, chem_force,
                             peak_density, eval_density_y)
from simulation_blowup import sample_u0, quantile_radii, core_counts

from ldg_diagnostics import recon_L2_norm, core_mass


# ---------------------------------------------------------------------------
# One-step transport map (same as simulation_blowup.make_step).
# ---------------------------------------------------------------------------
def make_step(K, chi, gamma, gamma_diff, D, L_min, q_window):
    def step(X, rng, mass, tau):
        x_c, L = compute_window(X, gamma=gamma, gamma_diff=gamma_diff, D=D,
                                tau=tau, L_min=L_min, q_window=q_window)
        Y = (X - x_c) * (jnp.pi / L)
        coeff = density_coeffs_y(Y, K)
        F = chem_force(X, coeff, x_c, L, mass, chi)        # = -chi grad v
        drift = -F                                          # = +chi grad v (inward)
        rng, key = jax.random.split(rng)
        noise = jax.random.normal(key, shape=X.shape, dtype=X.dtype)
        X_new = X + drift * tau + jnp.sqrt(2.0 * tau) * noise
        return X_new, rng, x_c, L, coeff

    return jax.jit(step)


# ---------------------------------------------------------------------------
# Reconstructed physical-u field on the adaptive-window grid (for snapshots).
# ---------------------------------------------------------------------------
def reconstruct_field(coeff, x_c, L, mass, n_grid):
    """Return (XX, YY, U) physical-u field on the side-2L window mesh.

    Mirrors adaptive_window.peak_density rescaling exactly:
    u_phys = mass*(pi/L)^2 * rho_y, evaluated on linspace(-pi,pi,n_grid)^2,
    mapped to physical coords x = x_c + (L/pi) y.
    """
    g = jnp.linspace(-jnp.pi, jnp.pi, n_grid)
    GX, GY = jnp.meshgrid(g, g, indexing="ij")
    pts = jnp.stack([GX.ravel(), GY.ravel()], axis=1)
    rho_y = jax.vmap(eval_density_y, in_axes=(0, None))(pts, coeff)
    U = (mass * (jnp.pi / L) ** 2 * rho_y).reshape(n_grid, n_grid)
    XX = x_c[0] + (L / jnp.pi) * GX                    # physical coordinates
    YY = x_c[1] + (L / jnp.pi) * GY
    return np.asarray(XX), np.asarray(YY), np.asarray(U)


# ---------------------------------------------------------------------------
def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    snap_dir = os.path.join(args.outdir, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    rng_np = np.random.default_rng(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    N = args.N
    mass = float(args.mass)               # physical mass of u0 (default 10*pi)
    chi = 1.0
    tau = args.dt
    n_steps = args.n_steps
    qs = [0.5, 0.8, 0.9, 0.99]
    core_radii = [0.01, 0.02, 0.04]

    # report times: caller list + the canonical LDG cross-check times.
    report_times = sorted(set(list(args.report_times) + [5e-5, 1e-4, 1.5e-4]))

    X = sample_u0(rng_np, N)

    step = make_step(args.K, chi, args.gamma, args.gamma_diff, args.D,
                     args.L_min, args.q_window)

    # save config + a small README copy into outdir
    cfg = vars(args).copy()
    cfg["report_times"] = report_times
    cfg.update(dict(mass=mass, chi=chi, model="2D parabolic-elliptic KS",
                    drift_sign="+chi grad v (inward, blow-up driving)",
                    core_radii=core_radii))
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(args.outdir, "README"), "w") as f:
        f.write(
            "KS 2D LDG-diagnostics run.\n"
            f"model: u_t = Delta u - chi div(u grad v), -Delta v + v = u, chi=1\n"
            f"IC: u0 = 840 exp(-84|x|^2), mass={mass} (=10pi, supercritical)\n"
            f"N={N} K={args.K} dt={tau} n_steps={n_steps} seed={args.seed} "
            f"q_window={args.q_window}\n"
            "CSV columns: step,t,N,K,L,h_eff,xc_x,xc_y,S_u,R2,"
            "R_q...,N_q...,R50_over_heff,peak_PK_u,S_L2,"
            "Mcore_0.01,Mcore_0.02,Mcore_0.04\n"
            "snapshots/: snap_<tag>_t<...>.npz with keys X,Y,U,x_c,L,t,mass,K,N\n"
            "Reuses ../blowup_time (imported, not copied).\n")

    rows = []
    header = (["step", "t", "N", "K", "L", "h_eff", "xc_x", "xc_y",
               "S_u", "R2"]
              + [f"R_{q}" for q in qs] + [f"N_{q}" for q in qs]
              + ["R50_over_heff", "peak_PK_u", "S_L2"]
              + [f"Mcore_{r}" for r in core_radii])

    # track which report times have been saved (save at first step at/after each)
    saved_reports = set()

    t0 = time.time()
    for i in range(n_steps + 1):
        t_now = i * tau
        # diagnostics at step i (BEFORE moving), so i=0 is the initial state
        x_c = jnp.mean(X, axis=0)
        d2 = jnp.sum((X - x_c) ** 2, axis=1)
        S_u = float(jnp.mean(d2))
        R2 = float(np.sqrt(S_u))
        radii = quantile_radii(X, x_c, qs)
        counts = core_counts(X, x_c, radii)
        _, L = compute_window(X, gamma=args.gamma, gamma_diff=args.gamma_diff,
                              D=args.D, tau=tau, L_min=args.L_min,
                              q_window=args.q_window)
        L = float(L)
        h_eff = L / args.K
        R50_over_heff = radii[0.5] / h_eff if h_eff > 0 else np.nan

        # decide whether to log this step
        is_log = (i % args.diag_every == 0) or (i == n_steps)
        # decide whether a report-time snapshot is due at this step
        due_reports = [rt for rt in report_times
                       if rt not in saved_reports and t_now >= rt]

        if is_log or due_reports:
            Y = (X - x_c) * (jnp.pi / L)
            coeff = density_coeffs_y(Y, args.K)
            peak = float(peak_density(coeff, x_c, L, mass, n_grid=65))
            S_L2 = float(recon_L2_norm(coeff, x_c, L, mass,
                                       n_grid=args.n_grid_diag))
            cm = core_mass(X, x_c, core_radii, N, mass)

            if is_log:
                row = ([i, t_now, N, args.K, L, h_eff,
                        float(x_c[0]), float(x_c[1]), S_u, R2]
                       + [radii[q] for q in qs] + [counts[q] for q in qs]
                       + [R50_over_heff, peak, S_L2]
                       + [cm[r] for r in core_radii])
                rows.append(row)
                if args.verbose:
                    print(f"step {i:6d} t={t_now:.3e} S_u={S_u:.4e} "
                          f"R50/h={R50_over_heff:5.2f} N50={counts[0.5]:7d} "
                          f"peak={peak:.2e} S_L2={S_L2:.3e} "
                          f"Mc.01={cm[0.01]:.3e} L={L:.4f}", flush=True)

            for rt in due_reports:
                saved_reports.add(rt)
                XX, YY, U = reconstruct_field(coeff, x_c, L, mass,
                                              args.n_grid_snap)
                tag = (f"N{N}_K{args.K}_dt{args.dt:.0e}_q{args.q_window}"
                       f"_seed{args.seed}")
                npz = os.path.join(snap_dir, f"snap_{tag}_t{rt:.2e}.npz")
                np.savez(npz, X=XX, Y=YY, U=U,
                         x_c=np.asarray(x_c), L=np.float64(L),
                         t=np.float64(t_now), report_t=np.float64(rt),
                         mass=np.float64(mass), K=np.int64(args.K),
                         N=np.int64(N), peak=np.float64(peak),
                         S_L2=np.float64(S_L2))
                if args.verbose:
                    print(f"  [snapshot] report_t={rt:.2e} (t={t_now:.3e}) "
                          f"-> {os.path.basename(npz)}", flush=True)

        if i == n_steps:
            break
        X, rng, _, _, _ = step(X, rng, mass, tau)
        X.block_until_ready()

    t1 = time.time()
    tag = f"N{N}_K{args.K}_dt{args.dt:.0e}_q{args.q_window}_seed{args.seed}"
    out_csv = os.path.join(args.outdir, f"diag_{tag}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    S0 = rows[0][header.index("S_L2")]
    SN = rows[-1][header.index("S_L2")]
    print(f"\n[{tag}] wrote {out_csv}")
    print(f"[{tag}] runtime {t1-t0:.1f}s  S_L2(0)={S0:.4e} S_L2(T)={SN:.4e} "
          f"rise={SN/S0:.2f}x")
    print(f"[{tag}] snapshots in {snap_dir}")
    return out_csv


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=200000)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--dt", type=float, default=1e-7)
    p.add_argument("--n_steps", type=int, default=2000)
    p.add_argument("--diag_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mass", type=float, default=10.0 * np.pi)
    p.add_argument("--gamma", type=float, default=3.0)
    p.add_argument("--gamma_diff", type=float, default=6.0)
    p.add_argument("--D", type=float, default=1.0)
    p.add_argument("--L_min", type=float, default=1e-3)
    p.add_argument("--q_window", type=float, default=0.8)
    p.add_argument("--n_grid_diag", type=int, default=129,
                   help="grid for the S_L2 reconstructed-norm integral")
    p.add_argument("--n_grid_snap", type=int, default=257,
                   help="grid for the saved snapshot field")
    p.add_argument("--report_times", type=float, nargs="*",
                   default=[],
                   help="extra snapshot report times (the canonical "
                        "5e-5,1e-4,1.5e-4 are always added)")
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--verbose", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
