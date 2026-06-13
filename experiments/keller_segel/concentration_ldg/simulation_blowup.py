"""2D parabolic-ELLIPTIC Keller-Segel blow-up-time stress test.

Model (cleanest for an actual T* estimate):
    u_t = Delta u - chi div(u grad v),     v slaved to u by   -Delta v + v = u.
chi = 1.  u is carried by N equal-weight particles (u is conservative; NO
u-branching).  v is solved SPECTRALLY on a core-adaptive Fourier window:
    v_hat = u_hat / (|k|^2 + 1).

Particle SDE for  u_t = Delta u - chi div(u grad v):
    dX = +chi grad v dt + sqrt(2 dt) xi.
NOTE ON DRIFT SIGN.  The brief wrote "dX = -chi grad v dt", but the PDE
u_t = Delta u - chi div(u grad v) is the transport-diffusion equation with
velocity field b = +chi grad v (advection term -div(u b) with b=+chi grad v),
which is also exactly what the existing chemotaxis code uses
(case2_test3/simulation.py:267-269: F_1 = nabla_v_X1; dX1 = F_1*dt + noise).
With v peaked at the cluster centre, +chi grad v points INWARD and drives
aggregation/blow-up; -chi grad v would point outward and PREVENT blow-up.
We therefore use the physically correct +chi grad v and flag this explicitly.

Initial data: canonical super-critical Gaussian  u0(x) = 840 exp(-84 |x|^2),
physical mass = 840*pi/84 = 10*pi > 8*pi critical  => finite-time blow-up
expected.  Effectively-unbounded domain via the core-adaptive window.

Reconstruction reuse: Fourier coefficient tensor pattern adapted from
  case2_test3/density.py:8-132 and case2_test3/simulation.py:21-160,
replacing the [min,max] data box with a core-adaptive window (see
adaptive_window.py).  case2_test3 files are NOT modified.

Diagnostics logged every `diag_every` steps and saved to results/.
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

from adaptive_window import compute_window, density_coeffs_y, chem_force, peak_density


# ---------------------------------------------------------------------------
def sample_u0(rng_np, N, a=84.0):
    """Sample N points from u0 ~ exp(-a |x|^2) (each coord N(0,1/(2a)))."""
    std = 1.0 / np.sqrt(2.0 * a)
    return jnp.asarray(rng_np.normal(0.0, std, size=(N, 2)))


def quantile_radii(X, x_c, qs):
    r = jnp.sqrt(jnp.sum((X - x_c) ** 2, axis=1))
    return {q: float(jnp.quantile(r, q)) for q in qs}


def core_counts(X, x_c, radii):
    r = jnp.sqrt(jnp.sum((X - x_c) ** 2, axis=1))
    return {q: int(jnp.sum(r <= radii[q])) for q in radii}


def make_step(K, chi, gamma, gamma_diff, D, L_min, q_window):
    """Build a jitted one-step transport map.  mass = total physical mass (const)."""

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

    return jax.jit(step, static_argnames=())


def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    rng_np = np.random.default_rng(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    N = args.N
    mass = 10.0 * np.pi                    # physical mass of u0 = 840 exp(-84 r^2)
    chi = 1.0
    tau = args.dt
    n_steps = args.n_steps
    qs = [0.5, 0.8, 0.9, 0.99]

    X = sample_u0(rng_np, N)

    step = make_step(args.K, chi, args.gamma, args.gamma_diff, args.D,
                     args.L_min, args.q_window)

    # save config
    cfg = vars(args).copy()
    cfg.update(dict(mass=float(mass), chi=chi, model="2D parabolic-elliptic KS",
                    drift_sign="+chi grad v (inward, blow-up driving)"))
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    rows = []
    header = (["step", "t", "N", "K", "L", "h_eff", "xc_x", "xc_y",
               "S_u", "R2"]
              + [f"R_{q}" for q in qs] + [f"N_{q}" for q in qs]
              + ["R50_over_heff", "peak_PK_u"])

    t0 = time.time()
    for i in range(n_steps + 1):
        # diagnostics at step i (BEFORE moving), so i=0 is the initial state
        x_c = jnp.mean(X, axis=0)
        d2 = jnp.sum((X - x_c) ** 2, axis=1)
        S_u = float(jnp.mean(d2))
        R2 = float(np.sqrt(S_u))
        radii = quantile_radii(X, x_c, qs)
        counts = core_counts(X, x_c, radii)
        # window half-width for h_eff and peak reconstruction
        _, L = compute_window(X, gamma=args.gamma, gamma_diff=args.gamma_diff,
                              D=args.D, tau=tau, L_min=args.L_min,
                              q_window=args.q_window)
        L = float(L)
        h_eff = L / args.K
        R50_over_heff = radii[0.5] / h_eff if h_eff > 0 else np.nan

        if i % args.diag_every == 0 or i == n_steps:
            # secondary peak cross-check (coarse grid)
            Y = (X - jnp.mean(X, axis=0)) * (jnp.pi / L)
            coeff = density_coeffs_y(Y, args.K)
            peak = float(peak_density(coeff, jnp.mean(X, axis=0), L, mass,
                                      n_grid=65))
            row = ([i, i * tau, N, args.K, L, h_eff,
                    float(x_c[0]), float(x_c[1]), S_u, R2]
                   + [radii[q] for q in qs] + [counts[q] for q in qs]
                   + [R50_over_heff, peak])
            rows.append(row)
            if args.verbose:
                print(f"step {i:5d} t={i*tau:.3e} S_u={S_u:.4e} "
                      f"R50/h_eff={R50_over_heff:5.2f} N50={counts[0.5]:6d} "
                      f"peak={peak:.2e} L={L:.4f}", flush=True)

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

    S0 = rows[0][header.index("S_u")]
    SN = rows[-1][header.index("S_u")]
    print(f"\n[{tag}] wrote {out_csv}")
    print(f"[{tag}] runtime {t1-t0:.1f}s  S_u(0)={S0:.4e} S_u(T)={SN:.4e} "
          f"ratio={S0/SN:.1f}x  (fixed-window ref reached ~6x => S/S0=0.17)")
    print(f"[{tag}] S_u(T)/S_u(0) = {SN/S0:.4f}  (deeper than fixed-window if < 0.17)")
    return out_csv


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=200000)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--dt", type=float, default=1e-7)
    p.add_argument("--n_steps", type=int, default=2000)
    p.add_argument("--diag_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gamma", type=float, default=3.0)
    p.add_argument("--gamma_diff", type=float, default=6.0)
    p.add_argument("--D", type=float, default=1.0)
    p.add_argument("--L_min", type=float, default=1e-3)
    p.add_argument("--q_window", type=float, default=0.99)
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--verbose", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
