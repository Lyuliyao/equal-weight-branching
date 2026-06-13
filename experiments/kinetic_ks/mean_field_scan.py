"""Mean-field parameter scan for the 6D field-coupled kinetic Keller-Segel.

Runs ONE weighted cloud (fixed N0, no branching -> no buffer overflow) through
the real dynamics (verified field solve + OU velocity + reaction) for a small
grid of parameter sets, and reports the trajectory of:
  total mass M(t)=sum w/N0, ||c||_inf, ||rho||_inf, core radius R_0.5(t),
  weighted mean reaction rate, mass-fraction with r>0, max:mean weight ratio.

Purpose: BEFORE committing a multi-hour 4-method pilot, find parameters where
the chemoattractant activation S_c switches on (c crosses c0) AND the blob
focuses (R_0.5 decreases) AND growth is moderate (M(T) ~ 2-10x, not explosive).
Reuses ONLY already-Codex-verified functions; writes a single JSON summary.

Run (CPU dev node):
  JAX_PLATFORMS=cpu python mean_field_scan.py
"""
import os
import sys
import json
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from field_kinetic import build_half_spectrum, density_coeffs, eval_field, eval_rho
from common_kinetic import (
    sample_initial, ou_velocity_step, reaction_rate_field, reaction_weighted,
    mass_centroid_x, quantile_core_radii, S_c,
)


def run_one(p, N0=4000, T=2.0, tau=2e-3, K_x=6, seed=0, n_diag=11):
    Kvecs_np, ksq_np = build_half_spectrum(K_x)
    Kvecs, ksq = jnp.asarray(Kvecs_np), jnp.asarray(ksq_np)
    kappa = p["kappa"]
    key = jax.random.PRNGKey(seed)
    key, ki = jax.random.split(key)
    Z = sample_initial(ki, N0, p["sigma_x"], p["Tv"], kind="single")  # (N0,6)
    X, V = Z[:, :3], Z[:, 3:]
    w = jnp.ones((N0,), jnp.float64)
    mask = jnp.ones((N0,), bool)
    steps = int(round(T / tau))

    def diagnose(X, V, w):
        A, B = density_coeffs(X, w, mask, N0, Kvecs)
        c, gc = eval_field(X, A, B, Kvecs, ksq, kappa)
        Mf = float(jnp.sum(w) / N0)
        rho = eval_rho(X, A, B, Kvecs, jnp.asarray(Mf))
        r = reaction_rate_field(c, rho, p["lambda_g"], p["alpha_rho"], p["beta"],
                                p["c0"], p["delta_c"], p["rho0"])
        Sc = S_c(c, p["c0"], p["delta_c"])
        xc = mass_centroid_x(X, w)
        R05, R09 = quantile_core_radii(X, w, xc, qs=(0.5, 0.9))
        c, rho, r, wn, Sc = map(np.asarray, (c, rho, r, w, Sc))
        Wt = float(np.sum(wn))
        return dict(mass=Mf,
                    c_max=float(np.max(c)),               # signed max (drives S_c)
                    c_inf=float(np.max(np.abs(c))),
                    rho_inf=float(np.max(np.abs(rho))), R05=float(R05),
                    mean_Sc=float(np.sum(wn * Sc) / Wt),  # weighted activation level
                    mean_r=float(np.sum(wn * r) / Wt),
                    frac_rpos=float(np.sum(wn[r > 0]) / Wt),
                    maxw_over_mean=float(np.max(wn) / np.mean(wn)))

    @jax.jit
    def step(X, V, w, xi):
        A, B = density_coeffs(X, w, mask, N0, Kvecs)
        c, gc = eval_field(X, A, B, Kvecs, ksq, kappa)
        Mf = jnp.sum(w) / N0
        rho = eval_rho(X, A, B, Kvecs, Mf)
        r = reaction_rate_field(c, rho, p["lambda_g"], p["alpha_rho"], p["beta"],
                                p["c0"], p["delta_c"], p["rho0"])
        Xn, Vn = ou_velocity_step(X, V, gc, p["gamma_v"], p["chi"], p["D_v"], tau, xi)
        wn = reaction_weighted(w, r, tau)
        return Xn, Vn, wn

    diag_steps = set(int(round(s)) for s in np.linspace(0, steps, n_diag))
    traj = []
    key2 = key
    for i in range(steps + 1):
        if i in diag_steps:
            d = diagnose(X, V, w)
            d["t"] = i * tau
            traj.append(d)
            if not np.isfinite(d["mass"]) or d["mass"] > 1e3:
                break  # explosive; stop early
        if i == steps:
            break
        key2, kt = jax.random.split(key2)
        xi = jax.random.normal(kt, (N0, 3), jnp.float64)
        X, V, w = step(X, V, w, xi)
    return traj


# Base = corrected-plan params; variations tighten the blob / lower c0 / raise
# chi so the chemoattractant activation switches on as the blob concentrates.
# Round 3: keep self-limiting alpha_rho (bounded growth from round 2) but REDUCE
# the velocity diffusion D_v so the spatial diffusion D_x=D_v/gamma_v^2 drops and
# chemotaxis can overcome spreading (round 2 all spread at D_v=1 -> D_x=0.25).
# Keep gamma_v=2-3 so the test stays genuinely KINETIC (not overdamped/parabolic).
# COMMON now sets D_v/Tv per-set (overridden in each row); Tv = D_v/gamma_v.
COMMON = dict(kappa=0.5, delta_c=0.05, rho0=0.2)
SETS = {
    "base_plan": dict(sigma_x=0.7, c0=0.10, chi=1.5, lambda_g=4.0, alpha_rho=1.0, beta=0.20,
                      gamma_v=2.0, D_v=1.0, Tv=0.50),
    "U1": dict(sigma_x=0.5, c0=0.05, chi=3.0, lambda_g=4.0, alpha_rho=3.0, beta=0.30,
               gamma_v=2.0, D_v=0.5, Tv=0.25),   # D_x=0.125
    "U2": dict(sigma_x=0.5, c0=0.05, chi=3.0, lambda_g=4.0, alpha_rho=3.0, beta=0.30,
               gamma_v=2.0, D_v=0.3, Tv=0.15),   # D_x=0.075
    "U3": dict(sigma_x=0.5, c0=0.05, chi=3.0, lambda_g=4.0, alpha_rho=3.0, beta=0.30,
               gamma_v=3.0, D_v=0.6, Tv=0.20),   # D_x=0.067, still kinetic
    "U4": dict(sigma_x=0.45, c0=0.04, chi=4.0, lambda_g=5.0, alpha_rho=3.5, beta=0.30,
               gamma_v=2.0, D_v=0.3, Tv=0.15),   # strong chi, low D_x
}


def main():
    out = {}
    for name, extra in SETS.items():
        p = dict(COMMON, **extra)
        t0 = time.time()
        traj = run_one(p)
        dt = time.time() - t0
        first, last = traj[0], traj[-1]
        out[name] = dict(params=p, traj=traj)
        print(f"\n=== {name} ({dt:.1f}s) ===  "
              f"sigma_x={p['sigma_x']} c0={p['c0']} chi={p['chi']} "
              f"lam_g={p['lambda_g']} beta={p['beta']}")
        print(f"  t=0   : mass={first['mass']:.2f} c_max={first['c_max']:+.3f} "
              f"mean_Sc={first['mean_Sc']:.3f} rho_inf={first['rho_inf']:.3f} "
              f"R05={first['R05']:.3f} mean_r={first['mean_r']:+.3f} "
              f"frac_rpos={first['frac_rpos']:.2f}")
        print(f"  t={last['t']:.1f} : mass={last['mass']:.2f} c_max={last['c_max']:+.3f} "
              f"mean_Sc={last['mean_Sc']:.3f} rho_inf={last['rho_inf']:.3f} "
              f"R05={last['R05']:.3f} mean_r={last['mean_r']:+.3f} "
              f"frac_rpos={last['frac_rpos']:.2f} maxw/mean={last['maxw_over_mean']:.1f}")
        # activation = does the chemoattractant term S_c become non-negligible;
        # focusing = does the core radius shrink.  Use signed c_max / mean_Sc.
        mid = traj[len(traj) // 2]
        act = "ON " if mid["mean_Sc"] > 0.1 else "off"
        foc = "FOCUS" if last["R05"] < 0.9 * first["R05"] else "spread"
        print(f"  mid c_max={mid['c_max']:+.3f} mean_Sc={mid['mean_Sc']:.3f} "
              f"(c0={p['c0']}) -> S_c {act};  "
              f"R05 {first['R05']:.3f}->{last['R05']:.3f} -> {foc}")
    with open("mean_field_scan_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote mean_field_scan_results.json")


if __name__ == "__main__":
    main()
