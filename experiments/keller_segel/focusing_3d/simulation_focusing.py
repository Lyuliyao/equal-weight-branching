"""
3D Keller-Segel focusing / self-convergence stress test (parabolic-elliptic).

Model on the periodic box [-L/2, L/2]^3 (L = 12):
    d_t rho = Delta rho - chi div(rho grad c),   -Delta c + kappa^2 c = rho - rho_bar,
with chi = 1, kappa = 0.1. rho is a SINGLE conservative particle cloud: pure
chemotaxis + diffusion, NO branching / reaction. The particle count is constant,
so total mass is conserved exactly (a bookkeeping sanity check, 5.5(a)).

Particle SDE (Euler-Maruyama):
    X_{n+1} = wrap( X_n + chi * grad c(X_n) * tau + sqrt(2 tau) * xi_n ),
    xi_n ~ N(0, I_3),   wrap to [-L/2, L/2]^3.
The drift is +chi grad c (INWARD, aggregating; verified by construction in
field3d_screened.selftest_field3d). grad c is solved spectrally each step from
the current cloud via the 3D screened-Poisson solver.

The physical mass M enters ONLY the field scale (rho = M * p); it does not change
the particle count. N controls Monte-Carlo resolution, H the reconstruction
bandwidth, M the chemotactic forcing strength.

Outputs (into out_dir):
  - diagnostics CSV (one row per snapshot time);
  - optional density-slice npz at report times;
  - a copy of the resolved config (config_used.json) and a short README.

CLI:
  python simulation_focusing.py --N 200000 --M 60 --sigma 0.45 --H 18 \
      --tau 1e-4 --T 1.0 --L 12 --kappa 0.1 --chi 1.0 \
      --ic_type radial --seed 0 --out_dir results/radial_M60_N2e5_H18 \
      [--sigma_c 0.25] [--n_report 11] [--save_slices]

Run only after Codex cold verification (see CLAUDE.md protocol). No existing file
is modified; this script lives in case_3d_focusing/.
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

import field3d_screened as fld
import diagnostics_focusing as diag
from ic_focusing import gaussian_ic, tetra_clusters_ic, TETRA_CENTERS


# ---------------------------------------------------------------------------
# CLI / config
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="3D KS focusing / self-convergence")
    p.add_argument("--N", type=float, required=True,
                   help="number of particles (float ok, e.g. 2e5)")
    p.add_argument("--M", type=float, required=True, help="physical mass")
    p.add_argument("--sigma", type=float, default=0.45,
                   help="radial Gaussian width (5.5b)")
    p.add_argument("--sigma_c", type=float, default=0.25,
                   help="per-cluster width (5.5c tetra)")
    p.add_argument("--H", type=int, default=18,
                   help="Fourier bandwidth (modes per axis)")
    p.add_argument("--tau", type=float, default=1e-4, help="time step")
    p.add_argument("--T", type=float, default=1.0, help="final time")
    p.add_argument("--L", type=float, default=12.0, help="box side")
    p.add_argument("--kappa", type=float, default=0.1, help="screening kappa")
    p.add_argument("--chi", type=float, default=1.0, help="chemotactic sensitivity")
    p.add_argument("--ic_type", choices=["radial", "tetra"], default="radial")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--n_report", type=int, default=11,
                   help="number of diagnostic snapshots (incl. t=0 and T)")
    p.add_argument("--save_slices", action="store_true",
                   help="save density-slice npz at report times")
    p.add_argument("--Qc_Hlo", type=int, default=12)
    p.add_argument("--Qc_Hhi", type=int, default=24)
    p.add_argument("--grid_half", type=float, default=2.0,
                   help="half-extent of the coarse eval grid for peaks")
    p.add_argument("--grid_nside", type=int, default=33,
                   help="points per axis of the coarse eval grid")
    p.add_argument("--slice_nside", type=int, default=129,
                   help="points per axis for the saved z=0 density slice")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# One transport step (jitted, closed over static scalars).
# ---------------------------------------------------------------------------
def make_step(H, L, M, kappa, chi, tau):
    sqrt2tau = jnp.sqrt(2.0 * tau)

    @jax.jit
    def step(X, key):
        coeff_p = fld.density_coeffs(X, H, L)
        coeff_c = fld.screened_solve(coeff_p, M, kappa)
        gc = fld.grad_c(X, coeff_c)                       # (N,3) inward drift
        xi = jax.random.normal(key, X.shape, dtype=jnp.float64)
        Xn = X + chi * gc * tau + sqrt2tau * xi
        Xn = (Xn + L / 2.0) % L - L / 2.0                 # wrap to box
        return Xn
    return step


# ---------------------------------------------------------------------------
# Density-slice helper (z = centroid plane) for visualization npz.
# ---------------------------------------------------------------------------
def density_slice(X, M, H, L, x_c, half, n_side):
    g = jnp.linspace(-half, half, n_side)
    GX, GY = jnp.meshgrid(g, g, indexing="ij")
    pts = jnp.stack([GX.ravel(), GY.ravel(),
                     jnp.zeros((GX.size,))], axis=1) + jnp.asarray(x_c)[None, :]
    pts = (pts + L / 2.0) % L - L / 2.0
    coeff_p = fld.density_coeffs(X, H, L)
    vals = fld.eval_density(pts, coeff_p, M)
    return np.asarray(g), np.asarray(vals).reshape(n_side, n_side)


# ---------------------------------------------------------------------------
# Diagnostics row at a snapshot.
# ---------------------------------------------------------------------------
def snapshot_row(X, t, args, n0, labels, n_clusters):
    L, M, kappa = args.L, args.M, args.kappa
    x_c = diag.torus_centroid(X, L)
    rad = diag.core_radii(X, x_c, L)
    R05, R09 = rad[0.5], rad[0.9]
    rc = diag.rho_core(R05, M)
    PH = diag.peak_density_PH(X, M, args.H, L, x_c,
                              grid_half=args.grid_half, n_side=args.grid_nside)
    CH = diag.peak_chem_CH(X, M, args.H, L, kappa, x_c,
                           grid_half=args.grid_half, n_side=args.grid_nside)
    Qc, c_lo, c_hi = diag.self_convergence_Qc(
        X, M, L, kappa, x_c, H_lo=args.Qc_Hlo, H_hi=args.Qc_Hhi,
        grid_half=args.grid_half, n_side=args.grid_nside)
    drift = diag.mass_drift(X.shape[0], n0)
    cfl = diag.drift_cfl(X, M, args.H, L, kappa, args.chi, args.tau)

    row = {
        "t": float(t), "n_active": int(X.shape[0]),
        "x_c_0": float(x_c[0]), "x_c_1": float(x_c[1]), "x_c_2": float(x_c[2]),
        "R_0.5": R05, "R_0.9": R09, "rho_core": rc,
        "P_H": PH, "C_H": CH,
        "Qc": Qc, "C_Hlo": c_lo, "C_Hhi": c_hi,
        "mass_drift": drift, "drift_cfl": cfl,
    }

    if labels is not None:
        cents = diag.cluster_centroids(X, labels, n_clusters, L)
        crad = diag.cluster_core_radii(X, labels, cents, n_clusters, L)
        dmin = diag.min_intercluster_distance(cents, L)
        row["min_intercluster_dist"] = dmin
        for m in range(n_clusters):
            row[f"cl{m}_c0"] = float(cents[m, 0])
            row[f"cl{m}_c1"] = float(cents[m, 1])
            row[f"cl{m}_c2"] = float(cents[m, 2])
            row[f"cl{m}_R0.5"] = float(crad[m, 0])
            row[f"cl{m}_R0.9"] = float(crad[m, 1])
    return row, x_c


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    args.N = int(round(args.N))
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- initial particles ------------------------------------------------
    rng = np.random.default_rng(args.seed)
    labels, n_clusters = None, 0
    if args.ic_type == "radial":
        X, M = gaussian_ic(rng, args.N, args.M, args.sigma, args.L)
    else:
        X, M, labels = tetra_clusters_ic(rng, args.N, args.M, args.sigma_c,
                                         TETRA_CENTERS, args.L)
        n_clusters = TETRA_CENTERS.shape[0]
    assert abs(M - args.M) < 1e-12
    n0 = X.shape[0]

    n_steps = int(round(args.T / args.tau))
    # report-step indices (snapshots), inclusive of 0 and n_steps
    n_report = max(2, args.n_report)
    report_steps = sorted(set(
        int(round(k)) for k in np.linspace(0, n_steps, n_report)))

    step = make_step(args.H, args.L, M, args.kappa, args.chi, args.tau)
    base_key = jax.random.PRNGKey(args.seed)

    rows = []
    slices = {}
    t0 = time.time()
    for i in range(n_steps + 1):
        t = i * args.tau
        if i in report_steps:
            row, x_c = snapshot_row(X, t, args, n0, labels, n_clusters)
            rows.append(row)
            print(f"[t={t:.4f}] n={row['n_active']} R0.5={row['R_0.5']:.4f} "
                  f"R0.9={row['R_0.9']:.4f} rho_core={row['rho_core']:.3e} "
                  f"P_H={row['P_H']:.3e} C_H={row['C_H']:.3e} Qc={row['Qc']:.4f} "
                  f"drift={row['mass_drift']:.2e}", flush=True)
            if args.save_slices:
                g, sl = density_slice(X, M, args.H, args.L, x_c,
                                      args.grid_half, args.slice_nside)
                slices[f"t{i:06d}_grid"] = g
                slices[f"t{i:06d}_slice"] = sl
                slices[f"t{i:06d}_xc"] = np.asarray(x_c)
        if i == n_steps:
            break
        key = jax.random.fold_in(base_key, i)
        X = step(X, key)

    elapsed = time.time() - t0

    # ---- write diagnostics CSV -------------------------------------------
    csv_path = os.path.join(args.out_dir, "diagnostics.csv")
    fieldnames = list(rows[0].keys())
    # union of keys (tetra rows may all share keys; guard anyway)
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if args.save_slices and slices:
        np.savez(os.path.join(args.out_dir, "density_slices.npz"), **slices)

    # ---- config copy + README -------------------------------------------
    cfg = vars(args).copy()
    cfg.update({"n_steps": n_steps, "report_steps": report_steps,
                "n_particles": n0, "elapsed_s": elapsed,
                "tetra_centers": TETRA_CENTERS.tolist()})
    with open(os.path.join(args.out_dir, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    with open(os.path.join(args.out_dir, "README.txt"), "w") as f:
        f.write(
            "3D Keller-Segel focusing / self-convergence run\n"
            f"  ic_type={args.ic_type} N={n0} M={M} H={args.H} tau={args.tau} "
            f"T={args.T} L={args.L} kappa={args.kappa} chi={args.chi} "
            f"seed={args.seed}\n"
            "  model: d_t rho = Delta rho - chi div(rho grad c), "
            "-Delta c + kappa^2 c = rho - rho_bar\n"
            "  single conservative cloud (no branching); drift +chi grad c "
            "(inward).\n"
            f"  elapsed {elapsed:.1f}s. See diagnostics.csv, config_used.json.\n")

    print(f"\n[done] wrote {csv_path}")
    print(f"[done] elapsed {elapsed:.1f}s, {n_steps} steps, {n0} particles")
    print(f"[done] final R_0.5={rows[-1]['R_0.5']:.4f} "
          f"rho_core={rows[-1]['rho_core']:.3e} "
          f"P_H={rows[-1]['P_H']:.3e} C_H={rows[-1]['C_H']:.3e} "
          f"Qc={rows[-1]['Qc']:.4f} max_drift="
          f"{max(r['mass_drift'] for r in rows):.2e}")


if __name__ == "__main__":
    main()
