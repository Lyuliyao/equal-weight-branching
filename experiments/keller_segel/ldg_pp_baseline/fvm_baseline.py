"""
Positivity-preserving finite-volume baseline for the fully parabolic-parabolic
Keller-Segel system (CLAUDE.md §7.1.3 / §7.2).
=============================================================================

System (Li-Shu-Yang LDG benchmark), on a square Omega = [-0.5, 0.5]^2:

    u_t = div( grad u - u grad v )            (cell density, conservative)
    v_t = Delta v + u - v                     (chemoattractant)

Initial data (super-critical, mass M_u = 10*pi > 8*pi):

    u0 = 840 * exp(-84 (x^2+y^2)),
    v0 = 420 * exp(-42 (x^2+y^2)).

Boundary condition: homogeneous Neumann (zero flux), matching the LDG study.
Because the Gaussian has standard deviation 1/sqrt(2*84) ~ 0.077, its value at
the domain boundary (|x|=0.5) is exp(-21) ~ 7e-10, so the Neumann/periodic/
whole-plane distinction is numerically negligible over the reported times.

Discretization:
  * cell-centered finite volume, n x n cells, dx = 1/n;
  * u-flux  F = -grad u + u grad v  (so u_t = -div F):
      - diffusive face flux  -(U_{i+1}-U_i)/dx                (central),
      - advective face flux  a * upwind(U),  a = (V_{i+1}-V_i)/dx (first-order
        upwind), which is positivity-preserving under the CFL condition;
  * v: central 5-point Laplacian + reaction (u - v);
  * explicit Euler with an adaptive step dt = cfl * min(diffusion, advection,
    reaction) limits.  Zero boundary flux => exact discrete mass conservation
    for u.

This is the deterministic reference baseline.  It is NOT the particle method.
Two resolutions n and 2n are run to form the LDG-style resolution-gap time
    t_b(n; theta) = inf{ t : S_{2n}(t) >= theta S_n(t) },   S_n(t) = ||u_n(t)||_L2,
computed by tb_from_pair.py.

Usage:
    python fvm_baseline.py --n 256 --T 2e-4 --cfl 0.25 \
        --report_times 6e-5 1.2e-4 2.0e-4 --out_dir results/n256
    python fvm_baseline.py --smoke
"""
import os
import sys
import json
import csv
import time
import argparse
import subprocess

import numpy as np

L = 1.0          # domain side length: Omega = [-0.5, 0.5]^2
HALF = 0.5


def initial_fields(n):
    """Cell-centered LDG initial data on [-0.5,0.5]^2."""
    dx = L / n
    xs = -HALF + (np.arange(n) + 0.5) * dx
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    r2 = X * X + Y * Y
    u0 = 840.0 * np.exp(-84.0 * r2)
    v0 = 420.0 * np.exp(-42.0 * r2)
    return xs, X, Y, u0, v0, dx


def neumann_pad(A):
    """Zero-gradient (Neumann) ghost layer via edge replication."""
    return np.pad(A, 1, mode="edge")


def laplacian(V, dx):
    Vp = neumann_pad(V)
    return (Vp[2:, 1:-1] + Vp[:-2, 1:-1] + Vp[1:-1, 2:] + Vp[1:-1, :-2]
            - 4.0 * V) / (dx * dx)


def u_rhs(U, V, dx):
    """du/dt = -div F, F = -grad u + u grad v, first-order upwind advection +
    central diffusion, zero-flux Neumann boundaries.  Returns dU/dt and the
    max advective speed (for the CFL limit)."""
    # ----- x-direction faces (between column i and i+1), interior faces only
    # advective velocity a = dV/dx at the face
    ax = (V[:, 1:] - V[:, :-1]) / dx                      # (n, n-1)
    up_x = np.where(ax > 0.0, U[:, :-1], U[:, 1:])        # upwind u at face
    Fadv_x = ax * up_x
    Fdiff_x = -(U[:, 1:] - U[:, :-1]) / dx
    Fx = Fadv_x + Fdiff_x                                 # interior x-faces
    # divergence contribution in x: -(F_{i+1/2}-F_{i-1/2})/dx, zero flux at walls
    divx = np.zeros_like(U)
    divx[:, 1:-1] += -(Fx[:, 1:] - Fx[:, :-1]) / dx
    divx[:, 0] += -(Fx[:, 0]) / dx                       # left wall flux = 0
    divx[:, -1] += -(-Fx[:, -1]) / dx                    # right wall flux = 0

    # ----- y-direction faces (between row j and j+1)
    ay = (V[1:, :] - V[:-1, :]) / dx
    up_y = np.where(ay > 0.0, U[:-1, :], U[1:, :])
    Fadv_y = ay * up_y
    Fdiff_y = -(U[1:, :] - U[:-1, :]) / dx
    Fy = Fadv_y + Fdiff_y
    divy = np.zeros_like(U)
    divy[1:-1, :] += -(Fy[1:, :] - Fy[:-1, :]) / dx
    divy[0, :] += -(Fy[0, :]) / dx
    divy[-1, :] += -(-Fy[-1, :]) / dx

    max_a = max(float(np.max(np.abs(ax))) if ax.size else 0.0,
                float(np.max(np.abs(ay))) if ay.size else 0.0)
    return divx + divy, max_a


def l2_norm(U, dx):
    return float(np.sqrt(np.sum(U * U) * dx * dx))


def total_mass(U, dx):
    return float(np.sum(U) * dx * dx)


def core_diagnostics(U, X, Y, dx, alpha=3.0):
    """Reconstruction-free core diagnostics directly from the grid density:
    centroid x_c, half-/80%-mass radii R_0.5,R_0.8, and the core-local L2 norm
    S_core = ||u||_{L2(W)}, W = B(x_c, alpha*R_0.8).  All from cell masses, no
    Fourier reconstruction."""
    w = U * dx * dx                      # cell masses (U >= 0)
    M = float(w.sum())
    cx = float(np.sum(X * w) / M)
    cy = float(np.sum(Y * w) / M)
    rr = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    order = np.argsort(rr, axis=None)
    r_sorted = rr.ravel()[order]
    cum = np.cumsum(w.ravel()[order]) / M
    R05 = float(r_sorted[np.searchsorted(cum, 0.5)])
    R08 = float(r_sorted[np.searchsorted(cum, 0.8)])
    in_core = rr <= alpha * R08
    S_core = float(np.sqrt(np.sum((U[in_core]) ** 2) * dx * dx))
    return cx, cy, R05, R08, S_core


def run(n, T, cfl, report_times, out_dir, dt_max=None, max_steps=4_000_000,
        n_save=400, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    xs, X, Y, U, V, dx = initial_fields(n)
    save_times = sorted(set(np.linspace(0.0, T, n_save).tolist()
                            + list(report_times)))
    save_times = [t for t in save_times if t <= T + 1e-15]
    diff_limit = dx * dx / 4.0          # 2D explicit diffusion (D=1)
    M_u0 = total_mass(U, dx)

    rows = []
    snaps = {}
    report_set = list(report_times)
    t = 0.0
    si = 0
    step = 0
    blew_up = False
    t0 = time.time()

    def record(t):
        cx, cy, R05, R08, S_core = core_diagnostics(U, X, Y, dx)
        rows.append(dict(t=t, S_L2=l2_norm(U, dx), peak=float(U.max()),
                         umin=float(U.min()), mass_u=total_mass(U, dx),
                         peak_v=float(V.max()), mass_v=total_mass(V, dx),
                         xc=cx, yc=cy, R_0_5=R05, R_0_8=R08, S_core=S_core))

    record(0.0)
    next_save_idx = 1
    while t < T - 1e-15 and step < max_steps:
        du, max_a = u_rhs(U, V, dx)
        adv_limit = dx / max_a if max_a > 0 else np.inf
        dt = cfl * min(diff_limit, adv_limit)
        dt = min(dt, 1.0)                      # reaction (-v) limit is O(1)
        if dt_max is not None:
            dt = min(dt, dt_max)
        # land exactly on the next save/report time
        if next_save_idx < len(save_times):
            dt = min(dt, save_times[next_save_idx] - t)
        if dt <= 0:
            dt = cfl * min(diff_limit, adv_limit)
        Un = U + dt * du
        Vn = V + dt * (laplacian(V, dx) + U - V)
        if not (np.all(np.isfinite(Un)) and np.all(np.isfinite(Vn))):
            blew_up = True
            break
        U, V = Un, Vn
        t += dt
        step += 1
        if next_save_idx < len(save_times) and t >= save_times[next_save_idx] - 1e-15:
            record(t)
            next_save_idx += 1
        for rt in report_set:
            if rt not in snaps and abs(t - rt) < 1e-12:
                snaps[rt] = (U.copy(), V.copy())
        if verbose and step % 2000 == 0:
            print(f"  n={n} step {step} t={t:.3e} peak_u={U.max():.3e} "
                  f"umin={U.min():.3e} S={l2_norm(U,dx):.2f} "
                  f"mass_drift={abs(total_mass(U,dx)-M_u0)/M_u0:.2e}", flush=True)

    # ensure report snapshots captured even if exact-time landing missed
    for rt in report_set:
        if rt not in snaps and t >= rt - 1e-12:
            snaps[rt] = (U.copy(), V.copy())

    runtime = time.time() - t0
    umin_global = min(r["umin"] for r in rows)
    mass_drift = max(abs(r["mass_u"] - M_u0) / M_u0 for r in rows)

    # ---- write S-curve CSV ----
    cols = ["t", "S_L2", "S_core", "peak", "umin", "mass_u", "peak_v", "mass_v",
            "xc", "yc", "R_0_5", "R_0_8"]
    with open(os.path.join(out_dir, "S_curves.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n"] + cols)
        w.writeheader()
        for r in rows:
            w.writerow(dict(n=n, **{c: r[c] for c in cols}))

    # ---- snapshots ----
    snap_arr = {}
    for rt, (us, vs) in snaps.items():
        key = f"{rt:.2e}"
        snap_arr[f"u_{key}"] = us.astype(np.float32)
        snap_arr[f"v_{key}"] = vs.astype(np.float32)
    np.savez_compressed(os.path.join(out_dir, "snapshots.npz"),
                        xs=xs, report_times=np.array(report_set), n=n, **snap_arr)

    cfg = dict(n=n, T=T, cfl=cfl, report_times=list(report_times), dx=dx,
               dt_max=dt_max, scheme="positivity-preserving upwind FVM, "
               "central diffusion, explicit Euler, Neumann",
               domain=[-HALF, HALF], M_u0=M_u0, blew_up=blew_up,
               final_t=t, steps=step, runtime_s=round(runtime, 2),
               umin_global=umin_global, max_mass_drift=mass_drift,
               numpy=np.__version__, python=sys.version.split()[0])
    with open(os.path.join(out_dir, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[n={n}] done: steps={step} final_t={t:.3e} blew_up={blew_up} "
          f"umin={umin_global:.3e} mass_drift={mass_drift:.2e} "
          f"runtime={runtime:.1f}s", flush=True)
    # report-time concentration summary
    for rt in report_set:
        rr = [r for r in rows if abs(r["t"] - rt) < 1e-12]
        if rr:
            print(f"    t={rt:.2e}: S_L2={rr[0]['S_L2']:.2f} "
                  f"peak={rr[0]['peak']:.2f} umin={rr[0]['umin']:.3e}")
    return cfg, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--T", type=float, default=2.0e-4)
    ap.add_argument("--cfl", type=float, default=0.25)
    ap.add_argument("--dt_max", type=float, default=None)
    ap.add_argument("--report_times", type=float, nargs="+",
                    default=[6e-5, 1.2e-4, 2.0e-4])
    ap.add_argument("--out_dir", default="results/n256")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n = 64
        args.T = 6e-5
        args.out_dir = "results/smoke_n64"
    print(f"=== FVM pp-KS baseline: n={args.n} T={args.T:.2e} cfl={args.cfl} ===")
    run(args.n, args.T, args.cfl, args.report_times, args.out_dir,
        dt_max=args.dt_max)


if __name__ == "__main__":
    main()
