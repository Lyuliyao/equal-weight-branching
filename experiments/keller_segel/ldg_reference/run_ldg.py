"""
Run the direct LDG reference (Li-Shu-Yang) on the Keller-Segel blow-up benchmark
(their Example 5.2) and compute the numerical-blow-up-time indicator (their (5.2)).
================================================================================

System: u_t - div(grad u - u grad v)=0, v_t = Delta v + u - v, chi=1, Neumann on
Omega=[-1/2,1/2]^2; IC u0=840 exp(-84 r^2), v0=420 exp(-42 r^2) (mass M_u=10pi).
Reporting times 6e-5, 1.2e-4, 2e-4 (their Figs 5.1-5.2); reference numerical
blow-up time ~1.21e-4.

Adaptive explicit time step (the concentrating core makes the chemotaxis speed
alpha=max|grad v| grow): dt = min(c_diff dx^2, c_conv dx/alpha), recomputed each
step; SSP-RK3 + Zhang-Shu positivity limiter (verified second-order on the heat
equation; chemotaxis consistency third order; u-mass exactly 10pi).

Outputs S_curves.csv (t, S_L2=||u||_L2, peak, umin, mass_u, mass_v) + snapshots.
Pair two resolutions (N)->(2N) and form tb(N)=inf{t: S(2N,t)>=1.05 S(N,t)} (5.2).

Usage:
  python run_ldg.py --N 160 --T 2e-4 --report_times 6e-5 1.2e-4 2e-4 --out_dir results/N160
"""
import os
import sys
import json
import csv
import time
import argparse

import numpy as np

from ldg_solver import (LDGMesh, LDGSolver, project_ic, field_L2, total_mass,
                        field_peak, field_min)

# core-collapse-time diagnostic: method-independent mass-quantile radii R_q(t).
_CCT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core_collapse_time")
if _CCT not in sys.path:
    sys.path.insert(0, _CCT)
from core_radii import ldg_core_radii  # noqa: E402


def u0_func(x, y):
    return 840.0 * np.exp(-84.0 * (x * x + y * y))


def v0_func(x, y):
    return 420.0 * np.exp(-42.0 * (x * x + y * y))


def run(N, T, report_times, out_dir, c_diff=0.05, c_conv=0.2, half=0.5,
        n_save=300, max_steps=2_000_000, verbose=True,
        core_radii=False, qs=(0.05, 0.1, 0.2, 0.3, 0.5, 0.8), quad_order=3):
    os.makedirs(os.path.join(out_dir, "snapshots"), exist_ok=True)
    m = LDGMesh(-half, half, -half, half, N, N)
    sol = LDGSolver(m, chi=1.0, positivity=True)
    U = project_ic(m, u0_func)
    U = sol.limit_positivity(U)
    V = project_ic(m, v0_func)
    M0 = total_mass(U, m)
    save_t = sorted(set(np.linspace(0, T, n_save).tolist() + list(report_times)))
    save_t = [t for t in save_t if t <= T + 1e-15]

    rows = []
    radii_rows = []
    snaps = {}
    qs = list(qs)

    def record(t):
        sL2 = field_L2(U, m); pk = field_peak(U, m); um = field_min(U, m)
        rows.append(dict(t=t, S_L2=sL2, peak=pk, umin=um,
                         mass_u=total_mass(U, m), mass_v=total_mass(V, m)))
        if core_radii:
            cr = ldg_core_radii(U, m, qs, quad_order=quad_order)  # raw + clip
            row = dict(t=t, N=N, S_L2=sL2, peak=pk, u_min=um,
                       M_raw=cr["raw"]["M"], M_clip=cr["clip"]["M"],
                       xc_x_raw=cr["raw"]["cx"], xc_y_raw=cr["raw"]["cy"],
                       xc_x_clip=cr["clip"]["cx"], xc_y_clip=cr["clip"]["cy"])
            for q in qs:
                row[f"R_{q}_raw"] = cr["raw"]["R"][float(q)]
                row[f"R_{q}_clip"] = cr["clip"]["R"][float(q)]
            radii_rows.append(row)

    record(0.0)
    t = 0.0
    si = 1
    step = 0
    blew = False
    t0 = time.time()
    report_set = list(report_times)
    while t < T - 1e-15 and step < max_steps:
        alpha = sol.max_alpha(V)
        dt = min(c_diff * m.dx ** 2, c_conv * m.dx / (alpha + 1e-30))
        if si < len(save_t):
            dt = min(dt, save_t[si] - t)
        if dt <= 0:
            dt = min(c_diff * m.dx ** 2, c_conv * m.dx / (alpha + 1e-30))
        Un, Vn, _ = sol.step(U, V, dt)
        if not (np.all(np.isfinite(Un)) and np.all(np.isfinite(Vn))):
            blew = True
            break
        U, V = Un, Vn
        t += dt
        step += 1
        if si < len(save_t) and t >= save_t[si] - 1e-15:
            record(t)
            si += 1
        for rt in report_set:
            if rt not in snaps and abs(t - rt) < 1e-12:
                snaps[rt] = (U[..., 0].copy(), V[..., 0].copy())
        if verbose and step % 5000 == 0:
            print(f"  N={N} step {step} t={t:.3e} dt={dt:.2e} peak={field_peak(U,m):.2e} "
                  f"S={field_L2(U,m):.1f} umin={field_min(U,m):.2e} "
                  f"massdrift={abs(total_mass(U,m)-M0)/M0:.2e}", flush=True)
    for rt in report_set:
        if rt not in snaps and t >= rt - 1e-12:
            snaps[rt] = (U[..., 0].copy(), V[..., 0].copy())

    rt_s = time.time() - t0
    cols = ["t", "S_L2", "peak", "umin", "mass_u", "mass_v"]
    with open(os.path.join(out_dir, "S_curves.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N"] + cols)
        w.writeheader()
        for r in rows:
            w.writerow(dict(N=N, **{c: r[c] for c in cols}))
    if core_radii and radii_rows:
        rcols = (["t", "N", "M_raw", "M_clip", "xc_x_raw", "xc_y_raw",
                  "xc_x_clip", "xc_y_clip"]
                 + [f"R_{q}_raw" for q in qs] + [f"R_{q}_clip" for q in qs]
                 + ["S_L2", "peak", "u_min"])
        with open(os.path.join(out_dir, f"ldg_core_radii_N{N}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rcols)
            w.writeheader()
            for r in radii_rows:
                w.writerow({c: r.get(c) for c in rcols})
        print(f"[N={N}] wrote ldg_core_radii_N{N}.csv ({len(radii_rows)} rows, "
              f"quad_order={quad_order})")
    snap_arr = {}
    for rt, (uu, vv) in snaps.items():
        snap_arr[f"u_{rt:.2e}"] = uu.astype(np.float32)
        snap_arr[f"v_{rt:.2e}"] = vv.astype(np.float32)
    np.savez_compressed(os.path.join(out_dir, "snapshots.npz"),
                        xc=m.xc, yc=m.yc, N=N, report_times=np.array(report_set), **snap_arr)
    cfg = dict(N=N, T=T, report_times=list(report_times), half=half, chi=1.0,
               bc="Neumann", scheme="P1 LDG (Li-Shu-Yang), alternating fluxes, LF "
               "chemotaxis, Zhang-Shu positivity limiter, SSP-RK3, adaptive dt",
               c_diff=c_diff, c_conv=c_conv, M_u0=M0, final_t=t, steps=step,
               blew_up=blew, umin_global=min(r["umin"] for r in rows),
               max_mass_drift=max(abs(r["mass_u"] - M0) / M0 for r in rows),
               runtime_s=round(rt_s, 1))
    with open(os.path.join(out_dir, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[N={N}] steps={step} final_t={t:.3e} blew_up={blew} "
          f"umin={cfg['umin_global']:.2e} massdrift={cfg['max_mass_drift']:.2e} "
          f"runtime={rt_s:.1f}s")
    for rt in report_set:
        rr = [r for r in rows if abs(r["t"] - rt) < 1e-12]
        if rr:
            print(f"    t={rt:.2e}: S_L2={rr[0]['S_L2']:.2f} peak={rr[0]['peak']:.1f} "
                  f"umin={rr[0]['umin']:.2e}")
    return cfg, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=160)
    ap.add_argument("--T", type=float, default=2.0e-4)
    ap.add_argument("--report_times", type=float, nargs="+", default=[6e-5, 1.2e-4, 2e-4])
    ap.add_argument("--out_dir", default="results/N160")
    ap.add_argument("--c_diff", type=float, default=0.05)
    ap.add_argument("--c_conv", type=float, default=0.2)
    ap.add_argument("--core_radii", action="store_true",
                    help="also write ldg_core_radii_N<N>.csv (mass-quantile radii "
                         "R_q(t), raw+clip) for the core-collapse-time diagnostic")
    ap.add_argument("--quad_order", type=int, default=3,
                    help="Gauss quadrature order per cell for R_q sampling")
    ap.add_argument("--qs", type=float, nargs="+",
                    default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.8])
    ap.add_argument("--n_save", type=int, default=300,
                    help="number of dense diagnostic save times over [0,T]")
    args = ap.parse_args()
    print(f"=== LDG Example 5.2 (blow-up): N={args.N} T={args.T:.2e} ===")
    run(args.N, args.T, args.report_times, args.out_dir,
        c_diff=args.c_diff, c_conv=args.c_conv, n_save=args.n_save,
        core_radii=args.core_radii, qs=args.qs, quad_order=args.quad_order)


if __name__ == "__main__":
    main()
