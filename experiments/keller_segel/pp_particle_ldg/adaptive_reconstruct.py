"""
Particle-adaptive reconstruction of the pp-KS cell field (CLAUDE.md solver-hybrid
plan §3 / blocker TODO §2 -- POST-PROCESSING audit).
================================================================================

Reads SAVED raw particle clouds from the fully parabolic--parabolic particle run
(ldg_comparison/simulation.py --save_cloud_snapshots) and reconstructs the cell
density u at the LDG reporting times with two families:

  * global Fourier  P_{Kg} mu        (the current solver output, bandwidth-sensitive)
  * particle-adaptive hybrid          P_{Kg} mu + chi_W [T_W(mu|_Wpad) - T_W(P_Kg mu)]
        with the local window W=B(x_c, alpha R_0.8) detected from the cloud, and
        T_W a local spectral window (Kl) or a local Gaussian blob.  The signed
        residual avoids double-counting the global low modes (verified reuse of
        experiments/resolution_hybrid/reconstructors.py).

For each it recomputes S_L2=||u||_L2, peak=||u||_inf, the core-local S_core, the
field minimum (negativity of the signed hybrid), and a mass check int u dx vs M.
The reconstruction-free radii R_0.5, R_0.8 come straight from the particles.

This is a POST-PROCESSING audit (no solver rerun, no dynamics change).  It asks
the blocker-TODO acceptance question: does adaptive reconstruction reduce the
global-bandwidth sensitivity of the reconstructed core, relative to raising the
global K?  The deterministic FVM baseline (n=512) is loaded only as an internal
sanity anchor -- it is NOT presented as the LDG reference.

Usage:
  python adaptive_reconstruct.py --clouds_dir <run>/base --tag base \
      --fvm_baseline <baseline_run> --out_dir <out>
"""
import os
import sys
import csv
import json
import glob
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RH = os.path.abspath(os.path.join(HERE, "..", "..", "resolution_hybrid"))
sys.path.insert(0, RH)

import reconstructors as rec                         # noqa: E402
from detect_windows import detect_core_window        # noqa: E402

REPORTS = [6e-5, 1.2e-4, 2.0e-4]


def grid(box, n):
    g = np.linspace(box[0][0], box[0][1], n)
    gy = np.linspace(box[1][0], box[1][1], n)
    XX, YY = np.meshgrid(g, gy)
    return XX, YY


def core_L2(u, XX, YY, center, radius):
    dx = XX[0, 1] - XX[0, 0]
    dy = YY[1, 0] - YY[0, 0]
    r = np.sqrt((XX - center[0]) ** 2 + (YY - center[1]) ** 2)
    m = r <= radius
    return float(np.sqrt(np.sum(u[m] ** 2) * abs(dx * dy)))


def diagnostics(u, XX, YY, center, R08, cell):
    return dict(S_L2=rec.field_L2(u, XX, YY), peak=rec.field_peak(u),
                S_core=core_L2(u, XX, YY, center, 3.0 * R08),
                umin=rec.field_min(u),
                mass=float(np.sum(u) * cell))


def load_cloud(path):
    d = np.load(path)
    return dict(X=np.asarray(d["X_u"], float), mass=float(d["mass_u_total"]),
                mpp=float(d["mass_per_particle_u"]), rt=float(d["report_time"]),
                N=int(d["N_u"]))


def fvm_anchor(baseline_dir):
    """FVM n=512 peak/S_L2 at report times (deterministic sanity anchor, NOT LDG)."""
    if not baseline_dir:
        return {}
    p = os.path.join(baseline_dir, "n512", "S_curves.csv")
    if not os.path.exists(p):
        return {}
    rows = list(csv.DictReader(open(p)))
    t = np.array([float(r["t"]) for r in rows])
    out = {}
    for rt in REPORTS:
        i = int(np.argmin(np.abs(t - rt)))
        out[rt] = dict(S_L2=float(rows[i]["S_L2"]), peak=float(rows[i]["peak"]),
                       R08=float(rows[i]["R_0_8"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clouds_dir", required=True, help="<run>/base or <run>/refined")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fvm_baseline", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--box_half", type=float, default=0.5)
    ap.add_argument("--ngrid", type=int, default=1025)
    ap.add_argument("--Kg_list", type=int, nargs="+", default=[5, 8, 16, 32])
    ap.add_argument("--Kg_hybrid", type=int, default=8)
    ap.add_argument("--Kl_list", type=int, nargs="+", default=[24, 32, 40])
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--alpha_pad", type=float, default=4.5)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, "plot_data"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)

    box = [[-args.box_half, args.box_half], [-args.box_half, args.box_half]]
    XX, YY = grid(box, args.ngrid)
    cell = (XX[0, 1] - XX[0, 0]) * (YY[1, 0] - YY[0, 0])
    anchor = fvm_anchor(args.fvm_baseline)

    rows = []
    for rt in REPORTS:
        matches = glob.glob(os.path.join(args.clouds_dir, "snapshots",
                                         f"snap_u_t{rt:.4e}_*.npz"))
        if not matches:
            print(f"[warn] no cloud snapshot for t={rt:.2e} in {args.clouds_dir}")
            continue
        c = load_cloud(matches[0])
        X, M, mpp = c["X"], c["mass"], c["mpp"]
        win = detect_core_window(X, None, mpp, c_window=args.alpha)
        center = np.array(win["center"]); R05 = win["R05"]; R08 = win["R08"]
        pad_half = args.alpha_pad * R08
        win_half = args.alpha * R08

        # ---- global Fourier at several Kg (bandwidth sensitivity) ----
        for Kg in args.Kg_list:
            u, _ = rec.global_recon(X, None, box, Kg, M, XX, YY)
            dd = diagnostics(u, XX, YY, center, R08, cell)
            rows.append(dict(tag=args.tag, t=rt, method="global_fourier",
                             recon=f"P_K", Kg=Kg, Kl="", local_type="",
                             R05=R05, R08=R08, N=c["N"], **dd))
        # ---- hybrid: low global Kg_hybrid + local spectral window Kl ----
        for Kl in args.Kl_list:
            u, u_lo, u_res, info = rec.hybrid_spectrum_window(
                X, None, box, M, mpp, center, win_half, args.Kg_hybrid, Kl,
                XX, YY, pad=args.alpha_pad / args.alpha)
            dd = diagnostics(u, XX, YY, center, R08, cell)
            rows.append(dict(tag=args.tag, t=rt, method="hybrid_spectrum",
                             recon=f"Kg{args.Kg_hybrid}+Kl{Kl}", Kg=args.Kg_hybrid,
                             Kl=Kl, local_type="spectrum", R05=R05, R08=R08,
                             N=c["N"], n_local=info["n_local"], **dd))
        # ---- hybrid: low global + local Gaussian blob (h from particle spacing) ----
        N08 = max(int(win["local_count"]), 1)
        h = 0.7 * R08 / np.sqrt(N08)
        u, u_lo, u_res, info = rec.hybrid_blob_residual(
            X, None, box, M, mpp, center, win_half, h, args.Kg_hybrid, XX, YY,
            pad=args.alpha_pad / args.alpha)
        dd = diagnostics(u, XX, YY, center, R08, cell)
        rows.append(dict(tag=args.tag, t=rt, method="hybrid_blob",
                         recon=f"Kg{args.Kg_hybrid}+blob", Kg=args.Kg_hybrid,
                         Kl="", local_type="blob", R05=R05, R08=R08,
                         N=c["N"], h=h, **dd))

    # ---- write CSV ----
    cols = ["tag", "t", "method", "recon", "Kg", "Kl", "local_type",
            "S_L2", "peak", "S_core", "umin", "mass", "R05", "R08", "N"]
    with open(os.path.join(args.out_dir, f"adaptive_S_curves_{args.tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- console acceptance summary ----
    print(f"\n=== particle-adaptive reconstruction audit [{args.tag}] ===")
    print("mass check (int u dx vs M=10pi=31.416): "
          f"{min(r['mass'] for r in rows):.3f}..{max(r['mass'] for r in rows):.3f}")
    for rt in REPORTS:
        sub = [r for r in rows if abs(r["t"] - rt) < 1e-12]
        if not sub:
            continue
        gf = {r["Kg"]: r for r in sub if r["method"] == "global_fourier"}
        hy = [r for r in sub if r["method"] == "hybrid_spectrum"]
        bl = [r for r in sub if r["method"] == "hybrid_blob"]
        a = anchor.get(rt, {})
        print(f"\n t={rt:.2e}  (FVM n=512 anchor: S_L2={a.get('S_L2', float('nan')):.0f} "
              f"peak={a.get('peak', float('nan')):.0f})  R0.8={sub[0]['R08']:.4f}")
        print("   global Fourier  peak / S_L2 vs Kg:")
        for Kg in sorted(gf):
            print(f"     Kg={Kg:2d}: peak={gf[Kg]['peak']:.0f}  S_L2={gf[Kg]['S_L2']:.0f}  "
                  f"S_core={gf[Kg]['S_core']:.0f}")
        for r in hy:
            print(f"   hybrid Kg{r['Kg']}+Kl{r['Kl']}: peak={r['peak']:.0f}  "
                  f"S_L2={r['S_L2']:.0f}  S_core={r['S_core']:.0f}  umin={r['umin']:.1f}")
        for r in bl:
            print(f"   hybrid Kg{r['Kg']}+blob: peak={r['peak']:.0f}  "
                  f"S_L2={r['S_L2']:.0f}  S_core={r['S_core']:.0f}  umin={r['umin']:.1f}")

    # ---- plot_data ----
    np.savez(os.path.join(args.out_dir, "plot_data", f"adaptive_{args.tag}.npz"),
             rows=np.array([json_safe(r) for r in rows], dtype=object),
             report_times=np.array(REPORTS))
    with open(os.path.join(args.out_dir, "config_used.json"), "w") as f:
        json.dump(dict(vars(args), reports=REPORTS), f, indent=2)
    print(f"\nwrote adaptive_S_curves.csv + plot_data to {args.out_dir}")


def json_safe(r):
    return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in r.items()}


if __name__ == "__main__":
    main()
