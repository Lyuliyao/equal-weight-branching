"""core_window_demo.py -- Demo 1: KS-core reconstruction from a particle cloud.
================================================================================

Reconstruct a CONCENTRATED Keller-Segel core from a particle cloud at several
bandwidths and compare (CLAUDE.md Sec. 6, Demo 1):

    A. global Fourier  K_g            (low global spectrum)
    B. global Fourier  K_full         (expensive high global spectrum baseline)
    C. hybrid          K_g + local spectral window K_l
    D. hybrid          K_g + local residual blob   (optional, --blob)

By default the cloud is the canonical concentrated KS core
    u(x) = (mass*a/pi) exp(-a |x|^2),    each coord ~ N(0, 1/(2a)),
for which the field has EXACT diagnostics, so the reconstruction error of each
scheme is known exactly (the strongest possible demonstration that the missing
resolution is LOCAL, not global):
    peak  = mass*a/pi
    L2    = mass*sqrt(a/(2pi))
    R_q   = sqrt(-ln(1-q)/a)                 (reconstruction-free)
    M(<R) = mass*(1 - exp(-a R^2))           (reconstruction-free)

Pass `--cloud <file.npz>` (keys X, w, mass_per_particle, box) to use a REAL saved
particle cloud instead (e.g. a Keller-Segel snapshot); then the "exact" columns
are omitted and a high-K_full reconstruction is the reference.

The local window is detected FROM PARTICLES (detect_windows.detect_core_window),
not from the final image.  Reports peak/L2 (bandwidth-SENSITIVE), R_0.5/R_0.8 and
core mass (reconstruction-FREE), residual energy fraction, mode/quadrature cost,
and reconstruction wall time.

Run:  python core_window_demo.py --out_dir results/core_demo
      python core_window_demo.py --a 84 --N 400000 --Kg 8 --Kfull 40 --Kl 40 --blob
"""
import os
import sys
import csv
import json
import time
import argparse
import datetime
import platform
import subprocess

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import reconstructors as rc
import detect_windows as dw


def git_hash(short=False):
    try:
        a = ["git", "-C", _HERE, "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        return subprocess.check_output(a, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def exact_gaussian_diagnostics(a, mass):
    return dict(
        peak=mass * a / np.pi,
        L2=mass * np.sqrt(a / (2.0 * np.pi)),
        R05=np.sqrt(np.log(2.0) / a),
        R08=np.sqrt(-np.log(0.2) / a),
        mass_in_R08=mass * (1.0 - 0.2),
    )


def build_cloud(args):
    if args.cloud:
        d = np.load(args.cloud, allow_pickle=True)
        X = np.asarray(d["X"]); w = np.asarray(d["w"]) if "w" in d.files else None
        mpp = float(d["mass_per_particle"]); box = d["box"].tolist()
        mass = mpp * (X.shape[0] if w is None else float(np.sum(w)))
        return X, w, mpp, box, mass, None
    rng = np.random.default_rng(args.seed)
    a = args.a; mass = args.mass
    std = 1.0 / np.sqrt(2.0 * a)
    X = rng.normal(0.0, std, size=(args.N, 2))
    w = None
    mpp = mass / args.N
    box = [[-args.box, args.box], [-args.box, args.box]]
    return X, w, mpp, box, mass, exact_gaussian_diagnostics(a, mass)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", type=str, default=None,
                    help="optional saved cloud npz (keys X,w,mass_per_particle,box)")
    ap.add_argument("--a", type=float, default=84.0, help="KS-core Gaussian width")
    ap.add_argument("--mass", type=float, default=10.0 * np.pi)
    ap.add_argument("--N", type=int, default=400000)
    ap.add_argument("--box", type=float, default=1.0, help="half-box for the grid")
    ap.add_argument("--grid", type=int, default=221)
    ap.add_argument("--Kg", type=int, default=8)
    ap.add_argument("--Kfull", type=int, default=40)
    ap.add_argument("--Kl", type=int, default=40)
    ap.add_argument("--c_window", type=float, default=3.0)
    ap.add_argument("--B_target", type=int, default=3000,
                    help="target retained residual-particle count for the HT sketch")
    ap.add_argument("--blob", action="store_true", help="also run Option A (blob)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="results/core_demo")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.N, args.grid, args.out_dir = 80000, 121, "results/core_demo_smoke"

    os.makedirs(args.out_dir, exist_ok=True)
    X, w, mpp, box, mass, exact = build_cloud(args)
    g = np.linspace(-args.box, args.box, args.grid)
    XX, YY = np.meshgrid(g, g, indexing="xy")

    # window detected FROM PARTICLES
    win = dw.detect_core_window(X, w, mpp, c_window=args.c_window)
    center = win["center"]; half = win["half"]
    # reconstruction-free core diagnostics
    R05 = win["R05"]; R08 = win["R08"]
    core_mass_R08 = win["local_mass_R08"]
    local_count = win["local_count"]

    recons = {}

    def timed(fn):
        t0 = time.time(); out = fn(); return out, time.time() - t0

    (u_lo, _), t_lo = timed(lambda: rc.global_recon(X, w, box, args.Kg, mass, XX, YY))
    recons["A_global_low"] = dict(u=u_lo, gmodes=args.Kg ** 2, lmodes=0, qnodes=0, wall=t_lo)
    (u_hi, _), t_hi = timed(lambda: rc.global_recon(X, w, box, args.Kfull, mass, XX, YY))
    recons["B_global_high"] = dict(u=u_hi, gmodes=args.Kfull ** 2, lmodes=0, qnodes=0, wall=t_hi)
    (hyb, t_hyb) = timed(lambda: rc.hybrid_spectrum_window(
        X, w, box, mass, mpp, center, half, args.Kg, args.Kl, XX, YY))
    u_hyb, ulo2, ures, info = hyb
    recons["C_hybrid_window"] = dict(
        u=u_hyb, u_res=ures, gmodes=args.Kg ** 2, lmodes=args.Kl ** 2, qnodes=0,
        wall=t_hyb, info=info)
    dx = g[1] - g[0]
    h_blob = max(1.5 * dx, 0.3 * R05)
    if args.blob:
        (blobout, t_blob) = timed(lambda: rc.hybrid_blob_residual(
            X, w, box, mass, mpp, center, half, h=h_blob, Kg=args.Kg, XX=XX, YY=YY))
        u_blob, _, ures_b, info_b = blobout
        recons["D_hybrid_blob"] = dict(
            u=u_blob, u_res=ures_b, gmodes=args.Kg ** 2, lmodes=0,
            qnodes=info_b["n_local"], wall=t_blob, info=info_b)

    # ---- Option E: residual-particle sketch (HT + positive-excess) ----
    window = dw.Window(center, half, pad=1.5, taper_frac=0.85)
    rng_ht = np.random.default_rng(args.seed + 7)
    (htout, t_ht) = timed(lambda: rc.ht_residual_reconstruction(
        X, w, box, mass, mpp, window, Kg=args.Kg, h=h_blob, XX=XX, YY=YY,
        B_target=args.B_target, rng=rng_ht))
    u_ht, _, ures_ht, aux_ht = htout
    recons["E_ht_residual"] = dict(u=u_ht, u_res=ures_ht, gmodes=args.Kg ** 2, lmodes=0,
                                   qnodes=aux_ht["n_retained"], wall=t_ht, aux=aux_ht)
    rng_pos = np.random.default_rng(args.seed + 11)
    (posout, t_pos) = timed(lambda: rc.positive_residual_reconstruction(
        X, w, box, mass, mpp, window, Kg=args.Kg, h=h_blob, XX=XX, YY=YY, rng=rng_pos))
    u_pos, _, ures_pos, aux_pos = posout
    recons["E_positive_residual"] = dict(u=u_pos, u_res=ures_pos, gmodes=args.Kg ** 2,
                                         lmodes=0, qnodes=aux_pos["n_retained"],
                                         wall=t_pos, aux=aux_pos)
    # retained residual particles for the overlay figure
    res_idx_ht, res_w_ht, _ = rc.residual_particle_acceptance(
        X, w, lambda pts: mass * rc.eval_points(rc.fourier_coeffs(X, w, box, args.Kg), pts),
        None, window, mode="ht", B_target=args.B_target,
        rng=np.random.default_rng(args.seed + 7))

    # residual energy fraction in the window for hybrid C
    wmask = ((np.abs(XX - center[0]) <= half) & (np.abs(YY - center[1]) <= half))
    def L2win(u):
        dx = g[1] - g[0]
        return float(np.sqrt(np.sum((u * wmask) ** 2) * dx * dx))
    res_frac = L2win(ures) / max(L2win(u_lo), 1e-30)

    # ---- metrics CSV ----
    rows = []
    for key, R in recons.items():
        u = R["u"]
        row = dict(scheme=key, peak=rc.field_peak(u), L2=rc.field_L2(u, XX, YY),
                   field_min=rc.field_min(u), global_modes=R["gmodes"],
                   local_modes=R.get("lmodes", 0),
                   local_quadrature_nodes=R.get("qnodes", 0), wall_s=R["wall"])
        if exact:
            row["peak_relerr"] = abs(row["peak"] - exact["peak"]) / exact["peak"]
            row["L2_relerr"] = abs(row["L2"] - exact["L2"]) / exact["L2"]
        rows.append(row)
    cols = ["scheme", "peak", "L2", "field_min", "global_modes", "local_modes",
            "local_quadrature_nodes", "wall_s"] + (
            ["peak_relerr", "L2_relerr"] if exact else [])
    with open(os.path.join(args.out_dir, "core_demo_metrics.csv"), "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=cols); wtr.writeheader(); wtr.writerows(rows)

    # ---- reconstruction-free core diagnostics CSV ----
    with open(os.path.join(args.out_dir, "core_free_diagnostics.csv"), "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["quantity", "particle_value", "exact_value"])
        ex = exact or {}
        wtr.writerow(["R_0.5", R05, ex.get("R05", "")])
        wtr.writerow(["R_0.8", R08, ex.get("R08", "")])
        wtr.writerow(["core_mass_R08", core_mass_R08, ex.get("mass_in_R08", "")])
        wtr.writerow(["local_count_R08", local_count, ""])
        wtr.writerow(["residual_energy_frac_C", res_frac, ""])

    # ---- plot data ----
    pd = os.path.join(args.out_dir, "plot_data"); os.makedirs(pd, exist_ok=True)
    save = dict(XX=XX, YY=YY, center=np.asarray(center), half=half, box=np.array(box),
                R05=R05, R08=R08, res_frac=res_frac,
                u_global_low=recons["A_global_low"]["u"],
                u_global_high=recons["B_global_high"]["u"],
                u_hybrid=recons["C_hybrid_window"]["u"],
                u_res=recons["C_hybrid_window"]["u_res"])
    if "D_hybrid_blob" in recons:
        save["u_hybrid_blob"] = recons["D_hybrid_blob"]["u"]
    save["u_ht"] = recons["E_ht_residual"]["u"]
    save["u_positive"] = recons["E_positive_residual"]["u"]
    save["retained_particles"] = np.asarray(X)[res_idx_ht]   # overlay panel
    save["all_particles_sub"] = np.asarray(X)[np.random.default_rng(0).choice(
        X.shape[0], size=min(X.shape[0], 4000), replace=False)]
    if exact:
        save["exact_peak"] = exact["peak"]; save["exact_L2"] = exact["L2"]
    np.savez(os.path.join(pd, "figure_core_reconstruction.npz"), **save)

    # ---- residual-particle outputs (CLAUDE.md Sec. 6) ----
    rp_dir = os.path.join(args.out_dir, "residual_particles"); os.makedirs(rp_dir, exist_ok=True)
    np.savez(os.path.join(rp_dir, "window_0.npz"),
             retained_X=np.asarray(X)[res_idx_ht], retained_w=res_w_ht,
             center=np.asarray(center), half=half, B_target=args.B_target)
    with open(os.path.join(args.out_dir, "residual_acceptance.csv"), "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["window_id", "mode", "N_window", "B_target", "N_retained",
                      "mean_accept_rate", "mass_accept_rate", "min_accept_rate",
                      "max_accept_rate", "effective_HT_sample_size", "N_res_expected",
                      "positive_residual_mass", "negative_residual_mass",
                      "residual_mass_imbalance", "residual_energy_fraction"])
        a = aux_ht
        wtr.writerow([0, "ht", a["n_window"], a["B_target"], a["n_retained"],
                      a["mean_accept_rate"], a["mass_accept_rate"], a["min_accept_rate"],
                      a["max_accept_rate"], a["effective_HT_sample_size"],
                      a["expected_residual_count"], "", "", "", a["residual_energy_fraction"]])
        a = aux_pos
        wtr.writerow([0, "positive", a["n_window"], "", a["n_retained"],
                      a["mean_accept_rate"], a["mass_accept_rate"], a["min_accept_rate"],
                      a["max_accept_rate"], a.get("effective_HT_sample_size", ""),
                      a["expected_residual_count"], a["positive_residual_mass"],
                      a["negative_residual_mass"], a["residual_mass_imbalance"],
                      a["residual_energy_fraction"]])

    # ---- config + manifest ----
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(dict(
            experiment="resolution_hybrid/core_window_demo",
            git_commit=git_hash(False), git_commit_short=git_hash(True),
            command_line=" ".join([sys.executable] + sys.argv),
            python_version=platform.python_version(), numpy_version=np.__version__,
            datetime=datetime.datetime.now().isoformat(timespec="seconds"),
            out_dir=os.path.abspath(args.out_dir),
            window_detected_from="particles (detect_core_window)",
            cloud=(args.cloud or "synthetic concentrated KS-core Gaussian"),
        ), f, indent=2)

    # ---- console summary ----
    print("=== Demo 1: KS-core reconstruction ===")
    print(f"window (from particles): center={np.round(center,4)} half={half:.4f} "
          f"R05={R05:.4f} R08={R08:.4f} local_count={local_count:.0f}")
    if exact:
        print(f"exact: peak={exact['peak']:.1f} L2={exact['L2']:.3f} "
              f"R05={exact['R05']:.4f} R08={exact['R08']:.4f}")
    print(f"{'scheme':16s} {'peak':>9s} {'L2':>8s} {'min':>7s} {'modes':>7s} "
          f"{'wall_s':>7s}" + (f" {'peakErr':>8s} {'L2Err':>8s}" if exact else ""))
    for r in rows:
        line = (f"{r['scheme']:16s} {r['peak']:9.1f} {r['L2']:8.3f} {r['field_min']:7.2f} "
                f"{r['global_modes']:7d} {r['wall_s']:7.2f}")
        if exact:
            line += f" {r['peak_relerr']:8.3f} {r['L2_relerr']:8.3f}"
        print(line)
    print(f"hybrid-C residual energy fraction in window: {res_frac:.3f}")
    print(f"residual particles (HT): N_window={aux_ht['n_window']} "
          f"B_target={args.B_target} N_retained={aux_ht['n_retained']} "
          f"mean_accept={aux_ht['mean_accept_rate']:.3f} "
          f"mass_accept={aux_ht['mass_accept_rate']:.3f} "
          f"HT_eff={aux_ht['effective_HT_sample_size']:.0f}")
    print(f"residual particles (positive-excess): N_retained={aux_pos['n_retained']} "
          f"mass_accept={aux_pos['mass_accept_rate']:.3f} "
          f"residual_mass_imbalance={aux_pos['residual_mass_imbalance']:.3e}")
    print("NOTE: accept rate = reconstruction-enrichment rate, NOT a Metropolis "
          "acceptance and NOT a particle-dynamics resampling step.")
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
