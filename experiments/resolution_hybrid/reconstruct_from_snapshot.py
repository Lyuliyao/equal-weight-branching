"""reconstruct_from_snapshot.py -- Demo 2: multi-island local reconstruction.
================================================================================

Apply the hybrid reconstructions to a SAVED multi-island particle cloud (from
experiments/branch_vs_weighted/multi_island.py, clouds/cloud_<method>_seed0.npz)
and ask, for each growth island B_m, how different reconstructions estimate the
island mass (CLAUDE.md Sec. 6, Demo 2):

    A. particle counting / weighted particle mass          (reconstruction-FREE)
    B. global Fourier reconstruction on the full domain    (bandwidth-limited)
       B_high: global Fourier at a high bandwidth
    C. local spectral window reconstruction on B_m + pad
    E. residual-particle (HT) sketch with accept-rate diagnostics

The reference island masses M_m^ref come from the multi-island run's
island_masses.csv (deterministic spectral reference).  We report how the
reconstruction changes E_m = |M_m^method - M_m^ref| / M_m^ref, especially for the
WEAKER islands, and we also DETECT the islands straight from the particle cloud
(detect_islands_from_particles) to show the windows can be found WITHOUT the
reference solution.

Run:
    python reconstruct_from_snapshot.py \
        --cloud ../branch_vs_weighted/results/multi_island_prod/clouds/cloud_minvar_branch_seed0.npz \
        --ref_island_masses ../branch_vs_weighted/results/multi_island_prod/island_masses.csv \
        --out_dir results/island_demo
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


def load_ref_masses(path, method):
    """M_ref[m] and amplitude[m] for the given method (M_ref is method-independent)."""
    rows = list(csv.DictReader(open(path)))
    by_m = {}
    for r in rows:
        if r["method"] != method:
            continue
        m = int(r["m"])
        by_m[m] = (float(r["M_ref"]), float(r["amplitude"]),
                   float(r["cx"]), float(r["cy"]))
    M = len(by_m)
    Mref = np.array([by_m[m][0] for m in range(M)])
    amp = np.array([by_m[m][1] for m in range(M)])
    centers = np.array([[by_m[m][2], by_m[m][3]] for m in range(M)])
    return Mref, amp, centers


def disk_mass(field, XX, YY, center, R, box):
    """Integral of `field` over the periodic disk B(center,R) on the grid."""
    dx = XX[0, 1] - XX[0, 0]; dy = YY[1, 0] - YY[0, 0]
    L = box[0][1] - box[0][0]
    ddx = XX - center[0]; ddy = YY - center[1]
    ddx -= L * np.round(ddx / L); ddy -= L * np.round(ddy / L)
    mask = (ddx ** 2 + ddy ** 2) <= R ** 2
    return float(np.sum(field[mask]) * dx * dy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--ref_island_masses", required=True)
    ap.add_argument("--method_in_ref", default=None,
                    help="method name in island_masses.csv (default: inferred from cloud)")
    ap.add_argument("--Kg", type=int, default=8)
    ap.add_argument("--Kfull", type=int, default=48)
    ap.add_argument("--Kl", type=int, default=40)
    ap.add_argument("--grid", type=int, default=288)
    ap.add_argument("--c_window", type=float, default=3.0)
    ap.add_argument("--B_target", type=int, default=2000)
    ap.add_argument("--out_dir", default="results/island_demo")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    d = np.load(args.cloud, allow_pickle=True)
    X = np.asarray(d["X"]); w = np.asarray(d["w"]) if "w" in d.files else None
    if w is not None and np.allclose(w, 1.0):
        w = None
    mpp = float(d["mass_per_particle"]); box = d["box"].tolist()
    sigma = float(d["sigma"]); centers_saved = np.asarray(d["centers"])
    method = (args.method_in_ref or os.path.basename(args.cloud)
              .replace("cloud_", "").replace("_seed0.npz", ""))
    Mref, amp, centers = load_ref_masses(args.ref_island_masses, method)
    M = centers.shape[0]
    R_B = sigma * np.sqrt(2.0 * np.log(2.0))            # half-height disk radius
    mass_total = mpp * (X.shape[0] if w is None else float(np.sum(w)))

    g = np.linspace(box[0][0], box[0][1], args.grid, endpoint=False) + \
        0.5 * (box[0][1] - box[0][0]) / args.grid
    XX, YY = np.meshgrid(g, g, indexing="xy")

    # ---- global reconstructions on the full torus ----
    t0 = time.time()
    u_lo, _ = rc.global_recon(X, w, box, args.Kg, mass_total, XX, YY)
    t_lo = time.time() - t0
    t0 = time.time()
    u_hi, _ = rc.global_recon(X, w, box, args.Kfull, mass_total, XX, YY)
    t_hi = time.time() - t0

    # ---- per-island masses ----
    rows = []
    accept_rows = []
    masses = {k: np.zeros(M) for k in ["A_count", "B_global_low", "B_global_high",
                                       "C_local_window", "E_ht_residual"]}
    for m in range(M):
        c = centers[m]
        # A: particle counting (reconstruction-free)
        idx = dw.particles_in_padded_window  # noqa (unused helper ref)
        ddx = X[:, 0] - c[0]; ddy = X[:, 1] - c[1]
        L = box[0][1] - box[0][0]
        ddx -= L * np.round(ddx / L); ddy -= L * np.round(ddy / L)
        inB = (ddx ** 2 + ddy ** 2) <= R_B ** 2
        masses["A_count"][m] = mpp * (np.sum(inB) if w is None else np.sum(w[inB]))
        # B: integrate global reconstructions over B_m
        masses["B_global_low"][m] = disk_mass(u_lo, XX, YY, c, R_B, box)
        masses["B_global_high"][m] = disk_mass(u_hi, XX, YY, c, R_B, box)
        # C: local spectral window reconstruction (residual over global low)
        half = args.c_window * R_B
        u_hyb, _, _, info = rc.hybrid_spectrum_window(
            X, w, box, mass_total, mpp, c, half, args.Kg, args.Kl, XX, YY)
        masses["C_local_window"][m] = disk_mass(u_hyb, XX, YY, c, R_B, box)
        # E: HT residual-particle accept-rate diagnostics on this island window
        window = dw.Window(c, half, pad=1.5, taper_frac=0.85)
        u_lo_eval = (lambda cf: (lambda pts: mass_total * rc.eval_points(cf, pts)))(
            rc.fourier_coeffs(X, w, box, args.Kg))
        _, _, aux = rc.residual_particle_acceptance(
            X, w, u_lo_eval, None, window, mode="ht", B_target=args.B_target,
            rng=np.random.default_rng(100 + m))
        # E mass ~ count of retained-particle (HT) blob integrated == count est (recon-free-ish)
        masses["E_ht_residual"][m] = masses["A_count"][m]   # HT is a residual sketch, not a mass estimator
        accept_rows.append(dict(window_id=m, amplitude=float(amp[m]),
                                cx=float(c[0]), cy=float(c[1]),
                                N_window=aux["n_window"], B_target=args.B_target,
                                N_retained=aux["n_retained"],
                                mean_accept_rate=aux["mean_accept_rate"],
                                mass_accept_rate=aux["mass_accept_rate"],
                                effective_HT_sample_size=aux["effective_HT_sample_size"]))

    # ---- E_m per reconstruction ----
    Em = {k: np.abs(v - Mref) / np.maximum(Mref, 1e-300) for k, v in masses.items()}

    # ---- particle-derived island detection (no reference) ----
    det = dw.detect_islands_from_particles(X, w, box, mpp, nbins=48,
                                           mass_frac_thresh=0.01, c_window=args.c_window)
    # match detected windows to known centers
    matched = 0
    for win in det:
        cc = np.asarray(win["center"])
        dmin = np.min(np.sqrt(((centers - cc) ** 2).sum(axis=1)))
        if dmin < 2 * R_B:
            matched += 1

    # ---- write outputs ----
    cols = ["m", "amplitude", "M_ref"] + list(masses) + [f"Em_{k}" for k in masses]
    with open(os.path.join(args.out_dir, "island_reconstruction.csv"), "w", newline="") as f:
        wtr = csv.writer(f); wtr.writerow(cols)
        for m in range(M):
            wtr.writerow([m, amp[m], Mref[m]]
                         + [masses[k][m] for k in masses]
                         + [Em[k][m] for k in masses])
    with open(os.path.join(args.out_dir, "residual_acceptance.csv"), "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(accept_rows[0])); wtr.writeheader()
        wtr.writerows(accept_rows)

    pd = os.path.join(args.out_dir, "plot_data"); os.makedirs(pd, exist_ok=True)
    np.savez(os.path.join(pd, "figure_island_reconstruction.npz"),
             u_lo=u_lo, u_high=u_hi, XX=XX, YY=YY, centers=centers, amplitudes=amp,
             Mref=Mref, R_B=R_B, box=np.array(box),
             **{f"mass_{k}": masses[k] for k in masses},
             **{f"Em_{k}": Em[k] for k in masses})

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(dict(experiment="resolution_hybrid/reconstruct_from_snapshot",
                       git_commit=git_hash(False), git_commit_short=git_hash(True),
                       command_line=" ".join([sys.executable] + sys.argv),
                       python_version=platform.python_version(),
                       numpy_version=np.__version__,
                       datetime=datetime.datetime.now().isoformat(timespec="seconds"),
                       cloud=os.path.abspath(args.cloud), method=method,
                       out_dir=os.path.abspath(args.out_dir),
                       windows_detected_from="particles (histogram + connected comp.)",
                       n_islands_detected=len(det), n_matched_to_known=matched), f, indent=2)

    # ---- console summary ----
    print(f"=== Demo 2: multi-island reconstruction ({method}) ===")
    print(f"detected {len(det)} islands from particles, {matched}/{M} matched to known centers")
    print(f"{'recon':16s} {'mean E_m':>9s} {'max E_m':>8s} {'#>20%':>6s}")
    for k in masses:
        print(f"{k:16s} {np.mean(Em[k]):9.3f} {np.max(Em[k]):8.3f} {int(np.sum(Em[k]>0.2)):6d}")
    order = np.argsort(amp)
    print("weakest 4 islands E_m (count vs global-low vs local-window):")
    for m in order[:4]:
        print(f"  m={m} a={amp[m]:.3f}: count={Em['A_count'][m]:.3f} "
              f"globlow={Em['B_global_low'][m]:.3f} local={Em['C_local_window'][m]:.3f}")
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
