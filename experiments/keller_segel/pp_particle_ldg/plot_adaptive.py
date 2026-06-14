"""
Figure for the particle-adaptive reconstruction audit (blocker TODO §2).
========================================================================

Reads adaptive_S_curves.csv (peak / S_L2 / S_core / umin / mass vs Kg and
reconstruction) written by adaptive_reconstruct.py, plus the saved base cloud for
one core radial profile.  Shows the honest mixed/limited result:

  (a) global-Fourier peak plateaus far below the resolved core as Kg grows, while
      the particle-adaptive hybrid jumps to the FVM-anchor scale at low global Kg;
  (b) same for S_L2;
  (c) core radial profile: global low-K smears the core, the hybrid (spectrum/blob)
      recovers the spike consistent with the reconstruction-free R_0.8, but the
      signed residual undershoots (negative lobe) -- the reported caveat.

Usage:
  python plot_adaptive.py --adir <adaptive_recon> --clouds_dir <run>/base \
      --fvm_baseline <baseline_run>
"""
import os
import sys
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "resolution_hybrid")))
import common_plot_style as cps                       # noqa: E402
from common_plot_style import TEXTWIDTH_IN, savefig_multi  # noqa: E402
import reconstructors as rec                          # noqa: E402
from detect_windows import detect_core_window         # noqa: E402

cps.apply_style()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adir", required=True)
    ap.add_argument("--clouds_dir", required=True)
    ap.add_argument("--fvm_baseline", default="")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--rt", type=float, default=2.0e-4)
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(os.path.join(args.adir, f"adaptive_S_curves_{args.tag}.csv")))
            if r["tag"] == args.tag]
    rt = args.rt
    sub = [r for r in rows if abs(float(r["t"]) - rt) < 1e-12]
    gf = sorted([r for r in sub if r["method"] == "global_fourier"], key=lambda r: int(r["Kg"]))
    hy = [r for r in sub if r["method"] == "hybrid_spectrum"]
    bl = [r for r in sub if r["method"] == "hybrid_blob"]

    # FVM anchor
    anchor = {}
    if args.fvm_baseline:
        p = os.path.join(args.fvm_baseline, "n512", "S_curves.csv")
        if os.path.exists(p):
            rws = list(csv.DictReader(open(p)))
            t = np.array([float(r["t"]) for r in rws])
            i = int(np.argmin(np.abs(t - rt)))
            anchor = dict(S_L2=float(rws[i]["S_L2"]), peak=float(rws[i]["peak"]))

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 0.34 * TEXTWIDTH_IN))

    # (a) peak vs Kg
    ax = axes[0]
    Kg = [int(r["Kg"]) for r in gf]
    ax.plot(Kg, [float(r["peak"]) for r in gf], "o-", color="tab:gray", ms=3,
            label="global Fourier")
    for r in hy:
        ax.axhline(float(r["peak"]), color="tab:blue", lw=0.7, alpha=0.5)
    ax.scatter([8] * len(hy), [float(r["peak"]) for r in hy], c="tab:blue", s=14,
               zorder=3, label=r"hybrid (Kg8+Kl)")
    ax.scatter([8] * len(bl), [float(r["peak"]) for r in bl], c="tab:green", s=14,
               marker="s", zorder=3, label="hybrid blob")
    if anchor:
        ax.axhline(anchor["peak"], color="tab:red", ls="--", lw=1.0, label="FVM $n{=}512$")
    ax.set_yscale("log"); ax.set_xlabel(r"global bandwidth $K_g$")
    ax.set_ylabel(r"peak $\|u\|_\infty$"); ax.set_title("peak", fontsize=7)
    ax.legend(fontsize=4.6, loc="lower right")

    # (b) S_L2 vs Kg
    ax = axes[1]
    ax.plot(Kg, [float(r["S_L2"]) for r in gf], "o-", color="tab:gray", ms=3)
    for r in hy:
        ax.axhline(float(r["S_L2"]), color="tab:blue", lw=0.7, alpha=0.5)
    ax.scatter([8] * len(hy), [float(r["S_L2"]) for r in hy], c="tab:blue", s=14, zorder=3)
    ax.scatter([8] * len(bl), [float(r["S_L2"]) for r in bl], c="tab:green", s=14,
               marker="s", zorder=3)
    if anchor:
        ax.axhline(anchor["S_L2"], color="tab:red", ls="--", lw=1.0)
    ax.set_yscale("log"); ax.set_xlabel(r"global bandwidth $K_g$")
    ax.set_ylabel(r"$S_{L^2}$"); ax.set_title(r"$S_{L^2}$", fontsize=7)

    # (c) core radial profile (recompute a few fields from the saved cloud)
    ax = axes[2]
    import glob
    f = glob.glob(os.path.join(args.clouds_dir, "snapshots", f"snap_u_t{rt:.4e}_*.npz"))[0]
    d = np.load(f)
    X = np.asarray(d["X_u"], float); M = float(d["mass_u_total"]); mpp = float(d["mass_per_particle_u"])
    win = detect_core_window(X, None, mpp, c_window=3.0)
    center = np.array(win["center"]); R08 = win["R08"]
    box = [[-0.5, 0.5], [-0.5, 0.5]]
    rmax = 6 * R08
    g = np.linspace(center[0] - rmax, center[0] + rmax, 801)
    dgx = g[1] - g[0]
    gy = center[1] + (np.arange(7) - 3) * dgx          # thin band so dy is defined
    XX, YY = np.meshgrid(g, gy)
    mid = 3
    u_lo, _ = rec.global_recon(X, None, box, 8, M, XX, YY)
    u_hi, _ = rec.global_recon(X, None, box, 32, M, XX, YY)
    u_hyb, _, _, _ = rec.hybrid_spectrum_window(X, None, box, M, mpp, center, 3 * R08, 8, 32, XX, YY, pad=1.5)
    u_bl, _, _, _ = rec.hybrid_blob_residual(X, None, box, M, mpp, center, 3 * R08,
                                             0.7 * R08 / np.sqrt(max(int(win["local_count"]), 1)),
                                             8, XX, YY, pad=1.5)
    xr = (g - center[0])
    ax.plot(xr, u_lo[mid], "-", color="tab:gray", lw=1.0, label=r"global $K_g{=}8$")
    ax.plot(xr, u_hi[mid], "-", color="0.4", lw=0.9, label=r"global $K{=}32$")
    ax.plot(xr, u_hyb[mid], "-", color="tab:blue", lw=1.1, label=r"hybrid Kl32")
    ax.plot(xr, u_bl[mid], "-", color="tab:green", lw=1.0, label="hybrid blob")
    ax.axhline(0, color="0.7", lw=0.4)
    ax.axvspan(-R08, R08, color="tab:orange", alpha=0.12)
    ax.set_xlabel(r"$x-x_c$"); ax.set_ylabel(r"$u$ (core ray)")
    ax.set_title(r"core profile (shaded $R_{0.8}$)", fontsize=6.4)
    ax.legend(fontsize=4.4, loc="upper right")

    fig.suptitle(f"Particle-adaptive reconstruction audit ({args.tag} cloud, "
                 f"$t={rt:.1e}$): global Fourier under-resolves the core; hybrid "
                 f"recovers it (signed-residual undershoot is the caveat)", fontsize=6.2)
    fig.tight_layout(pad=0.4, rect=[0, 0, 1, 0.93])
    out = os.path.join(args.adir, "figures", "adaptive_reconstruct")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    savefig_multi(fig, out, close=True)
    print("wrote", out + ".pdf/.png")


if __name__ == "__main__":
    main()
