"""plot_ldg_vs_spectrum.py -- reconstruct u from the SAME particle cloud with the
spectral operator P_K mu vs the LDG-matched P1-DG projection, and compare.

Spectral P_K mu : global Fourier, band-limited -> peak diverges with K, rings
(negative side-lobes).  LDG P1-DG : local projection on an n x n mesh (the LDG
reference's space), resolution = cell size -> positive-ish, no global ringing.

Uses the saved K-ladder clouds (X_u at t=1.2e-4). For each K we form:
  spectral  : peak_PK_u + min(u_field) from the snapshot (Fourier reconstruction)
  LDG match : P1-DG at n=2K (resolution matched to the K-mode spectrum)
  LDG fixed : P1-DG at a fixed fine mesh (isolates cloud convergence)
"""
import os, glob, argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from particle_dg_readout import project_particles_to_p1dg, dg_peak, dg_min, dg_l2_norm

KS = [5, 8, 10, 16, 32]
BOX = ((-0.5, 0.5), (-0.5, 0.5))


def eval_p1dg(coeffs, n, gx, gy):
    """evaluate the P1-DG field c0+c1*xi+c2*eta on grid points gx,gy (1D each)."""
    (x0, x1), (y0, y1) = BOX
    dx = (x1 - x0) / n; dy = (y1 - y0) / n
    X, Y = np.meshgrid(gx, gy)
    ix = np.clip(np.floor((X - x0) / dx).astype(int), 0, n - 1)
    iy = np.clip(np.floor((Y - y0) / dy).astype(int), 0, n - 1)
    xc = x0 + (ix + 0.5) * dx; yc = y0 + (iy + 0.5) * dy
    xi = 2 * (X - xc) / dx; eta = 2 * (Y - yc) / dy
    c = coeffs[iy, ix]
    return c[..., 0] + c[..., 1] * xi + c[..., 2] * eta


def load_cloud(rdir, K):
    g = glob.glob(f"{rdir}/sens_K{K}_*_seed0/snapshots/snap_u_t1.2000e-04_seed0.npz")
    return np.load(g[0]) if g else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clouds_dir", required=True)
    ap.add_argument("--hist_npz", default=None, help="optional hist_vs_K_data.npz for hist peak")
    ap.add_argument("--n_fixed", type=int, default=128)
    ap.add_argument("--out_prefix", default="ldg_vs_spectrum")
    args = ap.parse_args()

    rec = {}
    for K in KS:
        d = load_cloud(args.clouds_dir, K)
        if d is None:
            continue
        X = np.asarray(d["X_u"]); w = np.full(X.shape[0], float(d["mass_per_particle_u"]))
        cm, _ = project_particles_to_p1dg(X, w, 2 * K, BOX)        # matched n=2K
        cf, _ = project_particles_to_p1dg(X, w, args.n_fixed, BOX)  # fixed fine
        rec[K] = dict(
            f_peak=float(d["peak_PK_u"]), f_min=float(np.min(d["u_field"])),
            ldg_m_peak=dg_peak(cm), ldg_m_min=dg_min(cm),
            ldg_f_peak=dg_peak(cf), ldg_f_min=dg_min(cf),
            coeff_f=cf, X=X, w=w, snap=d)
    Ks = list(rec)

    hist_peak = {}
    if args.hist_npz and os.path.exists(args.hist_npz):
        H = np.load(args.hist_npz)
        hist_peak = dict(zip([int(k) for k in H["Ks"]], H["hist_peak"]))

    # ============ FIGURE ============
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0], hspace=0.33, wspace=0.28)

    # top: field heatmaps at K=32 -- spectral vs LDG(matched) vs LDG(fixed)
    Kshow = 32 if 32 in rec else Ks[-1]
    d = rec[Kshow]["snap"]
    zoom = 0.12
    # spectral field (from snapshot)
    axs = fig.add_subplot(gs[0, 0])
    xg, yg, U = d["x_grid"], d["y_grid"], d["u_field"]
    vmax = rec[Kshow]["f_peak"]
    axs.pcolormesh(xg, yg, U, cmap="inferno", vmin=0, vmax=vmax, shading="auto")
    axs.set_xlim(-zoom, zoom); axs.set_ylim(-zoom, zoom); axs.set_aspect("equal")
    axs.set_title(f"spectral $P_{{{Kshow}}}\\mu$\npeak={rec[Kshow]['f_peak']:.0f}, "
                  f"min={rec[Kshow]['f_min']:.0f}", fontsize=9)
    # LDG matched n=2K field
    gg = np.linspace(-zoom, zoom, 200)
    cm, _ = project_particles_to_p1dg(rec[Kshow]["X"], rec[Kshow]["w"], 2 * Kshow, BOX)
    Ulm = eval_p1dg(cm, 2 * Kshow, gg, gg)
    axm = fig.add_subplot(gs[0, 1])
    axm.pcolormesh(gg, gg, Ulm, cmap="inferno", vmin=0, vmax=vmax, shading="auto")
    axm.set_aspect("equal")
    axm.set_title(f"LDG P1-DG, n=2K={2*Kshow}\npeak={rec[Kshow]['ldg_m_peak']:.0f}, "
                  f"min={rec[Kshow]['ldg_m_min']:.0f}", fontsize=9)
    # LDG fixed fine field
    Ulf = eval_p1dg(rec[Kshow]["coeff_f"], args.n_fixed, gg, gg)
    axf = fig.add_subplot(gs[0, 2])
    im = axf.pcolormesh(gg, gg, Ulf, cmap="inferno", vmin=0, vmax=vmax, shading="auto")
    axf.set_aspect("equal")
    axf.set_title(f"LDG P1-DG, n={args.n_fixed} (fine)\npeak={rec[Kshow]['ldg_f_peak']:.0f}, "
                  f"min={rec[Kshow]['ldg_f_min']:.0f}", fontsize=9)
    fig.colorbar(im, ax=[axs, axm, axf], shrink=0.8, label="u", pad=0.01)

    # bottom-left: central line profile at K=32 (spectral negative dip vs LDG positive)
    axp = fig.add_subplot(gs[1, 0])
    j0 = np.argmin(np.abs(yg))
    axp.plot(xg, U[j0, :], "C3", lw=1.6, label=f"spectral $P_{{{Kshow}}}$")
    axp.plot(gg, Ulf[np.argmin(np.abs(gg)), :], "C0", lw=1.6, label=f"LDG n={args.n_fixed}")
    axp.axhline(0, color="k", lw=0.6)
    axp.set_xlim(-zoom, zoom); axp.set_xlabel("x (core slice)"); axp.set_ylabel("u")
    axp.set_title(f"(profile, K={Kshow}) spectral rings <0; LDG positive", fontsize=9)
    axp.legend(fontsize=8); axp.grid(alpha=0.3)

    # bottom-mid: peak vs K
    axpk = fig.add_subplot(gs[1, 1])
    Ka = np.array(Ks)
    axpk.loglog(Ka, [rec[K]["f_peak"] for K in Ks], "o-", color="C3", label="spectral $P_K\\mu$")
    axpk.loglog(Ka, [rec[K]["ldg_m_peak"] for K in Ks], "s-", color="C0", label="LDG n=2K")
    axpk.loglog(Ka, [rec[K]["ldg_f_peak"] for K in Ks], "^-", color="C2",
                label=f"LDG n={args.n_fixed} (fixed)")
    if hist_peak:
        axpk.loglog(Ka, [hist_peak.get(K, np.nan) for K in Ks], "d--", color="0.4",
                    label="histogram")
    axpk.set_xlabel("K"); axpk.set_ylabel("peak u"); axpk.set_xticks(Ka)
    axpk.set_xticklabels([str(k) for k in Ks])
    axpk.set_title("(peak vs K)", fontsize=9); axpk.legend(fontsize=7); axpk.grid(alpha=0.3, which="both")

    # bottom-right: min value vs K (positivity)
    axmn = fig.add_subplot(gs[1, 2])
    axmn.plot(Ka, [rec[K]["f_min"] for K in Ks], "o-", color="C3", label="spectral $P_K\\mu$ min")
    axmn.plot(Ka, [rec[K]["ldg_f_min"] for K in Ks], "^-", color="C2",
              label=f"LDG n={args.n_fixed} min")
    axmn.axhline(0, color="k", lw=0.6)
    axmn.set_xscale("log"); axmn.set_xticks(Ka); axmn.set_xticklabels([str(k) for k in Ks])
    axmn.set_xlabel("K"); axmn.set_ylabel("min u (negativity)")
    axmn.set_title("(positivity) spectral undershoots <0", fontsize=9)
    axmn.legend(fontsize=8); axmn.grid(alpha=0.3)

    fig.suptitle("Reconstruction: spectral $P_K\\mu$ vs LDG P1-DG projection "
                 "(same K-ladder clouds, t=1.2e-4)", fontsize=12)
    fd = os.path.join(args.clouds_dir, "figures"); os.makedirs(fd, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"{args.out_prefix}.{ext}"), dpi=200, bbox_inches="tight")
    print(f"{'K':>4} {'spec_peak':>10} {'spec_min':>9} {'LDG2K_peak':>11} {'LDGfix_peak':>12} {'LDGfix_min':>11}")
    for K in Ks:
        r = rec[K]
        print(f"{K:>4} {r['f_peak']:10.0f} {r['f_min']:9.0f} {r['ldg_m_peak']:11.0f} "
              f"{r['ldg_f_peak']:12.0f} {r['ldg_f_min']:11.1f}")
    print(f"wrote {fd}/{args.out_prefix}.pdf/.png")


if __name__ == "__main__":
    main()
