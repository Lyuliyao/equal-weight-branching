"""plot_hist_vs_K.py -- reconstruct u by a particle HISTOGRAM (not Fourier P_K mu).

Motivation: P_K mu is band-limited, so its peak diverges with K and it rings
(negative side-lobes). A histogram of the particle positions is positive and its
resolution is set by the BIN WIDTH, not by K, so it shows the cloud's actual
concentration. Because the solver dynamics still depend on K (drift = grad of the
K-mode v field), the clouds differ across K and the histogram peak CONVERGES with K
(like the core radius R_q), in contrast to the diverging Fourier peak.

Reads the raw particle clouds X_u saved in the snapshots (--save_cloud_snapshots).
Produces (1) histogram field heatmaps + central profiles per K, and (2) a peak-vs-K
figure contrasting the Fourier peak (diverges) with the histogram peak (converges).
"""
import os, glob, csv, argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

MASS = 10.0 * np.pi


def load_cloud(run_dir, K, t_tag, seed=0):
    g = glob.glob(f"{run_dir}/sens_K{K}_*_seed{seed}/snapshots/snap_u_t{t_tag}_seed{seed}.npz")
    if not g:
        return None
    d = np.load(g[0])
    if "X_u" not in d.files:
        return None
    return d


def hist_density(X, xc, half, nbins, mass_per_particle):
    """2D histogram density u(x)=count*mpp/binarea on a fixed grid around xc."""
    edges = [np.linspace(xc[0] - half, xc[0] + half, nbins + 1),
             np.linspace(xc[1] - half, xc[1] + half, nbins + 1)]
    H, xe, ye = np.histogram2d(X[:, 0], X[:, 1], bins=edges)
    binarea = (xe[1] - xe[0]) * (ye[1] - ye[0])
    u = H.T * mass_per_particle / binarea          # transpose -> [y,x] for pcolormesh
    xc_grid = 0.5 * (xe[:-1] + xe[1:]); yc_grid = 0.5 * (ye[:-1] + ye[1:])
    return xc_grid, yc_grid, u


def R08_of(run_dir, K, t=1.2e-4, seed=0):
    g = glob.glob(f"{run_dir}/sens_K{K}_*_seed{seed}/diag_*.csv")
    if not g:
        return np.nan
    R = list(csv.DictReader(open(g[0])))
    tt = np.array([float(r["t"]) for r in R]); y = np.array([float(r["R_0.8"]) for r in R])
    return y[np.argmin(np.abs(tt - t))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--Ks", type=int, nargs="+", default=[5, 8, 10, 16, 32])
    ap.add_argument("--t_tag", default="1.2000e-04")
    ap.add_argument("--half", type=float, default=0.12, help="half-width of fixed window")
    ap.add_argument("--nbins", type=int, default=80, help="bins per axis (fixed for all K)")
    ap.add_argument("--out_prefix", default="hist_vs_K")
    args = ap.parse_args()

    clouds = {K: load_cloud(args.run_dir, K, args.t_tag) for K in args.Ks}
    clouds = {K: d for K, d in clouds.items() if d is not None}
    if not clouds:
        raise SystemExit("no cloud snapshots (X_u) found; rerun with --save_cloud_snapshots")
    t = float(next(iter(clouds.values()))["t"])
    binw = 2 * args.half / args.nbins

    # histogram densities + peaks
    hist, hpeak, fpeak, r08 = {}, {}, {}, {}
    for K, d in clouds.items():
        xc = np.asarray(d["x_c"]); mpp = float(d["mass_per_particle_u"])
        xg, yg, u = hist_density(np.asarray(d["X_u"]), xc, args.half, args.nbins, mpp)
        hist[K] = (xg, yg, u, xc)
        hpeak[K] = float(u.max()); fpeak[K] = float(d["peak_PK_u"]); r08[K] = R08_of(args.run_dir, K)

    Ks = list(clouds)
    vmax = max(hpeak.values())

    # ---------- Figure 1: histogram field heatmaps + central profile ----------
    nK = len(Ks)
    fig = plt.figure(figsize=(3.4 * nK, 6.6))
    gs = fig.add_gridspec(2, nK, height_ratios=[1.25, 1.0], hspace=0.33, wspace=0.12)
    ims = []
    for j, K in enumerate(Ks):
        ax = fig.add_subplot(gs[0, j]); xg, yg, u, xc = hist[K]
        im = ax.pcolormesh(xg, yg, u, cmap="inferno", vmin=0, vmax=vmax, shading="auto")
        ims.append(im); ax.set_aspect("equal")
        ax.set_title(f"K={K}  hist peak={hpeak[K]:.0f}", fontsize=10)
        ax.set_xlabel("x")
        if j == 0: ax.set_ylabel("y")
    cax = fig.add_axes([0.92, 0.52, 0.013, 0.34])
    fig.colorbar(ims[-1], cax=cax, label=r"$u$ (histogram, shared scale)")
    axp = fig.add_subplot(gs[1, :])
    cols = plt.cm.viridis(np.linspace(0.15, 0.82, nK))
    for K, c in zip(Ks, cols):
        xg, yg, u, xc = hist[K]
        j0 = np.argmin(np.abs(yg - xc[1]))
        axp.plot(xg, u[j0, :], lw=1.8, color=c, label=f"K={K} (peak {hpeak[K]:.0f})")
    axp.set_xlabel("x  (slice through core)"); axp.set_ylabel(r"$u$ (histogram)")
    axp.set_title(f"histogram reconstruction (bin width={binw:.4f}, fixed for all K): "
                  "peak CONVERGES with K, positive, no ringing", fontsize=9)
    axp.legend(fontsize=8); axp.grid(alpha=0.3)
    fig.suptitle(f"Histogram reconstruction of $u$ vs K (fixed domain, $t$={t:.1e}, "
                 f"{args.nbins}$^2$ bins on $\\pm${args.half})", fontsize=11)
    fd = os.path.join(args.run_dir, "figures"); os.makedirs(fd, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"{args.out_prefix}.{ext}"), dpi=200, bbox_inches="tight")

    # ---------- Figure 2: peak vs K -- Fourier diverges, histogram converges ----------
    fig2, ax2 = plt.subplots(figsize=(6.4, 4.6))
    Karr = np.array(Ks)
    ax2.plot(Karr, [fpeak[K] for K in Ks], "o-", color="C3", lw=1.8,
             label="Fourier $P_K\\mu$ peak (diverges)")
    ax2.plot(Karr, [hpeak[K] for K in Ks], "s-", color="C0", lw=1.8,
             label=f"histogram peak (bin {binw:.4f}, converges)")
    ax2.set_xlabel("reconstruction bandwidth $K$"); ax2.set_ylabel("peak $u$")
    ax2.set_yscale("log"); ax2.set_xscale("log"); ax2.set_xticks(Karr)
    ax2.set_xticklabels([str(k) for k in Ks])
    ax2.set_title("Reconstructed peak vs K: Fourier diverges, histogram converges",
                  fontsize=10)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3, which="both")
    # annotate R_0.8 convergence on a twin axis
    ax3 = ax2.twinx()
    ax3.plot(Karr, [r08[K] for K in Ks], "^--", color="C2", lw=1.4, alpha=0.8,
             label="$R_{0.8}$ (converges)")
    ax3.set_ylabel("$R_{0.8}$", color="C2"); ax3.tick_params(axis="y", labelcolor="C2")
    ax3.legend(fontsize=9, loc="center right")
    for ext in ("pdf", "png"):
        fig2.savefig(os.path.join(fd, f"{args.out_prefix}_peak_vs_K.{ext}"),
                     dpi=200, bbox_inches="tight")

    # ---------- save plot data ----------
    np.savez(os.path.join(args.run_dir, f"{args.out_prefix}_data.npz"),
             Ks=Karr, fourier_peak=np.array([fpeak[K] for K in Ks]),
             hist_peak=np.array([hpeak[K] for K in Ks]),
             R_0_8=np.array([r08[K] for K in Ks]),
             bin_width=binw, t=t, nbins=args.nbins, half=args.half)
    print(f"t={t:.3e}  bin width={binw:.4f}")
    print(f"{'K':>4} {'Fourier_peak':>13} {'hist_peak':>11} {'R_0.8':>8}")
    for K in Ks:
        print(f"{K:>4} {fpeak[K]:13.0f} {hpeak[K]:11.0f} {r08[K]:8.4f}")
    print(f"wrote {fd}/{args.out_prefix}.pdf/.png and _peak_vs_K.pdf/.png")


if __name__ == "__main__":
    main()
