"""plot_field_vs_K.py -- reconstructed solution field u(x) for different K.

Shows WHY the core-collapse diagnostic R_q depends on K in a fixed/large domain:
the v/grad-v field is a K-mode Fourier series on the box, so a coarse K is
band-limited (cannot represent a sharp core), the chemotactic drift it produces
is too smeared, and the particle cloud concentrates less -> larger R_q. A finer
K reconstructs a taller/sharper peak -> stronger drift -> more collapse.

Reads the saved snapshot u_field (no solver run). Top row: u heatmaps per K on a
shared color scale (zoomed to the core). Bottom: central line profiles u(x,0).
"""
import os, glob, argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


def load_snap(run_dir, K, t_tag, seed, tau="2e-7", N="6400000"):
    f = (f"{run_dir}/sens_K{K}_tau{tau}_q0.8_N{N}_seed{seed}"
         f"/snapshots/snap_u_t{t_tag}_seed{seed}.npz")
    if not os.path.exists(f):
        g = glob.glob(f"{run_dir}/sens_K{K}_*_seed{seed}/snapshots/snap_u_t{t_tag}_seed{seed}.npz")
        f = g[0] if g else None
    return np.load(f) if f else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--Ks", type=int, nargs="+", default=[5, 8, 10])
    ap.add_argument("--t_tag", default="1.2000e-04", help="report-time tag in filename")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zoom", type=float, default=0.15, help="half-width of core zoom")
    ap.add_argument("--out_prefix", default="field_vs_K")
    args = ap.parse_args()

    snaps = {K: load_snap(args.run_dir, K, args.t_tag, args.seed) for K in args.Ks}
    snaps = {K: d for K, d in snaps.items() if d is not None}
    if not snaps:
        raise SystemExit("no snapshots found")
    t = float(next(iter(snaps.values()))["t"])
    vmax = max(float(d["peak_PK_u"]) for d in snaps.values())

    # save plot data (line profiles + peaks)
    out_npz = os.path.join(args.run_dir, f"{args.out_prefix}_data.npz")
    prof = {}
    for K, d in snaps.items():
        xg = d["x_grid"]; u = d["u_field"]
        j0 = np.argmin(np.abs(d["y_grid"] - float(d["x_c"][1])))
        prof[K] = (xg, u[j0, :])
    np.savez(out_npz, t=t, Ks=np.array(list(snaps)),
             peaks=np.array([float(snaps[K]["peak_PK_u"]) for K in snaps]),
             **{f"xprof_K{K}": prof[K][0] for K in snaps},
             **{f"uprof_K{K}": prof[K][1] for K in snaps})

    # ===================== FIGURE =====================
    nK = len(snaps)
    fig = plt.figure(figsize=(3.6 * nK, 6.6))
    gs = fig.add_gridspec(2, nK, height_ratios=[1.25, 1.0], hspace=0.32, wspace=0.12)

    # top row: u heatmaps, shared scale, zoomed to core
    ims = []
    for j, (K, d) in enumerate(snaps.items()):
        ax = fig.add_subplot(gs[0, j])
        xg, yg, u = d["x_grid"], d["y_grid"], d["u_field"]
        xc = d["x_c"]
        im = ax.pcolormesh(xg, yg, u, cmap="inferno", vmin=0, vmax=vmax, shading="auto")
        ims.append(im)
        ax.set_xlim(xc[0] - args.zoom, xc[0] + args.zoom)
        ax.set_ylim(xc[1] - args.zoom, xc[1] + args.zoom)
        ax.set_aspect("equal")
        ax.set_title(f"K={K}   peak={float(d['peak_PK_u']):.0f}", fontsize=10)
        ax.set_xlabel("x");
        if j == 0: ax.set_ylabel("y")
    cax = fig.add_axes([0.92, 0.52, 0.013, 0.34])
    fig.colorbar(ims[-1], cax=cax, label=r"$u(x)$ (shared scale)")

    # bottom: central line profiles u(x, y=y_c)
    axp = fig.add_subplot(gs[1, :])
    cols = plt.cm.viridis(np.linspace(0.15, 0.82, nK))
    for (K, (xg, up)), c in zip(prof.items(), cols):
        axp.plot(xg, up, lw=1.8, color=c, label=f"K={K} (peak {up.max():.0f})")
    axp.set_xlim(-args.zoom, args.zoom)
    axp.set_xlabel("x  (slice through core, y=$y_c$)")
    axp.set_ylabel(r"$u(x,y_c)$")
    axp.set_title("central line profile: coarse K is band-limited -> broad, low peak; "
                  "fine K -> sharp, tall peak", fontsize=9)
    axp.legend(fontsize=8); axp.grid(alpha=0.3)

    fig.suptitle(f"Reconstructed solution $u$ vs reconstruction bandwidth $K$ "
                 f"(fixed domain $L$=0.5, $t$={t:.1e}, baseline $\\tau$=2e-7, N=6.4M)",
                 fontsize=11)
    fd = os.path.join(args.run_dir, "figures")
    os.makedirs(fd, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"{args.out_prefix}.{ext}"),
                    dpi=200, bbox_inches="tight")
    print(f"t={t:.3e}  peaks: " +
          ", ".join(f"K{K}={float(d['peak_PK_u']):.0f}" for K, d in snaps.items()))
    print(f"wrote {out_npz}")
    print(f"wrote {fd}/{args.out_prefix}.pdf/.png")


if __name__ == "__main__":
    main()
