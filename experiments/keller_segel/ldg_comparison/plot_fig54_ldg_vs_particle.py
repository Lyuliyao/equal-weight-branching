"""plot_fig54_ldg_vs_particle.py -- SS5.4 figure: LDG reference vs particle method,
in the case2_test1 style (full-domain 2D histogram -> physical u via /C_u,
contourf RdBu_r), 2 rows x len(times) columns.

Reproducible: reads saved LDG snapshots.npz + a particle run's saved u-cloud
snapshots; regenerates the figure (and a plot_data .npz of the displayed fields)
with NO solver run.

Colorbars: 3-4 INTEGER ticks each, with the power-of-ten shown once as x10^n
(no long decimal tick lists).

    python plot_fig54_ldg_vs_particle.py --run_dir <RUNDIR> \
        --particle_key orig_match_K5_global --seed 0
"""
import os
import sys
import glob
import argparse

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

C_U = 1.0 / (10.0 * np.pi)            # M_u = 10 pi; density=True hist / C_U -> physical u
BINS = 640
RANGE = [[-0.5, 0.5], [-0.5, 0.5]]
TIMES = [3e-5, 6e-5, 9e-5, 1.2e-4]


def nice_int_ticks(vmax, ntarget=4):
    """Return (ticks, exponent) giving ~ntarget ticks whose mantissas (tick/10^e)
    are INTEGERS, so labels read '0, 2, 4, 6' with a single 'x10^e'."""
    if not np.isfinite(vmax) or vmax <= 0:
        return np.array([0.0]), 0
    e = int(np.floor(np.log10(vmax)))
    if vmax / 10.0 ** e < 2:          # leading digit 1 -> drop an order to keep integers
        e -= 1
    span = vmax / 10.0 ** e           # mantissa span, ~[2,20)
    step = 1
    for s in (1, 2, 3, 5, 10, 20, 30, 50, 100):
        if span / s <= ntarget:       # at most ~ntarget intervals
            step = s
            break
    ticks = np.arange(0.0, vmax * 1.0001, step * 10.0 ** e)
    return ticks, e


def load_ldg(npz_path):
    z = np.load(npz_path)
    X, Y = np.meshgrid(np.asarray(z["xc"], float), np.asarray(z["yc"], float))
    out = {}
    for k in z.files:
        if k.startswith("u_"):
            out[float(k[2:])] = (X, Y, np.asarray(z[k], float))
    return out


def load_particle(run_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "snapshots", "snap_u_t*.npz"))):
        z = np.load(f)
        out[float(z["report_time"]) if "report_time" in z.files else float(z["t"])] = \
            np.asarray(z["X_u"], float)
    return out


def nearest(d, t, tol=2e-6):
    ks = np.array(list(d))
    j = int(np.argmin(np.abs(ks - t)))
    return float(ks[j]) if abs(ks[j] - t) <= tol else None


def _title(t):
    if t == 0:
        return r"$t = 0$"
    s = "{:.0e}".format(t)
    b, e = s.split("e")
    return r"$t = {} \times 10^{{{}}}$".format(b, int(e))


def draw_panel(fig, ax, X, Y, U, title, ylabel):
    cf = ax.contourf(X, Y, U, levels=100, cmap="RdBu_r")
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([-0.3, 0.0, 0.3]); ax.set_yticks([-0.3, 0.0, 0.3])
    ax.tick_params(labelsize=10)
    if title:
        ax.set_title(title, fontsize=13, pad=2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    ticks, e = nice_int_ticks(float(np.nanmax(U)))
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0, pos.y1 - 0.016, pos.width, 0.018])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{int(round(tt / 10.0 ** e))}" for tt in ticks])
    cb.ax.tick_params(labelsize=9, color="white", labelcolor="white", length=2)
    # the shared x10^e, once, in the gap just right of the colorbar (no tick overlap)
    cax.text(1.02, 0.5, rf"$\times 10^{{{e}}}$", transform=cax.transAxes,
             ha="left", va="center", color="black", fontsize=9, clip_on=False)
    return cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--particle_key", default="orig_match_K5_global")
    ap.add_argument("--particle_N", default="6400000")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ldg_npz", default=None)
    ap.add_argument("--particle_label", default="particle (K=5, N=6.4M)")
    ap.add_argument("--bins", type=int, default=BINS)
    ap.add_argument("--times", type=float, nargs="+", default=TIMES)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--tag", default="fig54_ldg_vs_particle")
    args = ap.parse_args()

    ldg_npz = args.ldg_npz or os.path.join(args.run_dir, "ldg", "N320", "snapshots.npz")
    out_dir = args.out_dir or os.path.join(args.run_dir, "figures")
    pdata_dir = os.path.join(args.run_dir, "plot_data")
    os.makedirs(out_dir, exist_ok=True); os.makedirs(pdata_dir, exist_ok=True)

    ldg = load_ldg(ldg_npz)
    pdir = os.path.join(args.run_dir, f"{args.particle_key}_N{args.particle_N}_seed{args.seed}")
    part = load_particle(pdir)
    times = [t for t in args.times
             if nearest(ldg, t) is not None and nearest(part, t) is not None]
    if not times:
        print("[fig54] no common times"); return

    n = len(times)
    fig, axs = plt.subplots(2, n, figsize=(3.05 * n, 6.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.07, wspace=0.12, hspace=0.2)
    saved = {}
    for c, t in enumerate(times):
        # LDG row
        X, Y, U = ldg[nearest(ldg, t)]
        draw_panel(fig, axs[0][c], X, Y, U, _title(t), "LDG reference" if c == 0 else "")
        saved[f"ldg_u_{t:.2e}"] = U
        # particle row (histogram -> physical u)
        Xp = part[nearest(part, t)]
        H, xe, ye = np.histogram2d(Xp[:, 0], Xp[:, 1], bins=args.bins, range=RANGE, density=True)
        Up = H.T / C_U
        Xp2, Yp2 = np.meshgrid(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]))
        draw_panel(fig, axs[1][c], Xp2, Yp2, Up, "", args.particle_label if c == 0 else "")
        saved[f"part_u_{t:.2e}"] = Up

    stem = os.path.join(out_dir, args.tag)
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    np.savez(os.path.join(pdata_dir, f"{args.tag}_plot_data.npz"),
             times=np.array(times), bins=args.bins, C_U=C_U,
             particle_key=args.particle_key, **saved)
    print(f"[fig54] wrote {stem}.pdf/.png  (2 x {n}, particle={args.particle_key})")


if __name__ == "__main__":
    main()
