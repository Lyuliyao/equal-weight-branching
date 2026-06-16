"""plot_case2_style.py -- render our particle u-clouds with the ORIGINAL case2_test1
plotting recipe: raw 2D histogram on the FULL [-0.5,0.5] domain (bins=640,
density=True), divided by C_u = 1/(10 pi) to physical u, contourf(levels=100,
cmap='RdBu_r'), 1x4 over the report times, per-panel colorbar on top.

One 1x4 figure per method (faithful replica of the user's case2_u.pdf code).
"""
import os
import sys
import glob
import argparse

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from plot_pp_ldg_particle_snapshots import load_particle, load_ldg, match_time  # noqa: E402

C_U = 1.0 / (10.0 * np.pi)
TICKFONTSIZE = 11
LEGENDFONTSIZE = 8
BINS = 640
RANGE = [[-0.5, 0.5], [-0.5, 0.5]]


def _title(t):
    if t == 0:
        return r"$t = 0$"
    s = "{:.0e}".format(t)
    base, exp = s.split("e")
    return r"$t = {} \times 10^{{{}}}$".format(base, int(exp))


def render_1x4(times, clouds, out_stem, bins=BINS, ldg=None):
    """clouds: dict time -> (N,2) array of u-particle positions (or LDG (x,y,u))."""
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15, wspace=0.1, hspace=0)
    for i, t in enumerate(times):
        ax = axs[i]
        if ldg is None:
            X1 = clouds[t]
            H, xe, ye = np.histogram2d(X1[:, 0], X1[:, 1], bins=bins, range=RANGE, density=True)
            U = H.T / C_U
            X, Y = np.meshgrid(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]))
        else:                                        # LDG reference field row
            x, y, u = clouds[t]
            X, Y = np.meshgrid(x, y)
            U = u
        cf = ax.contourf(X, Y, U, levels=100, cmap="RdBu_r")
        ax.set_title(_title(t), fontsize=TICKFONTSIZE)
        ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5)
        ax.set_xticks([-0.3, 0.0, 0.3]); ax.tick_params(labelsize=TICKFONTSIZE)
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0, pos.y1 - 0.02, pos.width, 0.03])
        cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=LEGENDFONTSIZE)
        cb.ax.xaxis.set_major_formatter(formatter)
        cb.ax.xaxis.set_tick_params(color="white")
        plt.setp(cb.ax.get_xticklabels(), color="white")
        cb.ax.xaxis.get_offset_text().set_color("white")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", bbox_inches="tight", dpi=200 if ext == "png" else None)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+",
                    default=["current_fourier", "blob_ch009", "screened_kg8"])
    ap.add_argument("--bins", type=int, default=BINS)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--include_ldg", action="store_true")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.run_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    # find each method's run dir (largest N for this seed)
    import re
    def pick(key):
        cands = [d for d in glob.glob(os.path.join(args.run_dir, f"{key}_N*_seed{args.seed}"))
                 if glob.glob(os.path.join(d, "snapshots", "snap_u_t*.npz"))]
        if not cands:
            return None
        return max(cands, key=lambda d: int(re.search(r"_N(\d+)_", d).group(1)))

    for m in args.methods:
        d = pick(m)
        if d is None:
            print(f"[case2] no run for {m}"); continue
        N = int(re.search(r"_N(\d+)_", d).group(1))
        snaps = load_particle(d)
        times = sorted(snaps.keys())
        clouds = {t: np.asarray(snaps[t]["X_u"], float) for t in times}
        stem = os.path.join(out_dir, f"case2style_{m}_N{N}_seed{args.seed}")
        render_1x4(times, clouds, stem, bins=args.bins)
        print(f"[case2] {m} N={N}: wrote {stem}.pdf/.png  (bins={args.bins}, "
              f"{len(times)} times, particles/snap={len(next(iter(clouds.values())))})")

    if args.include_ldg:
        ldg = load_ldg(os.path.join(args.run_dir, "ldg", "N320", "snapshots.npz"))
        times = sorted(ldg.keys())
        stem = os.path.join(out_dir, "case2style_LDG_N320")
        render_1x4(times, ldg, stem, ldg=ldg)
        print(f"[case2] LDG: wrote {stem}.pdf/.png")


if __name__ == "__main__":
    main()
