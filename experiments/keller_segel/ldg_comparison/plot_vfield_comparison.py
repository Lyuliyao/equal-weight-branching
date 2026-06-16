"""plot_vfield_comparison.py -- compare the particle core across v-field (grad v)
reconstruction methods, each vs the LDG reference, for the SS5.4 KS benchmark.

Rows: LDG reference, then one row per particle v-reconstruction method (Fourier,
hybrid-spectral, Gaussian blob, screened-Poisson/Bessel-K0, ...).  Columns: the
report times.  Particle rows use a NONNEGATIVE Gaussian KDE of the saved u-cloud
(the dynamics never uses a u reconstruction; only the v-basis driving aggregation
differs between rows).  Per-column LINEAR colour scale shared across all rows so
the methods are directly comparable.  R_0.2 (solid) / R_0.5 (dashed) core circles
overlaid: LDG from field quadrature, particle from ordered cloud distances.

PURE POST-PROCESSING of saved snapshots; no solver is run.

    python plot_vfield_comparison.py --run_dir <RUNDIR> [--seed 0]
"""
import os
import sys
import glob
import argparse

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_EXP = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
import common_plot_style as cps  # noqa: E402
from plot_pp_ldg_particle_snapshots import (  # noqa: E402
    load_ldg, load_particle, kde_field, cloud_Rq, field_centroid_Rq,
    view_field, view_pos, QUANTILES, match_time)

CMAP = "viridis"
# preferred row order + display labels
METHOD_ORDER = [
    ("current_fourier", "Fourier\n(single-K)"),
    ("two_level_spectral_residual", "hybrid\nspectral"),
    ("blob_ch006", "blob\n$c_h{=}0.06$"),
    ("blob_ch009", "blob\n$c_h{=}0.09$"),
    ("screened_kg8", "screened\n$K_0$"),
]


def _fmt_t(t):
    return rf"$t={t*1e4:.1f}\times10^{{-4}}$" if t >= 1e-4 else rf"$t={t*1e5:.0f}\times10^{{-5}}$"


def _dir_N(d):
    import re
    m = re.search(r"_N(\d+)_seed", os.path.basename(d))
    return int(m.group(1)) if m else 0


def discover_methods(run_dir, seed):
    """For each known method, pick its run dir for this seed; if several N exist,
    prefer the LARGEST N (so a high-N rerun supersedes the lower-N one)."""
    out = []
    for key, label in METHOD_ORDER:
        cands = [d for d in glob.glob(os.path.join(run_dir, f"{key}_N*_seed{seed}"))
                 if glob.glob(os.path.join(d, "snapshots", "snap_u_t*.npz"))]
        if not cands:
            continue
        d = max(cands, key=_dir_N)
        out.append((key, f"{label}\nN={_dir_N(d):.0e}".replace("e+0", "e"), d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ldg_npz", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--times", type=float, nargs="+", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--plot_data_dir", default=None)
    ap.add_argument("--kde_h_frac", type=float, default=0.035)
    ap.add_argument("--kde_ngrid", type=int, default=221)
    ap.add_argument("--view_W", type=float, default=0.04,
                    help="display half-width (zoom on the core); clamped to the KDE window")
    ap.add_argument("--scale", choices=["row", "column", "panel"], default="row",
                    help="colour-scale sharing: row (per method, default; core SHAPE comparison "
                         "-- peak is bandwidth-sensitive), column (per time, all rows), panel (each).")
    ap.add_argument("--tag", default="vfield_comparison")
    args = ap.parse_args()

    ldg_npz = args.ldg_npz or os.path.join(args.run_dir, "ldg", "N320", "snapshots.npz")
    out_dir = args.out_dir or os.path.join(args.run_dir, "figures")
    pdata_dir = args.plot_data_dir or os.path.join(args.run_dir, "plot_data")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(pdata_dir, exist_ok=True)

    ldg = load_ldg(ldg_npz)
    methods = discover_methods(args.run_dir, args.seed)
    if not methods:
        print(f"[cmp] no method snapshot dirs in {args.run_dir} for seed {args.seed}")
        return
    parts = {key: load_particle(d) for key, _, d in methods}

    # times present everywhere
    if args.times is None:
        common = set(ldg)
        for key, _, _ in methods:
            common &= set(parts[key])
        times = sorted(common)
    else:
        times = sorted(args.times)
    times = [t for t in times if match_time(ldg.keys(), t) is not None
             and all(match_time(parts[k].keys(), t) is not None for k, _, _ in methods)]
    if not times:
        print("[cmp] no common times across LDG + all methods")
        return
    print(f"[cmp] methods: {[k for k,_,_ in methods]}")
    print(f"[cmp] times: {[f'{t:.2e}' for t in times]}")

    # common view window: just inside the smallest particle reconstruction window
    Lmin = min(float(parts[k][match_time(parts[k].keys(), t)]["L"])
               for k, _, _ in methods for t in times)
    W = 0.95 * Lmin

    # --- assemble KDE particle fields + LDG fields + overlays --------------
    # rows: ('LDG', ...) then methods
    rows = [("__ldg__", "LDG\nreference")] + [(k, lbl) for k, lbl, _ in methods]
    fields = {}     # (rowkey, t) -> (x, y, u)
    overlays = {}   # (rowkey, t) -> (center, Rq)
    for t in times:
        lt = match_time(ldg.keys(), t)
        lx, ly, lu = ldg[lt]
        fields[("__ldg__", t)] = (lx, ly, lu)
        overlays[("__ldg__", t)] = field_centroid_Rq(lx, ly, lu)
        for k, _, _ in methods:
            snap = parts[k][match_time(parts[k].keys(), t)]
            X = np.asarray(snap["X_u"], float)
            mpp = float(snap["mass_per_particle_u"]) if "mass_per_particle_u" in snap \
                else float(snap["M_u"]) / max(len(X), 1)
            xs, ys, fld = kde_field(X, mpp, W, ngrid=args.kde_ngrid, h=args.kde_h_frac * W)
            fields[(k, t)] = (xs, ys, fld)
            c = (float(snap["x_c"][0]), float(snap["x_c"][1]))
            overlays[(k, t)] = (c, cloud_Rq(X, c))

    # display zoom (tight on the core) and colour-scale sharing.  The reconstructed
    # PEAK is bandwidth-sensitive (not the reliable diagnostic); the core RADIUS is.
    # Default 'row' scaling shows each method's core SHAPE clearly (each row to its
    # own peak) and the overlaid R_q circles give the reconstruction-free size
    # comparison -- it does not let LDG's extreme peak dim the particle rows.
    Wd = float(min(args.view_W, W))
    PCT = 99.5

    def _vp(rk, t):
        x, y, u = fields[(rk, t)]
        return view_pos(x, y, u, Wd)

    def _pmax(rk, t):
        vp = _vp(rk, t)
        return float(np.percentile(vp, PCT)) if vp.size else 0.0

    if args.scale == "column":
        _sca = {t: max(_pmax(rk, t) for rk, _ in rows) for t in times}
        vmax_of = lambda rk, t: _sca[t]
    elif args.scale == "panel":
        vmax_of = lambda rk, t: (_pmax(rk, t) or 1.0)
    else:  # row (default)
        _sca = {rk: max(_pmax(rk, t) for t in times) for rk, _ in rows}
        vmax_of = lambda rk, t: (_sca[rk] or 1.0)

    # --- figure ------------------------------------------------------------
    cps.apply_style()
    nrow, ncol = len(rows), len(times)
    panel = 1.15
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(ncol * panel + 0.9, nrow * panel + 0.55),
                             squeeze=False)
    styles = ["-", "--", ":"]
    row_im = {}
    for i, (rk, rlabel) in enumerate(rows):
        for j, t in enumerate(times):
            ax = axes[i][j]
            x, y, u = fields[(rk, t)]
            im = ax.imshow(np.maximum(u, 0.0), origin="lower",
                           extent=[x.min(), x.max(), y.min(), y.max()],
                           vmin=0, vmax=vmax_of(rk, t), cmap=CMAP, aspect="equal",
                           interpolation="nearest" if rk == "__ldg__" else "bilinear")
            row_im[i] = im
            ax.set_xlim(-Wd, Wd); ax.set_ylim(-Wd, Wd)
            ax.set_xticks([]); ax.set_yticks([])
            c, Rq = overlays[(rk, t)]
            for q, ls in zip(QUANTILES, styles):
                rq = Rq.get(q, np.nan)
                if np.isfinite(rq) and rq > 0:
                    ax.add_patch(plt.Circle(c, rq, fill=False, ec="white", lw=0.7,
                                            ls=ls, alpha=0.9))
            if i == 0:
                ax.set_title(_fmt_t(t), fontsize=8, pad=3)
            if j == 0:
                ax.set_ylabel(rlabel, fontsize=7.5)
    fig.subplots_adjust(left=0.11, right=0.90, top=0.93, bottom=0.04,
                        wspace=0.05, hspace=0.05)
    # per-row vertical colorbar (row scaling): each row shows its own core to its
    # own peak; the white R_0.2 (solid) / R_0.5 (dashed) circles are the
    # reconstruction-free size comparison shared across all rows.
    for i, (rk, _) in enumerate(rows):
        p = axes[i][-1].get_position()
        cax = fig.add_axes([0.915, p.y0, 0.012, p.height])
        cb = fig.colorbar(row_im[i], cax=cax)
        cb.ax.tick_params(labelsize=5.5)
        cb.set_ticks([0, vmax_of(rk, times[-1])])
    fig.text(0.5, 0.005, r"view $[-%.3f,%.3f]^2$;  circles: $R_{0.2}$ (solid), "
             r"$R_{0.5}$ (dashed);  %s colour scale (reconstructed peak is bandwidth-sensitive; "
             r"core radius is the reliable comparison)" % (Wd, Wd, args.scale),
             ha="center", fontsize=6.5)

    stem = os.path.join(out_dir, args.tag)
    cps.savefig_multi(fig, stem, close=False)
    plt.close(fig)
    # plot data
    np.savez(os.path.join(pdata_dir, f"{args.tag}_plot_data.npz"),
             times=np.array(times), W=W, view_W=Wd, scale=args.scale,
             rows=np.array([rk for rk, _ in rows]),
             row_vmax=np.array([vmax_of(rk, times[-1]) for rk, _ in rows]),
             **{f"{rk}_u_{t:.2e}": fields[(rk, t)][2] for rk, _ in rows for t in times})
    print(f"[cmp] wrote {stem}.pdf/.png  ({nrow} rows x {ncol} cols, W={W:.4f})")


if __name__ == "__main__":
    main()
