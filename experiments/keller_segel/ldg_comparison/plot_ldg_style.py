"""plot_ldg_style.py -- publication figures for the LDG-style KS comparison.

PURE POST-PROCESSING.  Reads saved snapshots (snapshots/*.npz) + a diag_*.csv
produced by simulation.py; it NEVER reruns the solver.  Uses the shared
publication style experiments/common_plot_style.py.

Figures produced (each saved as .pdf + .png, with its plot data in
plot_data/figure_<name>.npz):

  (a) figure_u_snapshots   -- reconstructed-u snapshots at the LDG reporting
      times t = 6e-5, 1.2e-4, 2.0e-4.  Peaks grow fast, so each panel is
      auto-scaled and LABELS ITS OWN PEAK VALUE (bandwidth-sensitive), with a
      core-zoom inset (cps.add_zoom_inset) on the concentrating core.
  (b) figure_timeseries    -- S_L2(t) (reconstructed L2, BANDWIDTH-SENSITIVE)
      and the core radii R_0.5(t), R_0.8(t) (RECONSTRUCTION-FREE).
  (c) figure_mass          -- M_u(t) conserved and M_v(t) vs the exact law
      M_v(t) = M_u + (M_v0 - M_u) e^{-t}.

Usage:
    python plot_ldg_style.py --results_dir <dir>   [--out_dir <dir>/figures]
The <dir> must contain a diag_*.csv and a snapshots/ subdir with the .npz files.
"""
import os
import sys
import csv
import glob
import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# shared publication style (experiments/ dir is two levels up)
_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
import common_plot_style as cps  # noqa: E402

REPORT_TIMES = [6e-5, 1.2e-4, 2.0e-4]


def load_diag(results_dir):
    cands = sorted(glob.glob(os.path.join(results_dir, "diag_*.csv")))
    if not cands:
        raise FileNotFoundError(f"no diag_*.csv in {results_dir}")
    path = cands[0]
    with open(path) as f:
        r = list(csv.DictReader(f))
    # skip non-float columns (e.g. the string `solver_field_mode`); only numeric
    # columns are used downstream.
    cols = {}
    for k in r[0]:
        try:
            cols[k] = np.array([float(x[k]) for x in r])
        except (ValueError, TypeError):
            cols[k] = np.array([x[k] for x in r])
    return cols, os.path.basename(path)


def load_snapshots(results_dir):
    snap_dir = os.path.join(results_dir, "snapshots")
    out = {}
    for path in sorted(glob.glob(os.path.join(snap_dir, "snap_u_*.npz"))):
        d = np.load(path)
        rt = float(d["report_time"])
        out[rt] = {k: d[k] for k in d.files}
    return out


# ---------------------------------------------------------------------------
# (a) u snapshots at the three LDG reporting times, with core-zoom insets.
# ---------------------------------------------------------------------------
def fig_u_snapshots(snaps, out_dir, plot_data_dir):
    rts = [rt for rt in REPORT_TIMES if rt in snaps]
    if not rts:
        # fall back to whatever was saved
        rts = sorted(snaps.keys())
    if not rts:
        print("[plot] no snapshots to plot")
        return
    n = len(rts)
    fig, axes = plt.subplots(1, n, figsize=cps.fig_size(1.0, 0.50 if n > 1 else 0.9))
    if n == 1:
        axes = [axes]

    saved = {}
    for ax, rt in zip(axes, rts):
        s = snaps[rt]
        xg = np.asarray(s["x_grid"]); yg = np.asarray(s["y_grid"])
        field = np.asarray(s["u_field"])
        extent = [xg.min(), xg.max(), yg.min(), yg.max()]
        peak = float(field.max())
        im = ax.imshow(field.T, origin="lower", extent=extent, cmap="inferno",
                       vmin=0.0, vmax=peak, interpolation="nearest", aspect="equal")
        ax.set_title(rf"$t={rt:.1e}$,  peak$=${peak:.2e}", fontsize=7)
        ax.set_xlabel(r"$x$")
        if ax is axes[0]:
            ax.set_ylabel(r"$y$")
        ax.locator_params(axis="x", nbins=4)
        ax.locator_params(axis="y", nbins=4)
        # horizontal colorbar UNDER the panel -> leaves the upper-right free for
        # the zoom inset, and keeps each panel's own (bandwidth-sensitive) scale.
        cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                          fraction=0.052, pad=0.18)
        cb.ax.tick_params(labelsize=5)
        cb.locator = mpl.ticker.MaxNLocator(nbins=3)
        cb.update_ticks()
        # core-zoom inset: a tight box about the particle centroid (the core).
        xc = float(np.asarray(s["x_c"])[0]); yc = float(np.asarray(s["x_c"])[1])
        half = 0.18 * (xg.max() - xg.min())
        zoom = (xc - half, xc + half, yc - half, yc + half)
        try:
            cps.add_zoom_inset(ax, field.T, extent, zoom, loc="upper right",
                               zoom_frac=0.38, vmin=0.0, vmax=peak, cmap="inferno",
                               edge="white")
        except Exception as e:
            print(f"[plot] inset skipped at t={rt}: {e}")
        saved[f"t_{rt:.2e}_field"] = field
        saved[f"t_{rt:.2e}_extent"] = np.array(extent)
        saved[f"t_{rt:.2e}_peak"] = np.array(peak)

    fig.suptitle("Reconstructed cell density $u$ (peak is BANDWIDTH-SENSITIVE; "
                 "inset zooms the particle-detected core)", fontsize=7.5, y=1.04)
    stem = os.path.join(out_dir, "figure_u_snapshots")
    paths = cps.savefig_multi(fig, stem)
    np.savez(os.path.join(plot_data_dir, "figure_u_snapshots.npz"), **saved)
    print(f"[plot] wrote {paths}")


# ---------------------------------------------------------------------------
# (b) S_L2(t) + core radii R_0.5(t), R_0.8(t).
# ---------------------------------------------------------------------------
def fig_timeseries(diag, out_dir, plot_data_dir):
    t = diag["t"]
    S = diag["S_L2"]
    R05 = diag["R_0.5"]; R08 = diag["R_0.8"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=cps.fig_size(1.0, 0.40))

    ax1.plot(t, S, color="C3", marker="o", ms=2.5)
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$S_{K,N}(t)=\|P_K\mu_t^N\|_{L^2}$")
    ax1.set_title("Reconstructed $L^2$ norm\n(BANDWIDTH-SENSITIVE)", fontsize=7)
    for rt in REPORT_TIMES:
        if t.min() <= rt <= t.max():
            ax1.axvline(rt, color="gray", lw=0.5, ls=":")

    ax2.plot(t, R05, color="C0", marker="o", ms=2.5, label=r"$R_{0.5}(t)$")
    ax2.plot(t, R08, color="C1", marker="s", ms=2.5, label=r"$R_{0.8}(t)$")
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"core radius")
    ax2.set_title("Core radii\n(RECONSTRUCTION-FREE)", fontsize=7)
    ax2.legend()
    for rt in REPORT_TIMES:
        if t.min() <= rt <= t.max():
            ax2.axvline(rt, color="gray", lw=0.5, ls=":")

    fig.tight_layout()
    stem = os.path.join(out_dir, "figure_timeseries")
    paths = cps.savefig_multi(fig, stem)
    np.savez(os.path.join(plot_data_dir, "figure_timeseries.npz"),
             t=t, S_L2=S, R_0_5=R05, R_0_8=R08, report_times=np.array(REPORT_TIMES))
    print(f"[plot] wrote {paths}")


# ---------------------------------------------------------------------------
# (c) mass: M_u conserved, M_v vs exact law.
# ---------------------------------------------------------------------------
def fig_mass(diag, out_dir, plot_data_dir):
    t = diag["t"]
    Mu = diag["M_u"]; Mv = diag["M_v"]
    Mu0 = float(Mu[0]); Mv0 = float(Mv[0])
    Mv_law = Mu0 + (Mv0 - Mu0) * np.exp(-t)        # exact chemical mass law
    rel_err = np.abs(Mv - Mv_law) / np.maximum(np.abs(Mv_law), 1e-300)
    max_rel = float(np.max(rel_err))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=cps.fig_size(1.0, 0.40))
    ax1.plot(t, Mu, color="C0", marker="o", ms=2.5, label=r"$M_u(t)$ (particle)")
    ax1.axhline(Mu0, color="C0", lw=0.5, ls="--", label=r"$M_u(0)=10\pi$")
    ax1.plot(t, Mv, color="C3", marker="s", ms=2.5, label=r"$M_v(t)$ (particle)")
    ax1.plot(t, Mv_law, color="k", lw=1.0, ls=":",
             label=r"$M_u{+}(M_{v0}{-}M_u)e^{-t}$")
    ax1.set_xlabel(r"$t$"); ax1.set_ylabel("mass")
    ax1.set_title("Mass conservation + chemical mass law", fontsize=7)
    ax1.legend(fontsize=5)

    ax2.semilogy(t, np.maximum(rel_err, 1e-16), color="C3", marker="o", ms=2.5)
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$|M_v-M_v^{\rm law}|/|M_v^{\rm law}|$")
    ax2.set_title(f"$M_v$ law rel. err (max $={max_rel:.2e}$)", fontsize=7)

    fig.tight_layout()
    stem = os.path.join(out_dir, "figure_mass")
    paths = cps.savefig_multi(fig, stem)
    np.savez(os.path.join(plot_data_dir, "figure_mass.npz"),
             t=t, M_u=Mu, M_v=Mv, M_v_law=Mv_law, rel_err=rel_err,
             max_rel_err=np.array(max_rel))
    print(f"[plot] wrote {paths}  (M_v law max rel err = {max_rel:.3e})")
    return max_rel


def main():
    ap = argparse.ArgumentParser(description="plot LDG-style KS comparison figures")
    ap.add_argument("--results_dir", required=True,
                    help="dir with diag_*.csv and snapshots/")
    ap.add_argument("--out_dir", default=None,
                    help="figure output dir (default <results_dir>/figures)")
    args = ap.parse_args()

    cps.apply_style()
    out_dir = args.out_dir or os.path.join(args.results_dir, "figures")
    plot_data_dir = os.path.join(args.results_dir, "plot_data")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_data_dir, exist_ok=True)

    diag, diag_name = load_diag(args.results_dir)
    snaps = load_snapshots(args.results_dir)
    print(f"[plot] diag={diag_name}  snapshots at t={sorted(snaps.keys())}")

    fig_u_snapshots(snaps, out_dir, plot_data_dir)
    fig_timeseries(diag, out_dir, plot_data_dir)
    fig_mass(diag, out_dir, plot_data_dir)
    print(f"[plot] all figures in {out_dir}, plot data in {plot_data_dir}")


if __name__ == "__main__":
    main()
