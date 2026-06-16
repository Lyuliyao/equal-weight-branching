"""plot_pp_ldg_particle_snapshots.py -- compact LDG-vs-particle KS solution
snapshot figure for paper section 5.4 (fully parabolic-parabolic, LDG-aligned).

PURE POST-PROCESSING.  Reads saved LDG snapshots.npz + particle snap_u_t*.npz;
it NEVER reruns a solver.  Two rows (LDG reference, particle method), one column
per reporting time.  This is qualitative visual context shown BEFORE the scalar
t_b / T_core diagnostics: it shows LDG and the particle-field method form the
SAME central concentrating core.  It makes no continuum blow-up-time claim.

  u_t - div(grad u - u grad v) = 0,   v_t - Delta v = u - v
  u0 = 840 exp(-84|x|^2),  v0 = 420 exp(-42|x|^2)   (Li-Shu-Yang IC)

Two colour-scaling versions are written:
  Version 1 (manuscript candidate): log10(1+u) with ONE shared global colorbar.
  Version 2 (diagnostic):           per-time-column linear scale, LDG/particle
                                     rows share the column scale.

Notes on the data this figure consumes:
  * The particle u-field is a CORE-ADAPTIVE WINDOW Fourier reconstruction on a
    small (~[-0.1,0.1]) window that tracks/shrinks to the core; the LDG field is
    on the full [-0.5,0.5] Neumann domain.  We plot a common, fixed, core-focused
    view window covered by the particle reconstruction at every reporting time.
  * The particle Fourier reconstruction can ring NEGATIVE near the steep core;
    for display we clip u at 0 (recorded in the plot-data README).  Core radii
    R_q and the centroid are reconstruction-free and are NOT clipped.

Usage (defaults reproduce the saved figure with NO solver run):
    python plot_pp_ldg_particle_snapshots.py
    python plot_pp_ldg_particle_snapshots.py --times 3e-5 6e-5 9e-5 1.2e-4   # after a rerun
"""
import os
import sys
import csv
import glob
import json
import argparse

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.abspath(os.path.join(_HERE, "..", ".."))
_REPO = os.path.abspath(os.path.join(_EXP, ".."))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
import common_plot_style as cps  # noqa: E402

KS = os.path.join(_REPO, "reference_results", "keller_segel_ldg_pp")
DEF_LDG = os.path.join(KS, "ldg_20260614_2074_b41f6d4_ldg_fixed_flux", "N320", "snapshots.npz")
DEF_PART = os.path.join(KS, "solver_field_tb_20260614_2256_0293dd7", "current_fourier_N320000_seed2")
DEF_OUT = os.path.join(_REPO, "paper", "cmame", "figure")
DEF_PLOTDATA = os.path.join(KS, "plot_data")
CMAP = "viridis"
# core mass-quantile radii to overlay (solid, dashed).  The snapshot run logs
# R_0.1/0.2/0.5/0.8/0.9 (ordered-particle-distance radii) but NOT R_0.3, so we use
# R_0.2 and R_0.5; R_0.5 is the half-mass radius the later T_core analysis uses.
QUANTILES = (0.2, 0.5)


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def load_ldg(npz_path):
    """Return dict time -> (x, y, u) for the LDG reference (full [-0.5,0.5] grid)."""
    z = np.load(npz_path)
    x = np.asarray(z["xc"], float)
    y = np.asarray(z["yc"], float)
    out = {}
    for k in z.files:
        if k.startswith("u_"):
            t = float(k[2:])
            out[t] = (x, y, np.asarray(z[k], float))
    return out


def load_particle(run_dir):
    """Return dict time -> dict(x_grid,y_grid,u_field,x_c,L) for the particle run."""
    out = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "snapshots", "snap_u_t*.npz"))):
        z = np.load(path)
        t = float(z["report_time"]) if "report_time" in z.files else float(z["t"])
        out[t] = {k: z[k] for k in z.files}
    return out


def load_diag(run_dir):
    cands = sorted(glob.glob(os.path.join(run_dir, "diag_*.csv")))
    if not cands:
        return None
    with open(cands[0]) as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for k in rows[0]:
        try:
            cols[k] = np.array([float(r[k]) for r in rows])
        except (ValueError, TypeError):
            cols[k] = np.array([r[k] for r in rows])
    return cols


def match_time(available, t, tol=2e-6):
    """Return the available key closest to t within tol, else None."""
    if not available:
        return None
    arr = np.array(list(available))
    j = int(np.argmin(np.abs(arr - t)))
    return float(arr[j]) if abs(arr[j] - t) <= tol else None


# ---------------------------------------------------------------------------
# diagnostics: mass-quantile radius of a gridded nonnegative field
# ---------------------------------------------------------------------------
def field_centroid_Rq(x, y, u, qs=QUANTILES):
    """Centroid and mass-quantile radii R_q of a gridded field u(y,x) (clip<0)."""
    uu = np.maximum(np.asarray(u, float), 0.0)
    X, Y = np.meshgrid(x, y)              # u indexed [iy, ix]
    m = uu.sum()
    if m <= 0:
        return (0.0, 0.0), {q: np.nan for q in qs}
    cx = float((X * uu).sum() / m)
    cy = float((Y * uu).sum() / m)
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).ravel()
    w = uu.ravel()
    order = np.argsort(r)
    cum = np.cumsum(w[order]) / m
    rr = r[order]
    Rq = {}
    for q in qs:
        idx = np.searchsorted(cum, q)
        Rq[q] = float(rr[min(idx, len(rr) - 1)])
    return (cx, cy), Rq


def view_field(x, y, u, W):
    """Mask a gridded field to the square view [-W,W]^2; return (xv, yv, uv)."""
    ix = np.where(np.abs(x) <= W)[0]
    iy = np.where(np.abs(y) <= W)[0]
    if ix.size == 0 or iy.size == 0:
        return x, y, np.asarray(u, float)
    return x[ix], y[iy], np.asarray(u, float)[np.ix_(iy, ix)]


def view_pos(x, y, u, W):
    """Nonnegative field values inside the [-W,W]^2 view (flat array)."""
    _, _, uv = view_field(x, y, u, W)
    return np.maximum(uv, 0.0)


def cloud_Rq(X, center, qs=QUANTILES):
    """Reconstruction-free mass-quantile radii from ordered particle distances
    (equal-mass particles), about `center`."""
    d = np.sort(np.sqrt(np.sum((np.asarray(X, float) - np.asarray(center, float)) ** 2, axis=1)))
    n = len(d)
    return {q: float(d[min(int(q * n), n - 1)]) for q in qs}


def kde_field(X, mass_per_particle, W, ngrid=221, h=None):
    """Nonnegative Gaussian KDE of the u-cloud on a uniform [-W,W]^2 grid.

    Deposits equal-mass particles to a histogram then convolves with an isotropic
    Gaussian of bandwidth h (density units, so u integrates to the enclosed mass).
    Nonnegative by construction -> no ringing, no negative values.  This is a
    post-hoc VISUALISATION of the evolved measure mu_u; the dynamics never uses a
    u reconstruction (drift uses grad v from the v-cloud).
    """
    if h is None:
        h = 0.06 * W
    edges = np.linspace(-W, W, ngrid + 1)
    xs = 0.5 * (edges[:-1] + edges[1:])
    X = np.asarray(X, float)
    H, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=[edges, edges])   # H[ix, iy]
    dx = xs[1] - xs[0]
    rho = H * float(mass_per_particle) / (dx * dx)
    sigma = h / dx
    try:
        from scipy.ndimage import gaussian_filter
        field = gaussian_filter(rho, sigma=sigma, mode="constant")
    except Exception:                                                # FFT fallback
        ny, nx = rho.shape
        ky = np.fft.fftfreq(ny)[:, None]; kx = np.fft.fftfreq(nx)[None, :]
        g = np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * (kx ** 2 + ky ** 2))
        field = np.real(np.fft.ifft2(np.fft.fft2(rho) * g))
    return xs, xs.copy(), field.T                                    # -> [iy, ix]


# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------
def build_panels(ldg, part, times):
    """Assemble per-(row,time) FULL fields + overlays; choose the view window.

    Full fields are imshown on their native extent and the axes are clipped to the
    common [-W,W] view, so every panel fills its box (no white margin) and the two
    rows share an identical physical scale.
    """
    # view half-width = just inside the smallest particle reconstruction window
    Lmin = min(float(part[t]["L"]) for t in times)
    W = 0.95 * Lmin
    panels = {"ldg": {}, "part": {}}
    overlays = {"ldg": {}, "part": {}}
    for t in times:
        lx, ly, lu = ldg[t]
        panels["ldg"][t] = (lx, ly, lu)              # FULL LDG field ([-0.5,0.5])
        c, Rq = field_centroid_Rq(lx, ly, lu)        # full field for stable R_q
        overlays["ldg"][t] = (c, Rq)

        p = part[t]
        px, py, pu = np.asarray(p["x_grid"], float), np.asarray(p["y_grid"], float), \
            np.asarray(p["u_field"], float)
        panels["part"][t] = (px, py, pu)             # FULL native-window field
    return W, panels, overlays


def add_rq_overlay(ax, center, Rq, color="white"):
    cx, cy = center
    styles = ["-", "--", ":"]
    for q, ls in zip(QUANTILES, styles):
        rq = Rq.get(q, np.nan)
        if np.isfinite(rq) and rq > 0:
            ax.add_patch(Circle((cx, cy), rq, fill=False, ec=color, lw=0.6, ls=ls,
                                alpha=0.85))


ROW_LABEL = {"ldg": "LDG\nreference", "part": "particle\nmethod"}


def _fmt_t(t):
    return rf"$t={t*1e4:.1f}\times10^{{-4}}$" if t >= 1e-4 else rf"$t={t*1e5:.0f}\times10^{{-5}}$"


def figure_log(W, panels, overlays, times, out_stem, plotdata, rq=True):
    """Version 1: log10(1+u), one shared global colorbar."""
    cps.apply_style()
    n = len(times)
    gmax = 0.0
    for row in ("ldg", "part"):
        for t in times:
            x, y, u = panels[row][t]
            vp = view_pos(x, y, u, W)
            gmax = max(gmax, float(vp.max()) if vp.size else 0.0)
    vmin, vmax = 0.0, float(np.log10(1.0 + gmax))

    fw = cps.TEXTWIDTH_IN
    fh = fw * (2.0 / n) * 1.04 + 0.30
    fig, axes = plt.subplots(2, n, figsize=(fw, fh), squeeze=False)
    im = None
    for i, row in enumerate(("ldg", "part")):
        for j, t in enumerate(times):
            ax = axes[i][j]
            x, y, u = panels[row][t]
            d = np.log10(1.0 + np.maximum(u, 0.0))
            im = ax.imshow(d, origin="lower",
                           extent=[x.min(), x.max(), y.min(), y.max()],
                           vmin=vmin, vmax=vmax, cmap=CMAP, aspect="equal",
                           interpolation="nearest" if row == "ldg" else "bilinear")
            ax.set_xlim(-W, W); ax.set_ylim(-W, W)
            ax.set_xticks([]); ax.set_yticks([])
            if rq:
                c, Rq = overlays[row][t] if row == "ldg" else overlays["part"][t]
                add_rq_overlay(ax, c, Rq)
            if i == 0:
                ax.set_title(_fmt_t(t), fontsize=8, pad=3)
            if j == 0:
                ax.set_ylabel(ROW_LABEL[row], fontsize=8)
    fig.subplots_adjust(left=0.085, right=0.88, top=0.90, bottom=0.03, wspace=0.06, hspace=0.06)
    cax = fig.add_axes([0.895, 0.06, 0.02, 0.84])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\log_{10}(1+u)$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cps.savefig_multi(fig, out_stem, close=False)
    plt.close(fig)
    np.savez(plotdata,
             times=np.array(times), W=W, version="log10(1+u) shared global",
             vmin=vmin, vmax=vmax,
             **{f"ldg_u_{t:.2e}": panels["ldg"][t][2] for t in times},
             **{f"part_u_{t:.2e}": panels["part"][t][2] for t in times},
             **{f"ldg_x_{t:.2e}": panels["ldg"][t][0] for t in times},
             **{f"ldg_y_{t:.2e}": panels["ldg"][t][1] for t in times},
             **{f"part_x_{t:.2e}": panels["part"][t][0] for t in times},
             **{f"part_y_{t:.2e}": panels["part"][t][1] for t in times})
    return vmax


def figure_linear(W, panels, overlays, times, out_stem, rq=True):
    """Version 2: per-column linear scale shared by the two rows; one colorbar/col."""
    cps.apply_style()
    n = len(times)
    col_vmax = {}
    for t in times:
        lview = view_pos(*panels["ldg"][t], W)
        pview = view_pos(*panels["part"][t], W)
        # high percentile of the particle field to avoid a lone ringing spike
        pv = float(np.percentile(pview, 99.9)) if pview.size else 0.0
        col_vmax[t] = float(max(lview.max(), pv))

    fw = cps.TEXTWIDTH_IN
    fh = fw * (2.0 / n) * 1.04 + 0.52
    fig, axes = plt.subplots(2, n, figsize=(fw, fh), squeeze=False)
    ims = {}
    for i, row in enumerate(("ldg", "part")):
        for j, t in enumerate(times):
            ax = axes[i][j]
            x, y, u = panels[row][t]
            ims[(i, j)] = ax.imshow(np.maximum(u, 0.0), origin="lower",
                                    extent=[x.min(), x.max(), y.min(), y.max()],
                                    vmin=0, vmax=col_vmax[t], cmap=CMAP, aspect="equal",
                                    interpolation="nearest" if row == "ldg" else "bilinear")
            ax.set_xlim(-W, W); ax.set_ylim(-W, W)
            ax.set_xticks([]); ax.set_yticks([])
            if rq:
                c, Rq = overlays[row][t] if row == "ldg" else overlays["part"][t]
                add_rq_overlay(ax, c, Rq)
            if i == 0:
                ax.set_title(_fmt_t(t), fontsize=8, pad=3)
            if j == 0:
                ax.set_ylabel(ROW_LABEL[row], fontsize=8)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.14, wspace=0.06, hspace=0.06)
    # one slim horizontal colorbar under each column (column-wise scale)
    for j, t in enumerate(times):
        p = axes[1][j].get_position()
        cax = fig.add_axes([p.x0, 0.075, p.width, 0.025])
        cb = fig.colorbar(ims[(1, j)], cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=6)
        cb.set_label(rf"$u$  (max ${col_vmax[t]:.2g}$)", fontsize=6.5)
    cps.savefig_multi(fig, out_stem, close=False)
    plt.close(fig)
    return col_vmax


def write_readme(path, args, times, W, vmax_log, col_vmax, part_dir, ldg_npz):
    with open(path, "w") as f:
        f.write("# ks_pp_ldg_particle_snapshots -- plot data + provenance\n\n")
        f.write("Compact LDG-vs-particle KS solution snapshots for section 5.4.\n")
        f.write("Pure post-processing of saved fields; no solver was run.\n\n")
        f.write("## Sources\n")
        f.write(f"- LDG reference: `{os.path.relpath(ldg_npz, _REPO)}` (N=320, Neumann, full [-0.5,0.5]).\n")
        f.write(f"- Particle: `{os.path.relpath(part_dir, _REPO)}`\n")
        f.write("  (current_fourier, N=320000, K=10, tau=2e-7, q_window=0.8; REPRESENTATIVE seed, "
                "the one reaching every reporting time).  Quantitative t_b / T_core diagnostics "
                "use seed bootstrap, NOT this single seed.\n\n")
        f.write("## Reporting times\n")
        f.write("  " + ", ".join(f"{t:.2e}" for t in times) + "\n\n")
        f.write("## Display conventions\n")
        f.write(f"- Common fixed view window: [-{W:.4f}, {W:.4f}]^2, centered at origin, "
                "chosen just inside the smallest particle reconstruction window so the particle "
                "field covers the view at every time.\n")
        f.write("- Particle field is a core-adaptive WINDOW Fourier reconstruction; the LDG field "
                "is the full-domain positivity-preserving solution.\n")
        f.write("- The particle Fourier reconstruction rings NEGATIVE near the steep core; for "
                "DISPLAY ONLY u is clipped at 0 (max(u,0)).  R_q radii and the centroid are "
                "reconstruction-free and are not clipped.\n")
        f.write(f"- Version 1 (manuscript): log10(1+u), one shared colorbar, vmax={vmax_log:.3f}.\n")
        f.write("- Version 2 (diagnostic, *_linear_colscale): per-column linear scale, "
                "particle vmax taken at the 99.9 percentile to avoid a lone ringing spike; "
                "column maxima:\n")
        for t in times:
            f.write(f"    t={t:.2e}: vmax={col_vmax[t]:.4g}\n")
        f.write(f"- Mass-quantile core circles overlaid: R_{QUANTILES[0]:g} (solid) and "
                f"R_{QUANTILES[1]:g} (dashed).  Particle radii from its diag_*.csv (ordered "
                "particle distances); LDG radii from field quadrature about the field centroid.  "
                "NOTE: the snapshot run did not log R_0.3, so R_0.5 (the half-mass radius used by "
                "the T_core analysis) is shown as the second circle instead of R_0.3.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ldg_npz", default=DEF_LDG)
    ap.add_argument("--particle_run_dir", default=DEF_PART)
    ap.add_argument("--times", type=float, nargs="+", default=None,
                    help="reporting times; default = intersection of LDG & particle saved times")
    ap.add_argument("--out_dir", default=DEF_OUT)
    ap.add_argument("--plot_data_dir", default=DEF_PLOTDATA)
    ap.add_argument("--no_rq", action="store_true")
    ap.add_argument("--particle_recon", choices=["auto", "spectral", "kde"], default="auto",
                    help="auto: KDE if the snapshot saved the cloud (X_u), else the saved "
                         "spectral P_K-mu field.  kde: force nonnegative Gaussian KDE.  "
                         "spectral: force the saved current_fourier readout.")
    ap.add_argument("--kde_h_frac", type=float, default=0.06,
                    help="KDE bandwidth as a fraction of the view half-width W.")
    ap.add_argument("--kde_ngrid", type=int, default=221)
    args = ap.parse_args()

    ldg = load_ldg(args.ldg_npz)
    part = load_particle(args.particle_run_dir)
    diag = load_diag(args.particle_run_dir)

    # resolve requested times present in BOTH sources
    if args.times is None:
        req = sorted(set(ldg) & set(part))
    else:
        req = list(args.times)
    used, missing = [], []
    ldg_m, part_m = {}, {}
    for t in req:
        lt = match_time(ldg.keys(), t)
        pt = match_time(part.keys(), t)
        if lt is None or pt is None:
            missing.append((t, lt is not None, pt is not None))
            continue
        ldg_m[t] = ldg[lt]
        part_m[t] = part[pt]
        used.append(t)
    used = sorted(used)
    if missing:
        print("[plot] MISSING reporting times (skipped):")
        for t, has_l, has_p in missing:
            print(f"    t={t:.2e}: LDG={'ok' if has_l else 'MISSING'}, "
                  f"particle={'ok' if has_p else 'MISSING'}")
    if not used:
        print("[plot] no times present in BOTH LDG and particle data; nothing to plot.")
        return
    print(f"[plot] plotting times: {', '.join(f'{t:.2e}' for t in used)}")

    # view half-width (needed up front so the KDE grid can be built on it)
    W = 0.95 * min(float(part_m[t]["L"]) for t in used)

    # decide the particle reconstruction: nonnegative KDE of the saved cloud (X_u),
    # or the saved spectral P_K-mu readout.
    has_cloud = all("X_u" in part_m[t] for t in used)
    use_kde = args.particle_recon == "kde" or (args.particle_recon == "auto" and has_cloud)
    if args.particle_recon == "kde" and not has_cloud:
        print("[plot] --particle_recon kde requested but the snapshots have no cloud (X_u); "
              "falling back to the spectral field.")
        use_kde = False
    recon_label = "KDE(cloud)" if use_kde else "spectral P_K-mu"
    print(f"[plot] particle reconstruction: {recon_label}")

    overlays_part = {}
    for t in used:
        snap = part_m[t]
        if use_kde:
            X = np.asarray(snap["X_u"], float)
            mpp = float(snap["mass_per_particle_u"]) if "mass_per_particle_u" in snap \
                else float(snap["M_u"]) / max(len(X), 1)
            xs, ys, fld = kde_field(X, mpp, W, ngrid=args.kde_ngrid, h=args.kde_h_frac * W)
            snap = dict(snap)
            snap["x_grid"], snap["y_grid"], snap["u_field"] = xs, ys, fld
            part_m[t] = snap
            c = (float(snap["x_c"][0]), float(snap["x_c"][1]))
            Rq = cloud_Rq(X, c)                       # reconstruction-free, ordered distances
        elif diag is not None and "t" in diag:
            j = int(np.argmin(np.abs(diag["t"] - t)))
            c = (float(diag["xc_x"][j]), float(diag["xc_y"][j]))
            Rq = {q: (float(diag[f"R_{q:g}"][j]) if f"R_{q:g}" in diag else np.nan)
                  for q in QUANTILES}
        else:
            c, Rq = (float(snap["x_c"][0]), float(snap["x_c"][1])), {q: np.nan for q in QUANTILES}
        overlays_part[t] = (c, Rq)

    W, panels, overlays = build_panels(ldg_m, part_m, used)
    overlays["part"] = overlays_part

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.plot_data_dir, exist_ok=True)
    stem_log = os.path.join(args.out_dir, "ks_pp_ldg_particle_snapshots")
    stem_lin = os.path.join(args.out_dir, "ks_pp_ldg_particle_snapshots_linear_colscale")
    pdata = os.path.join(args.plot_data_dir, "ks_pp_ldg_particle_snapshots_plot_data.npz")

    vmax_log = figure_log(W, panels, overlays, used, stem_log, pdata, rq=not args.no_rq)
    col_vmax = figure_linear(W, panels, overlays, used, stem_lin, rq=not args.no_rq)
    write_readme(os.path.join(args.plot_data_dir, "ks_pp_ldg_particle_snapshots_README.md"),
                 args, used, W, vmax_log, col_vmax, args.particle_run_dir, args.ldg_npz)
    print(f"[plot] wrote {stem_log}.pdf/.png")
    print(f"[plot] wrote {stem_lin}.pdf/.png")
    print(f"[plot] wrote {pdata}")
    print(f"[plot] view half-width W = {W:.4f}; log vmax = {vmax_log:.3f}")


if __name__ == "__main__":
    main()
