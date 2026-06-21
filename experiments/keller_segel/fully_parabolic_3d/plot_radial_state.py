"""plot_radial_state.py -- validation-closure Task C: u/v state-evolution figure from
SAVED particle clouds (never runs the solver).

Computes the 2D MASS MARGINALS  int u dz, int v dz  directly from the particles by a
mass-conserving periodic histogram on the full box [-L/2,L/2]^2:

    sigma(x,y) = (omega / (dx*dy)) * (# particles in the (x,y) bin),

so  sum_bins sigma * dx*dy = omega * N = M (the species mass).  This is a marginal, NOT
a 3D slice.  Row 1 = u marginal, row 2 = v marginal; columns = saved times.  Each
panel uses a LINEAR color scale (vmin=0, vmax=panel max); NO percentile clipping, and
panel maxima are annotated.  A per-panel-normalized companion figure is also saved
to reveal shape evolution (explicitly labelled).

  python plot_radial_state.py --clouds <clouds_seed0.npz> --out_root <validation_closure dir>
  python plot_radial_state.py --from_plotdata <figure_radial_state_evolution.npz> --out_root ...
"""
import os, argparse, sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", ".."))
if EXP not in sys.path:
    sys.path.insert(0, EXP)
from paper_style import apply_style, TEXTWIDTH_IN  # noqa: E402

apply_style()
mpl.rcParams.update({"axes.titlesize": 7.0, "axes.labelsize": 7.0,
                     "xtick.labelsize": 6.0, "ytick.labelsize": 6.0})


def _fmt_tick(x):
    x = float(x)
    if abs(x) < 1e-12:
        return "0"
    if abs(x) >= 1000:
        return f"{x / 1000:.1f}k"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}".rstrip("0").rstrip(".")
    if abs(x) >= 1:
        return f"{x:.2g}"
    return f"{x:.1g}"


def marginal(P, omega, L, bins, rng):
    """int (.) dz mass-marginal histogram on [-rng,rng]^2 over the FULL box; returns
    (sigma (bins,bins), xedges, in_window_mass_fraction, total_mass)."""
    if P.shape[0] == 0:
        e = np.linspace(-rng, rng, bins + 1)
        return np.zeros((bins, bins)), e, 0.0, 0.0
    H, xe, ye = np.histogram2d(P[:, 0], P[:, 1], bins=bins, range=[[-rng, rng], [-rng, rng]])
    dx = (2 * rng) / bins
    sigma = H.T * (omega / (dx * dx))                # transpose -> [y,x] for imshow origin lower
    total = omega * P.shape[0]
    in_win = float(H.sum() * omega)
    return sigma, xe, in_win / max(total, 1e-30), total


def build(clouds_npz, bins, rng):
    d = np.load(clouds_npz)
    times = d["times"]; omega = float(d["omega"]); L = float(d["L"])
    nt = len(times)
    U, Vv, fracU, fracV, peakU, peakV = [], [], [], [], [], []
    for i in range(nt):
        su, xe, fu, _ = marginal(d[f"u_{i}"].astype(float), omega, L, bins, rng)
        sv, _, fv, _ = marginal(d[f"v_{i}"].astype(float), omega, L, bins, rng)
        U.append(su); Vv.append(sv); fracU.append(fu); fracV.append(fv)
        peakU.append(float(su.max())); peakV.append(float(sv.max()))
    return dict(times=times, U=np.array(U), V=np.array(Vv), xe=xe, rng=rng, bins=bins,
                omega=omega, L=L, fracU=np.array(fracU), fracV=np.array(fracV),
                peakU=np.array(peakU), peakV=np.array(peakV),
                M=float(d["M"]), sigma0=float(d["sigma"]), K_dyn=int(d["K_dyn"]),
                seed=int(d["seed"]))


def render(S, out_root, prefix, disp=3.0):
    times, U, Vv = S["times"], S["U"], S["V"]
    nt = len(times)
    ext = [-S["rng"], S["rng"], -S["rng"], S["rng"]]
    fig_w = TEXTWIDTH_IN
    fig_h = 0.43 * TEXTWIDTH_IN
    left = 0.46
    right = 0.08
    panel_gap = 0.070
    cbar_pad = 0.012
    cbar_w = 0.030
    panel = (fig_w - left - right - (nt - 1) * panel_gap
             - nt * (cbar_pad + cbar_w)) / nt
    row_gap = 0.09
    bottom = 0.18
    top_y = bottom + panel + row_gap
    for variant in ("physical", "normalized"):
        fig = plt.figure(figsize=(fig_w, fig_h))
        for ri, (data, lab, cmap) in enumerate([(U, r"$\int u\,dz$", "viridis"),
                                                (Vv, r"$\int v\,dz$", "magma")]):
            y = top_y if ri == 0 else bottom
            fig.text(0.024, (y + 0.5 * panel) / fig_h,
                     lab + ("\n(norm.)" if variant == "normalized" else ""),
                     rotation=90, va="center", ha="center", fontsize=7)
            for ci in range(nt):
                x = left + ci * (panel + cbar_pad + cbar_w + panel_gap)
                ax = fig.add_axes([x / fig_w, y / fig_h, panel / fig_w, panel / fig_h])
                cax = fig.add_axes([(x + panel + cbar_pad) / fig_w, y / fig_h,
                                    cbar_w / fig_w, panel / fig_h])
                A = data[ci]
                panel_max = float(A.max())
                if variant == "normalized":
                    A = A / (A.max() if A.max() > 0 else 1.0); vlim = 1.0
                else:
                    vlim = panel_max if panel_max > 0 else 1.0
                im = ax.imshow(A, origin="lower", extent=ext, cmap=cmap,
                               vmin=0, vmax=vlim, aspect="equal")
                ax.set_xlim(-disp, disp); ax.set_ylim(-disp, disp)
                if ri == 0:
                    ax.set_title(rf"$t={times[ci]:.2f}$", fontsize=7, pad=2)
                ax.text(0.04, 0.91, f"max={_fmt_tick(panel_max)}", transform=ax.transAxes,
                        fontsize=4.8, color="w",
                        bbox=dict(facecolor="k", edgecolor="none", alpha=0.45, pad=0.6))
                ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
                if ri == 0:
                    ax.set_xticklabels([])
                if ci != 0:
                    ax.set_yticklabels([])
                ax.tick_params(labelsize=5.4, length=2, pad=1)
                ax.grid(False)
                cb = fig.colorbar(im, cax=cax)
                cb.set_ticks([])
        suffix = "" if variant == "physical" else "_normalized"
        fd = os.path.join(out_root, "figures"); os.makedirs(fd, exist_ok=True)
        for e in ("pdf", "png"):
            with mpl.rc_context({"savefig.bbox": "standard"}):
                fig.savefig(os.path.join(fd, f"{prefix}{suffix}.{e}"), dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clouds", help="snapshots/clouds_seed0.npz")
    ap.add_argument("--from_plotdata", help="regenerate from saved plot_data npz")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--bins", type=int, default=240)
    ap.add_argument("--range", type=float, default=6.0, help="histogram half-width (full box L/2)")
    ap.add_argument("--disp", type=float, default=3.0, help="display half-width (zoom)")
    ap.add_argument("--prefix", default="figure_radial_state_evolution")
    args = ap.parse_args()

    if args.from_plotdata:
        d = np.load(args.from_plotdata, allow_pickle=True)
        S = {k: d[k] for k in d.files}
        S["times"] = np.asarray(S["times"]); S["rng"] = float(S["rng"])
    else:
        if not args.clouds:
            raise SystemExit("need --clouds or --from_plotdata")
        S = build(args.clouds, args.bins, args.range)
        pdd = os.path.join(args.out_root, "plot_data"); os.makedirs(pdd, exist_ok=True)
        np.savez_compressed(os.path.join(pdd, f"{args.prefix}.npz"),
                            times=S["times"], U=S["U"].astype(np.float32),
                            V=S["V"].astype(np.float32), xe=S["xe"], rng=S["rng"],
                            bins=S["bins"], omega=S["omega"], L=S["L"], M=S["M"],
                            sigma0=S["sigma0"], K_dyn=S["K_dyn"], seed=S["seed"],
                            fracU=S["fracU"], fracV=S["fracV"], peakU=S["peakU"],
                            peakV=S["peakV"], disp=args.disp)
    render(S, args.out_root, args.prefix, disp=float(args.disp if "disp" not in S else S["disp"]))
    print(f"times={np.asarray(S['times'])}")
    print(f"u in-window mass frac (disp): see fracU; peak u={np.asarray(S['peakU'])}")
    print(f"peak v={np.asarray(S['peakV'])}")
    print(f"wrote figures/{args.prefix}.pdf/.png (+ _normalized) and plot_data/{args.prefix}.npz")


if __name__ == "__main__":
    main()
