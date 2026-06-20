"""plot_radial_state.py -- validation-closure Task C: u/v state-evolution figure from
SAVED particle clouds (never runs the solver).

Computes the 2D MASS MARGINALS  int u dz, int v dz  directly from the particles by a
mass-conserving periodic histogram on the full box [-L/2,L/2]^2:

    sigma(x,y) = (omega / (dx*dy)) * (# particles in the (x,y) bin),

so  sum_bins sigma * dx*dy = omega * N = M (the species mass).  This is a marginal, NOT
a 3D slice.  Row 1 = u marginal, row 2 = v marginal; columns = saved times.  Shared
LINEAR color scale within each row (vmin=0, vmax=row max); NO percentile clipping (the
true peak is shown and annotated).  A per-panel-normalized companion figure is also saved
to reveal shape evolution (explicitly labelled).

  python plot_radial_state.py --clouds <clouds_seed0.npz> --out_root <validation_closure dir>
  python plot_radial_state.py --from_plotdata <figure_radial_state_evolution.npz> --out_root ...
"""
import os, argparse, sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


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
    for variant in ("physical", "normalized"):
        fig, ax = plt.subplots(2, nt, figsize=(2.7 * nt, 5.6), squeeze=False)
        for ri, (data, lab, cmap) in enumerate([(U, r"$\int u\,dz$", "viridis"),
                                                (Vv, r"$\int v\,dz$", "magma")]):
            vmax = float(data.max()) if data.max() > 0 else 1.0
            for ci in range(nt):
                A = data[ci]
                if variant == "normalized":
                    A = A / (A.max() if A.max() > 0 else 1.0); vlim = 1.0
                else:
                    vlim = vmax
                im = ax[ri, ci].imshow(A, origin="lower", extent=ext, cmap=cmap,
                                       vmin=0, vmax=vlim, aspect="equal")
                ax[ri, ci].set_xlim(-disp, disp); ax[ri, ci].set_ylim(-disp, disp)
                if ri == 0:
                    ax[ri, ci].set_title(f"t={times[ci]:.2f}", fontsize=9)
                if ci == 0:
                    ax[ri, ci].set_ylabel(lab + ("\n(per-panel norm.)" if variant == "normalized"
                                                  else ""), fontsize=9)
                pk = S["peakU"][ci] if ri == 0 else S["peakV"][ci]
                ax[ri, ci].text(0.04, 0.92, f"peak={pk:.1f}", transform=ax[ri, ci].transAxes,
                                fontsize=6, color="w")
                ax[ri, ci].tick_params(labelsize=6)
            cb = fig.colorbar(im, ax=ax[ri, :].tolist(), fraction=0.012, pad=0.01)
            cb.ax.tick_params(labelsize=6)
        ttl = ("physical mass marginals (shared linear scale per row, true peak shown)"
               if variant == "physical" else
               "normalized shape (each panel scaled to its own peak)")
        fig.suptitle(f"Radial state evolution (M={S['M']:g}, K={S['K_dyn']}, seed={S['seed']}): "
                     + ttl, fontsize=11)
        suffix = "" if variant == "physical" else "_normalized"
        fd = os.path.join(out_root, "figures"); os.makedirs(fd, exist_ok=True)
        for e in ("pdf", "png"):
            fig.savefig(os.path.join(fd, f"{prefix}{suffix}.{e}"), dpi=200, bbox_inches="tight")
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
