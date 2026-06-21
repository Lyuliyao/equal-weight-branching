"""Create the manuscript two-panel radial Keller--Segel figure from saved diagnostics.

This script reads existing diagnostics.csv files only.  It never runs the solver.
The selected observable is R_0_5(t) / R_0_5(0), normalized per seed before
forming ensemble means and standard deviations.
"""
import argparse
import os
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", ".."))
if EXP not in sys.path:
    sys.path.insert(0, EXP)
from paper_style import apply_style, TEXTWIDTH_IN  # noqa: E402
from plot_radial_response import load, _match  # noqa: E402

apply_style()
mpl.rcParams.update({
    "axes.titlesize": 5.9,
    "axes.labelsize": 6.0,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "legend.fontsize": 5.1,
})


def normalized_seed_ensemble(runs, sel, col="R_0_5"):
    grp = [r for r in runs if _match(r, sel) and col in r["cols"]]
    if not grp:
        raise ValueError(f"no runs match {sel}")

    tg = grp[0]["cols"]["t"]
    ratios = []
    seeds = []
    for r in grp:
        y = np.interp(tg, r["cols"]["t"], r["cols"][col])
        if y.size == 0 or not np.isfinite(y[0]) or y[0] == 0:
            raise ValueError(f"invalid initial {col} for {sel}, seed={r['seed']}")
        ratios.append(y / y[0])
        seeds.append(r["seed"])

    Y = np.vstack(ratios)
    return {
        "t": tg,
        "mean": Y.mean(axis=0),
        "std": Y.std(axis=0) if Y.shape[0] > 1 else np.zeros_like(tg),
        "nseed": Y.shape[0],
        "seeds": np.array(seeds, dtype=int),
    }


def add_curve(ax, ens, label, color, band=True):
    t, m, s = ens["t"], ens["mean"], ens["std"]
    ax.plot(t, m, color=color, lw=1.25, label=label)
    if band and ens["nseed"] > 1:
        ax.fill_between(t, m - s, m + s, color=color, alpha=0.18, lw=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--base_N", type=int, default=100000)
    ap.add_argument("--base_K", type=int, default=12)
    ap.add_argument("--base_tau", type=float, default=1e-3)
    ap.add_argument("--delayed_M", type=float, default=96.0)
    ap.add_argument("--weak_M", type=float, default=72.0)
    ap.add_argument("--out_prefix", default="figure_radial_two_panel")
    args = ap.parse_args()

    runs = load(args.run_dir)
    if not runs:
        raise SystemExit(f"no radial diagnostics found in {args.run_dir}")

    K = args.base_K
    tau = args.base_tau
    base_N = args.base_N
    Md = args.delayed_M
    Mw = args.weak_M

    weak = normalized_seed_ensemble(
        runs, dict(label="weak", M=Mw, N=base_N, K=K, tau=tau)
    )
    delayed = normalized_seed_ensemble(
        runs, dict(label="delayed", M=Md, N=base_N, K=K, tau=tau)
    )

    refine_Ns = sorted({
        r["N"] for r in runs
        if r["label"] == "delayed"
        and np.isclose(r["M"], Md, rtol=1e-6, atol=1e-12)
        and r["K"] == K
        and np.isclose(r["tau"], tau, rtol=1e-6, atol=1e-12)
    })
    refine = {
        N: normalized_seed_ensemble(
            runs, dict(label="delayed", M=Md, N=N, K=K, tau=tau)
        )
        for N in refine_Ns
    }

    fig_w = 0.5 * TEXTWIDTH_IN
    fig, ax = plt.subplots(1, 2, figsize=(fig_w, 0.68 * fig_w), sharey=False)

    colors_a = {Mw: "#1f77b4", Md: "#d62728"}
    add_curve(ax[0], weak, rf"$M={Mw:g}$", colors_a[Mw])
    add_curve(ax[0], delayed, rf"$M={Md:g}$", colors_a[Md])
    ax[0].set_title("(a) Diffusion and\ndelayed focusing", linespacing=0.9, pad=1.0)
    ax[0].set_xlabel(r"$t$")
    ax[0].set_ylabel(r"$R_{0.5}(t)/R_{0.5}(0)$")
    ax[0].legend(loc="lower right", handlelength=1.1, labelspacing=0.2,
                 borderaxespad=0.2)

    colors_b = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    for color, N in zip(colors_b, refine_Ns):
        if N == 20000:
            label = r"$2\times10^4$"
        elif N == 100000:
            label = r"$10^5$"
        elif N == 320000:
            label = r"$3.2\times10^5$"
        else:
            label = rf"$N={N:g}$"
        add_curve(ax[1], refine[N], label, color)
    ax[1].set_title("(b) Particle-number\nrefinement", linespacing=0.9, pad=1.0)
    ax[1].set_xlabel(r"$t$")
    ax[1].legend(title=r"$N$", loc="upper right", handlelength=1.1,
                 labelspacing=0.2, borderaxespad=0.2, title_fontsize=5.1)

    for a in ax:
        a.axhline(1.0, color="0.35", lw=0.7, ls=":")
        a.set_xlim(0, 2.0)
        a.set_box_aspect(1)
        a.grid(True, alpha=0.28)
    ax[0].set_ylim(0.15, 4.05)
    ax[1].set_ylim(0.15, 1.35)

    fig.tight_layout(pad=0.18, w_pad=0.55)

    fig_dir = os.path.join(args.run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fig_dir, f"{args.out_prefix}.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    plot_dir = os.path.join(args.run_dir, "plot_data")
    os.makedirs(plot_dir, exist_ok=True)
    out = {
        "base_N": base_N,
        "K": K,
        "tau": tau,
        "weak_M": Mw,
        "delayed_M": Md,
        "panel_a_M72_t": weak["t"],
        "panel_a_M72_mean": weak["mean"],
        "panel_a_M72_std": weak["std"],
        "panel_a_M72_seeds": weak["seeds"],
        "panel_a_M96_t": delayed["t"],
        "panel_a_M96_mean": delayed["mean"],
        "panel_a_M96_std": delayed["std"],
        "panel_a_M96_seeds": delayed["seeds"],
        "refine_Ns": np.array(refine_Ns, dtype=int),
    }
    for N, ens in refine.items():
        key = f"panel_b_N{N}"
        out[f"{key}_t"] = ens["t"]
        out[f"{key}_mean"] = ens["mean"]
        out[f"{key}_std"] = ens["std"]
        out[f"{key}_seeds"] = ens["seeds"]
    np.savez(os.path.join(plot_dir, f"{args.out_prefix}.npz"), **out)

    print(f"wrote {fig_dir}/{args.out_prefix}.pdf/.png")
    print(f"baseline: N={base_N}, K={K}, tau={tau:g}")
    print(f"M={Mw:g}: {weak['nseed']} seeds, final ratio={weak['mean'][-1]:.3f}")
    print(f"M={Md:g}: {delayed['nseed']} seeds, final ratio={delayed['mean'][-1]:.3f}")
    for N in refine_Ns:
        ens = refine[N]
        print(f"refine N={N}: {ens['nseed']} seeds, final ratio={ens['mean'][-1]:.3f}")


if __name__ == "__main__":
    main()
