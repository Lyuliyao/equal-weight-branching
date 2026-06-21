"""Create the manuscript tetrahedral 3D Keller--Segel dynamics figure.

The script reads existing diagnostics.csv files only.  It never runs the
solver.  The main-panel observables are restricted to centroid trajectories,
normalized minimum centroid separation, and normalized mean per-cluster
half-mass radius.
"""
import argparse
import itertools
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
from paper_style import TEXTWIDTH_IN, apply_style  # noqa: E402
import vc_load as V  # noqa: E402

apply_style()
mpl.rcParams.update({
    "axes.titlesize": 6.9,
    "axes.labelsize": 6.7,
    "xtick.labelsize": 5.8,
    "ytick.labelsize": 5.8,
    "legend.fontsize": 5.7,
    "lines.linewidth": 1.05,
    "lines.markersize": 3.0,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})

ACTIVE_COLOR = "#c44e52"
CONTROL_COLOR = "#4c78a8"
CLUSTER_COLORS = ("#4c78a8", "#f58518", "#54a24b", "#b279a2")


def add_derived(runs):
    for r in runs:
        c = r["cols"]
        r05 = [f"R05_c{m}" for m in range(4) if f"R05_c{m}" in c]
        if len(r05) == 4:
            c["R05_mean"] = np.mean([c[k] for k in r05], axis=0)
        amin = [f"A_c{m}" for m in range(4) if f"A_c{m}" in c]
        if len(amin) == 4:
            c["A_min"] = np.min([c[k] for k in amin], axis=0)
    return runs


def require_baseline(runs, sel, label):
    grp = sorted(V.select(runs, sel), key=lambda r: r["seed"])
    if not grp:
        raise SystemExit(f"no {label} runs match {sel}")
    return grp


def normalized_ensemble(runs, sel, col, tgrid=None, tmax=None):
    grp = require_baseline(runs, sel, sel.get("arm", "selected"))
    if any(col not in r["cols"] for r in grp):
        missing = [r["seed"] for r in grp if col not in r["cols"]]
        raise SystemExit(f"missing column {col} for seeds {missing}")

    tg = np.asarray(grp[0]["cols"]["t"] if tgrid is None else tgrid, dtype=float)
    if tmax is not None:
        tg = tg[tg <= tmax + 10.0 * np.finfo(float).eps]
    if tg.size == 0:
        raise SystemExit(f"empty time grid for {col}")

    ratios = []
    seeds = []
    for r in grp:
        y = np.interp(tg, r["cols"]["t"], r["cols"][col])
        y0 = float(y[0])
        if not np.isfinite(y0) or y0 == 0.0:
            raise SystemExit(f"invalid initial {col} for seed {r['seed']}")
        ratios.append(y / y0)
        seeds.append(r["seed"])

    Y = np.vstack(ratios)
    return {
        "t": tg,
        "mean": Y.mean(axis=0),
        "std": Y.std(axis=0) if Y.shape[0] > 1 else np.zeros_like(tg),
        "nseed": Y.shape[0],
        "seeds": np.array(seeds, dtype=int),
    }


def reliable_tmax(control_runs, threshold):
    if any("A_min" not in r["cols"] for r in control_runs):
        missing = [r["seed"] for r in control_runs if "A_min" not in r["cols"]]
        raise SystemExit(
            "control reliability A_min is required for panel (b); "
            f"missing for seeds {missing}"
        )

    tg = np.asarray(control_runs[0]["cols"]["t"], dtype=float)
    A = np.vstack([
        np.interp(tg, r["cols"]["t"], r["cols"]["A_min"])
        for r in control_runs
    ])
    ok = np.all(A >= threshold, axis=0)
    if not bool(ok[0]):
        return float(tg[0])
    first_bad = np.flatnonzero(~ok)
    if first_bad.size:
        return float(tg[max(int(first_bad[0]) - 1, 0)])
    return float(tg[-1])


def add_curve(ax, ens, label, color, band=True):
    t, mean, std = ens["t"], ens["mean"], ens["std"]
    ax.plot(t, mean, color=color, lw=1.25, label=label)
    if band and ens["nseed"] > 1:
        lo = mean - std
        hi = mean + std
        if ax.get_yscale() == "log":
            positive = mean[mean > 0.0]
            floor = positive.min() * 0.25 if positive.size else 1.0e-6
            lo = np.maximum(lo, floor)
        ax.fill_between(t, lo, hi, color=color, alpha=0.18, lw=0)


def set_equal_3d_limits(ax, xyz, pad=0.08):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = radius * (1.0 + pad) if radius > 0.0 else 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def plot_centroid_panel(ax, run):
    c = run["cols"]
    trajectories = []
    initials = []
    finals = []
    for m in range(4):
        xyz = np.column_stack((c[f"c{m}_x"], c[f"c{m}_y"], c[f"c{m}_z"]))
        trajectories.append(xyz)
        initials.append(xyz[0])
        finals.append(xyz[-1])

    initials = np.asarray(initials)
    for i, j in itertools.combinations(range(4), 2):
        ax.plot(
            [initials[i, 0], initials[j, 0]],
            [initials[i, 1], initials[j, 1]],
            [initials[i, 2], initials[j, 2]],
            color="0.76",
            lw=0.55,
            zorder=0,
        )

    for m, xyz in enumerate(trajectories):
        color = CLUSTER_COLORS[m]
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, lw=1.15,
                label=rf"$m={m + 1}$")
        ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], s=15, marker="o",
                   color=color, edgecolor="0.15", linewidth=0.25, depthshade=False)
        ax.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], s=22, marker="^",
                   color=color, edgecolor="0.15", linewidth=0.25, depthshade=False)

    all_xyz = np.vstack(trajectories)
    set_equal_3d_limits(ax, all_xyz)
    ax.view_init(elev=18, azim=38)
    ax.set_title("(a) Cluster-centroid\ntrajectories", linespacing=0.9, pad=1.0)
    ax.set_xlabel(r"$x$", labelpad=-4)
    ax.set_ylabel(r"$y$", labelpad=-4)
    ax.set_zlabel(r"$z$", labelpad=-6)
    ax.tick_params(axis="both", which="major", pad=-2)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), handlelength=1.0,
              borderaxespad=0.0, labelspacing=0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="directory containing tetra baseline diagnostics")
    ap.add_argument("--out_root", required=True,
                    help="reference result root for figures/ and plot_data/")
    ap.add_argument("--paper_out", default=None,
                    help="optional manuscript PDF path, e.g. paper/figure/ks3d_tetra.pdf")
    ap.add_argument("--base_N", type=int, default=80000)
    ap.add_argument("--base_K", type=int, default=12)
    ap.add_argument("--base_tau", type=float, default=1e-3)
    ap.add_argument("--M", type=float, default=240.0)
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--representative_seed", type=int, default=0)
    ap.add_argument("--reliability_threshold", type=float, default=0.2)
    ap.add_argument("--out_prefix", default="figure_tetra_dynamics")
    args = ap.parse_args()

    runs = add_derived(V.load_runs(args.run_dir))
    if not runs:
        raise SystemExit(f"no tetra diagnostics found in {args.run_dir}")

    base = dict(N=args.base_N, K=args.base_K, tau=args.base_tau,
                M=args.M, a=args.a)
    active_sel = dict(arm="active", **base)
    control_sel = dict(arm="control", **base)
    active_runs = require_baseline(runs, active_sel, "active")
    control_runs = require_baseline(runs, control_sel, "control")

    active_rep = next(
        (r for r in active_runs if r["seed"] == args.representative_seed),
        None,
    )
    if active_rep is None:
        seeds = [r["seed"] for r in active_runs]
        raise SystemExit(
            f"representative active seed {args.representative_seed} not found; "
            f"available seeds: {seeds}"
        )

    t_reliable = reliable_tmax(control_runs, args.reliability_threshold)
    control_t = control_runs[0]["cols"]["t"]
    panel_b_t = control_t[control_t <= t_reliable + 10.0 * np.finfo(float).eps]

    active_d = normalized_ensemble(runs, active_sel, "d_min", tgrid=panel_b_t)
    control_d = normalized_ensemble(runs, control_sel, "d_min", tgrid=panel_b_t)
    active_r = normalized_ensemble(runs, active_sel, "R05_mean")
    control_r = normalized_ensemble(runs, control_sel, "R05_mean")

    fig = plt.figure(figsize=(1.018 * TEXTWIDTH_IN, 0.49 * TEXTWIDTH_IN),
                     constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=(1.08, 1.0, 1.0),
                          wspace=0.18)
    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    plot_centroid_panel(ax0, active_rep)

    add_curve(ax1, active_d, r"chemotaxis, $\chi=1$", ACTIVE_COLOR)
    add_curve(ax1, control_d, r"diffusion control, $\chi=0$", CONTROL_COLOR)
    ax1.axhline(1.0, color="0.35", lw=0.65, ls=":")
    ax1.set_title("(b) Inter-cluster\nattraction", linespacing=0.9, pad=1.0)
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$d_{\min}(t)/d_{\min}(0)$")
    ax1.set_xlim(float(panel_b_t[0]), float(panel_b_t[-1]))
    ax1.legend(loc="lower left", handlelength=1.2, labelspacing=0.18,
               borderaxespad=0.2)

    ax2.set_yscale("log")
    add_curve(ax2, active_r, r"chemotaxis, $\chi=1$", ACTIVE_COLOR)
    add_curve(ax2, control_r, r"diffusion control, $\chi=0$", CONTROL_COLOR)
    ax2.axhline(1.0, color="0.35", lw=0.65, ls=":")
    ax2.set_title("(c) Individual-cluster\nfocusing", linespacing=0.9, pad=1.0)
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$R_{0.5}(t)/R_{0.5}(0)$")
    ax2.set_xlim(float(active_r["t"][0]), float(active_r["t"][-1]))
    ax2.set_ylim(0.32, 14.0)
    ax2.text(control_r["t"][-1] * 0.98, control_r["mean"][-1],
             r"$\chi=0$", color=CONTROL_COLOR, ha="right", va="bottom",
             fontsize=5.7)
    ax2.text(active_r["t"][-1] * 0.98, active_r["mean"][-1],
             r"$\chi=1$", color=ACTIVE_COLOR, ha="right", va="top",
             fontsize=5.7)

    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.28)
        ax.set_box_aspect(1.0)

    fig_dir = os.path.join(args.out_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fig_dir, f"{args.out_prefix}.{ext}"),
                    dpi=300, bbox_inches="tight", pad_inches=0.01)
    if args.paper_out:
        os.makedirs(os.path.dirname(args.paper_out), exist_ok=True)
        fig.savefig(args.paper_out, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    plot_dir = os.path.join(args.out_root, "plot_data")
    os.makedirs(plot_dir, exist_ok=True)
    out = {
        "base_N": args.base_N,
        "base_K": args.base_K,
        "base_tau": args.base_tau,
        "M": args.M,
        "a": args.a,
        "representative_seed": args.representative_seed,
        "reliability_threshold": args.reliability_threshold,
        "t_reliable": t_reliable,
        "panel_b_active_t": active_d["t"],
        "panel_b_active_mean": active_d["mean"],
        "panel_b_active_std": active_d["std"],
        "panel_b_active_seeds": active_d["seeds"],
        "panel_b_control_t": control_d["t"],
        "panel_b_control_mean": control_d["mean"],
        "panel_b_control_std": control_d["std"],
        "panel_b_control_seeds": control_d["seeds"],
        "panel_c_active_t": active_r["t"],
        "panel_c_active_mean": active_r["mean"],
        "panel_c_active_std": active_r["std"],
        "panel_c_active_seeds": active_r["seeds"],
        "panel_c_control_t": control_r["t"],
        "panel_c_control_mean": control_r["mean"],
        "panel_c_control_std": control_r["std"],
        "panel_c_control_seeds": control_r["seeds"],
    }
    for m in range(4):
        for ax in ("x", "y", "z"):
            out[f"panel_a_c{m}_{ax}"] = active_rep["cols"][f"c{m}_{ax}"]
    np.savez(os.path.join(plot_dir, f"{args.out_prefix}.npz"), **out)

    print(f"wrote {fig_dir}/{args.out_prefix}.pdf/.png")
    if args.paper_out:
        print(f"wrote {args.paper_out}")
    print(
        f"baseline: N={args.base_N}, K={args.base_K}, "
        f"tau={args.base_tau:g}, M={args.M:g}, a={args.a:g}"
    )
    print(
        f"panel (a): active seed {args.representative_seed}; "
        f"panel (b): t <= {t_reliable:.3f} from control A_min >= "
        f"{args.reliability_threshold:g}"
    )
    print(
        f"panel (b) final ratios: active={active_d['mean'][-1]:.3f}, "
        f"control={control_d['mean'][-1]:.3f}; "
        f"panel (c) final ratios: active={active_r['mean'][-1]:.3f}, "
        f"control={control_r['mean'][-1]:.3f}"
    )


if __name__ == "__main__":
    main()
