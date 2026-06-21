"""Create the manuscript tetrahedral 3D Keller--Segel dynamics figure.

Panel (a) uses a saved raw u-particle snapshot.  Panels (b)--(c) use archived
production diagnostics only.  This script never runs the solver and never
reconstructs particles from centroids or radii.
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
        "values": Y,
        "nseed": Y.shape[0],
        "seeds": np.array(seeds, dtype=int),
    }


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


def load_snapshot(path, target_time):
    d = np.load(path)
    times = np.asarray(d["times"], dtype=float)
    if times.size == 0:
        raise SystemExit(f"no snapshot times in {path}")
    idx = int(np.argmin(np.abs(times - target_time)))
    if not np.isclose(times[idx], target_time, atol=1e-10, rtol=0.0):
        raise SystemExit(
            f"snapshot {path} has no time {target_time:g}; available {times.tolist()}"
        )
    X = np.asarray(d[f"u_{idx}"], dtype=float)
    labels = np.asarray(d["labels"], dtype=int)
    if X.ndim != 2 or X.shape[1] != 3 or labels.shape[0] != X.shape[0]:
        raise SystemExit(f"bad snapshot shapes: X={X.shape}, labels={labels.shape}")
    return {
        "path": path,
        "time": float(times[idx]),
        "time_index": idx,
        "X": X,
        "labels": labels,
        "L": float(d["L"]),
        "a": float(d["a"]),
        "M": float(d["M"]),
        "N": int(X.shape[0]),
        "K": int(d["K"]),
        "tau": float(d["tau"]),
        "chi": float(d["chi"]),
        "seed": int(d["seed"]),
    }


def deterministic_subsample(labels, per_cluster, seed):
    rng = np.random.default_rng(seed)
    pieces = []
    for m in range(4):
        pool = np.flatnonzero(labels == m)
        if pool.size < per_cluster:
            raise SystemExit(
                f"cluster {m} has only {pool.size} particles; need {per_cluster}"
            )
        chosen = rng.choice(pool, size=per_cluster, replace=False)
        pieces.append(np.sort(chosen))
    return np.concatenate(pieces)


def displayed_cube(X, vertices, L, min_fraction=0.99):
    half = max(1.25 * float(np.max(np.abs(vertices))), 1.5)
    max_half = 0.5 * float(L)
    while True:
        inside = np.all((X >= -half) & (X <= half), axis=1)
        fraction = float(np.mean(inside))
        if fraction >= min_fraction or half >= max_half:
            break
        half = min(max_half, half + 0.25)
    return (-half, half), fraction


def plot_particle_panel(ax, snapshot, per_cluster, subsample_seed):
    X = snapshot["X"]
    labels = snapshot["labels"]
    a = snapshot["a"]
    L = snapshot["L"]
    vertices = a * np.array(
        [[1.0, 1.0, 1.0],
         [1.0, -1.0, -1.0],
         [-1.0, 1.0, -1.0],
         [-1.0, -1.0, 1.0]]
    )
    scatter_indices = deterministic_subsample(labels, per_cluster, subsample_seed)
    scatter_xyz = X[scatter_indices]
    scatter_labels = labels[scatter_indices]
    (lo, hi), inside_fraction = displayed_cube(X, vertices, L)

    ax.set_proj_type("ortho")
    for i, j in itertools.combinations(range(4), 2):
        ax.plot(
            [vertices[i, 0], vertices[j, 0]],
            [vertices[i, 1], vertices[j, 1]],
            [vertices[i, 2], vertices[j, 2]],
            color="0.70",
            lw=0.55,
            ls="--",
            zorder=0,
        )
    for m in range(4):
        pts = scatter_xyz[scatter_labels == m]
        color = CLUSTER_COLORS[m]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.7, color=color,
                   alpha=0.34, linewidths=0, depthshade=False,
                   rasterized=True, label=rf"$m={m + 1}$")
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
               s=24, marker="o", facecolors="none", edgecolors="k",
               linewidths=0.65, depthshade=False, rasterized=False)

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=18, azim=38)
    ax.set_title(r"(a) Active cell particles" "\n" r"at $T=3$",
                 linespacing=0.9, pad=1.0)
    ax.set_xlabel(r"$x$", labelpad=-4)
    ax.set_ylabel(r"$y$", labelpad=-4)
    ax.set_zlabel(r"$z$", labelpad=-6)
    ticks = [-5, 0, 5] if hi >= 5.0 else [-2, 0, 2]
    ax.set_xticks(ticks); ax.set_yticks(ticks); ax.set_zticks(ticks)
    ax.tick_params(axis="both", which="major", pad=-2)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 0.98), handlelength=1.0,
              borderaxespad=0.0, labelspacing=0.1)
    return {
        "indices": scatter_indices,
        "xyz": scatter_xyz,
        "labels": scatter_labels,
        "vertices": vertices,
        "axis_limits": np.array([lo, hi], dtype=float),
        "inside_fraction": inside_fraction,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="directory containing archived tetra production diagnostics")
    ap.add_argument("--clouds", required=True,
                    help="saved u-cloud snapshot NPZ from the representative active run")
    ap.add_argument("--out_root", required=True,
                    help="reference result root for figures/ and plot_data/")
    ap.add_argument("--paper_out", default=None,
                    help="optional manuscript PDF path")
    ap.add_argument("--paper_png", default=None,
                    help="optional manuscript PNG path")
    ap.add_argument("--trace_out", default=None,
                    help="optional traceability NPZ path")
    ap.add_argument("--base_N", type=int, default=80000)
    ap.add_argument("--base_K", type=int, default=12)
    ap.add_argument("--base_tau", type=float, default=1e-3)
    ap.add_argument("--M", type=float, default=240.0)
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--representative_seed", type=int, default=0)
    ap.add_argument("--snapshot_time", type=float, default=3.0)
    ap.add_argument("--subsample_per_cluster", type=int, default=1000)
    ap.add_argument("--subsample_seed", type=int, default=24012026)
    ap.add_argument("--out_prefix", default="figure_tetra_particle_dynamics")
    args = ap.parse_args()

    runs = add_derived(V.load_runs(args.run_dir))
    if not runs:
        raise SystemExit(f"no tetra diagnostics found in {args.run_dir}")
    snapshot = load_snapshot(args.clouds, args.snapshot_time)
    if snapshot["seed"] != args.representative_seed:
        raise SystemExit(
            f"snapshot seed {snapshot['seed']} does not match "
            f"representative seed {args.representative_seed}"
        )
    if (snapshot["N"] != args.base_N or snapshot["K"] != args.base_K
            or not np.isclose(snapshot["tau"], args.base_tau)
            or not np.isclose(snapshot["M"], args.M)
            or not np.isclose(snapshot["a"], args.a)
            or not np.isclose(snapshot["chi"], 1.0)):
        raise SystemExit("snapshot metadata do not match the requested production config")

    base = dict(N=args.base_N, K=args.base_K, tau=args.base_tau,
                M=args.M, a=args.a)
    active_sel = dict(arm="active", **base)
    control_sel = dict(arm="control", **base)
    active_runs = require_baseline(runs, active_sel, "active")
    control_runs = require_baseline(runs, control_sel, "control")

    active_d = normalized_ensemble(runs, active_sel, "d_min")
    control_d = normalized_ensemble(runs, control_sel, "d_min", tgrid=active_d["t"])
    active_r = normalized_ensemble(runs, active_sel, "R05_mean")
    control_r = normalized_ensemble(runs, control_sel, "R05_mean", tgrid=active_r["t"])

    fig = plt.figure(figsize=(1.018 * TEXTWIDTH_IN, 0.49 * TEXTWIDTH_IN),
                     constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=(1.08, 1.0, 1.0),
                          wspace=0.18)
    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    scatter = plot_particle_panel(
        ax0, snapshot, args.subsample_per_cluster, args.subsample_seed
    )

    add_curve(ax1, active_d, r"chemotaxis, $\chi=1$", ACTIVE_COLOR)
    add_curve(ax1, control_d, r"diffusion control, $\chi=0$", CONTROL_COLOR)
    ax1.axhline(1.0, color="0.35", lw=0.65, ls=":")
    ax1.set_title("(b) Inter-cluster\nattraction", linespacing=0.9, pad=1.0)
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$d_{\min}(t)/d_{\min}(0)$")
    ax1.set_xlim(float(active_d["t"][0]), float(active_d["t"][-1]))
    ax1.legend(loc="lower left", handlelength=1.2, labelspacing=0.18,
               borderaxespad=0.2)

    ax2.set_yscale("log")
    add_curve(ax2, active_r, r"chemotaxis, $\chi=1$", ACTIVE_COLOR)
    add_curve(ax2, control_r, r"diffusion control, $\chi=0$", CONTROL_COLOR)
    ax2.axhline(1.0, color="0.35", lw=0.65, ls=":")
    ax2.set_title("(c) Individual-cluster\nfocusing", linespacing=0.9, pad=1.0)
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$\overline{R}_{0.5}(t)/\overline{R}_{0.5}(0)$")
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
    if args.paper_png:
        os.makedirs(os.path.dirname(args.paper_png), exist_ok=True)
        fig.savefig(args.paper_png, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    plot_dir = os.path.join(args.out_root, "plot_data")
    os.makedirs(plot_dir, exist_ok=True)
    trace_path = args.trace_out or os.path.join(plot_dir, f"{args.out_prefix}.npz")
    out = {
        "base_N": args.base_N,
        "base_K": args.base_K,
        "base_tau": args.base_tau,
        "M": args.M,
        "a": args.a,
        "representative_seed": args.representative_seed,
        "snapshot_time": snapshot["time"],
        "snapshot_path": np.array(args.clouds),
        "subsample_per_cluster": args.subsample_per_cluster,
        "subsample_seed": args.subsample_seed,
        "panel_a_scatter_indices": scatter["indices"],
        "panel_a_scatter_xyz": scatter["xyz"],
        "panel_a_scatter_labels": scatter["labels"],
        "panel_a_initial_vertices": scatter["vertices"],
        "panel_a_axis_limits": scatter["axis_limits"],
        "panel_a_cloud_fraction_inside_axes": scatter["inside_fraction"],
        "panel_b_active_t": active_d["t"],
        "panel_b_active_mean": active_d["mean"],
        "panel_b_active_std": active_d["std"],
        "panel_b_active_values": active_d["values"],
        "panel_b_active_seeds": active_d["seeds"],
        "panel_b_control_t": control_d["t"],
        "panel_b_control_mean": control_d["mean"],
        "panel_b_control_std": control_d["std"],
        "panel_b_control_values": control_d["values"],
        "panel_b_control_seeds": control_d["seeds"],
        "panel_c_active_t": active_r["t"],
        "panel_c_active_mean": active_r["mean"],
        "panel_c_active_std": active_r["std"],
        "panel_c_active_values": active_r["values"],
        "panel_c_active_seeds": active_r["seeds"],
        "panel_c_control_t": control_r["t"],
        "panel_c_control_mean": control_r["mean"],
        "panel_c_control_std": control_r["std"],
        "panel_c_control_values": control_r["values"],
        "panel_c_control_seeds": control_r["seeds"],
    }
    np.savez(trace_path, **out)

    print(f"wrote {fig_dir}/{args.out_prefix}.pdf/.png")
    if args.paper_out:
        print(f"wrote {args.paper_out}")
    if args.paper_png:
        print(f"wrote {args.paper_png}")
    print(f"wrote traceability {trace_path}")
    print(
        f"baseline: N={args.base_N}, K={args.base_K}, "
        f"tau={args.base_tau:g}, M={args.M:g}, a={args.a:g}"
    )
    print(
        f"panel (a): active seed {args.representative_seed}, "
        f"{args.subsample_per_cluster} particles per cluster, "
        f"cloud fraction inside displayed axes={scatter['inside_fraction']:.5f}"
    )
    print(
        f"panel (b) final ratios: active={active_d['mean'][-1]:.3f}, "
        f"control={control_d['mean'][-1]:.3f}; "
        f"panel (c) final ratios: active={active_r['mean'][-1]:.3f}, "
        f"control={control_r['mean'][-1]:.3f}"
    )


if __name__ == "__main__":
    main()
