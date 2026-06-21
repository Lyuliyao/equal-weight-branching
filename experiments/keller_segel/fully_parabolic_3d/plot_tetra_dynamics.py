"""Create the manuscript tetrahedral 3D Keller--Segel dynamics figure.

Panel (a) uses a saved raw u-particle snapshot.  Panels (b)--(c) use archived
production diagnostics only.  This script never runs the solver and never
reconstructs particles from centroids or radii.
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
from paper_style import TEXTWIDTH_IN, apply_style  # noqa: E402
import diagnostics_pp3d as D  # noqa: E402
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


def load_snapshots(path, target_times):
    d = np.load(path)
    times = np.asarray(d["times"], dtype=float)
    if times.size == 0:
        raise SystemExit(f"no snapshot times in {path}")
    labels = np.asarray(d["labels"], dtype=int)
    clouds = []
    matched_times = []
    matched_indices = []
    for target_time in target_times:
        idx = int(np.argmin(np.abs(times - target_time)))
        if not np.isclose(times[idx], target_time, atol=1e-10, rtol=0.0):
            raise SystemExit(
                f"snapshot {path} has no time {target_time:g}; "
                f"available {times.tolist()}"
            )
        X = np.asarray(d[f"u_{idx}"], dtype=float)
        if X.ndim != 2 or X.shape[1] != 3 or labels.shape[0] != X.shape[0]:
            raise SystemExit(f"bad snapshot shapes: X={X.shape}, labels={labels.shape}")
        clouds.append(X)
        matched_times.append(float(times[idx]))
        matched_indices.append(idx)
    return {
        "path": path,
        "times": np.array(matched_times, dtype=float),
        "time_indices": np.array(matched_indices, dtype=int),
        "clouds": clouds,
        "labels": labels,
        "L": float(d["L"]),
        "a": float(d["a"]),
        "M": float(d["M"]),
        "N": int(clouds[0].shape[0]),
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


def unit(v):
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("zero vector")
    return np.asarray(v, dtype=float) / n


def tangent_basis(n):
    n = unit(n)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(ref, n))) > 0.85:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = unit(ref - float(np.dot(ref, n)) * n)
    e2 = unit(np.cross(n, e1))
    return e1, e2


def fibonacci_sphere(count):
    i = np.arange(count, dtype=float)
    z = 1.0 - 2.0 * (i + 0.5) / count
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((r * np.cos(phi), r * np.sin(phi), z))


def min_projected_centroid_distance(centroids, n):
    n = unit(n)
    vals = []
    for i in range(centroids.shape[0]):
        for j in range(i + 1, centroids.shape[0]):
            d = centroids[j] - centroids[i]
            sq = float(np.dot(d, d) - np.dot(d, n) ** 2)
            vals.append(np.sqrt(max(sq, 0.0)))
    return float(min(vals))


def choose_projection_basis(initial_centroids):
    candidates = [
        fibonacci_sphere(24000),
        np.eye(3),
        -np.eye(3),
        np.array([unit(v) for v in initial_centroids - initial_centroids.mean(axis=0)]),
    ]
    dirs = np.vstack(candidates)
    scores = np.array([min_projected_centroid_distance(initial_centroids, n)
                       for n in dirs])
    best = unit(dirs[int(np.argmax(scores))])
    best_score = float(np.max(scores))

    for step in (0.20, 0.08, 0.03, 0.012, 0.004):
        e1, e2 = tangent_basis(best)
        offsets = np.linspace(-step, step, 17)
        local = []
        for a in offsets:
            for b in offsets:
                local.append(unit(best + a * e1 + b * e2))
        local = np.array(local)
        scores = np.array([min_projected_centroid_distance(initial_centroids, n)
                           for n in local])
        idx = int(np.argmax(scores))
        if float(scores[idx]) > best_score:
            best = local[idx]
            best_score = float(scores[idx])

    # Orient the plotting basis by the widest projected centroid pair.
    best = unit(best)
    pair_vec = None
    pair_norm = -np.inf
    for i in range(initial_centroids.shape[0]):
        for j in range(i + 1, initial_centroids.shape[0]):
            d = initial_centroids[j] - initial_centroids[i]
            p = d - float(np.dot(d, best)) * best
            pn = float(np.linalg.norm(p))
            if pn > pair_norm:
                pair_norm = pn
                pair_vec = p
    e1 = unit(pair_vec)
    e2 = unit(np.cross(best, e1))
    origin = initial_centroids.mean(axis=0)
    if float(np.dot(initial_centroids[0] - origin, e2)) < 0.0:
        e2 = -e2
    basis = np.vstack((e1, e2))
    return basis, best, origin, best_score


def project_points(X, basis, origin):
    return (X - origin) @ basis.T


def square_limits(projected_sets, margin=0.045):
    P = np.vstack(projected_sets)
    mins = P.min(axis=0)
    maxs = P.max(axis=0)
    center = 0.5 * (mins + maxs)
    half = 0.5 * float(np.max(maxs - mins))
    half *= 1.0 + margin
    return np.array([center[0] - half, center[0] + half,
                     center[1] - half, center[1] + half])


def plot_snapshot(ax, projected, labels, centroids_projected, indices, title):
    pts_all = projected[indices]
    labels_all = labels[indices]
    for m in range(4):
        pts = pts_all[labels_all == m]
        color = CLUSTER_COLORS[m]
        ax.scatter(pts[:, 0], pts[:, 1], s=2.0, color=color, alpha=0.34,
                   linewidths=0, rasterized=True)
    ax.scatter(centroids_projected[:, 0], centroids_projected[:, 1],
               marker="x", s=15, color="k", linewidths=0.7, rasterized=False)
    ax.set_title(title, pad=1.0)
    ax.set_xlabel(r"$p_1$")
    ax.set_ylabel(r"$p_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)


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
    ap.add_argument("--subsample_per_cluster", type=int, default=800)
    ap.add_argument("--subsample_seed", type=int, default=24012026)
    ap.add_argument("--out_prefix", default="figure_tetra_particle_dynamics")
    args = ap.parse_args()

    runs = add_derived(V.load_runs(args.run_dir))
    if not runs:
        raise SystemExit(f"no tetra diagnostics found in {args.run_dir}")
    snapshots = load_snapshots(args.clouds, [0.0, args.snapshot_time])
    if snapshots["seed"] != args.representative_seed:
        raise SystemExit(
            f"snapshot seed {snapshots['seed']} does not match "
            f"representative seed {args.representative_seed}"
        )
    if (snapshots["N"] != args.base_N or snapshots["K"] != args.base_K
            or not np.isclose(snapshots["tau"], args.base_tau)
            or not np.isclose(snapshots["M"], args.M)
            or not np.isclose(snapshots["a"], args.a)
            or not np.isclose(snapshots["chi"], 1.0)):
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

    labels = snapshots["labels"]
    X0, XT = snapshots["clouds"]
    L = snapshots["L"]
    c0 = D.cluster_centroids(X0, labels, 4, L)
    cT = D.cluster_centroids(XT, labels, 4, L)
    basis, view_direction, projection_origin, projection_score = choose_projection_basis(c0)
    P0 = project_points(X0, basis, projection_origin)
    PT = project_points(XT, basis, projection_origin)
    C0 = project_points(c0, basis, projection_origin)
    CT = project_points(cT, basis, projection_origin)
    scatter_indices = deterministic_subsample(
        labels, args.subsample_per_cluster, args.subsample_seed
    )
    axis_limits = square_limits([P0, PT])

    fig = plt.figure(figsize=(1.096 * TEXTWIDTH_IN, 0.86 * TEXTWIDTH_IN),
                     constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=(1.05, 1.0),
                          hspace=0.10, wspace=0.17)
    ax0 = fig.add_subplot(gs[0, 0])
    axT = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    plot_snapshot(ax0, P0, labels, C0, scatter_indices, r"(a) Active particles, $t=0$")
    plot_snapshot(axT, PT, labels, CT, scatter_indices, r"Active particles, $t=3$")
    for ax in (ax0, axT):
        ax.set_xlim(axis_limits[0], axis_limits[1])
        ax.set_ylim(axis_limits[2], axis_limits[3])
        ax.tick_params(labelsize=5.4, length=2.0, pad=1.0)

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
        "snapshot_times": snapshots["times"],
        "snapshot_path": np.array(args.clouds),
        "subsample_per_cluster": args.subsample_per_cluster,
        "subsample_seed": args.subsample_seed,
        "projection_basis": basis,
        "projection_origin": projection_origin,
        "projection_view_direction": view_direction,
        "projection_min_initial_centroid_separation": projection_score,
        "panel_a_scatter_indices": scatter_indices,
        "panel_a_scatter_labels": labels[scatter_indices],
        "panel_a_t0_full_projected": P0,
        "panel_a_t3_full_projected": PT,
        "panel_a_t0_projected": P0[scatter_indices],
        "panel_a_t3_projected": PT[scatter_indices],
        "panel_a_t0_centroids_3d": c0,
        "panel_a_t3_centroids_3d": cT,
        "panel_a_t0_centroids_projected": C0,
        "panel_a_t3_centroids_projected": CT,
        "panel_a_axis_limits": axis_limits,
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
        f"projection min initial centroid separation={projection_score:.4f}"
    )
    print(
        f"panel (b) final ratios: active={active_d['mean'][-1]:.3f}, "
        f"control={control_d['mean'][-1]:.3f}; "
        f"panel (c) final ratios: active={active_r['mean'][-1]:.3f}, "
        f"control={control_r['mean'][-1]:.3f}"
    )


if __name__ == "__main__":
    main()
