"""
Figures for the 3D Keller-Segel focusing / self-convergence study.

Reads diagnostics.csv files produced by simulation_focusing.py and renders PDF
figures (Agg backend, no display needed):

  (1) core_radii_vs_t.pdf : R_0.5(t), R_0.9(t) per mass M (radial sweep).
  (2) rho_core_vs_t.pdf   : rho_core(t) per mass M (focusing surrogate).
  (3) peaks_vs_t.pdf      : P_H(t), C_H(t) per mass M.
  (4) self_convergence.pdf: Q_c(t) over the (N,H) grid (under-resolution check).
  (5) density_slice.pdf   : reconstructed z=centroid density slices at report
                            times (if a density_slices.npz is provided).
  (6) cluster_trajectories.pdf : per-cluster centroid tracks + min inter-cluster
                            distance vs t (tetra runs only).

The script is grid-agnostic: pass a list of run directories (each with a
diagnostics.csv and a config_used.json). It groups radial runs by mass M and
tetra runs separately. No existing file is modified.

CLI:
  python plot_focusing.py --runs DIR1 DIR2 ... --out_dir figures
"""
import os
import csv
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_run(run_dir):
    """Load (config, rows) from a run directory."""
    with open(os.path.join(run_dir, "config_used.json")) as f:
        cfg = json.load(f)
    rows = []
    with open(os.path.join(run_dir, "diagnostics.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if v != "" else np.nan)
                         for k, v in r.items()})
    return cfg, rows


def col(rows, name):
    return np.array([r.get(name, np.nan) for r in rows], dtype=float)


def plot_core_radii(runs, out_dir):
    """R_0.5(t) and R_0.9(t) per mass M (radial runs only)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for cfg, rows in runs:
        if cfg.get("ic_type") != "radial":
            continue
        t = col(rows, "t")
        lbl = f"M={cfg['M']:.0f}, N={cfg['n_particles']:.0g}, H={cfg['H']}"
        ax1.plot(t, col(rows, "R_0.5"), marker="o", ms=3, label=lbl)
        ax2.plot(t, col(rows, "R_0.9"), marker="s", ms=3, label=lbl)
    ax1.set_xlabel("t"); ax1.set_ylabel(r"$R_{0.5}(t)$")
    ax1.set_title("Inner half-mass core radius")
    ax2.set_xlabel("t"); ax2.set_ylabel(r"$R_{0.9}(t)$")
    ax2.set_title("90% core radius")
    for ax in (ax1, ax2):
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "core_radii_vs_t.pdf"))
    plt.close(fig)


def plot_rho_core(runs, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4.4))
    for cfg, rows in runs:
        if cfg.get("ic_type") != "radial":
            continue
        t = col(rows, "t")
        ax.semilogy(t, col(rows, "rho_core"), marker="o", ms=3,
                    label=f"M={cfg['M']:.0f}")
    ax.set_xlabel("t"); ax.set_ylabel(r"$\rho_{\rm core}(t)$")
    ax.set_title("Core density (focusing surrogate)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rho_core_vs_t.pdf"))
    plt.close(fig)


def plot_peaks(runs, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for cfg, rows in runs:
        t = col(rows, "t")
        lbl = f"M={cfg['M']:.0f}"
        ax1.semilogy(t, col(rows, "P_H"), marker="o", ms=3, label=lbl)
        ax2.semilogy(t, col(rows, "C_H"), marker="s", ms=3, label=lbl)
    ax1.set_xlabel("t"); ax1.set_ylabel(r"$\|P_H\rho\|_\infty$")
    ax1.set_title("Reconstructed density peak")
    ax2.set_xlabel("t"); ax2.set_ylabel(r"$\|c_H\|_\infty$")
    ax2.set_title("Chemical-field peak")
    for ax in (ax1, ax2):
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "peaks_vs_t.pdf"))
    plt.close(fig)


def plot_self_convergence(runs, out_dir):
    """Q_c(t) over the (N,H) grid."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for cfg, rows in runs:
        t = col(rows, "t")
        lbl = (f"M={cfg['M']:.0f}, N={cfg['n_particles']:.0g}, H={cfg['H']}")
        ax.plot(t, col(rows, "Qc"), marker="o", ms=3, label=lbl)
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$Q_c=\|c_{H_{\rm hi}}\|_\infty/\|c_{H_{\rm lo}}\|_\infty$")
    ax.set_title("Self-convergence of the chemical peak")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "self_convergence.pdf"))
    plt.close(fig)


def plot_density_slices(run_dir, out_dir):
    npz_path = os.path.join(run_dir, "density_slices.npz")
    if not os.path.exists(npz_path):
        return
    data = np.load(npz_path)
    times = sorted({k.split("_")[0] for k in data.files if k.endswith("_slice")})
    n = len(times)
    if n == 0:
        return
    ncol = min(n, 4)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow),
                             squeeze=False)
    for idx, tkey in enumerate(times):
        ax = axes[idx // ncol][idx % ncol]
        g = data[f"{tkey}_grid"]
        sl = data[f"{tkey}_slice"]
        im = ax.imshow(sl.T, origin="lower",
                       extent=[g[0], g[-1], g[0], g[-1]], aspect="auto")
        ax.set_title(tkey)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Reconstructed density slice (z = centroid plane)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "density_slice.pdf"))
    plt.close(fig)


def plot_cluster_trajectories(runs, out_dir):
    """Per-cluster centroid tracks + min inter-cluster distance (tetra)."""
    tetra = [(c, r) for c, r in runs if c.get("ic_type") == "tetra"]
    if not tetra:
        return
    cfg, rows = tetra[0]
    n_cl = len(cfg.get("tetra_centers", [[0, 0, 0]] * 4))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    # project centroid tracks onto (x,y)
    for m in range(n_cl):
        cx = col(rows, f"cl{m}_c0")
        cy = col(rows, f"cl{m}_c1")
        ax1.plot(cx, cy, marker="o", ms=3, label=f"cluster {m}")
        ax1.scatter([cx[0]], [cy[0]], marker="*", s=90, zorder=5)
    ax1.set_xlabel(r"$x$"); ax1.set_ylabel(r"$y$")
    ax1.set_title("Cluster centroid tracks (xy projection; * = t0)")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3); ax1.set_aspect("equal")
    t = col(rows, "t")
    ax2.plot(t, col(rows, "min_intercluster_dist"), marker="o", ms=3, color="C3")
    ax2.set_xlabel("t"); ax2.set_ylabel("min inter-cluster distance")
    ax2.set_title("Cluster merging indicator")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cluster_trajectories.pdf"))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                   help="run directories (each with diagnostics.csv)")
    p.add_argument("--out_dir", type=str, default="figures")
    p.add_argument("--slice_run", type=str, default=None,
                   help="run dir whose density_slices.npz to render")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = [load_run(d) for d in args.runs]
    plot_core_radii(runs, args.out_dir)
    plot_rho_core(runs, args.out_dir)
    plot_peaks(runs, args.out_dir)
    plot_self_convergence(runs, args.out_dir)
    plot_cluster_trajectories(runs, args.out_dir)
    slice_run = args.slice_run or args.runs[0]
    plot_density_slices(slice_run, args.out_dir)
    print(f"[plot] wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
