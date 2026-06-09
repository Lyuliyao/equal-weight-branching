"""
Plotting for the high-dimensional (4D / 6D) particle experiment.

Reads results/highdim/{metrics.csv, marginals_d{d}_seed{seed}.npz,
fht_d{d}_seed{seed}.npz} and writes publication-style PDFs into the same dir,
one set per dimension d present in metrics.csv:

  (i)   metrics_d{d}.pdf       : moment m(t), total_mass(t), N_active(t),
                                 N_local_B(t), nESS(t), max_w/mean_w(t)
                                 for the 3 methods (seed mean +/- std).
  (ii)  marginals_d{d}.pdf     : 1D marginals (FHT low-rank vs raw histogram)
                                 per coordinate, plus a 2D marginal panel.
  (iii) diagonal_d{d}.pdf      : FHT diagonal profile f(s,...,s).

Run:  python plot.py
"""
import os
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = "results/highdim"
METHODS = ["weighted", "poisson", "minvar"]
COLORS = {"weighted": "tab:red", "poisson": "tab:blue", "minvar": "tab:green"}


def load_metrics():
    rows = list(csv.DictReader(open(os.path.join(RD, "metrics.csv"))))
    def f(x):
        try:
            return float(x)
        except (ValueError, TypeError):
            return np.nan
    for r in rows:
        for k in r:
            if k != "method":
                r[k] = f(r[k])
    return rows


def _agg(rows, d, method, col):
    """Return (ts, mean, std) over seeds for (d, method, col)."""
    sub = [r for r in rows if r["method"] == method and int(r["d"]) == d]
    ts = sorted(set(r["t"] for r in sub))
    mean, std = [], []
    for t in ts:
        vals = [r[col] for r in sub if r["t"] == t]
        mean.append(np.nanmean(vals))
        std.append(np.nanstd(vals))
    return np.array(ts), np.array(mean), np.array(std)


def plot_metrics(rows, d):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    # (a) moment m(t)
    ax = axes[0, 0]
    for m in METHODS:
        ts, mu, sd = _agg(rows, d, m, "moment_m")
        ax.plot(ts, mu, "-o", color=COLORS[m], label=m, ms=3)
        ax.fill_between(ts, mu - sd, mu + sd, color=COLORS[m], alpha=0.2)
    ax.set_xlabel("t"); ax.set_ylabel("moment m(t)")
    ax.set_title("moment m[f] (logistic stabilization)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (b) total mass
    ax = axes[0, 1]
    for m in METHODS:
        ts, mu, sd = _agg(rows, d, m, "total_mass")
        ax.plot(ts, mu, "-o", color=COLORS[m], label=m, ms=3)
        ax.fill_between(ts, mu - sd, mu + sd, color=COLORS[m], alpha=0.2)
    ax.set_xlabel("t"); ax.set_ylabel("total mass = N_active/N0")
    ax.set_title("total mass(t)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (c) N_active
    ax = axes[0, 2]
    for m in METHODS:
        ts, mu, sd = _agg(rows, d, m, "N_active")
        ax.plot(ts, mu, "-o", color=COLORS[m], label=m, ms=3)
        ax.fill_between(ts, mu - sd, mu + sd, color=COLORS[m], alpha=0.2)
    ax.set_xlabel("t"); ax.set_ylabel("N_active")
    ax.set_title("active particle count(t)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (d) N_local_B
    ax = axes[1, 0]
    for m in METHODS:
        ts, mu, sd = _agg(rows, d, m, "N_local_B")
        ax.plot(ts, mu, "-o", color=COLORS[m], label=m, ms=3)
        ax.fill_between(ts, mu - sd, mu + sd, color=COLORS[m], alpha=0.2)
    ax.set_xlabel("t"); ax.set_ylabel("N_local_B (G_d >= eta)")
    ax.set_title("local particle count(t)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (e) nESS (weighted only)
    ax = axes[1, 1]
    ts, mu, sd = _agg(rows, d, "weighted", "global_nESS")
    ax.plot(ts, mu, "-o", color="tab:red", label="global nESS", ms=3)
    ax.fill_between(ts, mu - sd, mu + sd, color="tab:red", alpha=0.2)
    ts, mu, sd = _agg(rows, d, "weighted", "local_nESS_B")
    ax.plot(ts, mu, "--s", color="tab:orange", label="local nESS (B)", ms=3)
    ax.fill_between(ts, mu - sd, mu + sd, color="tab:orange", alpha=0.2)
    ax.set_xlabel("t"); ax.set_ylabel("normalized ESS")
    ax.set_ylim(0, 1.05)
    ax.set_title("weighted effective sample size collapse")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (f) max_w / mean_w (weighted only)
    ax = axes[1, 2]
    ts, mu, sd = _agg(rows, d, "weighted", "max_w_over_mean_w")
    ax.plot(ts, mu, "-^", color="tab:purple", label="max_w/mean_w", ms=3)
    ax.fill_between(ts, mu - sd, mu + sd, color="tab:purple", alpha=0.2)
    ax.set_xlabel("t"); ax.set_ylabel("max_w / mean_w")
    ax.set_yscale("log")
    ax.set_title("weight degeneracy (weighted)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(f"High-dim particle metrics (d={d}, seed mean +/- std)")
    fig.savefig(os.path.join(RD, f"metrics_d{d}.pdf"))
    plt.close(fig)


def plot_marginals(d):
    mg_path = os.path.join(RD, f"marginals_d{d}_seed0.npz")
    fht_path = os.path.join(RD, f"fht_d{d}_seed0.npz")
    if not (os.path.exists(mg_path) and os.path.exists(fht_path)):
        return
    mg = np.load(mg_path, allow_pickle=True)
    fht = np.load(fht_path, allow_pickle=True)
    centers = mg["centers"]
    h1 = mg["hist1d_poisson"]          # (d, nbins) raw histogram (branching)
    zg = fht["zgrid"]
    m1 = fht["fht_marg1d"]             # (d, ngrid) FHT low-rank
    used_fht = bool(fht["used_fht"])

    ncol = 3
    nrow = int(np.ceil(d / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow),
                             constrained_layout=True, squeeze=False)
    for j in range(d):
        ax = axes[j // ncol][j % ncol]
        ax.bar(centers, h1[j], width=(centers[1] - centers[0]),
               alpha=0.35, color="tab:gray", label="histogram (poisson)")
        if used_fht:
            ax.plot(zg, m1[j], "-", color="tab:blue", lw=2,
                    label="FHT low-rank")
        coord_name = (f"x{j+1}" if j < d // 2 else f"v{j+1 - d//2}")
        ax.set_title(f"1D marginal coord {j} ({coord_name})")
        ax.set_xlabel("z"); ax.set_ylabel("density")
        if j == 0:
            ax.legend(fontsize=8)
    for j in range(d, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    title = f"1D marginals d={d} (FHT vs histogram, seed 0)"
    if not used_fht:
        title += " [FHT FAILED -> fallback]"
    fig.suptitle(title)
    fig.savefig(os.path.join(RD, f"marginals_d{d}.pdf"))
    plt.close(fig)

    # 2D marginal panel: histogram vs FHT for the (x1,x2) pair and (x1,v1) pair
    if used_fht:
        H2x = mg["hist2d_x_poisson"]
        H2xv = mg["hist2d_xv_poisson"]
        g2x = fht["fht_marg2d_x"]
        g2xv = fht["fht_marg2d_xv"]
        ZZ0, ZZ1 = fht["fht_ZZ0"], fht["fht_ZZ1"]
        edges = mg["edges"]
        ec = 0.5 * (edges[:-1] + edges[1:])
        EX, EY = np.meshgrid(ec, ec, indexing="ij")
        fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
        panels = [
            (axes[0, 0], EX, EY, H2x, "histogram (x1,x2)"),
            (axes[0, 1], ZZ0, ZZ1, g2x, "FHT (x1,x2)"),
            (axes[1, 0], EX, EY, H2xv, "histogram (x1,v1)"),
            (axes[1, 1], ZZ0, ZZ1, g2xv, "FHT (x1,v1)"),
        ]
        for ax, XX, YY, Z, name in panels:
            im = ax.pcolormesh(XX, YY, Z, shading="auto", cmap="viridis")
            ax.set_title(name); ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(f"2D marginals d={d} (seed 0)")
        fig.savefig(os.path.join(RD, f"marginals2d_d{d}.pdf"))
        plt.close(fig)


def plot_diagonal(d):
    fht_path = os.path.join(RD, f"fht_d{d}_seed0.npz")
    if not os.path.exists(fht_path):
        return
    fht = np.load(fht_path, allow_pickle=True)
    if not bool(fht["used_fht"]):
        return
    zg = fht["zgrid"]
    diag = fht["fht_diag"]
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.plot(zg, diag, "-o", color="tab:green", ms=3)
    ax.set_xlabel("s"); ax.set_ylabel("f(s, s, ..., s)")
    ax.set_title(f"FHT diagonal profile (d={d}, seed 0)")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(RD, f"diagonal_d{d}.pdf"))
    plt.close(fig)


def main():
    rows = load_metrics()
    ds = sorted(set(int(r["d"]) for r in rows))
    for d in ds:
        plot_metrics(rows, d)
        plot_marginals(d)
        plot_diagonal(d)
    print("wrote PDFs for dims", ds, "to", RD)


if __name__ == "__main__":
    main()
