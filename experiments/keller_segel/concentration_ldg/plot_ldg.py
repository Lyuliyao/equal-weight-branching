"""LDG-diagnostics plots for the 2D Keller-Segel particle method.

Reads the CSVs (diag_*.csv) and the snapshot .npz files written by
simulation_ldg.py, plus the optional t_gap table from tgap.py, and produces:
  (a) contour/heatmap snapshots of the reconstructed physical-u field at the
      report times (same spatial window per run + shared color scale across the
      report times of that run), saved as PDF;
  (b) curves of S_L2(t), peak_PK_u(t) and R_0.5^2(t) vs time (one PDF per run);
  (c) a t_gap(N,K) table figure (PDF) from tgap_table.csv.

matplotlib Agg backend; all PDFs land in the run's outdir.
"""
import os
import sys
import csv
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path):
    with open(path) as f:
        r = list(csv.DictReader(f))
    return {k: np.array([float(x[k]) for x in r]) for k in r[0]}


# ---------------------------------------------------------------------------
# (a) snapshot heatmaps at the report times (shared color scale per run)
# ---------------------------------------------------------------------------
def plot_snapshots(snap_dir, out_dir, run_tag=None):
    pat = f"snap_{run_tag}_t*.npz" if run_tag else "snap_*_t*.npz"
    files = sorted(glob.glob(os.path.join(snap_dir, pat)))
    if not files:
        print(f"[snapshots] none found in {snap_dir} (pattern {pat})")
        return None
    # group by run tag (everything between 'snap_' and the trailing '_t<time>')
    groups = {}
    for fpath in files:
        base = os.path.basename(fpath)[len("snap_"):-len(".npz")]
        tag = base.rsplit("_t", 1)[0]
        groups.setdefault(tag, []).append(fpath)

    outs = []
    for tag, fl in groups.items():
        data = []
        for fpath in fl:
            d = np.load(fpath)
            data.append((float(d["report_t"]), d))
        data.sort(key=lambda z: z[0])
        vmax = max(float(d["U"].max()) for _, d in data)
        vmax = vmax if vmax > 0 else 1.0
        n = len(data)
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), squeeze=False)
        im = None
        for ax, (rt, d) in zip(axes[0], data):
            X, Y, U = d["X"], d["Y"], d["U"]
            im = ax.pcolormesh(X, Y, U, shading="auto", vmin=0.0, vmax=vmax,
                               cmap="inferno")
            ax.set_aspect("equal")
            ax.set_title(rf"$t={rt:.2e}$  (peak={float(d['U'].max()):.1e})",
                         fontsize=9)
            ax.set_xlabel("x"); ax.set_ylabel("y")
        if im is not None:
            fig.colorbar(im, ax=list(axes[0]), shrink=0.85,
                         label=r"$u$ (reconstructed)")
        fig.suptitle(f"reconstructed u snapshots — {tag}")
        out = os.path.join(out_dir, f"snapshots_{tag}.pdf")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out)
        outs.append(out)
    return outs


# ---------------------------------------------------------------------------
# (b) S_L2, peak, R_0.5^2 curves vs time
# ---------------------------------------------------------------------------
def plot_curves(csv_path, out_dir):
    c = load_csv(csv_path)
    t = c["t"]
    tag = os.path.basename(csv_path).replace("diag_", "").replace(".csv", "")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

    a = ax[0]
    a.semilogy(t, c["S_L2"], "C0")
    a.set_xlabel("t"); a.set_ylabel(r"$S_{K,N}=\|P_K\mu\|_{L^2}$")
    a.set_title("reconstructed L2 norm (focusing)")

    a = ax[1]
    a.semilogy(t, c["peak_PK_u"], "C2")
    a.set_xlabel("t"); a.set_ylabel(r"$\|P_K u\|_\infty$")
    a.set_title("reconstructed peak")

    a = ax[2]
    a.semilogy(t, c["R_0.5"] ** 2, "C3", label=r"$R_{0.5}^2$ (core)")
    if "R_0.8" in c:
        a.semilogy(t, c["R_0.8"] ** 2, "C1", label=r"$R_{0.8}^2$")
    a.set_xlabel("t"); a.set_ylabel(r"$R_q^2$")
    a.set_title("core radius collapse")
    a.legend(fontsize=8)

    fig.suptitle(tag)
    fig.tight_layout()
    out = os.path.join(out_dir, f"curves_{tag}.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


# ---------------------------------------------------------------------------
# (c) t_gap(N,K) table figure
# ---------------------------------------------------------------------------
def plot_tgap_table(tgap_csv, out_dir):
    if not os.path.exists(tgap_csv):
        print(f"[tgap] {tgap_csv} not found; skipping table figure")
        return None
    with open(tgap_csv) as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        print(f"[tgap] {tgap_csv} empty; skipping")
        return None
    header, body = rows[0], rows[1:]
    fig, ax = plt.subplots(figsize=(min(2 + 1.3 * len(header), 18),
                                    1.0 + 0.45 * len(body)))
    ax.axis("off")
    tbl = ax.table(cellText=body, colLabels=header, loc="center",
                   cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    ax.set_title(r"resolution-gap focusing time $t_{\rm gap}(N,K)$", pad=12)
    out = os.path.join(out_dir, "tgap_table.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def main():
    p = argparse.ArgumentParser(description="LDG diagnostics plots")
    p.add_argument("--csvs", nargs="*", default=None,
                   help="diag_*.csv files (default: all in --out_dir)")
    p.add_argument("--snap_dir", type=str, default=None,
                   help="snapshots dir (default: <out_dir>/snapshots)")
    p.add_argument("--tgap_csv", type=str, default=None,
                   help="tgap table CSV (default: <out_dir>/tgap_table.csv)")
    p.add_argument("--out_dir", type=str, default="results")
    args = p.parse_args()

    csvs = args.csvs or sorted(glob.glob(os.path.join(args.out_dir,
                                                      "diag_*.csv")))
    snap_dir = args.snap_dir or os.path.join(args.out_dir, "snapshots")
    tgap_csv = args.tgap_csv or os.path.join(args.out_dir, "tgap_table.csv")

    for cpath in csvs:
        plot_curves(cpath, args.out_dir)
    plot_snapshots(snap_dir, args.out_dir)
    plot_tgap_table(tgap_csv, args.out_dir)


if __name__ == "__main__":
    main()
