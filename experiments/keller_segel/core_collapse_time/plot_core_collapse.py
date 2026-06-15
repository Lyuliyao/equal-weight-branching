"""
Figures for the core-collapse-time diagnostic (core_collapse plan §9.4).
=======================================================================
Reads the LDG ldg_core_radii_N<N>.csv (+ optional particle diag) and core_fit_all.csv;
produces:
  core_Rq2_fits.{pdf,png}     -- R_q^2(t) vs t for q=0.1,0.2,0.3 per N, with fit lines.
  core_Tq_scatter.{pdf,png}   -- T_q(window) scatter over q and fitting windows per N.
  core_T_summary.{pdf,png}    -- T_core(N) with [p10,p90] bars (+ particle if present).
  core_halo_sep.{pdf,png}     -- R_0.8/R_0.2 (core-halo decoupling) per N.

Usage:  python plot_core_collapse.py --ldg_dir <run>/ldg --fitdir <run> [--mass raw]
"""
import os
import csv
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt

import sys
_EXP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from common_plot_style import apply_style, savefig_multi  # noqa: E402

NCOL = {80: "#1f77b4", 160: "#2ca02c", 320: "#d62728"}
QCOL = {0.1: "#1f77b4", 0.2: "#2ca02c", 0.3: "#d62728"}
WINDOWS = [(4e-5, 9e-5), (5e-5, 1.0e-4), (6e-5, 1.1e-4), (7e-5, 1.2e-4)]


def load_ldg(ldg_dir, mass):
    out = {}
    for f in sorted(glob.glob(os.path.join(ldg_dir, "N*", "ldg_core_radii_N*.csv"))):
        r = list(csv.DictReader(open(f)))
        if not r:
            continue
        N = int(float(r[0]["N"]))
        t = np.array([float(x["t"]) for x in r])
        d = dict(t=t, R={}, S_L2=np.array([float(x["S_L2"]) for x in r]),
                 peak=np.array([float(x["peak"]) for x in r]))
        for q in [0.1, 0.2, 0.3, 0.5, 0.8]:
            c = f"R_{q}_{mass}"
            if c in r[0]:
                d["R"][q] = np.array([float(x[c]) for x in r])
        out[N] = d
    return out


def load_fits(fitdir):
    f = os.path.join(fitdir, "core_fit_all.csv")
    if not os.path.exists(f):
        return []
    return list(csv.DictReader(open(f)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ldg_dir", required=True)
    ap.add_argument("--fitdir", required=True)
    ap.add_argument("--mass", default="raw")
    args = ap.parse_args()
    apply_style()
    figdir = os.path.join(args.fitdir, "figures"); os.makedirs(figdir, exist_ok=True)
    ldg = load_ldg(args.ldg_dir, args.mass)
    fits = load_fits(args.fitdir)
    Ns = sorted(ldg)

    # ---- Fig 1: R_q^2 vs t with fit lines (finest N) ----
    fig, ax = plt.subplots(figsize=(3.7, 2.9))
    Nf = Ns[-1] if Ns else None
    if Nf is not None:
        d = ldg[Nf]
        for q in [0.1, 0.2, 0.3]:
            if q not in d["R"]:
                continue
            ax.plot(d["t"] * 1e4, d["R"][q] ** 2 * 1e4, color=QCOL[q], lw=1.2,
                    label=f"$R_{{{q}}}^2$")
        # overlay fit lines for the LDG finest N
        for fr in fits:
            if fr["method"] != "LDG" or int(float(fr["resolution"])) != Nf:
                continue
            if fr["quantity"] != "Rq2" or fr["valid_fit"] != "True":
                continue
            if fr.get("q_set") not in (None, "", "0.1,0.2,0.3"):
                continue
            q = float(fr["q"]); a = float(fr["alpha"]); b = -float(fr["beta"])
            w0, w1 = float(fr["window_start"]), float(fr["window_end"])
            tt = np.linspace(w0, w1, 20)
            ax.plot(tt * 1e4, (a + b * tt) * 1e4, color=QCOL.get(q, "k"), lw=0.7, ls="--")
    ax.set_xlabel(r"$t\,[\times10^{-4}]$"); ax.set_ylabel(r"$R_q^2\,[\times10^{-4}]$")
    ax.set_title(f"(a) $R_q^2(t)$ + linear fits (LDG N={Nf})")
    ax.legend(fontsize=6); ax.set_ylim(bottom=0)
    fig.tight_layout(); savefig_multi(fig, os.path.join(figdir, "core_Rq2_fits")); plt.close(fig)

    # ---- Fig 2: T_q(window) scatter ----
    fig, ax = plt.subplots(figsize=(3.7, 2.9))
    for fr in fits:
        if fr["method"] != "LDG" or fr["quantity"] != "Rq2":
            continue
        if fr.get("q_set") not in (None, "", "0.1,0.2,0.3"):
            continue
        if fr["valid_fit"] != "True":
            continue
        N = int(float(fr["resolution"])); q = float(fr["q"]); T = float(fr["T_est"])
        ax.scatter([N], [T * 1e4], color=QCOL.get(q, "k"),
                   marker={0.1: "o", 0.2: "s", 0.3: "^"}.get(q, "x"), s=20,
                   label=f"q={q}" if N == Ns[0] else None)
    ax.set_xlabel("LDG $N$"); ax.set_ylabel(r"$T_q(I)\,[\times10^{-4}]$")
    ax.set_title("(b) collapse-time estimates (valid fits)")
    h, l = ax.get_legend_handles_labels()
    if h:
        ax.legend(fontsize=6)
    fig.tight_layout(); savefig_multi(fig, os.path.join(figdir, "core_Tq_scatter")); plt.close(fig)

    # ---- Fig 3: T_core(N) summary, default q-set; particle uses SEED BOOTSTRAP CI ----
    fig, ax = plt.subplots(figsize=(3.9, 2.9))
    srows = list(csv.DictReader(open(os.path.join(args.fitdir, "core_fit_summary.csv"))))
    bpath = os.path.join(args.fitdir, "core_fit_bootstrap.csv")
    boot = list(csv.DictReader(open(bpath))) if os.path.exists(bpath) else []
    def boot_ci(method, N):
        for b in boot:
            if b["method"] == method and b["resolution"] == str(N) and b["q_set"] == "0.1,0.2,0.3":
                return float(b["T_boot_median"]), float(b["T_boot_p10"]), float(b["T_boot_p90"])
        return None
    xs, meds, los, his, cols = [], [], [], [], []
    for r in srows:
        if r["quantity"] != "T_core" or r.get("q_set") != "0.1,0.2,0.3":
            continue
        try:
            med = float(r["T_median"]); p10 = float(r["T_p10"]); p90 = float(r["T_p90"])
        except ValueError:
            continue
        if not np.isfinite(med):
            continue
        ci = boot_ci(r["method"], r["resolution"]) if r["method"] == "particle" else None
        if ci is not None:                       # particle: seed-bootstrap CI (the honest band)
            med, p10, p90 = ci
            cols.append("#2ca02c")
        else:
            cols.append("k")
        xs.append(f"{r['method'][:4]}{r['resolution']}")
        meds.append(med * 1e4); los.append((med - p10) * 1e4); his.append((p90 - med) * 1e4)
    for i in range(len(xs)):
        ax.errorbar(i, meds[i], yerr=[[los[i]], [his[i]]], fmt="o", capsize=3, color=cols[i])
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, rotation=30, ha="right", fontsize=6)
    ax.axhline(1.21, color="0.6", ls="-.", lw=0.8); ax.text(0, 1.23, "LDG $t_b\\sim$1.21", fontsize=6, color="0.5")
    ax.set_ylabel(r"$T_{core}\,[\times10^{-4}]$")
    ax.set_title("(c) $T_{core}$: LDG q/window spread, particle seed-bootstrap CI")
    fig.tight_layout(); savefig_multi(fig, os.path.join(figdir, "core_T_summary")); plt.close(fig)

    # ---- Fig 5: q-set sensitivity (LDG N=320 + particle N=3.2e5) ----
    fig, ax = plt.subplots(figsize=(3.9, 2.9))
    qset_order = ["0.1,0.2,0.3", "0.2,0.3", "0.2", "0.3"]
    for method, N, col, mk in (("LDG", "320", "#d62728", "s"), ("particle", "320000", "#2ca02c", "o")):
        xs2, ys2 = [], []
        for qs in qset_order:
            for r in srows:
                if r["quantity"] == "T_core" and r["method"] == method and r["resolution"] == N and r["q_set"] == qs:
                    try:
                        v = float(r["T_median"])
                    except ValueError:
                        v = np.nan
                    if np.isfinite(v):
                        xs2.append(qset_order.index(qs)); ys2.append(v * 1e4)
        ax.plot(xs2, ys2, marker=mk, color=col, lw=1.0, label=f"{method} N={N}")
    ax.axhline(1.21, color="0.6", ls="-.", lw=0.8)
    ax.set_xticks(range(len(qset_order))); ax.set_xticklabels(qset_order, rotation=20, fontsize=6)
    ax.set_xlabel("q-set"); ax.set_ylabel(r"$T_{core}\,[\times10^{-4}]$")
    ax.set_title("(e) q-set sensitivity"); ax.legend(fontsize=6)
    fig.tight_layout(); savefig_multi(fig, os.path.join(figdir, "core_T_qset_sensitivity")); plt.close(fig)

    # ---- Fig 4: core-halo separation R_0.8/R_0.2 ----
    fig, ax = plt.subplots(figsize=(3.7, 2.9))
    for N in Ns:
        d = ldg[N]
        if 0.8 in d["R"] and 0.2 in d["R"]:
            ratio = d["R"][0.8] / np.maximum(d["R"][0.2], 1e-30)
            ax.plot(d["t"] * 1e4, ratio, color=NCOL.get(N, "k"), lw=1.2, label=f"N={N}")
    ax.set_xlabel(r"$t\,[\times10^{-4}]$"); ax.set_ylabel(r"$R_{0.8}/R_{0.2}$")
    ax.set_title("(d) core-halo separation")
    ax.legend(fontsize=6)
    fig.tight_layout(); savefig_multi(fig, os.path.join(figdir, "core_halo_sep")); plt.close(fig)
    print(f"wrote 4 figures to {figdir}")


if __name__ == "__main__":
    main()
