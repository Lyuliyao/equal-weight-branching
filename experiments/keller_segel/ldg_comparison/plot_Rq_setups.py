"""plot_Rq_setups.py -- mass-quantile radii R_q(t) across sensitivity setups.

R_q(t) = q-th quantile of the particle radius |X-x_c|, i.e. the radius enclosing
a fraction q of the particles (recon-free, straight from particle positions).

The user asked for the radii enclosing 90% / 80% / 70% of particles. The saved
diagnostics logged q in {0.05,0.1,0.2,0.3,0.5,0.8,0.9,0.99}; R_0.7 (and R_0.6)
were NOT logged, so the third curve defaults to the nearest saved bulk quantile
R_0.5. Pass --qs to choose; missing columns are skipped with a warning.

Layout: rows = quantiles, cols = sensitivity axes (K / tau / q_window / N_p).
Each panel overlays the configs that vary that one axis (baseline always shown).
Plot only -- reads diag_*.csv, never runs the solver.
"""
import os, re, csv, glob, argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

LIT = 1.21e-4
BASE = "sens_K5_tau2e-7_q0.8_N6400000"
ALL_AXES = ["K", "tau", "q", "N"]
FIELD_IDX = {"K": 0, "tau": 1, "q": 2, "N": 3}   # index into cfg_fields()
AXLABEL = {"K": "readout K", "tau": r"$\tau$", "q": "window q", "N": r"$N_p$"}


def parse_cfg(name):
    m = re.search(r"(sens_K\d+_tau[0-9.e-]+_q[0-9.]+_N\d+)_seed(\d+)", name)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def cfg_fields(cfg):
    m = re.search(r"K(\d+)_tau([0-9.e-]+)_q([0-9.]+)_N(\d+)", cfg)
    return m.groups()  # (K, tau, q, N) as strings


def axis_of(cfg):
    if cfg == BASE:
        return "baseline"
    K, t, q, N = cfg_fields(cfg); bK, bt, bq, bN = cfg_fields(BASE)
    if (t, q, N) == (bt, bq, bN):
        return "K"
    if (K, q, N) == (bK, bq, bN):
        return "tau"
    if (K, t, N) == (bK, bt, bN):
        return "q"
    if (K, t, q) == (bK, bt, bq):
        return "N"
    return "other"


def axis_value(cfg, axis):
    K, t, q, N = cfg_fields(cfg)
    return {"K": f"K={K}", "tau": rf"$\tau$={float(t):.0e}", "q": f"q={q}",
            "N": f"N={int(N)/1e6:g}M"}[axis]


def seedmean_Rq(csvs, q):
    key = f"R_{q:g}"
    per = []
    for f in csvs:
        R = list(csv.DictReader(open(f)))
        if not R or key not in R[0]:        # empty/failed run or missing column
            return None, None
        t = np.array([float(r["t"]) for r in R])
        y = np.array([float(r[key]) for r in R])
        per.append((t, y))
    tg = np.linspace(max(t.min() for t, _ in per), min(t.max() for t, _ in per), 700)
    yg = np.mean([np.interp(tg, t, y) for t, y in per], axis=0)
    return tg, yg


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--qs", type=float, nargs="+", default=[0.8, 0.9, 0.99])
    ap.add_argument("--axes", nargs="+", default=["K", "tau", "N"],
                    choices=ALL_AXES, help="sensitivity axes to show as columns")
    ap.add_argument("--base", default=BASE,
                    help="baseline config dir prefix (override for non-q0.8 runs)")
    ap.add_argument("--out_prefix", default="Rq_setups")
    ap.add_argument("--t_cap", type=float, default=1.2e-4)
    args = ap.parse_args()
    BASE = args.base

    # group seeds per config
    groups = {}
    for d in sorted(glob.glob(os.path.join(args.run_dir, "sens_K*_seed*"))):
        cfg, seed = parse_cfg(os.path.basename(d))
        if cfg is None:
            continue
        cs = glob.glob(os.path.join(d, "diag_*.csv"))
        if cs:
            groups.setdefault(cfg, []).append(cs[0])
    if not groups:
        raise SystemExit(f"no sens_K*_seed*/diag_*.csv under {args.run_dir}")

    # which qs are actually present
    any_csv = next(iter(groups.values()))[0]
    hdr = list(csv.DictReader(open(any_csv)).fieldnames)
    qs_present = [q for q in args.qs if f"R_{q:g}" in hdr]
    qs_missing = [q for q in args.qs if f"R_{q:g}" not in hdr]
    if qs_missing:
        print(f"[warn] not logged in these runs (skipped): "
              f"{', '.join('R_%g' % q for q in qs_missing)}")
    if not qs_present:
        raise SystemExit("none of the requested R_q are present")

    # configs grouped by axis (only the requested axes -> columns)
    axes_list = args.axes
    by_axis = {ax: [] for ax in axes_list}
    for cfg in groups:
        ax = axis_of(cfg)
        if ax == "baseline":
            for a in axes_list:
                by_axis[a].append(cfg)
        elif ax in by_axis:
            by_axis[ax].append(cfg)
    for ax in axes_list:
        by_axis[ax] = sorted(set(by_axis[ax]),
                             key=lambda c: float(cfg_fields(c)[FIELD_IDX[ax]]))

    # precompute seed-mean R_q curves
    curves = {}   # (cfg,q) -> (t,y)
    for cfg in groups:
        for q in qs_present:
            curves[(cfg, q)] = seedmean_Rq(groups[cfg], q)

    # ---- save plot-data CSV ----
    out_csv = os.path.join(args.run_dir, f"{args.out_prefix}_data.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "q", "t", "R_q"])
        for (cfg, q), (t, y) in curves.items():
            if t is None:
                continue
            for ti, yi in zip(t, y):
                w.writerow([cfg, q, f"{ti:.6e}", f"{yi:.6e}"])

    # extend x-axis to the true end of the simulated time
    tmax_plot = max(t.max() for (t, y) in curves.values() if t is not None)

    # ============================ FIGURE ============================
    nq, na = len(qs_present), len(axes_list)
    fig, axes = plt.subplots(nq, na, figsize=(3.4 * na, 2.9 * nq),
                             sharex=True, squeeze=False)
    cmap = plt.cm.viridis
    for i, q in enumerate(qs_present):
        # shared y-limits per quantile row (over visible window t<=1.3e-4)
        def _vis(c):
            t, y = curves[(c, q)]
            return y[t <= tmax_plot] if t is not None else np.array([np.nan])
        ymin = np.nanmin([np.nanmin(_vis(c)) for c in groups
                          if curves[(c, q)][0] is not None])
        ymax = np.nanmax([np.nanmax(_vis(c)) for c in groups
                          if curves[(c, q)][0] is not None])
        for j, ax in enumerate(axes_list):
            a = axes[i][j]
            cfgs = by_axis[ax]
            cols = cmap(np.linspace(0.12, 0.85, len(cfgs)))
            for cfg, col in zip(cfgs, cols):
                t, y = curves[(cfg, q)]
                if t is None:
                    continue
                is_base = (cfg == BASE)
                a.semilogy(t * 1e4, y, lw=2.2 if is_base else 1.4,
                           color="k" if is_base else col,
                           ls="--" if is_base else "-",
                           zorder=5 if is_base else 3,
                           label=("baseline " if is_base else "") + axis_value(cfg, ax))
            a.axvline(LIT * 1e4, color="r", ls=":", lw=0.9)
            a.axvspan(0, args.t_cap * 1e4, color="0.95", zorder=0)
            a.set_xlim(0, tmax_plot * 1e4)  # full simulated time
            a.set_ylim(ymin * 0.8, ymax * 1.2)
            a.grid(alpha=0.3, which="both")
            a.legend(fontsize=6, loc="lower left", ncol=1)
            if i == 0:
                a.set_title(f"vary {AXLABEL[ax]}", fontsize=9)
            if i == nq - 1:
                a.set_xlabel(r"$t\ (\times10^{-4})$")
            if j == 0:
                pct = int(round(q * 100))
                a.set_ylabel(f"$R_{{{q:g}}}(t)$\n(radius of {pct}% particles)",
                             fontsize=8)
    note = ""
    if qs_missing:
        note = ("   [R_%s not logged in these runs; "
                "showing nearest saved bulk quantiles]"
                % "/".join("%g" % q for q in qs_missing))
    fig.suptitle("Mass-quantile particle radii $R_q(t)$ across setups "
                 "(recon-free, from particle positions)" + note, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fd = os.path.join(args.run_dir, "figures")
    os.makedirs(fd, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"{args.out_prefix}.{ext}"),
                    dpi=200, bbox_inches="tight")
    print(f"plotted q = {', '.join('%g' % q for q in qs_present)}")
    print(f"wrote {out_csv}")
    print(f"wrote {fd}/{args.out_prefix}.pdf/.png")


if __name__ == "__main__":
    main()
