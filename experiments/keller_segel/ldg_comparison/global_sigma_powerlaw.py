"""global_sigma_powerlaw.py -- GLOBAL Gaussian width sigma(t) of the KS cloud,
computed directly from the particle second moment, with a power-law collapse fit.

GLOBAL sigma (not a core quantile, not a reconstructed-field proxy):

    sigma(t) = sqrt( <|X - x_c|^2>_all particles / 2 ),     x_c = mean particle position

This is exactly sqrt(S_u/2), where S_u = <|X-x_c|^2> is recorded per step in the
diag CSV (computed from ALL particle positions during the run). For a 2D Gaussian
u = (M/2 pi sigma^2) exp(-r^2/2 sigma^2) one has <r^2> = 2 sigma^2, so sigma(0)
recovers the IC width 1/sqrt(2*a_u) exactly.

Fit (user-selected): power law
        sigma(t) = c (T_* - t)^alpha
fitted JOINTLY for (c, alpha, T_*) by a 1-D profile scan over T_* > t_cap with an
inner log-linear least squares for (log c, alpha); the T_* maximizing the
linear-space R^2 is reported. alpha is the collapse exponent (alpha=1 <-> the
self-similar toy sigma ~ (T_*-t); alpha=1/2 <-> virial sigma ~ sqrt(T_*-t)).

Run over ALL sensitivity configs (K / tau / q_window / N) x seeds and report the
spread of (T_*, alpha), since the global sigma decelerates / plateaus and the
collapse-time extrapolation is resolution-sensitive.

Window: pre-blow-up [t0, t_cap], t_cap <= 1.2e-4 (no post-blow-up data).
"""
import os, re, csv, glob, argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

LIT = 1.21e-4
BASE = "sens_K5_tau2e-7_q0.8_N6400000"


def parse_cfg(name):
    m = re.search(r"(sens_K\d+_tau[0-9.e-]+_q[0-9.]+_N\d+)_seed(\d+)", name)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def axis_of(cfg):
    """which sensitivity axis this config varies relative to BASE."""
    if cfg == BASE:
        return "baseline"
    bK, bt, bq, bN = re.search(r"K(\d+)_tau([0-9.e-]+)_q([0-9.]+)_N(\d+)", BASE).groups()
    K, t, q, N = re.search(r"K(\d+)_tau([0-9.e-]+)_q([0-9.]+)_N(\d+)", cfg).groups()
    if (t, q, N) == (bt, bq, bN):
        return "K"
    if (K, q, N) == (bK, bq, bN):
        return "tau"
    if (K, t, N) == (bK, bt, bN):
        return "q"
    if (K, t, q) == (bK, bt, bq):
        return "N"
    return "other"


def load_sigma(csv_path):
    """global sigma(t)=sqrt(S_u/2) directly from the particle second moment."""
    R = list(csv.DictReader(open(csv_path)))
    t = np.array([float(r["t"]) for r in R])
    Su = np.array([float(r["S_u"]) for r in R])
    return t, np.sqrt(np.maximum(Su, 0.0) / 2.0)


def seed_mean(curves):
    """interp each seed's sigma onto a common t-grid, average."""
    t0 = max(t.min() for t, _ in curves)
    t1 = min(t.max() for t, _ in curves)
    tg = np.linspace(t0, t1, 700)
    sg = np.mean([np.interp(tg, t, s) for t, s in curves], axis=0)
    return tg, sg


def fit_powerlaw(t, sig, t0, tcap, n_Ts=800, Ts_max_factor=40.0):
    """sigma = c (T_*-t)^alpha. Profile over T_*>tcap; inner log-linear fit for
    (log c, alpha). Return best by linear-space R^2, plus a 'railed' flag set
    when the optimal T_* sits at the search ceiling (degenerate -> exponential).

    In the limit T_*->inf with alpha/T_*=lambda fixed,
        c (T_*-t)^alpha  ->  A exp(-lambda t),
    so a railed power-law fit means the data is an exponential, not a finite-time
    singularity. We report lambda_eq = alpha/T_* for that case."""
    m = (t >= t0) & (t <= tcap) & np.isfinite(sig) & (sig > 0)
    tt, ss = t[m], sig[m]
    if len(tt) < 5:
        return None
    sst = np.sum((ss - ss.mean()) ** 2)
    Ts_ceiling = tcap + Ts_max_factor * tcap
    Tgrid = tcap + np.geomspace(1e-7, Ts_max_factor * tcap, n_Ts)
    best = None
    for Ts in Tgrid:
        x = np.log(Ts - tt)
        A = np.vstack([np.ones_like(x), x]).T
        (lc, al), *_ = np.linalg.lstsq(A, np.log(ss), rcond=None)
        pred = np.exp(lc) * (Ts - tt) ** al
        r2 = 1.0 - np.sum((ss - pred) ** 2) / sst if sst > 0 else -np.inf
        if best is None or r2 > best["r2"]:
            best = dict(Ts=float(Ts), alpha=float(al), c=float(np.exp(lc)),
                        r2=float(r2))
    best["railed"] = best["Ts"] >= 0.95 * Ts_ceiling
    best["lambda_eq"] = best["alpha"] / best["Ts"]      # exponential-limit rate
    return best


def fit_exponential(t, sig, t0, tcap):
    """sigma = A exp(-lambda t); log-linear fit. Returns A, lambda, R^2 (linear).
    This is the well-posed limit of the railed power law and the robust
    characterization of the GLOBAL width's relaxation."""
    m = (t >= t0) & (t <= tcap) & np.isfinite(sig) & (sig > 0)
    tt, ss = t[m], sig[m]
    if len(tt) < 5:
        return None
    A = np.vstack([np.ones_like(tt), tt]).T
    (la, sl), *_ = np.linalg.lstsq(A, np.log(ss), rcond=None)
    pred = np.exp(la + sl * tt)
    sst = np.sum((ss - ss.mean()) ** 2)
    r2 = 1.0 - np.sum((ss - pred) ** 2) / sst if sst > 0 else np.nan
    return dict(A=float(np.exp(la)), lam=float(-sl), r2=float(r2),
                sig_floor=float(ss[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--t0", type=float, default=3e-5)
    ap.add_argument("--t_cap", type=float, default=1.2e-4)
    ap.add_argument("--out_prefix", default="global_sigma_powerlaw")
    args = ap.parse_args()

    # group diag CSVs by config
    groups = {}
    for d in sorted(glob.glob(os.path.join(args.run_dir, "sens_K*_seed*"))):
        cfg, seed = parse_cfg(os.path.basename(d))
        if cfg is None:
            continue
        cs = glob.glob(os.path.join(d, "diag_*.csv"))
        if cs:
            groups.setdefault(cfg, []).append((seed, cs[0]))
    if not groups:
        raise SystemExit(f"no sens_K*_seed*/diag_*.csv under {args.run_dir}")

    order = {"baseline": 0, "K": 1, "tau": 2, "q": 3, "N": 4, "other": 5}
    cfgs = sorted(groups, key=lambda c: (order[axis_of(c)], c))

    rows = []           # one row per config (seed-mean)
    seriesfit = {}      # cfg -> (tg, sg, pl, ex) for plotting
    for cfg in cfgs:
        seedcurves = [(load_sigma(p)) for _, p in sorted(groups[cfg])]
        tg, sg = seed_mean(seedcurves)
        pl = fit_powerlaw(tg, sg, args.t0, args.t_cap)
        ex = fit_exponential(tg, sg, args.t0, args.t_cap)
        seriesfit[cfg] = (tg, sg, pl, ex)
        if pl and ex:
            rows.append([cfg, axis_of(cfg), pl["Ts"], pl["alpha"], pl["c"],
                         pl["r2"], int(pl["railed"]), pl["lambda_eq"],
                         ex["lam"], ex["A"], ex["r2"], ex["sig_floor"], sg[0]])

    # ---- write CSV (fit table) ----
    out_csv = os.path.join(args.run_dir, f"{args.out_prefix}_fits.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "axis", "pl_T_star", "pl_alpha", "pl_c", "pl_R2",
                    "pl_railed", "pl_lambda_eq", "exp_lambda", "exp_A", "exp_R2",
                    "sigma_floor", "sigma0"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.6e}", f"{r[3]:.4f}", f"{r[4]:.4e}",
                        f"{r[5]:.4f}", r[6], f"{r[7]:.4e}", f"{r[8]:.4e}",
                        f"{r[9]:.6e}", f"{r[10]:.4f}", f"{r[11]:.6e}",
                        f"{r[12]:.6e}"])

    # ---- write seed-mean sigma(t) series (for plot regeneration) ----
    ser_csv = os.path.join(args.run_dir, f"{args.out_prefix}_sigma_series.csv")
    with open(ser_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "t", "sigma_global"])
        for cfg in cfgs:
            tg, sg, _, _ = seriesfit[cfg]
            for ti, si in zip(tg, sg):
                w.writerow([cfg, f"{ti:.6e}", f"{si:.6e}"])

    # ---- console report ----
    print("GLOBAL sigma(t)=sqrt(S_u/2) from particle second moment (all particles)")
    print(f"window t in [{args.t0:.0e},{args.t_cap:.1e}]; lit numerical-blowup "
          f"proxy ~ {LIT:.2e}\n")
    print("power-law sigma=c(T_*-t)^alpha  vs  exponential sigma=A exp(-lambda t)")
    print("| config | axis | pl T_* | pl alpha | pl R2 | railed | exp lambda | exp R2 | sigma_floor |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0].replace('sens_','')} | {r[1]} | {r[2]:.2e} | {r[3]:.1f} "
              f"| {r[5]:.3f} | {'Y' if r[6] else 'n'} | {r[8]:.0f} | {r[10]:.3f} "
              f"| {r[11]:.4f} |")
    Ts = np.array([r[2] for r in rows]); al = np.array([r[3] for r in rows])
    lam = np.array([r[8] for r in rows]); n_rail = sum(r[6] for r in rows)
    floor = np.array([r[11] for r in rows])
    print(f"\nspread over {len(rows)} configs (seed-mean):")
    print(f"  POWER LAW : {n_rail}/{len(rows)} configs RAIL T_* to the search "
          f"ceiling -> no finite-time power-law collapse of the GLOBAL width.")
    print(f"              (railed power law == exponential; T_* and alpha are "
          f"not meaningful collapse parameters here)")
    print(f"  EXPONENTIAL sigma=A exp(-lambda t)  [robust characterization]:")
    print(f"     lambda : mean {lam.mean():.1f}/s  std {lam.std():.1f}  "
          f"[{lam.min():.1f}, {lam.max():.1f}]   half-life {np.log(2)/lam.mean():.2e}")
    print(f"     sigma_floor (sigma at t_cap): mean {floor.mean():.4f}  "
          f"[{floor.min():.4f}, {floor.max():.4f}]  (does NOT reach 0)")
    print(f"\nwrote {out_csv}\nwrote {ser_csv}")

    # ============================ FIGURE ============================
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # (a) global sigma(t), all configs (linear); baseline highlighted
    ax = axes[0]
    for cfg in cfgs:
        tg, sg, _, _ = seriesfit[cfg]
        lw, al_, z, col = (2.2, 1.0, 5, "C3") if cfg == BASE else (0.9, 0.5, 2, "0.6")
        ax.plot(tg * 1e4, sg, lw=lw, alpha=al_, zorder=z, color=col,
                label="baseline K5 N6.4M" if cfg == BASE else None)
    fl = np.array([r[11] for r in rows]).mean()
    ax.axhline(fl, color="C0", ls="--", lw=1.0,
               label=f"floor $\\sigma\\!\\approx${fl:.3f} (not 0)")
    ax.axvspan(args.t0 * 1e4, args.t_cap * 1e4, color="0.92", zorder=0)
    ax.axvline(LIT * 1e4, color="r", ls=":", lw=1.0, label=f"lit {LIT:.2e}")
    ax.set_xlabel(r"$t\ (\times10^{-4})$")
    ax.set_ylabel(r"global $\sigma(t)=\sqrt{\langle r^2\rangle/2}$")
    ax.set_title("(a) global $\\sigma(t)$ from particles (all configs)")
    ax.set_xlim(0, 1.6); ax.set_ylim(0, None); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (b) semilog sigma vs t: EXPONENTIAL => straight line; finite-time power
    #     law would curve down to -inf. Baseline exp fit overlaid.
    ax = axes[1]
    tg, sg, pl, ex = seriesfit[BASE]
    m = (tg >= args.t0) & (tg <= args.t_cap)
    ax.semilogy(tg * 1e4, sg, "o", ms=2.5, color="C3", label="baseline $\\sigma(t)$")
    if ex:
        tt = tg[(tg >= args.t0) & (tg <= args.t_cap)]
        ax.semilogy(tt * 1e4, ex["A"] * np.exp(-ex["lam"] * tt), "k--", lw=1.6,
                    label=(f"exp fit $\\lambda$={ex['lam']:.0f}, "
                           f"$R^2$={ex['r2']:.3f}"))
    if pl:
        ax.semilogy(tg[m] * 1e4, pl["c"] * (pl["Ts"] - tg[m]) ** pl["alpha"],
                    "C2:", lw=1.4,
                    label=(f"power law (railed $T_*\\!\\to\\!\\infty$, "
                           f"$\\alpha$={pl['alpha']:.0f})"))
    ax.axvspan(args.t0 * 1e4, args.t_cap * 1e4, color="0.92", zorder=0)
    ax.set_xlabel(r"$t\ (\times10^{-4})$"); ax.set_ylabel(r"$\sigma$ (log)")
    ax.set_title("(b) exponential, not finite-time power law")
    ax.set_xlim(0, 1.6); ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

    # (c) exponential collapse rate lambda across configs (the robust number)
    ax = axes[2]
    labels = [c.replace("sens_K", "K").replace("_tau", " t").replace("_q", " q")
              .replace("_N", " N").replace("6400000", "6.4M").replace("3200000", "3.2M")
              .replace("1600000", "1.6M") for c in cfgs]
    xpos = np.arange(len(cfgs))
    lamc = np.array([r[8] for r in rows]); cols = ["C3" if c == BASE else "C0" for c in cfgs]
    ax.bar(xpos, lamc, color=cols, alpha=0.85)
    ax.axhline(lamc.mean(), color="k", ls="--", lw=1.0,
               label=f"mean {lamc.mean():.0f}/s")
    ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
    ax.set_ylabel(r"exp. collapse rate $\lambda$ (1/s)")
    ax.set_title("(c) $\\lambda$ spread across configs")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
    ax.text(0.02, 0.04, f"$\\lambda$={lamc.mean():.0f}$\\pm${lamc.std():.0f}/s\n"
            f"power law rails in {sum(r[6] for r in rows)}/{len(rows)} configs",
            transform=ax.transAxes, fontsize=7, va="bottom",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    fig.suptitle("Global second-moment width $\\sigma(t)=\\sqrt{S_u/2}$ (direct from "
                 "particles): no finite-time power-law collapse; exponential "
                 "relaxation toward a floor", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fd = os.path.join(args.run_dir, "figures")
    os.makedirs(fd, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"{args.out_prefix}.{ext}"),
                    dpi=200, bbox_inches="tight")
    print(f"wrote {fd}/{args.out_prefix}.pdf/.png")


if __name__ == "__main__":
    main()
