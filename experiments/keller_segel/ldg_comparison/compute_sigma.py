"""compute_sigma.py -- actual Gaussian width sigma(t) of the KS core, and the
blow-up time T_* defined by  sigma(T_*) = 0  (equivalently 1/sigma -> infinity).

For a 2D mass-M_u Gaussian  u(r) = (M/2 pi sigma^2) exp(-r^2/2 sigma^2):

  second moment   <|X-x_c|^2> = 2 sigma^2            -> sigma = sqrt(S_u/2)     [recon-free, all particles]
  core quantile   R_q = sigma*sqrt(-2 ln(1-q))       -> sigma = R_q / c_q       [recon-free, inner core]
  L2 norm         S   = M / (2 sqrt(pi) sigma)        -> sigma = M/(2 sqrt(pi) S)[bandwidth-sensitive]
  peak            P   = M / (2 pi sigma^2)            -> sigma = sqrt(M/(2 pi P))[bandwidth-sensitive]

All estimators are an ACTUAL length sigma(t) (physical units), not a proxy.

blow-up:  sigma(t) -> 0.  We fit the self-similar line sigma(t) = a + b t on the
pre-blow-up window [t0, t_cap<=1.2e-4] and take T_* = -a/b  (where sigma=0, i.e.
where 1/sigma diverges).  No t > 1.2e-4 (post-floor data) is used.
"""
import os, csv, glob, argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

LIT = 1.21e-4


def lf(t, y):
    A = np.vstack([np.ones_like(t), t]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = a + b * t
    ssr = np.sum((y - yh) ** 2); sst = np.sum((y - np.mean(y)) ** 2)
    return a, b, (1 - ssr / sst if sst > 0 else np.nan)


def seedmean(csvs, cols):
    per = []
    for f in csvs:
        R = list(csv.DictReader(open(f)))
        t = np.array([float(r["t"]) for r in R])
        per.append((t, {c: np.array([float(r[c]) for r in R]) for c in cols}))
    tg = np.linspace(max(t.min() for t, _ in per), min(t.max() for t, _ in per), 600)
    out = {c: np.mean([np.interp(tg, t, d[c]) for t, d in per], axis=0) for c in cols}
    return tg, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--config", default="sens_K5_tau2e-7_q0.8_N6400000",
                    help="config subdir prefix (default = baseline)")
    ap.add_argument("--t0", type=float, default=3e-5)
    ap.add_argument("--t_cap", type=float, default=1.2e-4)
    ap.add_argument("--M", type=float, default=10 * np.pi)
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.run_dir, args.config + "_seed*", "diag_*.csv")))
    if not csvs:
        raise SystemExit(f"no diag csv for {args.config}")
    cols = ["S_u", "R_0.1", "R_0.2", "R_0.3", "peak_PK_u", "S_L2_u"]
    tg, d = seedmean(csvs, cols)
    M = args.M

    # ---- actual sigma(t) from each estimator (physical length) ----
    sig = {
        "sig_mom  (<r^2>, recon-free)": np.sqrt(np.maximum(d["S_u"], 0) / 2.0),
        "sig_R0.1 (core, recon-free)": d["R_0.1"] / np.sqrt(-2 * np.log(0.9)),
        "sig_R0.2 (core, recon-free)": d["R_0.2"] / np.sqrt(-2 * np.log(0.8)),
        "sig_R0.3 (core, recon-free)": d["R_0.3"] / np.sqrt(-2 * np.log(0.7)),
        "sig_S    (L2, bandwidth)": M / (2 * np.sqrt(np.pi) * np.maximum(d["S_L2_u"], 1e-300)),
        "sig_peak (peak, bandwidth)": np.sqrt(M / (2 * np.pi * np.maximum(d["peak_PK_u"], 1e-300))),
    }

    # ---- table of sigma, 1/sigma, 1/sigma^2 at sample times (primary = sig_R0.2) ----
    samp = np.array([3e-5, 4.5e-5, 6e-5, 7.5e-5, 9e-5, 1.05e-4, 1.2e-4])
    prim = "sig_R0.2 (core, recon-free)"
    sp = np.interp(samp, tg, sig[prim])
    print(f"=== sigma(t) primary estimator: {prim}  (config {args.config}) ===")
    print("|   t        | sigma     | 1/sigma   | 1/sigma^2 |")
    print("|------------|-----------|-----------|-----------|")
    for t, s in zip(samp, sp):
        print(f"| {t:.2e} | {s:.3e} | {1/s:.3e} | {1/s**2:.3e} |")

    # ---- blow-up: fit sigma(t)=a+bt on [t0,t_cap], T_* where sigma=0 ----
    m = (tg >= args.t0) & (tg <= args.t_cap)
    print(f"\n=== blow-up T_* : sigma(t)=a+bt on [{args.t0:.0e},{args.t_cap:.1e}], "
          f"T_*=-a/b (sigma->0); lit ~ {LIT:.2e} ===")
    print("| estimator | sigma slope b | T_* (sigma=0) | R^2 |")
    print("|---|---|---|---|")
    Ts = {}
    for name, s in sig.items():
        a, b, r2 = lf(tg[m], s[m])
        T = -a / b if b < 0 else np.nan
        Ts[name] = (a, b, r2, T)
        print(f"| {name} | {b:.3e} | {T:.3e} | {r2:.3f} |")

    # ---- save CSV ----
    out = os.path.join(args.run_dir, f"sigma_of_t_{args.config}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["t"] + [n.split(" ")[0] for n in sig] + ["inv_sig_R0.2", "inv_sig2_R0.2"]
        w.writerow(hdr)
        for i in range(len(tg)):
            row = [f"{tg[i]:.6e}"] + [f"{sig[n][i]:.6e}" for n in sig]
            row += [f"{1/sig[prim][i]:.6e}", f"{1/sig[prim][i]**2:.6e}"]
            w.writerow(row)
    print(f"\nwrote {out}")

    # ---- figure: sigma, 1/sigma, 1/sigma^2 vs t ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    recon_free = [n for n in sig if "recon-free" in n]
    bw = [n for n in sig if "bandwidth" in n]
    win = (tg >= args.t0) & (tg <= args.t_cap)

    # panel 1: sigma(t) with linear fits crossing zero
    ax = axes[0]
    for name in sig:
        ls = "-" if "recon-free" in name else "--"
        ax.plot(tg * 1e4, sig[name], ls, lw=1.3, label=name.split(" ")[0])
    # fit line + T_* marker for the primary recon-free estimator
    a, b, r2, T = Ts[prim]
    tt = np.linspace(args.t0, T, 50)
    ax.plot(tt * 1e4, a + b * tt, "k:", lw=1.6)
    ax.axhline(0, color="k", lw=0.6)
    ax.plot([T * 1e4], [0], "k*", ms=12,
            label=f"$T_*$={T*1e4:.2f}e-4 ($\\sigma$=0)")
    ax.axvline(LIT * 1e4, color="r", ls="--", lw=1.0, label=f"lit {LIT:.2e}")
    ax.axvspan(args.t0 * 1e4, args.t_cap * 1e4, color="0.92", zorder=0)
    ax.set_xlabel(r"$t\ (\times10^{-4})$"); ax.set_ylabel(r"$\sigma(t)$")
    ax.set_title(r"(a) $\sigma(t)\to0$ at $T_*$"); ax.set_xlim(0, 1.6)
    ax.legend(fontsize=6.5, loc="upper right"); ax.grid(alpha=0.3)

    # panel 2: 1/sigma -> infinity
    ax = axes[1]
    for name in sig:
        ls = "-" if "recon-free" in name else "--"
        ax.plot(tg * 1e4, 1 / sig[name], ls, lw=1.3, label=name.split(" ")[0])
    ax.axvline(T * 1e4, color="k", ls=":", lw=1.2, label=f"$T_*$={T*1e4:.2f}e-4")
    ax.axvline(LIT * 1e4, color="r", ls="--", lw=1.0)
    ax.set_xlabel(r"$t\ (\times10^{-4})$"); ax.set_ylabel(r"$1/\sigma(t)$")
    ax.set_title(r"(b) $1/\sigma\to\infty$ at $T_*$"); ax.set_xlim(0, 1.6)
    ax.legend(fontsize=6.5, loc="upper left"); ax.grid(alpha=0.3)

    # panel 3: 1/sigma^2 -> infinity (peak-like)
    ax = axes[2]
    for name in sig:
        ls = "-" if "recon-free" in name else "--"
        ax.plot(tg * 1e4, 1 / sig[name] ** 2, ls, lw=1.3, label=name.split(" ")[0])
    ax.axvline(T * 1e4, color="k", ls=":", lw=1.2)
    ax.axvline(LIT * 1e4, color="r", ls="--", lw=1.0)
    ax.set_xlabel(r"$t\ (\times10^{-4})$"); ax.set_ylabel(r"$1/\sigma^2(t)$")
    ax.set_title(r"(c) $1/\sigma^2\to\infty$ at $T_*$"); ax.set_xlim(0, 1.6)
    ax.legend(fontsize=6.5, loc="upper left"); ax.grid(alpha=0.3)

    fig.suptitle(r"Core width $\sigma(t)$ and blow-up $T_*$ ($\sigma=0$): "
                 f"{args.config}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fd = os.path.join(args.run_dir, "figures")
    os.makedirs(fd, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"sigma_of_t.{ext}"), dpi=200, bbox_inches="tight")
    print(f"wrote {fd}/sigma_of_t.pdf/.png")


if __name__ == "__main__":
    main()
