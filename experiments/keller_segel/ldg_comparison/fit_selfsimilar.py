"""fit_selfsimilar.py -- self-similar collapse time T_* from the exponent-1 ansatz

The user's profile  f(x,t) = (1/(1-t)) exp(-x^2 / (2 (t-1)^2))  is a Gaussian whose
width sigma(t) = (T_* - t) shrinks LINEARLY (exponent 1), with T_* = 1 in that toy.
For a 2D mass-conserving Gaussian u(r) = (M/2 pi sigma^2) exp(-r^2/2sigma^2):

    core radius   R_q   ~ sigma^{ 1}          (reconstruction-free)
    L2 norm       S     ~ sigma^{-1}          (bandwidth-sensitive)
    peak          P     ~ sigma^{-2}          (bandwidth-sensitive)

So EVERY observable becomes linear in t once mapped to a sigma-proportional quantity:

    R_q          = a + b t        ("sigma"   , i.e. fit R_q directly)
    1 / S        = a + b t        ("1/sigma" , inverse-L2)
    P^{-1/2}     = a + b t        ("1/sigma^2", inverse-peak)

and the SAME blow-up time is the common x-intercept  T_* = -a/b.  Agreement across
the three is the real test of a finite-time self-similar singularity.

Fit window: t in [t0, t_cap] with t_cap <= 1.2e-4 (no post-blow-up data), per the
standing rule "do not use time larger than blow-up time".
"""
import os, re, csv, glob, argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

LIT = 1.21e-4
BASE = (5, "2e-7", 0.8, 6400000)


def lf(t, y):
    """least squares y = a + b t; return a, b, R^2."""
    A = np.vstack([np.ones_like(t), t]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = a + b * t
    ssr = np.sum((y - yh) ** 2); sst = np.sum((y - np.mean(y)) ** 2)
    return a, b, (1 - ssr / sst if sst > 0 else np.nan)


def parse(n):
    m = re.search(r"sens_K(\d+)_tau([0-9.e-]+)_q([0-9.]+)_N(\d+)_seed(\d+)", n)
    return (int(m[1]), m[2], float(m[3]), int(m[4])) if m else None


def axis_of(k):
    K, tau, q, N = k
    if k == BASE: return "baseline"
    if (tau, q, N) == (BASE[1], BASE[2], BASE[3]): return "K"
    if (K, q, N) == (BASE[0], BASE[2], BASE[3]): return "tau"
    if (K, tau, N) == (BASE[0], BASE[1], BASE[3]): return "q"
    return "N"


def seedmean(csvs, cols):
    """interp each requested column onto a common grid, average over seeds."""
    per = []
    for f in csvs:
        rows = list(csv.DictReader(open(f)))
        t = np.array([float(r["t"]) for r in rows])
        per.append((t, {c: np.array([float(r[c]) for r in rows]) for c in cols}))
    tg = np.linspace(max(t.min() for t, _ in per), min(t.max() for t, _ in per), 600)
    out = {c: np.mean([np.interp(tg, t, d[c]) for t, d in per], axis=0) for c in cols}
    return tg, out


def fit_obs(t, y, t0, tcap):
    """map an observable to a sigma-proportional quantity Y, fit Y=a+bt, T*=-a/b.
    returns (T_star, R2, a, b, Y, mask) ; Y is the transformed (sigma-like) series."""
    m = (t >= t0) & (t <= tcap) & np.isfinite(y) & (y > 0)
    if m.sum() < 4:
        return np.nan, np.nan, None, None, None, m
    a, b, r2 = lf(t[m], y[m])
    Tstar = -a / b if b != 0 else np.nan
    return Tstar, r2, a, b, y, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--t0", type=float, default=3e-5)
    ap.add_argument("--t_cap", type=float, default=1.2e-4)
    ap.add_argument("--qs", nargs="+", default=["0.1", "0.2", "0.3"])
    args = ap.parse_args()

    cols = [f"R_{q}" for q in args.qs] + ["S_L2_u", "peak_PK_u"]
    groups = {}
    for d in sorted(glob.glob(os.path.join(args.run_dir, "sens_K*_seed*"))):
        c = parse(os.path.basename(d))
        if not c:
            continue
        cs = glob.glob(os.path.join(d, "diag_*.csv"))
        if cs:
            groups.setdefault(c, []).append(cs[0])

    # observable -> transform to a sigma-proportional Y, with a label of the 1/sigma^p form
    def transforms(d):
        out = {}
        for q in args.qs:
            out[f"R_{q} (sigma)"] = d[f"R_{q}"]                    # ~ sigma
        out["1/S_L2 (1/sigma)"] = 1.0 / np.maximum(d["S_L2_u"], 1e-300)   # ~ sigma
        out["peak^-1/2 (1/sigma^2)"] = 1.0 / np.sqrt(np.maximum(d["peak_PK_u"], 1e-300))  # ~ sigma
        return out

    out_csv = os.path.join(args.run_dir, "selfsimilar_Tstar.csv")
    rows = []
    base_curves = None
    order = {"baseline": 0, "K": 1, "tau": 2, "q": 3, "N": 4}
    for k in sorted(groups, key=lambda k: (order[axis_of(k)], k[0], k[3])):
        tg, d = seedmean(groups[k], cols)
        tf = transforms(d)
        for name, y in tf.items():
            Ts, r2, a, b, _, m = fit_obs(tg, y, args.t0, args.t_cap)
            rows.append((axis_of(k), k[0], k[1], k[2], k[3], name, Ts, r2))
        if k == BASE:
            base_curves = (tg, d, tf)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["axis", "K", "tau", "q", "N", "observable", "T_star", "R2"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], r[4], r[5], f"{r[6]:.4e}", f"{r[7]:.3f}"])

    print(f"self-similar (exponent-1) collapse: fit sigma-proportional Y = a + b t,  "
          f"T* = -a/b,  t in [{args.t0:.0e}, {args.t_cap:.1e}]")
    print(f"literature continuum proxy ~ {LIT:.2e}\n")
    print("| axis | K | tau | q | N | observable | T* | R2 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]/1e6:.2f}M | {r[5]} | "
              f"{r[6]:.3e} | {r[7]:.2f} |")
    print(f"\nwrote {out_csv}")

    # baseline figure: the three sigma-proportional fits sharing one x-intercept
    if base_curves is not None:
        tg, d, tf = base_curves
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        groups_fig = [
            ("sigma : core radius $R_q$ (recon-free)",
             [n for n in tf if n.startswith("R_")]),
            ("$1/\\sigma$ : inverse $L^2$  $1/S$",
             ["1/S_L2 (1/sigma)"]),
            ("$1/\\sigma^2$ : inverse peak  $P^{-1/2}$",
             ["peak^-1/2 (1/sigma^2)"]),
        ]
        for ax, (title, names) in zip(axes, groups_fig):
            for name in names:
                y = tf[name]
                Ts, r2, a, b, _, m = fit_obs(tg, y, args.t0, args.t_cap)
                # normalize each curve to its own max-in-window for shared panel scale
                sc = np.max(y[m]) if m.any() else 1.0
                ax.plot(tg * 1e4, y / sc, ".", ms=3, alpha=0.5)
                tt = np.linspace(args.t0, Ts if np.isfinite(Ts) else args.t_cap, 50)
                ax.plot(tt * 1e4, (a + b * tt) / sc, "-", lw=1.4,
                        label=f"{name.split(' ')[0]}: $T_*$={Ts*1e4:.2f}, $R^2$={r2:.2f}")
                if np.isfinite(Ts):
                    ax.axvline(Ts * 1e4, ls=":", lw=0.8,
                               color=ax.lines[-1].get_color())
            ax.axvline(LIT * 1e4, color="r", ls="--", lw=1.0, label=f"lit {LIT:.2e}")
            ax.axhline(0, color="k", lw=0.5)
            ax.axvspan(args.t0 * 1e4, args.t_cap * 1e4, color="0.9", zorder=0)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(r"$t\ (\times 10^{-4})$")
            ax.set_ylabel("normalized $\\propto\\sigma$")
            ax.set_xlim(0, max(2.0, LIT * 1e4 * 1.3))
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(alpha=0.3)
        fig.suptitle("Self-similar collapse $\\sigma(t)\\propto(T_*-t)$: "
                     "radius, $1/L^2$, $1/\\sqrt{\\rm peak}$ share the x-intercept $T_*$ "
                     "(baseline K=5, $\\tau$=2e-7, q=0.8, N=6.4M)", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fd = os.path.join(args.run_dir, "figures")
        os.makedirs(fd, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(fd, f"selfsimilar_fits.{ext}"),
                        dpi=200, bbox_inches="tight")
        print(f"wrote {fd}/selfsimilar_fits.pdf/.png")


if __name__ == "__main__":
    main()
