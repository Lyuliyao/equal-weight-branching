"""plot_radial_response.py -- Figure B: radial delayed chemical response (plan 11).
Reads radial-production diagnostics.csv (never runs the solver).  Weak and delayed
arms share chi=1 and differ only in initial mass M (effective coupling F=M), so the
arms are keyed by LABEL+M, not chi.  Panels:
 (a) M_v(t) vs exact law + G_v(t)          -- chemical buildup (delayed arm)
 (b) R_0.2/0.5/0.8(t) for the delayed cfg  -- delayed cell response (t_turn, t_focus10)
 (c) R_0.5(t): weak vs delayed             -- forcing contrast across critical mass
 (d) R_0.5(t): N-refinement (delayed)      -- reconstruction-free convergence

CONFIG-SAFE GROUPING (validation-closure section 7.1): the baseline (N,K,tau) is taken
from EXPLICIT CLI selectors (--base_N/--base_K/--base_tau), never by auto-picking the
smallest tau present.  Seed means are grouped by the FULL config tuple
(label, M, N, K_dyn, tau); runs with a different time step or bandwidth are never pooled.
"""
import os, csv, glob, re, argparse, json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


def load(run_dir):
    runs = []
    for d in sorted(glob.glob(os.path.join(run_dir, "*_chi*_N*_K*_tau*_seed*"))):
        m = re.search(r"(\w+?)_chi([0-9.]+)_N(\d+)_K(\d+)_tau([0-9.e+-]+)_seed(\d+)",
                      os.path.basename(d))
        cs = os.path.join(d, "diagnostics.csv")
        if not m or not os.path.exists(cs):
            continue
        R = list(csv.DictReader(open(cs)))
        if not R:
            continue
        cols = {c: np.array([float(r[c]) for r in R]) for c in R[0]}
        Mval = np.nan
        mf = os.path.join(d, "manifest.json")
        if os.path.exists(mf):
            try:
                Mval = float(json.load(open(mf)).get("M", np.nan))
            except Exception:
                pass
        runs.append(dict(label=m[1], chi=float(m[2]), N=int(m[3]), K=int(m[4]),
                         tau=float(m[5]), seed=int(m[6]), M=Mval, cols=cols))
    return runs


def _match(r, sel):
    """Full-config match: ints/strs exact, floats (M, tau) with tolerance."""
    for k, v in sel.items():
        rv = r[k]
        if isinstance(v, float):
            if not np.isclose(rv, v, rtol=1e-6, atol=1e-12, equal_nan=True):
                return False
        elif rv != v:
            return False
    return True


def seedmean(runs, sel, col):
    grp = [r for r in runs if _match(r, sel) and col in r["cols"]]
    if not grp:
        return None, None, None
    tg = grp[0]["cols"]["t"]
    Y = np.array([np.interp(tg, r["cols"]["t"], r["cols"][col]) for r in grp])
    return tg, Y.mean(0), (Y.std(0) if len(Y) > 1 else np.zeros_like(tg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--base_N", type=int, default=100000, help="baseline particle count")
    ap.add_argument("--base_K", type=int, default=12, help="baseline bandwidth K_dyn")
    ap.add_argument("--base_tau", type=float, default=1e-3, help="baseline time step")
    ap.add_argument("--delayed_M", type=float, default=96.0)
    ap.add_argument("--weak_M", type=float, default=72.0)
    ap.add_argument("--out_prefix", default="figureB_radial_response")
    args = ap.parse_args()
    runs = load(args.run_dir)
    if not runs:
        raise SystemExit("no radial runs found")
    K, tau0, Md, Mw = args.base_K, args.base_tau, args.delayed_M, args.weak_M
    sd = dict(label="delayed", M=Md, N=args.base_N, K=K, tau=tau0)
    sw = dict(label="weak", M=Mw, N=args.base_N, K=K, tau=tau0)
    # N-refinement: delayed arm, baseline K & tau, all N present
    Ns = sorted(set(r["N"] for r in runs if r["label"] == "delayed"
                    and r["K"] == K and np.isclose(r["tau"], tau0)
                    and np.isclose(r["M"], Md, equal_nan=True)))
    nseed = len(set(r["seed"] for r in runs if _match(r, sd)))

    fig, ax = plt.subplots(1, 4, figsize=(17, 3.9))
    t, Mv, _ = seedmean(runs, sd, "M_v"); _, Mex, _ = seedmean(runs, sd, "M_v_exact")
    _, Gv, _ = seedmean(runs, sd, "G_v")
    if t is not None:
        ax[0].plot(t, Mv, "C0-", label="$M_v$"); ax[0].plot(t, Mex, "k--", label="exact law")
        a2 = ax[0].twinx(); a2.plot(t, Gv, "C2-", lw=1.2); a2.set_ylabel("$G_v$", color="C2")
        ax[0].set_title("(a) chemical buildup (delayed)"); ax[0].set_xlabel("t")
        ax[0].set_ylabel("$M_v$"); ax[0].legend(fontsize=8, loc="lower right")
    for q, c in [("R_0_2", "C0"), ("R_0_5", "C1"), ("R_0_8", "C3")]:
        t, m, s = seedmean(runs, sd, q)
        if t is not None:
            ax[1].plot(t, m, c, label=q.replace("R_0_", "$R_{0.") + "}$")
            ax[1].fill_between(t, m - s, m + s, color=c, alpha=0.2, lw=0)
    t5, m5, _ = seedmean(runs, sd, "R_0_5")
    if t5 is not None:
        tt = t5[np.argmax(m5)]; ax[1].axvline(tt, color="0.5", ls=":", lw=1)
        ax[1].text(tt, ax[1].get_ylim()[1] * 0.9, "$t_{turn}$", fontsize=7)
    ax[1].set_title(f"(b) delayed response (M={Md:g})"); ax[1].set_xlabel("t")
    ax[1].set_ylabel("$R_q^u$"); ax[1].legend(fontsize=8)
    for sel, lab, c in [(sw, f"weak (M={Mw:g})", "C0"), (sd, f"delayed (M={Md:g})", "C3")]:
        t, m, s = seedmean(runs, sel, "R_0_5")
        if t is not None:
            ax[2].plot(t, m, c, label=lab)
            ax[2].fill_between(t, m - s, m + s, color=c, alpha=0.2, lw=0)
    ax[2].set_title("(c) $R_{0.5}^u$: weak vs delayed"); ax[2].set_xlabel("t")
    ax[2].set_ylabel("$R_{0.5}^u$"); ax[2].legend(fontsize=8)
    for N in Ns:
        t, m, _ = seedmean(runs, dict(label="delayed", M=Md, N=N, K=K, tau=tau0), "R_0_5")
        if t is not None:
            ax[3].plot(t, m, label=f"N={N/1e3:g}k")
    ax[3].set_title("(d) $R_{0.5}^u$ vs $N$ (delayed)"); ax[3].set_xlabel("t")
    ax[3].set_ylabel("$R_{0.5}^u$"); ax[3].legend(fontsize=8)

    fig.suptitle(f"Figure B: 3D fully-parabolic radial delayed chemical response "
                 f"(reconstruction-free $R_q$; K={K}, tau={tau0:g}, N={args.base_N}, "
                 f"{nseed} seeds)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fd = os.path.join(args.run_dir, "figures"); os.makedirs(fd, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fd, f"{args.out_prefix}.{ext}"), dpi=200, bbox_inches="tight")
    pdd = os.path.join(args.run_dir, "plot_data"); os.makedirs(pdd, exist_ok=True)
    nref = {f"refine_N{N}_R_0_5": np.array(
                seedmean(runs, dict(label="delayed", M=Md, N=N, K=K, tau=tau0), "R_0_5")[:2],
                dtype=object) for N in Ns}
    np.savez(os.path.join(pdd, f"{args.out_prefix}.npz"),
             K=K, base_tau=tau0, baseN=args.base_N, Md=Md, Mw=Mw, Ns=np.array(Ns),
             **{f"delayed_{q}": np.array(seedmean(runs, sd, q)[:2], dtype=object)
                for q in ("R_0_2", "R_0_5", "R_0_8", "M_v", "M_v_exact", "G_v")},
             weak_R_0_5=np.array(seedmean(runs, sw, "R_0_5")[:2], dtype=object),
             **nref)
    if m5 is not None:
        R0 = m5[0]; foc = t5[m5 <= 0.9 * R0]; tturn = t5[np.argmax(m5)]
        print(f"delayed M={Md:g} K={K} tau={tau0:g}: R0.5 {R0:.4f}->{m5[-1]:.4f} "
              f"(ratio {m5[-1]/R0:.3f}), t_turn={tturn:.2f}, "
              f"t_focus10={foc[0] if foc.size else 'none'}")
    tw, mw, _ = seedmean(runs, sw, "R_0_5")
    if mw is not None:
        print(f"weak    M={Mw:g} K={K} tau={tau0:g}: R0.5 {mw[0]:.4f}->{mw[-1]:.4f} "
              f"(ratio {mw[-1]/mw[0]:.3f})")
    print(f"N-refinement (delayed): Ns={Ns}")
    print(f"wrote {fd}/{args.out_prefix}.pdf/.png and plot_data/{args.out_prefix}.npz")


if __name__ == "__main__":
    main()
