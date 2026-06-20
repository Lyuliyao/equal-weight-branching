"""run_radial_pilot.py -- Experiment B pilot (plan section 6.2).

Logged forcing sweep F = chi*alpha*M/beta, v0=0, radial wrapped-Gaussian u0.
The normalized dynamics depend only on the effective coupling F, so we sweep it
two equivalent ways:

  --chis ...           fix M, vary chi   (original mode);
  --Ms   ... [--chi 1] fix chi=alpha=beta=D=1, vary the initial mass M
                       (preferred: F = M, the cleanest single knob).

Records EVERY attempted (forcing, seed), then picks two production configs by a
rule fixed BEFORE the runs (written to selection.json), keyed on the forcing F:

  weak-response   = smallest F with NO 10% focusing  (R_0.5(T) > 0.9 R_0.5(0));
  delayed-focusing= smallest F that DOES show >=10% focusing
                    (R_0.5 drops below 0.9 R_0.5(0)) with a clear turnover
                    (t_turn > 0: initial diffusion then attraction) AND stays
                    numerically stable (max drift_resolution_number < 2, v-buffer
                    occupancy < 1.5 N_u, no abort).

--fast uses the JITTED fixed-capacity grad-v buffer (verified equivalent to the
eager dynamic-cloud path in test_buffer_equiv.py).
"""
import os, csv, json, argparse, subprocess, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import simulation_pp3d as S

SELECTION_RULE = (
    "weak-response = smallest chi with R_0.5(T) > 0.9 R_0.5(0) (no 10% focusing); "
    "delayed-focusing = smallest chi with R_0.5(T) <= 0.9 R_0.5(0) AND t_turn>0 "
    "AND max drift_resolution_number<2 AND max v-occupancy<1.5 N_u AND no abort.")


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", _HERE, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def run_one(chi, M, seed, args, out_dir, fast):
    F = chi * M                                 # effective forcing (alpha=beta=1)
    n_steps = int(round(args.T / args.tau)); diag_every = max(1, int(round(0.04 / args.tau)))
    cfg = dict(experiment="radial", ic="radial", L=args.L, Du=args.D, Dv=args.D,
               alpha=1.0, beta=1.0, chi=chi, M=M, sigma=args.sigma, v0=0.0,
               K_dyn=args.K_dyn, K_test=4, Nu=args.N, tau=args.tau, n_steps=n_steps,
               kernel="minvar", buffer_factor=args.buffer_factor)
    aborted = False
    try:
        recs, summ, _ = S.simulate(cfg, seed=seed, diag_every=diag_every, fast=fast)
    except RuntimeError as e:
        aborted = True; recs, summ = [], {"max_v_occupancy": -1}
    d = os.path.join(out_dir, f"chi{chi:g}_M{M:g}_seed{seed}"); os.makedirs(d, exist_ok=True)
    if recs:
        cols = list(recs[0].keys())
        with open(os.path.join(d, "diagnostics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
            for r in recs:
                w.writerow(r)
    # per-run summary metrics
    if not recs:
        return dict(F=F, chi=chi, M=M, seed=seed, aborted=True, R05_ratio=np.nan,
                    t_focus10=np.nan, t_turn=np.nan, Cdrift_max=np.nan,
                    occ_max=summ["max_v_occupancy"], Nu=args.N)
    t = np.array([r["t"] for r in recs]); R05 = np.array([r["R_0_5"] for r in recs])
    Cd = np.array([r["drift_resolution_number"] for r in recs])
    R0 = R05[0]
    foc = t[R05 <= 0.9 * R0]
    return dict(F=F, chi=chi, M=M, seed=seed, aborted=False, R05_0=float(R0),
                R05_T=float(R05[-1]), R05_ratio=float(R05[-1] / R0),
                t_focus10=float(foc[0]) if foc.size else np.nan,
                t_turn=float(t[np.argmax(R05)]), Cdrift_max=float(np.nanmax(Cd)),
                occ_max=int(summ["max_v_occupancy"]), Nu=args.N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chis", type=float, nargs="+", default=None,
                    help="fix M, sweep chi (original mode)")
    ap.add_argument("--Ms", type=float, nargs="+", default=None,
                    help="fix chi=alpha=beta=D=1, sweep initial mass M (preferred)")
    ap.add_argument("--chi", type=float, default=1.0, help="fixed chi when sweeping --Ms")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--N", type=int, default=20000)
    ap.add_argument("--K_dyn", type=int, default=8)
    ap.add_argument("--tau", type=float, default=1e-3)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--L", type=float, default=12.0)
    ap.add_argument("--D", type=float, default=1.0)
    ap.add_argument("--M", type=float, default=10.0, help="fixed M when sweeping --chis")
    ap.add_argument("--sigma", type=float, default=0.45)
    ap.add_argument("--buffer_factor", type=float, default=1.6)
    ap.add_argument("--fast", action="store_true", help="JITTED fixed-capacity grad-v buffer")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # build the (chi, M) sweep points; both reduce to the forcing F = chi*M
    if args.Ms is not None:
        points = [(args.chi, M) for M in args.Ms]; sweep = "M"; forcing_desc = f"chi={args.chi:g}*M"
    elif args.chis is not None:
        points = [(chi, args.M) for chi in args.chis]; sweep = "chi"; forcing_desc = f"chi*M={args.M:g}"
    else:
        points = [(chi, args.M) for chi in [0.5, 1, 2, 4, 8, 16]]; sweep = "chi"; forcing_desc = "chi*M"
    # record the selection rule BEFORE running
    json.dump({"selection_rule": SELECTION_RULE, "git": git_hash(), "sweep": sweep,
               "points_chi_M": points, "seeds": args.seeds, "N": args.N, "K_dyn": args.K_dyn,
               "tau": args.tau, "T": args.T, "D": args.D, "sigma": args.sigma, "L": args.L,
               "fast": bool(args.fast), "buffer_factor": args.buffer_factor,
               "forcing": "F = chi*alpha*M/beta", "forcing_desc": forcing_desc},
              open(os.path.join(args.out_dir, "pilot_protocol.json"), "w"), indent=2)

    rows = []
    for chi, M in points:
        for seed in args.seeds:
            r = run_one(chi, M, seed, args, args.out_dir, args.fast)
            rows.append(r)
            print(f"  F={r['F']:g} (chi={chi:g},M={M:g}) seed={seed}: "
                  f"R05 ratio={r['R05_ratio']:.3f} t_focus10={r['t_focus10']} "
                  f"t_turn={r.get('t_turn', np.nan):.2f} Cdrift_max={r['Cdrift_max']:.2f} "
                  f"occ={r['occ_max']}/{int(1.5*args.N)} {'ABORT' if r['aborted'] else ''}",
                  flush=True)
    with open(os.path.join(args.out_dir, "pilot_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)

    # seed-mean per forcing F, apply rule
    Fs = sorted(set(r["F"] for r in rows))
    agg = {}
    for F in Fs:
        rr = [r for r in rows if r["F"] == F]
        focused = np.nanmean([r["R05_ratio"] for r in rr]) <= 0.9
        stable = (not any(r["aborted"] for r in rr)
                  and np.nanmax([r["Cdrift_max"] for r in rr]) < 2.0
                  and max(r["occ_max"] for r in rr) < 1.5 * args.N)
        turn = np.nanmean([r["t_turn"] for r in rr]) > 0
        valid = (not any(r["aborted"] for r in rr)
                 and np.all(np.isfinite([r["R05_ratio"] for r in rr])))
        agg[F] = dict(chi=rr[0]["chi"], M=rr[0]["M"],
                      R05_ratio=float(np.nanmean([r["R05_ratio"] for r in rr])),
                      valid=bool(valid), focused=bool(focused), stable=bool(stable),
                      turnover=bool(turn),
                      Cdrift_max=float(np.nanmax([r["Cdrift_max"] for r in rr])),
                      occ_max=int(max(r["occ_max"] for r in rr)))
    # weak = smallest VALID (non-aborted, finite) F that does not focus
    weak = next((F for F in Fs if agg[F]["valid"] and not agg[F]["focused"]), None)
    delayed = next((F for F in Fs if agg[F]["valid"] and agg[F]["focused"]
                    and agg[F]["stable"] and agg[F]["turnover"]), None)
    sel = dict(selection_rule=SELECTION_RULE, sweep=sweep, per_F={str(k): v for k, v in agg.items()},
               weak_response_F=weak, delayed_focusing_F=delayed,
               weak_response=(None if weak is None else dict(chi=agg[weak]["chi"], M=agg[weak]["M"])),
               delayed_focusing=(None if delayed is None else dict(chi=agg[delayed]["chi"], M=agg[delayed]["M"])))
    json.dump(sel, open(os.path.join(args.out_dir, "selection.json"), "w"), indent=2)
    print("\n=== pilot selection ===")
    for F in Fs:
        print(f"  F={F:g} (chi={agg[F]['chi']:g},M={agg[F]['M']:g}): "
              f"R05_ratio={agg[F]['R05_ratio']:.3f} focused={agg[F]['focused']} "
              f"stable={agg[F]['stable']} turn={agg[F]['turnover']} "
              f"Cd_max={agg[F]['Cdrift_max']:.2f} occ={agg[F]['occ_max']}")
    print(f"  -> weak-response F = {weak};  delayed-focusing F = {delayed}")


if __name__ == "__main__":
    main()
