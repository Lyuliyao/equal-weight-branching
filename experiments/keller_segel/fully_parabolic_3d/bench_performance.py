"""bench_performance.py -- validation-closure Task A.2 performance benchmark.

After one warm-up compilation, time 3 reps of the coupled solver:
  case 1: N=20000,  K_dyn=8,  T=0.1, tau=1e-3 -- fast=False vs fast=True (speedup)
  case 2: N=100000, K_dyn=12, T=0.2, tau=1e-3 -- fast=True
Reports compile/warm-up time, steady-state runtime, seconds/step, peak v occupancy,
buffer capacity, JAX device, and the fast/slow speedup for case 1.

Run on GPU via submit_bench.sb.  Writes benchmark.csv + benchmark.json to --out_dir.
"""
import os, csv, json, time, argparse, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import simulation_pp3d as S
import repro


def _cfg(N, K, T, tau):
    return dict(experiment="radial", ic="radial", L=12.0, Du=1.0, Dv=1.0, alpha=1.0,
                beta=1.0, chi=1.0, M=96.0, sigma=0.45, v0=0.0, K_dyn=K, K_test=4, Nu=N,
                tau=tau, n_steps=int(round(T / tau)), kernel="minvar", buffer_factor=1.6)


def time_runs(cfg, fast, reps=3, seed=0):
    """Return (warmup_s, steady_mean_s, steady_std_s, summary). First run includes the
    JIT compile (fast=True) / first-trace cost; the next `reps` are steady-state."""
    t0 = time.perf_counter()
    _, summ, _ = S.simulate(cfg, seed=seed, diag_every=cfg["n_steps"], fast=fast)
    warm = time.perf_counter() - t0
    ts = []
    for r in range(reps):
        t0 = time.perf_counter()
        _, summ, _ = S.simulate(cfg, seed=seed, diag_every=cfg["n_steps"], fast=fast)
        ts.append(time.perf_counter() - t0)
    return warm, float(np.mean(ts)), float(np.std(ts)), summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    try:
        import jax
        device = str(jax.devices()[0])
    except Exception:
        device = "unknown"

    rows = []
    # ---- case 1: fast vs slow ----
    c1 = _cfg(20000, 8, 0.1, 1e-3)
    Ncap1 = int(np.ceil(1.6 * c1["Nu"]))
    res = {}
    for fast in (False, True):
        warm, mean, std, summ = time_runs(c1, fast, reps=args.reps)
        res[fast] = mean
        rows.append(dict(case="case1", N=c1["Nu"], K_dyn=c1["K_dyn"], T=0.1, tau=1e-3,
                         n_steps=c1["n_steps"], fast=fast, warmup_s=round(warm, 3),
                         steady_mean_s=round(mean, 3), steady_std_s=round(std, 3),
                         sec_per_step=round(mean / c1["n_steps"], 5),
                         peak_v_occupancy=summ["max_v_occupancy"],
                         buffer_capacity=(Ncap1 if fast else None), device=device))
    speedup1 = res[False] / res[True] if res[True] > 0 else float("nan")

    # ---- case 2: fast only ----
    c2 = _cfg(100000, 12, 0.2, 1e-3)
    Ncap2 = int(np.ceil(1.6 * c2["Nu"]))
    warm, mean, std, summ = time_runs(c2, True, reps=args.reps)
    rows.append(dict(case="case2", N=c2["Nu"], K_dyn=c2["K_dyn"], T=0.2, tau=1e-3,
                     n_steps=c2["n_steps"], fast=True, warmup_s=round(warm, 3),
                     steady_mean_s=round(mean, 3), steady_std_s=round(std, 3),
                     sec_per_step=round(mean / c2["n_steps"], 5),
                     peak_v_occupancy=summ["max_v_occupancy"],
                     buffer_capacity=Ncap2, device=device))

    with open(os.path.join(args.out_dir, "benchmark.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    out = dict(rows=rows, case1_speedup_fast_over_slow=round(speedup1, 3), device=device)
    out.update(repro.env_record(_HERE))
    json.dump(out, open(os.path.join(args.out_dir, "benchmark.json"), "w"), indent=2)
    repro.write_command_txt(args.out_dir)
    print(f"case1 fast/slow speedup = {speedup1:.2f}x  (slow {res[False]:.2f}s -> "
          f"fast {res[True]:.2f}s); device={device}")
    for r in rows:
        print(f"  {r['case']} fast={r['fast']}: warmup {r['warmup_s']}s, steady "
              f"{r['steady_mean_s']}+-{r['steady_std_s']}s, {r['sec_per_step']*1e3:.2f} ms/step, "
              f"occ {r['peak_v_occupancy']}/{r['buffer_capacity']}")


if __name__ == "__main__":
    main()
