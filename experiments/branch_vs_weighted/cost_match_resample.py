"""Archive the cost-matched ESS-resampling row of the §5.2 table (tab:bw_cost).

The localized-growth cost-match table reports a `weighted + resample (ESS)` row at
N0 = 3.8e4 (particle-steps 1.9e7) to compare against branching at matched work.
This driver reproduces exactly that run with the same machinery as
`resample_baseline.py` (ESS-triggered systematic resampling on the §5.2 PDE) and
writes a per-seed CSV so the table row is traceable (CLAUDE.md §9.4).

Run:  python cost_match_resample.py --N0 38000 --seeds 0 1 2 3
"""
import os
import sys
import csv
import json
import argparse

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment import (CONFIG, PERIOD, reference_solver, sample_initial_particles,
                        reconstruct_field, grid_metrics, G_of, r_of, L)
from common_particle import (generate_density_estimation, em_transport, wrap_torus,
                             reaction_weighted, nESS)


def systematic_resample(rng, w):
    N = w.shape[0]
    p = w / np.sum(w)
    positions = (rng.random() + np.arange(N)) / N
    cumsum = np.cumsum(p); cumsum[-1] = 1.0
    return np.clip(np.searchsorted(cumsum, positions, side="right"), 0, N - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N0", type=int, default=38000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--ess_threshold", type=float, default=0.5)
    ap.add_argument("--out_dir", type=str, default="results/cost_match_resample")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = dict(CONFIG)
    period = jnp.asarray(PERIOD)
    n = cfg["grid"]; cell_area = (L / n) ** 2; eta = cfg["eta"]
    tau = cfg["tau"]; steps = int(round(cfg["T"] / tau))
    density_estimation, _, density_evaluate_grid = generate_density_estimation(
        n_freq=cfg["K"], period=PERIOD)
    ref_u0, advance_ref, xs, XX, YY, Gg = reference_solver(cfg)
    u_ref = ref_u0.copy()
    for _ in range(steps):
        u_ref = advance_ref(u_ref)
    M0 = float(np.sum(ref_u0) * cell_area)

    rows = []
    for seed in args.seeds:
        key = jax.random.PRNGKey(seed); key, k_init = jax.random.split(key)
        X, _ = sample_initial_particles(k_init, args.N0, cfg)
        w = jnp.ones((args.N0,), dtype=jnp.float64)
        rng_np = np.random.default_rng(10_000 + seed)
        n_resamples = 0; ps = 0
        for s in range(steps):
            key, kT = jax.random.split(key)
            dW = jax.random.normal(kT, shape=(args.N0, 2), dtype=jnp.float64)
            rW = r_of(X, cfg)
            X = wrap_torus(em_transport(X, jnp.zeros_like(X), cfg["D"], tau, dW), period)
            w = reaction_weighted(w, rW, tau)
            ps += args.N0
            wn = np.asarray(w)
            g_ness = float((wn.sum() ** 2) / (args.N0 * np.sum(wn ** 2)))
            if g_ness < args.ess_threshold:
                idx = systematic_resample(rng_np, wn)
                X = jnp.asarray(np.asarray(X)[idx])
                w = jnp.full((args.N0,), wn.sum() / args.N0, dtype=jnp.float64)
                n_resamples += 1
        sum_w = float(jnp.sum(w)); mass_w = (sum_w / args.N0) * M0
        u = reconstruct_field(density_estimation, density_evaluate_grid, X, w,
                              jnp.ones((args.N0,), dtype=bool), mass_w, XX, YY)
        m = grid_metrics(u, u_ref, XX, YY, Gg, eta, cell_area)
        rows.append(dict(N0=args.N0, seed=seed, particle_steps=ps,
                         L2_rel_err=m["L2_rel_err"], peak_height_err=m["peak_height_err"],
                         global_nESS=float(nESS(jnp.asarray(np.asarray(w)))),
                         n_resamples=n_resamples))
        print(f"N0={args.N0} seed={seed}: L2rel={m['L2_rel_err']:.4f} ps={ps:.2e} "
              f"resamples={n_resamples}", flush=True)

    cols = ["N0", "seed", "particle_steps", "L2_rel_err", "peak_height_err",
            "global_nESS", "n_resamples"]
    out_csv = os.path.join(args.out_dir, f"cost_match_resample_N{args.N0}.csv")
    with open(out_csv, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=cols); wtr.writeheader(); wtr.writerows(rows)
    with open(os.path.join(args.out_dir, "config_used.json"), "w") as f:
        json.dump(dict(N0=args.N0, seeds=args.seeds, ess_threshold=args.ess_threshold,
                       PDE_config=cfg, particle_steps_per_seed=steps * args.N0), f, indent=2)
    l2 = [r["L2_rel_err"] for r in rows]
    print(f"\nN0={args.N0} ESS-resample: rel L2 = {np.mean(l2):.4f} +/- {np.std(l2):.4f} "
          f"at {steps*args.N0:.2e} particle-steps")
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
