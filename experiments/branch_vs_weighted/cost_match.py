"""
Cost-matched weighted baseline for the localized-growth comparison.
=====================================================================

Referee point: in experiment.py the branching schemes grow from N0=2e4 to ~1.3e5
particles, so their lower error could partly reflect more work. This script runs
the WEIGHTED scheme alone at several N0 (including N0 matched to the branching
final count) and reports the final-time relative L2 error, local nESS(B), max/mean
weight, and runtime. It demonstrates that increasing the global particle count
lowers the GLOBAL error (Monte-Carlo N^-1/2) but does NOT cure the LOCAL weight
degeneracy: nESS(B) and max/mean-weight are essentially N0-independent, because the
weights compound the same way regardless of how many particles carry them.

Run:  python cost_match.py        (full)
      python cost_match.py --smoke
Reuses experiment.py building blocks (same PDE, reference, reconstruction).
"""
import os, sys, json, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from experiment import (
    CONFIG, PERIOD, reference_solver, sample_initial_particles, reconstruct_field,
    grid_metrics, G_of, r_of, L,
)
from common_particle import (
    generate_density_estimation, em_transport, wrap_torus, reaction_weighted, nESS,
)

N0_LIST = [20000, 60000, 131072]
SEEDS = [0, 1, 2, 3]
if "--smoke" in sys.argv:
    N0_LIST = [2000, 8000]
    SEEDS = [0, 1]

cfg = dict(CONFIG)
cfg["smoke"] = False
rd = "results/cost_match"
os.makedirs(rd, exist_ok=True)

period = jnp.asarray(PERIOD)
n = cfg["grid"]; cell_area = (L / n) ** 2; eta = cfg["eta"]
tau = cfg["tau"]; steps = int(round(cfg["T"] / tau))
density_estimation, _, density_evaluate_grid = generate_density_estimation(n_freq=cfg["K"], period=PERIOD)
ref_u0, advance_ref, xs, XX, YY, Gg = reference_solver(cfg)

# reference final field (deterministic, N0-independent)
u_ref = ref_u0.copy()
for _ in range(steps):
    u_ref = advance_ref(u_ref)
M0 = float(np.sum(ref_u0) * cell_area)
Bmask_part_thresh = eta

rows = []
for N0 in N0_LIST:
    for seed in SEEDS:
        key = jax.random.PRNGKey(seed)
        key, k_init = jax.random.split(key)
        X, _ = sample_initial_particles(k_init, N0, cfg)
        w = jnp.ones((N0,), dtype=jnp.float64)
        mask = jnp.ones((N0,), dtype=bool)
        t0 = time.time()
        for s in range(steps):
            key, kT = jax.random.split(key)
            dW = jax.random.normal(kT, shape=(N0, 2), dtype=jnp.float64)
            rW = r_of(X, cfg)
            X = wrap_torus(em_transport(X, jnp.zeros_like(X), cfg["D"], tau, dW), period)
            w = reaction_weighted(w, rW, tau)
        runtime = time.time() - t0
        sum_w = float(jnp.sum(w))
        mass_w = (sum_w / N0) * M0
        u = reconstruct_field(density_estimation, density_evaluate_grid, X, w, mask, mass_w, XX, YY)
        m = grid_metrics(u, u_ref, XX, YY, Gg, eta, cell_area)
        w_np = np.asarray(w)
        gl = float(nESS(jnp.asarray(w_np)))
        mw = float(np.max(w_np) / np.mean(w_np))
        Gpart = np.asarray(G_of(X, cfg))
        loc = Gpart >= eta
        neb = float(nESS(jnp.asarray(w_np[loc]))) if np.sum(loc) > 0 else np.nan
        rec = dict(N0=N0, seed=seed, L2_rel_err=m["L2_rel_err"], peak_height_err=m["peak_height_err"],
                   global_nESS=gl, local_nESS_B=neb, max_w_over_mean_w=mw, runtime_s=runtime)
        rows.append(rec)
        print(f"N0={N0} seed={seed}: L2rel={m['L2_rel_err']:.4f} nESS={gl:.4f} nESS_B={neb:.4f} "
              f"maxw/mean={mw:.1f} peakErr={m['peak_height_err']:.2f} rt={runtime:.1f}s", flush=True)

cols = ["N0", "seed", "L2_rel_err", "peak_height_err", "global_nESS", "local_nESS_B",
        "max_w_over_mean_w", "runtime_s"]
with open(os.path.join(rd, "cost_match.csv"), "w") as f:
    f.write(",".join(cols) + "\n")
    for r in rows:
        f.write(",".join(str(r[c]) for c in cols) + "\n")

# aggregate
import statistics as st
print("\n=== cost-matched weighted summary (mean over seeds) ===")
for N0 in N0_LIST:
    sub = [r for r in rows if r["N0"] == N0]
    def mean(c): return st.mean(r[c] for r in sub)
    print(f"N0={N0:7d}: L2rel={mean('L2_rel_err'):.4f}  nESS_B={mean('local_nESS_B'):.4f}  "
          f"maxw/mean={mean('max_w_over_mean_w'):.1f}  peakErr={mean('peak_height_err'):.2f}  "
          f"rt={mean('runtime_s'):.1f}s")
print("wrote", os.path.join(rd, "cost_match.csv"))
