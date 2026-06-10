"""
Weighted-particles-with-resampling baseline for the localized-growth comparison.
================================================================================

Referee request (recurring): compare branching not only against raw weighted
particles but against weighted particles WITH standard resampling, since
branching is closely related to local resampling.

This script runs the weighted scheme with SYSTEMATIC resampling on the same
PDE/configuration as experiment.py, under two policies:
  - 'ess':    resample when global nESS < ess_threshold (standard adaptive SMC)
  - 'always': resample every step
Mass handling (finite measure): resampling draws N0 indices proportional to the
weights and assigns every offspring the equal weight wbar = (sum w)/N0, so the
total represented mass (sum w)/N0 * M0 is exactly conserved through the step.

Diagnostics mirror experiment.py / cost_match.py: final-time relative L2 and
peak errors against the deterministic spectral reference, global nESS,
max/mean weight, number of resample events, the local particle count N_B in
the predefined region B={G>=eta} (after resampling, duplicated particles DO
concentrate in B, so N_B is the relevant local-resolution diagnostic), and
runtime. Same seeds and shared transport stream as the other baselines.

Run:  python resample_baseline.py          (full: N0 in {2e4, 1.31e5} x 4 seeds x 2 policies)
      python resample_baseline.py --smoke
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

N0_LIST = [20000, 131072]
SEEDS = [0, 1, 2, 3]
POLICIES = ["ess", "always"]
ESS_THRESHOLD = 0.5
if "--smoke" in sys.argv:
    N0_LIST = [2000]
    SEEDS = [0, 1]

cfg = dict(CONFIG)
rd = "results/resample_baseline"
os.makedirs(rd, exist_ok=True)
with open(os.path.join(rd, "config_used.json"), "w") as f:
    json.dump(dict(cfg=cfg, N0_LIST=N0_LIST, SEEDS=SEEDS, POLICIES=POLICIES,
                   ESS_THRESHOLD=ESS_THRESHOLD, resample_seed_offset=10_000,
                   numpy=np.__version__, jax=jax.__version__), f, indent=2)

period = jnp.asarray(PERIOD)
n = cfg["grid"]; cell_area = (L / n) ** 2; eta = cfg["eta"]
tau = cfg["tau"]; steps = int(round(cfg["T"] / tau))
density_estimation, _, density_evaluate_grid = generate_density_estimation(n_freq=cfg["K"], period=PERIOD)
ref_u0, advance_ref, xs, XX, YY, Gg = reference_solver(cfg)

u_ref = ref_u0.copy()
for _ in range(steps):
    u_ref = advance_ref(u_ref)
M0 = float(np.sum(ref_u0) * cell_area)


def systematic_resample(rng, w):
    """Systematic resampling: returns N0 indices drawn proportional to w."""
    assert np.isfinite(w).all() and w.sum() > 0, "invalid weights in resample"
    N = w.shape[0]
    p = w / np.sum(w)
    positions = (rng.random() + np.arange(N)) / N
    cumsum = np.cumsum(p)
    cumsum[-1] = 1.0  # guard against round-off
    return np.clip(np.searchsorted(cumsum, positions, side="right"), 0, N - 1)


rows = []
for N0 in N0_LIST:
    for policy in POLICIES:
        for seed in SEEDS:
            key = jax.random.PRNGKey(seed)
            key, k_init = jax.random.split(key)
            X, _ = sample_initial_particles(k_init, N0, cfg)
            w = jnp.ones((N0,), dtype=jnp.float64)
            rng_np = np.random.default_rng(10_000 + seed)  # resampling stream
            n_resamples = 0
            t0 = time.time()
            for s in range(steps):
                key, kT = jax.random.split(key)
                dW = jax.random.normal(kT, shape=(N0, 2), dtype=jnp.float64)
                rW = r_of(X, cfg)
                X = wrap_torus(em_transport(X, jnp.zeros_like(X), cfg["D"], tau, dW), period)
                w = reaction_weighted(w, rW, tau)
                # --- resampling policy ---
                w_np = np.asarray(w)
                g_ness = float((w_np.sum() ** 2) / (N0 * np.sum(w_np ** 2)))
                if policy == "always" or (policy == "ess" and g_ness < ESS_THRESHOLD):
                    idx = systematic_resample(rng_np, w_np)
                    X = jnp.asarray(np.asarray(X)[idx])
                    wbar = w_np.sum() / N0          # equal weights, mass conserved exactly
                    w = jnp.full((N0,), wbar, dtype=jnp.float64)
                    n_resamples += 1
            runtime = time.time() - t0

            sum_w = float(jnp.sum(w))
            mass_w = (sum_w / N0) * M0
            mask = jnp.ones((N0,), dtype=bool)
            u = reconstruct_field(density_estimation, density_evaluate_grid, X, w, mask, mass_w, XX, YY)
            m = grid_metrics(u, u_ref, XX, YY, Gg, eta, cell_area)
            w_np = np.asarray(w)
            gl = float(nESS(jnp.asarray(w_np)))
            mw = float(np.max(w_np) / np.mean(w_np))
            Gpart = np.asarray(G_of(X, cfg))
            loc = Gpart >= eta
            N_B = int(np.sum(loc))
            neb = float(nESS(jnp.asarray(w_np[loc]))) if N_B > 0 else np.nan
            rec = dict(N0=N0, policy=policy, seed=seed, L2_rel_err=m["L2_rel_err"],
                       peak_height_err=m["peak_height_err"], global_nESS=gl, local_nESS_B=neb,
                       max_w_over_mean_w=mw, N_local_B=N_B, n_resamples=n_resamples,
                       runtime_s=runtime)
            rows.append(rec)
            print(f"N0={N0} {policy:6s} seed={seed}: L2rel={m['L2_rel_err']:.4f} "
                  f"peakErr={m['peak_height_err']:.2f} nESS={gl:.3f} N_B={N_B} "
                  f"resamples={n_resamples} rt={runtime:.1f}s", flush=True)

cols = ["N0", "policy", "seed", "L2_rel_err", "peak_height_err", "global_nESS",
        "local_nESS_B", "max_w_over_mean_w", "N_local_B", "n_resamples", "runtime_s"]
with open(os.path.join(rd, "resample_baseline.csv"), "w") as f:
    f.write(",".join(cols) + "\n")
    for r in rows:
        f.write(",".join(str(r[c]) for c in cols) + "\n")

import statistics as st
print("\n=== weighted + systematic resampling, summary (mean +/- std over seeds) ===")
for N0 in N0_LIST:
    for policy in POLICIES:
        sub = [r for r in rows if r["N0"] == N0 and r["policy"] == policy]
        if not sub:
            continue
        def ms(c):
            v = [r[c] for r in sub]
            return st.mean(v), (st.pstdev(v) if len(v) > 1 else 0.0)
        l2 = ms("L2_rel_err"); pk = ms("peak_height_err"); nb = ms("N_local_B"); rs = ms("n_resamples")
        print(f"N0={N0:7d} {policy:6s}: L2rel={l2[0]:.4f}±{l2[1]:.4f} peakErr={pk[0]:.2f}±{pk[1]:.2f} "
              f"N_B={nb[0]:.0f} resamples={rs[0]:.0f}")
print("wrote", os.path.join(rd, "resample_baseline.csv"))
