"""
Figure-data driver for Sec. 5.2 (stationary localized growth): the weighted+ESS pieces
that the original runs did not save.
=====================================================================================

experiment.py saved fields_seed*.npz (reference/weighted/poisson/minvar) and metrics.csv
(weighted/poisson/minvar L2-vs-t); resample_baseline.py saved only FINAL-time ESS scalars.
The revised Fig. 2 (2x2 fields incl. weighted+ESS) and Fig. 4 (L2-vs-t incl. weighted+ESS)
need the ESS field and the ESS L2-vs-time, which were never written.  This driver, on the
SAME config/reference, produces exactly those, cheaply:

  fig52_ess_l2_vs_t.csv      ESS relative-L2 error vs time, seeds 0..7  (Fig. 4 ESS curve).
  fig52_fields_seed0.npz     seed-0 FINAL fields reference/weighted/weighted_ess/minvar
                             on one shared color scale                  (Fig. 2 four panels).

The Fig. 4 weighted/minvar curves stay from the committed metrics.csv (8 seeds, canonical);
only the ESS curve is new here.  Weighted and ESS share one transport (CRN) stream so the
two differ only by the systematic resampling.  Minvar branching is run once (seed 0) just
for its final field.  Reuses experiment.py / resample_baseline.py machinery verbatim.

Run:  python figdata_52.py        (8 seeds; minutes)
      python figdata_52.py --smoke
"""
import os
import sys
import csv

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from experiment import (
    CONFIG, PERIOD, reference_solver, sample_initial_particles, reconstruct_field,
    grid_metrics, r_of, L, branch_compact,
)
from common_particle import (
    generate_density_estimation, em_transport, wrap_torus, reaction_weighted,
    reaction_minvar,
)

# Inlined from resample_baseline.py (importing it would run its module-level experiment).
ESS_THRESHOLD = 0.5


def systematic_resample(rng, w):
    """Systematic resampling: N0 indices drawn proportional to w (verbatim copy of
    resample_baseline.systematic_resample so the ESS dynamics match exactly)."""
    assert np.isfinite(w).all() and w.sum() > 0, "invalid weights in resample"
    N = w.shape[0]
    p = w / np.sum(w)
    positions = (rng.random() + np.arange(N)) / N
    cumsum = np.cumsum(p)
    cumsum[-1] = 1.0
    return np.clip(np.searchsorted(cumsum, positions, side="right"), 0, N - 1)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(REPO, "reference_results", "branch_vs_weighted")

cfg = dict(CONFIG)
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
if "--smoke" in sys.argv:
    cfg = dict(cfg, tau=0.05, N0=2000); SEEDS = [0, 1]

N0 = cfg["N0"]; tau = cfg["tau"]; steps = int(round(cfg["T"] / tau))
n = cfg["grid"]; cell_area = (L / n) ** 2; eta = cfg["eta"]
buffer_size = cfg["buffer_mult"] * N0
period = jnp.asarray(PERIOD)
snap_steps = sorted(set(np.linspace(0, steps, cfg["n_snapshots"], dtype=int).tolist()))

de, _, deg = generate_density_estimation(n_freq=cfg["K"], period=PERIOD)
ref_u0, advance_ref, xs, XX, YY, Gg = reference_solver(cfg)
M0 = float(np.sum(ref_u0) * cell_area)

# reference timeline at the snapshot steps
u_ref = ref_u0.copy(); ref_snap = {0: u_ref.copy()}
for s in range(1, steps + 1):
    u_ref = advance_ref(u_ref)
    if s in snap_steps:
        ref_snap[s] = u_ref.copy()


def l2(u, u_ref_s):
    return float(grid_metrics(u, u_ref_s, XX, YY, Gg, eta, cell_area)["L2_rel_err"])


def run_seed(seed, do_minvar):
    """Return (times, ess_L2[t]) and, if do_minvar, the seed's final fields dict."""
    key = jax.random.PRNGKey(seed)
    key, k_init = jax.random.split(key)
    X_init, _ = sample_initial_particles(k_init, N0, cfg)
    Xw = X_init; ww = jnp.ones((N0,), dtype=jnp.float64)           # weighted
    Xe = X_init; we = jnp.ones((N0,), dtype=jnp.float64)           # weighted + ESS resample
    rng_e = np.random.default_rng(10_000 + seed)
    if do_minvar:
        Xb = np.zeros((buffer_size, 2)); Xb[:N0] = np.asarray(X_init)
        mb = np.zeros((buffer_size,), bool); mb[:N0] = True
        Xm = jnp.asarray(Xb); maskm = jnp.asarray(mb)

    ts, ess_l2 = [], []
    fields = None

    def snap(s):
        u_ref_s = ref_snap[s]
        ue = reconstruct_field(de, deg, Xe, we, jnp.ones((N0,), bool),
                               (float(jnp.sum(we)) / N0) * M0, XX, YY)
        ts.append(s * tau); ess_l2.append(l2(ue, u_ref_s))
        if do_minvar and s == steps:
            uw = reconstruct_field(de, deg, Xw, ww, jnp.ones((N0,), bool),
                                   (float(jnp.sum(ww)) / N0) * M0, XX, YY)
            nm = int(jnp.sum(maskm))
            um = reconstruct_field(de, deg, Xm, jnp.ones((buffer_size,)), maskm,
                                   (nm / N0) * M0, XX, YY)
            return dict(reference=np.asarray(u_ref_s), weighted=np.asarray(uw),
                        weighted_ess=np.asarray(ue), minvar=np.asarray(um))
        return None

    if 0 in snap_steps:
        snap(0)
    for s in range(1, steps + 1):
        key, kT, km = jax.random.split(key, 3)
        dW = jax.random.normal(kT, shape=(buffer_size, 2), dtype=jnp.float64)
        # weighted (fixed N0)
        Xw = wrap_torus(em_transport(Xw, jnp.zeros_like(Xw), cfg["D"], tau, dW[:N0]), period)
        ww = reaction_weighted(ww, r_of(Xw, cfg), tau)
        # weighted + ESS: SAME transport increment as weighted (CRN), resample on nESS
        Xe = wrap_torus(em_transport(Xe, jnp.zeros_like(Xe), cfg["D"], tau, dW[:N0]), period)
        we = reaction_weighted(we, r_of(Xe, cfg), tau)
        we_np = np.asarray(we)
        g_ness = float((we_np.sum() ** 2) / (N0 * np.sum(we_np ** 2)))
        if g_ness < ESS_THRESHOLD:
            idx = systematic_resample(rng_e, we_np)
            Xe = jnp.asarray(np.asarray(Xe)[idx])
            we = jnp.full((N0,), we_np.sum() / N0, dtype=jnp.float64)
        # minvar branching (seed 0 only, for the final field)
        if do_minvar:
            Xm = wrap_torus(em_transport(Xm, jnp.zeros_like(Xm), cfg["D"], tau, dW), period)
            nu = jnp.where(maskm, reaction_minvar(km, r_of(Xm, cfg), tau), 0)
            Xmb, mmb, ov, _ = branch_compact(Xm, nu, buffer_size)
            if ov:
                raise RuntimeError(f"minvar buffer overflow at step {s}")
            Xm, maskm = jnp.asarray(Xmb), jnp.asarray(mmb)
        if s in snap_steps:
            out = snap(s)
            if out is not None:
                fields = out
    return np.array(ts), np.array(ess_l2), fields


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rows = []
    fields0 = None
    for seed in SEEDS:
        ts, el2, fields = run_seed(seed, do_minvar=(seed == 0))
        if fields is not None:
            fields0 = fields
        for t, e in zip(ts, el2):
            rows.append(dict(seed=seed, method="weighted_ess", t=float(t), L2_rel_err=float(e)))
        print(f"  seed {seed}: ESS final L2 = {el2[-1]:.4f}", flush=True)
    with open(os.path.join(OUT, "fig52_ess_l2_vs_t.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "method", "t", "L2_rel_err"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    np.savez_compressed(os.path.join(OUT, "fig52_fields_seed0.npz"),
                        **{k: v.astype(np.float32) for k, v in fields0.items()},
                        extent=np.array([-np.pi, np.pi, -np.pi, np.pi]))
    print("wrote", os.path.join(OUT, "fig52_ess_l2_vs_t.csv"))
    print("wrote", os.path.join(OUT, "fig52_fields_seed0.npz"))
