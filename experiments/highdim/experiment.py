"""
High-dimensional (4D / 6D) particle experiment -- kinetic localized growth.
=========================================================================

PDE on the phase-space torus [-pi,pi]^d, z = (x, v), d = d_x + d_v, d_x = d_v:

    d_t f + v . grad_x f = D_v Lap_v f + r[f](z) f ,
    r[f](z) = lambda * G_d(z) * (1 - alpha * m[f]) - beta ,
    m[f]    = integral G_d f dz ,   G_d(z) = prod_j (1 + cos z_j)/2 .

The reaction uses ONLY the scalar moment m[f], estimated grid-free as a
Monte-Carlo average over active particles -- no dense grid, no field solve,
in any dimension.

Three particle representations of the reaction are compared under IDENTICAL
initial particles and IDENTICAL transport (velocity-noise) increments per seed:
    weighted   : positions evolve; w_i *= exp(r_i tau).
    poisson    : equal-weight unbiased integer branching.
    minvar     : equal-weight minimum-variance integer branching.

Story to reproduce in high-D: weighted max/mean-weight grows and (local)
effective sample size collapses, while branching grows the active / local
particle count and keeps equal weights; the logistic factor (1 - alpha m)
stabilizes m(t).

Diagnostics are grid-free -> results/highdim/metrics.csv.
Final-time low-rank FHT reconstruction -> results/highdim/fht_d{d}_seed{seed}.npz.

Run:
    python experiment.py                 # full d=4 config
    python experiment.py --d6            # full d=6 config
    python experiment.py --smoke         # tiny d=4 smoke
    python experiment.py --smoke --d6    # tiny d=6 smoke
    python experiment.py --config my.json
"""

import os
import sys
import json
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from common_highdim import (
    G_d, G_d_np,
    moment_estimate,
    reaction_rate,
    phase_step,
    wrap_torus,
    reaction_weighted,
    reaction_poisson,
    reaction_minvar,
    branch_compact,
    nESS,
    sample_initial,
    build_fht,
    fht_marginal_1d,
    fht_marginal_2d,
    fht_diagonal,
    empirical_fourier_coeffs,
)

# ---------------------------------------------------------------------------
# CONFIGS
# ---------------------------------------------------------------------------
CONFIG_D4 = {
    "d": 4,                 # total phase-space dim (d_x = d_v = 2)
    "D_v": 0.1,
    "lambda": 6.0,
    "alpha": 1.0,
    "beta": 0.5,
    "eta": 0.5,             # local region B = {G_d >= eta}
    "T": 2.0,
    "tau": 0.02,            # steps = T/tau = 100
    "N0": 20000,
    "buffer_mult": 8,
    "sigma0": 1.0,          # initial wrapped-normal std per coord
    "n_snapshots": 21,
    "seeds": [0, 1, 2],
    # FHT reconstruction knobs (final time only)
    "fht_deg": 8,
    "fht_rank": 6,
    "fht_sketch": 5,
    "fht_grid": 41,         # points per axis for marginals / diagonal
    "results_dir": "results/highdim",
    "smoke": False,
}

CONFIG_D6 = dict(CONFIG_D4)
CONFIG_D6.update({
    "d": 6,                 # d_x = d_v = 3
})

SMOKE_OVERRIDES = {
    "N0": 2000,
    "tau": 0.04,            # steps = T/tau = 50 ; smoke trims T below too
    "T": 0.6,               # steps = 15
    "seeds": [0, 1],
    "n_snapshots": 8,
    "fht_grid": 25,
    "smoke": True,
}


def resolve_config(argv):
    if "--d6" in argv:
        cfg = dict(CONFIG_D6)
    else:
        cfg = dict(CONFIG_D4)
    if "--smoke" in argv:
        cfg.update(SMOKE_OVERRIDES)
    if "--config" in argv:
        i = argv.index("--config")
        with open(argv[i + 1]) as f:
            cfg.update(json.load(f))
    return cfg


# ---------------------------------------------------------------------------
# Run all three methods for one seed
# ---------------------------------------------------------------------------
def run_seed(seed, cfg, records):
    t0 = time.time()
    d = cfg["d"]
    d_x = d // 2
    d_v = d - d_x
    D_v = cfg["D_v"]
    lam = cfg["lambda"]
    alpha = cfg["alpha"]
    beta = cfg["beta"]
    eta = cfg["eta"]
    tau = cfg["tau"]
    steps = int(round(cfg["T"] / tau))
    N0 = cfg["N0"]
    buffer_size = cfg["buffer_mult"] * N0
    snap_steps = sorted(set(np.linspace(0, steps, cfg["n_snapshots"], dtype=int).tolist()))

    key = jax.random.PRNGKey(seed)
    key, k_init = jax.random.split(key)
    Z_init = sample_initial(k_init, N0, d, cfg["sigma0"])  # (N0, d) on torus

    # ----- WEIGHTED state -----
    Zw = Z_init
    ww = jnp.ones((N0,), dtype=jnp.float64)
    maskw = jnp.ones((N0,), dtype=bool)

    # ----- branching buffers -----
    def init_buffer():
        Zb = np.zeros((buffer_size, d), dtype=np.float64)
        Zb[:N0] = np.asarray(Z_init)
        mb = np.zeros((buffer_size,), dtype=bool)
        mb[:N0] = True
        return jnp.asarray(Zb), jnp.asarray(mb)

    Zp, maskp = init_buffer()
    Zm, maskm = init_buffer()
    overflow_p = False
    overflow_m = False

    onesbuf = jnp.ones((buffer_size,), dtype=jnp.float64)

    def record_snapshot(s):
        t = s * tau
        # ---- WEIGHTED diagnostics ----
        m_w = float(moment_estimate(Zw, ww, maskw, N0))
        sum_w = float(jnp.sum(ww * maskw))
        total_mass_w = sum_w / N0
        w_act = np.asarray(ww)[np.asarray(maskw)]
        global_nESS_w = float(nESS(jnp.asarray(w_act)))
        max_over_mean_w = float(np.max(w_act) / np.mean(w_act)) if w_act.size else np.nan
        Gw = np.asarray(G_d(Zw))
        loc_idx = (Gw >= eta) & np.asarray(maskw)
        n_local_w = int(np.sum(loc_idx))
        if np.sum(loc_idx) > 0:
            local_nESS_w = float(nESS(jnp.asarray(np.asarray(ww)[loc_idx])))
            local_mass_w = float(np.sum(np.asarray(ww)[loc_idx]) / N0)
        else:
            local_nESS_w = np.nan
            local_mass_w = 0.0

        # ---- POISSON diagnostics ----
        m_p = float(moment_estimate(Zp, onesbuf, maskp, N0))
        np_act = int(jnp.sum(maskp))
        Gp = np.asarray(G_d(Zp))
        loc_p = (Gp >= eta) & np.asarray(maskp)
        n_local_p = int(np.sum(loc_p))
        local_mass_p = n_local_p / N0

        # ---- MINVAR diagnostics ----
        m_m = float(moment_estimate(Zm, onesbuf, maskm, N0))
        nm_act = int(jnp.sum(maskm))
        Gm = np.asarray(G_d(Zm))
        loc_m = (Gm >= eta) & np.asarray(maskm)
        n_local_m = int(np.sum(loc_m))
        local_mass_m = n_local_m / N0

        rows = [
            dict(method="weighted", t=t, total_mass=total_mass_w, moment_m=m_w,
                 local_mass_B=local_mass_w, N_active=N0, N_local_B=n_local_w,
                 global_nESS=global_nESS_w, local_nESS_B=local_nESS_w,
                 max_w_over_mean_w=max_over_mean_w),
            dict(method="poisson", t=t, total_mass=np_act / N0, moment_m=m_p,
                 local_mass_B=local_mass_p, N_active=np_act, N_local_B=n_local_p,
                 global_nESS=np.nan, local_nESS_B=np.nan, max_w_over_mean_w=np.nan),
            dict(method="minvar", t=t, total_mass=nm_act / N0, moment_m=m_m,
                 local_mass_B=local_mass_m, N_active=nm_act, N_local_B=n_local_m,
                 global_nESS=np.nan, local_nESS_B=np.nan, max_w_over_mean_w=np.nan),
        ]
        for r in rows:
            rec = dict(seed=seed, d=d, runtime_s=np.nan)
            rec.update(r)
            records.append(rec)

    if 0 in snap_steps:
        record_snapshot(0)

    for s in range(1, steps + 1):
        key, kT, kp_r, km_r = jax.random.split(key, 4)

        # Common random numbers on FIXED-SIZE buffers: one velocity-noise stream
        # kT shared by all methods (since normal(kT,(M,dv))[:k]==normal(kT,(k,dv)),
        # the shared front particles get identical increments). All JAX ops act on
        # the full buffer + mask so XLA compiles each step once (dynamic slicing
        # Z[:nact] would recompile every step and exhaust memory). Inactive slots
        # are carried but their offspring count is forced to 0.
        xi_buf = jax.random.normal(kT, shape=(buffer_size, d_v), dtype=jnp.float64)

        # ---- WEIGHTED (fixed N0) ----
        m_w = moment_estimate(Zw, ww, maskw, N0)
        rW = reaction_rate(Zw, m_w, lam, alpha, beta)
        Zw = phase_step(Zw, d_x, d_v, D_v, tau, xi_buf[:N0])
        ww = reaction_weighted(ww, rW, tau)

        # ---- POISSON branching (full buffer + mask) ----
        m_p = moment_estimate(Zp, onesbuf, maskp, N0)
        rP = reaction_rate(Zp, m_p, lam, alpha, beta)
        Zp = phase_step(Zp, d_x, d_v, D_v, tau, xi_buf)
        nu_p = jnp.where(maskp, reaction_poisson(kp_r, rP, tau), 0)
        Zpb, mpb, ov_p, n_new_p = branch_compact(Zp, nu_p, buffer_size, d)
        if ov_p:
            raise RuntimeError(f"poisson buffer overflow at step {s} (n_new>{buffer_size}); increase buffer_mult")
        Zp, maskp = jnp.asarray(Zpb), jnp.asarray(mpb)

        # ---- MINVAR branching (full buffer + mask) ----
        m_m = moment_estimate(Zm, onesbuf, maskm, N0)
        rM = reaction_rate(Zm, m_m, lam, alpha, beta)
        Zm = phase_step(Zm, d_x, d_v, D_v, tau, xi_buf)
        nu_m = jnp.where(maskm, reaction_minvar(km_r, rM, tau), 0)
        Zmb, mmb, ov_m, n_new_m = branch_compact(Zm, nu_m, buffer_size, d)
        if ov_m:
            raise RuntimeError(f"minvar buffer overflow at step {s} (n_new>{buffer_size}); increase buffer_mult")
        Zm, maskm = jnp.asarray(Zmb), jnp.asarray(mmb)

        if s in snap_steps:
            record_snapshot(s)

    runtime = time.time() - t0
    for rec in records:
        if rec["seed"] == seed and np.isnan(rec["runtime_s"]):
            rec["runtime_s"] = runtime

    # ----- final-time clouds for marginals + FHT -----
    final = dict(
        Zw=np.asarray(Zw)[np.asarray(maskw)],
        ww=np.asarray(ww)[np.asarray(maskw)],
        Zp=np.asarray(Zp)[np.asarray(maskp)],
        Zm=np.asarray(Zm)[np.asarray(maskm)],
    )
    return overflow_p, overflow_m, runtime, final


# ---------------------------------------------------------------------------
# Marginals (raw histograms) + FHT reconstruction at final time
# ---------------------------------------------------------------------------
def save_marginals(seed, cfg, final):
    d = cfg["d"]
    rd = cfg["results_dir"]
    nb = 40  # histogram bins per coord
    edges = np.linspace(-np.pi, np.pi, nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    out = dict(d=d, seed=seed, edges=edges, centers=centers)
    for tag, Z, w in [("weighted", final["Zw"], final["ww"]),
                      ("poisson", final["Zp"], np.ones(final["Zp"].shape[0])),
                      ("minvar", final["Zm"], np.ones(final["Zm"].shape[0]))]:
        # 1D marginals (normalized to integrate to 1 over [-pi,pi])
        h1 = np.zeros((d, nb))
        for j in range(d):
            hh, _ = np.histogram(Z[:, j], bins=edges, weights=w, density=True)
            h1[j] = hh
        out[f"hist1d_{tag}"] = h1
        # one 2D marginal (coords 0,1) -- x-space pair
        H2_x, _, _ = np.histogram2d(Z[:, 0], Z[:, 1], bins=[edges, edges],
                                    weights=w, density=True)
        out[f"hist2d_x_{tag}"] = H2_x
        # one 2D marginal (coords 0, d_x) -- x0 vs v0 pair
        H2_xv, _, _ = np.histogram2d(Z[:, 0], Z[:, d // 2], bins=[edges, edges],
                                     weights=w, density=True)
        out[f"hist2d_xv_{tag}"] = H2_xv

    np.savez(os.path.join(rd, f"marginals_d{d}_seed{seed}.npz"), **out)
    return centers


def save_fht(seed, cfg, final):
    """Build the FHT low-rank reconstruction from the final POISSON cloud
    (equal-weight, branching) and extract 1D / 2D marginals + diagonal.

    Falls back to the empirical product-Fourier coefficient diagnostic if FHT
    construction raises.
    """
    d = cfg["d"]
    rd = cfg["results_dir"]
    deg = cfg["fht_deg"]
    ngrid = cfg["fht_grid"]
    grid_y = np.linspace(-1.0, 1.0, ngrid)
    zgrid = np.pi * grid_y

    Z = final["Zp"]  # equal-weight branching cloud
    y = np.asarray(Z) / np.pi  # to [-1,1]
    # subsample for tractable sketch if huge
    if y.shape[0] > 200000:
        idx = np.random.default_rng(seed).choice(y.shape[0], 200000, replace=False)
        y = y[idx]

    out = dict(d=d, seed=seed, grid_y=grid_y, zgrid=zgrid)
    used_fht = False
    fht_err = ""
    try:
        htn, meta = build_fht(y, d, deg=deg, r_val=cfg["fht_rank"],
                              s_val=cfg["fht_sketch"], verbose=True)
        m1 = np.zeros((d, ngrid))
        for j in range(d):
            m1[j] = fht_marginal_1d(htn, meta, j, grid_y)
        out["fht_marg1d"] = m1
        # 2D marginal coords (0,1)
        g2, ZZ0, ZZ1 = fht_marginal_2d(htn, meta, 0, 1, grid_y)
        out["fht_marg2d_x"] = g2
        out["fht_ZZ0"] = ZZ0
        out["fht_ZZ1"] = ZZ1
        # 2D marginal x0 vs v0
        g2xv, _, _ = fht_marginal_2d(htn, meta, 0, d // 2, grid_y)
        out["fht_marg2d_xv"] = g2xv
        # diagonal profile
        out["fht_diag"] = fht_diagonal(htn, meta, grid_y)
        used_fht = True
    except Exception as e:  # documented fallback
        fht_err = repr(e)
        print(f"[FHT] reconstruction failed: {fht_err}; using fallback Fourier coeffs")

    # always also store the cheap empirical product-Fourier coefficients
    a, b = empirical_fourier_coeffs(Z, np.ones(Z.shape[0]),
                                    np.ones(Z.shape[0], dtype=bool), n_modes=deg)
    out["emp_fourier_cos"] = a
    out["emp_fourier_sin"] = b
    out["used_fht"] = used_fht
    out["fht_err"] = fht_err

    np.savez(os.path.join(rd, f"fht_d{d}_seed{seed}.npz"), **out)
    return used_fht


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    argv = sys.argv[1:]
    cfg = resolve_config(argv)
    d = cfg["d"]
    print(f"=== High-dim particle experiment (d={d}) ===")
    print("backend:", jax.default_backend(), "| devices:", jax.devices())
    rd = cfg["results_dir"]
    os.makedirs(rd, exist_ok=True)
    with open(os.path.join(rd, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    records = []
    overflow_any = False
    used_fht_any = False
    t_all = time.time()
    for seed in cfg["seeds"]:
        ov_p, ov_m, rt, final = run_seed(seed, cfg, records)
        overflow_any = overflow_any or ov_p or ov_m
        save_marginals(seed, cfg, final)
        used = save_fht(seed, cfg, final)
        used_fht_any = used_fht_any or used
        print(f"seed {seed}: runtime {rt:.2f}s  N_active(poisson final)="
              f"{final['Zp'].shape[0]}  N_active(minvar final)={final['Zm'].shape[0]}  "
              f"overflow_p={ov_p} overflow_m={ov_m}  fht_used={used}")

    # write metrics CSV (append d into filename-independent single csv)
    cols = ["seed", "d", "method", "t", "total_mass", "moment_m", "local_mass_B",
            "N_active", "N_local_B", "global_nESS", "local_nESS_B",
            "max_w_over_mean_w", "runtime_s"]
    csv_path = os.path.join(rd, "metrics.csv")
    # if file exists and has matching header AND no rows for this d, append;
    # simplest robust behaviour: write per-d then merge -> we write d-tagged csv
    # and also a combined metrics.csv. Here: write/overwrite metrics_d{d}.csv and
    # rebuild metrics.csv by concatenating all metrics_d*.csv present.
    d_csv = os.path.join(rd, f"metrics_d{d}.csv")
    with open(d_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for rec in records:
            f.write(",".join(str(rec.get(c, "")) for c in cols) + "\n")
    print("wrote", d_csv, "rows:", len(records))

    # rebuild combined metrics.csv from all metrics_d*.csv
    import glob
    all_lines = [",".join(cols)]
    for fp in sorted(glob.glob(os.path.join(rd, "metrics_d*.csv"))):
        with open(fp) as fin:
            lines = fin.read().splitlines()
        all_lines.extend(lines[1:])  # skip header
    with open(csv_path, "w") as f:
        f.write("\n".join(all_lines) + "\n")
    print("wrote combined", csv_path)

    if overflow_any:
        print("WARNING: branching buffer overflow occurred. "
              "Increase buffer_mult or reduce lambda/T or strengthen alpha.")
    print(f"FHT reconstruction used: {used_fht_any} (fallback Fourier coeffs always saved)")
    print(f"TOTAL wallclock: {time.time() - t_all:.2f}s")


if __name__ == "__main__":
    main()
