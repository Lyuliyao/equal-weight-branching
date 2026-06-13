"""
Field-coupled 6D kinetic Keller-Segel particle experiment.
==========================================================

Phase space z = (x, v), x in T^3 = [-pi,pi]^3, v in R^3.  d = 6, d_x = d_v = 3.

PDE (corrected plan section 5.3):

    d_t f + v . grad_x f
        = gamma_v grad_v . ((v - chi grad_x c) f) + D_v Lap_v f + r[rho,c](x) f ,
    - Lap_x c + kappa^2 c = rho - rho_bar ,   rho(t,x) = integral f dv ,
    r[rho,c](x) = lambda_g S_c(c(x)) - alpha_rho S_rho(rho(x)) - beta .

The chemical field c and its gradient are solved SPECTRALLY from the particle
x-positions (field_kinetic): a small Fourier sketch with K_x modes per spatial
dim, the screened-Poisson divide  c_hat_k = rho_hat_k/(|k|^2+kappa^2), c_hat_0=0,
and real cos/sin evaluation at the particle positions.  No dense phase-space grid.

Four particle representations of the reaction are compared under IDENTICAL
initial particles and IDENTICAL transport (velocity-noise) increments per seed:
    weighted          : w_i *= exp(r_i tau).
    weighted_resample : weighted + ESS resample when global nESS < ess_thresh.
    poisson           : equal-weight unbiased integer branching.
    minvar            : equal-weight minimum-variance integer branching.

Diagnostics (grid-free where possible) -> CSV per snapshot; final clouds -> npz.

Run:
    python experiment_kinetic.py --config config_pilot.json
    python experiment_kinetic.py --config config_prod.json
    python experiment_kinetic.py --smoke           # tiny in-place smoke
"""

import os
import sys
import json
import time
import shutil

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from common_kinetic import (
    reaction_weighted,
    reaction_poisson,
    reaction_minvar,
    branch_compact,
    nESS,
    reaction_rate_field,
    S_c, S_rho,
    ou_velocity_step,
    euler_velocity_step,
    wrap_torus_x,
    sample_initial,
    mass_centroid_x,
    quantile_core_radii,
    local_region_mask,
    safe_corr,
    weighted_corr,
    reaction_histogram,
    mass_fraction_pos_neg,
    ess_resample,
    cfl_proxy,
)
from field_kinetic import (
    build_half_spectrum,
    density_coeffs,
    eval_field,
    eval_rho,
)

# ---------------------------------------------------------------------------
# DEFAULT CONFIG (authoritative plan section 5.3 parameters)
# ---------------------------------------------------------------------------
DEFAULT_CFG = {
    "d": 6,                 # d_x = d_v = 3 (fixed for this benchmark)
    "L": 2.0 * np.pi,       # spatial period (informational; torus is [-pi,pi]^3)
    "gamma_v": 2.0,
    "D_v": 1.0,
    "chi": 1.5,
    "kappa": 0.5,
    "lambda_g": 4.0,
    "alpha_rho": 1.0,
    "beta": 0.2,
    "c0": 0.1,
    "delta_c": 0.05,
    "rho0": 0.2,
    "T": 2.0,
    "tau": 2.0e-3,
    "N0": 20000,
    "seeds": [0, 1],
    "K_x": 8,               # Fourier modes per spatial dim (wavenumbers -K..K)
    "buffer_mult": 8,
    "init_kind": "single",  # "single" or "four"
    "sigma_x": 0.7,         # initial spatial blob std
    "Tv": 0.5,              # initial / stationary velocity temperature = D_v/gamma_v
    "velocity_scheme": "ou",  # "ou" (exact OU) or "euler" (Euler-Maruyama)
    "n_snapshots": 21,
    "ess_thresh": 0.5,      # resample trigger for weighted_resample
    "r_x": 1.0,             # local region B: spatial radius about x_c
    "r_v": 1.5,             # local region B: velocity radius about v_c
    "eval_grid_n": 32,      # coarse 32^3 eval grid for ||c||_inf, ||rho||_inf
    "dx_cfl": None,         # CFL dx; None => 2pi/(2 K_x + 1)
    "results_dir": "results/run",
    "save_clouds": True,
    "smoke": False,
}

SMOKE_OVERRIDES = {
    "N0": 3000,
    "tau": 2.0e-3,
    "T": 0.1,               # steps = T/tau = 50
    "seeds": [0],
    "n_snapshots": 6,
    "K_x": 6,
    "eval_grid_n": 16,
    "results_dir": "results/scratch_smoke",
    "smoke": True,
}


def resolve_config(argv):
    cfg = dict(DEFAULT_CFG)
    if "--config" in argv:
        i = argv.index("--config")
        with open(argv[i + 1]) as f:
            cfg.update(json.load(f))
    if "--smoke" in argv:
        cfg.update(SMOKE_OVERRIDES)
    return cfg


# ---------------------------------------------------------------------------
# Velocity step dispatch
# ---------------------------------------------------------------------------
def velocity_step(scheme, X, V, grad_c, gamma_v, chi, D_v, tau, xi):
    if scheme == "euler":
        return euler_velocity_step(X, V, grad_c, gamma_v, chi, D_v, tau, xi)
    return ou_velocity_step(X, V, grad_c, gamma_v, chi, D_v, tau, xi)


# ---------------------------------------------------------------------------
# Field solve at particle x-positions (shared helper).
#   Returns c (N,), grad_c (N,3), rho (N,) and the half-spectrum coeffs.
# ---------------------------------------------------------------------------
def field_at_particles(Xbuf, w, mask, N0, Kvecs, ksq, kappa):
    A, B = density_coeffs(Xbuf, w, mask, N0, Kvecs)
    M_f = jnp.sum(w * mask.astype(w.dtype)) / N0
    c, grad_c = eval_field(Xbuf, A, B, Kvecs, ksq, kappa)
    rho = eval_rho(Xbuf, A, B, Kvecs, M_f)
    return c, grad_c, rho, A, B, M_f


# ---------------------------------------------------------------------------
# CSV columns (one row per method per snapshot)
# ---------------------------------------------------------------------------
CSV_COLS = [
    "seed", "d", "method", "t", "step",
    "total_mass", "M_v_chem",          # M_v_chem: total "chemical" mass proxy (= M_f)
    "c_inf_grid", "rho_inf_grid",      # ||c||_inf, ||rho||_inf on coarse eval grid
    "c_inf_part", "rho_inf_part",      # same, evaluated at particle positions
    "R_core_0p5", "R_core_0p9",        # quantile core radii of x-marginal
    "xc0", "xc1", "xc2",               # spatial mass centroid
    "N_active", "global_nESS", "max_w_over_mean_w",
    "N_local_B", "local_ESS_B",        # local equal-weight count / weighted ESS in B
    "local_mass_B",
    "corr_r_c", "corr_r_rho",          # reaction-coupling correlations
    "mean_Sc", "mean_Srho", "mean_r",  # global mean activation / crowding / rate
    "mean_Sc_in", "mean_Sc_out",       # inside-core vs outside-core mean S_c
    "mean_r_in", "mean_r_out",         # inside-core vs outside-core mean r
    "frac_mass_rpos", "frac_mass_rneg",
    "cfl_proxy", "n_resample_events", "mass_jump_resample",
    "runtime_s",
]


# ---------------------------------------------------------------------------
# Per-method snapshot diagnostics
# ---------------------------------------------------------------------------
def snapshot_method(method, seed, d, t, step, X, V, w, mask, N0, cfg,
                    Kvecs, ksq, eval_grid, n_resample_events, mass_jump_resample,
                    runtime_s):
    """Compute all diagnostics for one method at one snapshot; return a dict row."""
    kappa = cfg["kappa"]
    lam_g = cfg["lambda_g"]
    alpha_rho = cfg["alpha_rho"]
    beta = cfg["beta"]
    c0 = cfg["c0"]
    delta_c = cfg["delta_c"]
    rho0 = cfg["rho0"]

    Xa = np.asarray(X)[np.asarray(mask)]
    Va = np.asarray(V)[np.asarray(mask)]
    wa = np.asarray(w)[np.asarray(mask)]
    n_act = Xa.shape[0]

    total_mass = float(np.sum(wa)) / N0
    M_f = total_mass

    # ---- field coefficients from the ACTIVE particles ----
    A, B = density_coeffs(jnp.asarray(X), jnp.asarray(w), jnp.asarray(mask),
                          N0, Kvecs)

    # ---- c, rho at particle positions ----
    c_part, grad_c_part = eval_field(jnp.asarray(Xa), A, B, Kvecs, ksq, kappa)
    rho_part = eval_rho(jnp.asarray(Xa), A, B, Kvecs, jnp.asarray(M_f))
    c_part = np.asarray(c_part)
    rho_part = np.asarray(rho_part)
    c_inf_part = float(np.max(np.abs(c_part))) if n_act else np.nan
    rho_inf_part = float(np.max(np.abs(rho_part))) if n_act else np.nan

    # ---- c, rho on the coarse eval grid ----
    c_g, _ = eval_field(eval_grid, A, B, Kvecs, ksq, kappa)
    rho_g = eval_rho(eval_grid, A, B, Kvecs, jnp.asarray(M_f))
    c_inf_grid = float(np.max(np.abs(np.asarray(c_g))))
    rho_inf_grid = float(np.max(np.abs(np.asarray(rho_g))))

    # ---- spatial centroid + core radii ----
    if n_act:
        xc = mass_centroid_x(Xa, wa)
        R05, R09 = quantile_core_radii(Xa, wa, xc, qs=(0.5, 0.9))
    else:
        xc = np.array([np.nan, np.nan, np.nan])
        R05 = R09 = np.nan

    # ---- reaction rate, activation, crowding at particles ----
    r_part = np.asarray(reaction_rate_field(
        jnp.asarray(c_part), jnp.asarray(rho_part),
        lam_g, alpha_rho, beta, c0, delta_c, rho0))
    Sc_part = np.asarray(S_c(jnp.asarray(c_part), c0, delta_c))
    Srho_part = np.asarray(S_rho(jnp.asarray(rho_part), rho0))

    # weighted means use wa
    Wsum = np.sum(wa) if n_act else 0.0
    if Wsum > 0:
        mean_Sc = float(np.sum(wa * Sc_part) / Wsum)
        mean_Srho = float(np.sum(wa * Srho_part) / Wsum)
        mean_r = float(np.sum(wa * r_part) / Wsum)
    else:
        mean_Sc = mean_Srho = mean_r = np.nan

    corr_r_c = weighted_corr(r_part, c_part, wa)
    corr_r_rho = weighted_corr(r_part, rho_part, wa)
    frac_pos, frac_neg = mass_fraction_pos_neg(r_part, wa, N0)

    # ---- global ESS / weight diagnostics (meaningful for weighted methods) ----
    if n_act:
        global_nESS = float(nESS(jnp.asarray(wa)))
        max_over_mean = float(np.max(wa) / np.mean(wa))
    else:
        global_nESS = np.nan
        max_over_mean = np.nan

    # ---- local phase-space region B (about x_c and v_c) ----
    if n_act:
        vc = np.average(Va, axis=0, weights=wa)
        in_B = local_region_mask(Xa, Va, xc, vc, cfg["r_x"], cfg["r_v"])
        n_local = int(np.sum(in_B))
        local_mass_B = float(np.sum(wa[in_B]) / N0)
        if n_local > 0:
            local_ESS_B = float(nESS(jnp.asarray(wa[in_B])))
        else:
            local_ESS_B = np.nan
    else:
        n_local = 0
        local_mass_B = 0.0
        local_ESS_B = np.nan

    # inside-vs-outside core for S_c and r (core = spatial torus ball radius R05)
    if n_act and np.isfinite(R05):
        from common_kinetic import torus_disp
        disp = torus_disp(Xa, xc)
        rr = np.sqrt(np.sum(disp ** 2, axis=1))
        inside = rr <= R05
        outside = ~inside
        def wmean(vals, msk):
            if np.sum(wa[msk]) > 0:
                return float(np.sum(wa[msk] * vals[msk]) / np.sum(wa[msk]))
            return np.nan
        mean_Sc_in = wmean(Sc_part, inside)
        mean_Sc_out = wmean(Sc_part, outside)
        mean_r_in = wmean(r_part, inside)
        mean_r_out = wmean(r_part, outside)
    else:
        mean_Sc_in = mean_Sc_out = mean_r_in = mean_r_out = np.nan

    # ---- CFL proxy ----
    dx_cfl = cfg["dx_cfl"] if cfg["dx_cfl"] else (2.0 * np.pi / (2 * cfg["K_x"] + 1))
    cfl = cfl_proxy(Va, cfg["tau"], dx_cfl) if n_act else np.nan

    row = dict(
        seed=seed, d=d, method=method, t=t, step=step,
        total_mass=total_mass, M_v_chem=M_f,
        c_inf_grid=c_inf_grid, rho_inf_grid=rho_inf_grid,
        c_inf_part=c_inf_part, rho_inf_part=rho_inf_part,
        R_core_0p5=R05, R_core_0p9=R09,
        xc0=float(xc[0]), xc1=float(xc[1]), xc2=float(xc[2]),
        N_active=n_act, global_nESS=global_nESS,
        max_w_over_mean_w=max_over_mean,
        N_local_B=n_local, local_ESS_B=local_ESS_B, local_mass_B=local_mass_B,
        corr_r_c=corr_r_c, corr_r_rho=corr_r_rho,
        mean_Sc=mean_Sc, mean_Srho=mean_Srho, mean_r=mean_r,
        mean_Sc_in=mean_Sc_in, mean_Sc_out=mean_Sc_out,
        mean_r_in=mean_r_in, mean_r_out=mean_r_out,
        frac_mass_rpos=frac_pos, frac_mass_rneg=frac_neg,
        cfl_proxy=cfl, n_resample_events=n_resample_events,
        mass_jump_resample=mass_jump_resample,
        runtime_s=runtime_s,
    )
    # also stash the reaction histogram counts for this method/snapshot
    hist_counts, hist_edges = reaction_histogram(r_part)
    return row, hist_counts, hist_edges


# ---------------------------------------------------------------------------
# Run all four methods for one seed under shared transport CRN
# ---------------------------------------------------------------------------
def run_seed(seed, cfg, records, hist_store):
    t0 = time.time()
    d = cfg["d"]
    assert d == 6, "this benchmark is fixed to d=6 (d_x=d_v=3)"
    d_x = 3
    d_v = 3
    gamma_v = cfg["gamma_v"]
    D_v = cfg["D_v"]
    chi = cfg["chi"]
    kappa = cfg["kappa"]
    tau = cfg["tau"]
    steps = int(round(cfg["T"] / tau))
    N0 = cfg["N0"]
    buffer_size = cfg["buffer_mult"] * N0
    scheme = cfg["velocity_scheme"]
    ess_thresh = cfg["ess_thresh"]
    snap_steps = sorted(set(np.linspace(0, steps, cfg["n_snapshots"], dtype=int).tolist()))

    # half-spectrum field operators (host -> device once)
    Kvecs_np, ksq_np = build_half_spectrum(cfg["K_x"])
    Kvecs = jnp.asarray(Kvecs_np)
    ksq = jnp.asarray(ksq_np)

    # coarse eval grid (cell-centered) for ||c||_inf, ||rho||_inf
    ng = cfg["eval_grid_n"]
    gs = (np.arange(ng) + 0.5) / ng * (2 * np.pi) - np.pi
    G1, G2, G3 = np.meshgrid(gs, gs, gs, indexing="ij")
    eval_grid = jnp.asarray(np.stack([G1.ravel(), G2.ravel(), G3.ravel()], axis=1))

    key = jax.random.PRNGKey(seed)
    key, k_init = jax.random.split(key)
    Z_init = sample_initial(k_init, N0, cfg["sigma_x"], cfg["Tv"], cfg["init_kind"])
    X_init = Z_init[:, :d_x]
    V_init = Z_init[:, d_x:]

    # ----- WEIGHTED state (fixed N0) -----
    Xw = X_init
    Vw = V_init
    ww = jnp.ones((N0,), dtype=jnp.float64)
    maskw = jnp.ones((N0,), dtype=bool)

    # ----- WEIGHTED + RESAMPLE state (fixed N0) -----
    Xr = X_init
    Vr = V_init
    wr = jnp.ones((N0,), dtype=jnp.float64)
    maskr = jnp.ones((N0,), dtype=bool)
    n_resample_events = 0
    mass_jump_resample = 0.0

    # ----- branching buffers (poisson, minvar) -----
    def init_buffer(Xi, Vi):
        Xb = np.zeros((buffer_size, d_x), dtype=np.float64)
        Vb = np.zeros((buffer_size, d_v), dtype=np.float64)
        Xb[:N0] = np.asarray(Xi)
        Vb[:N0] = np.asarray(Vi)
        mb = np.zeros((buffer_size,), dtype=bool)
        mb[:N0] = True
        return jnp.asarray(Xb), jnp.asarray(Vb), jnp.asarray(mb)

    Xp, Vp, maskp = init_buffer(X_init, V_init)
    Xm, Vm, maskm = init_buffer(X_init, V_init)
    onesbuf = jnp.ones((buffer_size,), dtype=jnp.float64)
    overflow_p = False
    overflow_m = False

    def record_snapshot(step):
        t = step * tau
        # weighted
        row_w, hc_w, he = snapshot_method(
            "weighted", seed, d, t, step, Xw, Vw, ww, maskw, N0, cfg,
            Kvecs, ksq, eval_grid, 0, 0.0, np.nan)
        # weighted + resample
        row_r, hc_r, _ = snapshot_method(
            "weighted_resample", seed, d, t, step, Xr, Vr, wr, maskr, N0, cfg,
            Kvecs, ksq, eval_grid, n_resample_events, mass_jump_resample, np.nan)
        # poisson
        row_p, hc_p, _ = snapshot_method(
            "poisson", seed, d, t, step, Xp, Vp, onesbuf, maskp, N0, cfg,
            Kvecs, ksq, eval_grid, 0, 0.0, np.nan)
        # minvar
        row_m, hc_m, _ = snapshot_method(
            "minvar", seed, d, t, step, Xm, Vm, onesbuf, maskm, N0, cfg,
            Kvecs, ksq, eval_grid, 0, 0.0, np.nan)
        for row in (row_w, row_r, row_p, row_m):
            records.append(row)
        # stash histograms (method, step) -> counts
        hist_store.append(dict(seed=seed, step=step, t=t, edges=he,
                               weighted=hc_w, weighted_resample=hc_r,
                               poisson=hc_p, minvar=hc_m))

    if 0 in snap_steps:
        record_snapshot(0)

    for s in range(1, steps + 1):
        key, kT, kp_r, km_r, k_rs = jax.random.split(key, 5)

        # Shared transport CRN on the FULL buffer; weighted uses xi_buf[:N0],
        # branching uses the full buffer (front N0 rows identical by construction
        # since normal(kT,(M,dv))[:k] == normal(kT,(k,dv))). All JAX ops act on
        # the full buffer + mask so XLA compiles each step once.
        xi_buf = jax.random.normal(kT, shape=(buffer_size, d_v), dtype=jnp.float64)
        xi_N0 = xi_buf[:N0]

        # ===== WEIGHTED (fixed N0) =====
        cW, gW, rhoW, _, _, _ = field_at_particles(Xw, ww, maskw, N0, Kvecs, ksq, kappa)
        rW = reaction_rate_field(cW, rhoW, cfg["lambda_g"], cfg["alpha_rho"],
                                 cfg["beta"], cfg["c0"], cfg["delta_c"], cfg["rho0"])
        Xw, Vw = velocity_step(scheme, Xw, Vw, gW, gamma_v, chi, D_v, tau, xi_N0)
        ww = reaction_weighted(ww, rW, tau)

        # ===== WEIGHTED + RESAMPLE (fixed N0) =====
        cR, gR, rhoR, _, _, _ = field_at_particles(Xr, wr, maskr, N0, Kvecs, ksq, kappa)
        rRr = reaction_rate_field(cR, rhoR, cfg["lambda_g"], cfg["alpha_rho"],
                                  cfg["beta"], cfg["c0"], cfg["delta_c"], cfg["rho0"])
        Xr, Vr = velocity_step(scheme, Xr, Vr, gR, gamma_v, chi, D_v, tau, xi_N0)
        wr = reaction_weighted(wr, rRr, tau)
        cur_nESS = float(nESS(wr))
        if cur_nESS < ess_thresh:
            Xr_np, Vr_np, wr_np, mb, ma = ess_resample(
                k_rs, np.asarray(Xr), np.asarray(Vr), np.asarray(wr), N0)
            Xr = jnp.asarray(Xr_np)
            Vr = jnp.asarray(Vr_np)
            wr = jnp.asarray(wr_np)
            n_resample_events += 1
            mass_jump_resample += abs(ma - mb)

        # ===== POISSON branching (full buffer + mask) =====
        cP, gP, rhoP, _, _, _ = field_at_particles(Xp, onesbuf, maskp, N0, Kvecs, ksq, kappa)
        rP = reaction_rate_field(cP, rhoP, cfg["lambda_g"], cfg["alpha_rho"],
                                 cfg["beta"], cfg["c0"], cfg["delta_c"], cfg["rho0"])
        Xp, Vp = velocity_step(scheme, Xp, Vp, gP, gamma_v, chi, D_v, tau, xi_buf)
        nu_p = jnp.where(maskp, reaction_poisson(kp_r, rP, tau), 0)
        # branch X and V together: replicate offspring of the SAME parent.
        Zp_cat = jnp.concatenate([Xp, Vp], axis=1)
        Zpb, mpb, ov_p, n_new_p = branch_compact(Zp_cat, nu_p, buffer_size, d)
        if ov_p:
            raise RuntimeError(
                f"poisson buffer overflow at step {s} (n_new={n_new_p}>{buffer_size}); "
                f"increase buffer_mult or strengthen alpha_rho/beta")
        Zpb = jnp.asarray(Zpb)
        Xp, Vp, maskp = Zpb[:, :d_x], Zpb[:, d_x:], jnp.asarray(mpb)

        # ===== MINVAR branching (full buffer + mask) =====
        cM, gM, rhoM, _, _, _ = field_at_particles(Xm, onesbuf, maskm, N0, Kvecs, ksq, kappa)
        rM = reaction_rate_field(cM, rhoM, cfg["lambda_g"], cfg["alpha_rho"],
                                 cfg["beta"], cfg["c0"], cfg["delta_c"], cfg["rho0"])
        Xm, Vm = velocity_step(scheme, Xm, Vm, gM, gamma_v, chi, D_v, tau, xi_buf)
        nu_m = jnp.where(maskm, reaction_minvar(km_r, rM, tau), 0)
        Zm_cat = jnp.concatenate([Xm, Vm], axis=1)
        Zmb, mmb, ov_m, n_new_m = branch_compact(Zm_cat, nu_m, buffer_size, d)
        if ov_m:
            raise RuntimeError(
                f"minvar buffer overflow at step {s} (n_new={n_new_m}>{buffer_size}); "
                f"increase buffer_mult or strengthen alpha_rho/beta")
        Zmb = jnp.asarray(Zmb)
        Xm, Vm, maskm = Zmb[:, :d_x], Zmb[:, d_x:], jnp.asarray(mmb)

        if s in snap_steps:
            record_snapshot(s)

    runtime = time.time() - t0
    # stamp runtime onto this seed's rows
    for rec in records:
        if rec["seed"] == seed and (isinstance(rec["runtime_s"], float)
                                    and np.isnan(rec["runtime_s"])):
            rec["runtime_s"] = runtime

    final = dict(
        Xw=np.asarray(Xw)[np.asarray(maskw)], Vw=np.asarray(Vw)[np.asarray(maskw)],
        ww=np.asarray(ww)[np.asarray(maskw)],
        Xr=np.asarray(Xr)[np.asarray(maskr)], Vr=np.asarray(Vr)[np.asarray(maskr)],
        wr=np.asarray(wr)[np.asarray(maskr)],
        Xp=np.asarray(Xp)[np.asarray(maskp)], Vp=np.asarray(Vp)[np.asarray(maskp)],
        Xm=np.asarray(Xm)[np.asarray(maskm)], Vm=np.asarray(Vm)[np.asarray(maskm)],
    )
    return overflow_p, overflow_m, runtime, final


# ---------------------------------------------------------------------------
# IO: write CSV, npz clouds, histogram npz, config copy, README
# ---------------------------------------------------------------------------
def write_csv(rd, records):
    csv_path = os.path.join(rd, "metrics.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(CSV_COLS) + "\n")
        for rec in records:
            f.write(",".join(str(rec.get(c, "")) for c in CSV_COLS) + "\n")
    return csv_path


def write_clouds(rd, seed, final):
    np.savez(
        os.path.join(rd, f"cloud_d6_seed{seed}.npz"),
        Xw=final["Xw"], Vw=final["Vw"], ww=final["ww"],
        Xr=final["Xr"], Vr=final["Vr"], wr=final["wr"],
        Xp=final["Xp"], Vp=final["Vp"],
        Xm=final["Xm"], Vm=final["Vm"],
    )


def write_histograms(rd, hist_store):
    # pack per-seed histogram time series into a single npz
    if not hist_store:
        return
    edges = hist_store[0]["edges"]
    seeds = np.array([h["seed"] for h in hist_store])
    steps = np.array([h["step"] for h in hist_store])
    ts = np.array([h["t"] for h in hist_store])
    methods = ["weighted", "weighted_resample", "poisson", "minvar"]
    packed = dict(edges=edges, seeds=seeds, steps=steps, t=ts)
    for mth in methods:
        packed[f"hist_{mth}"] = np.stack([h[mth] for h in hist_store], axis=0)
    np.savez(os.path.join(rd, "reaction_histograms.npz"), **packed)


def write_readme(rd, cfg):
    txt = f"""# kinetic_ks run output

Field-coupled 6D kinetic Keller-Segel particle experiment.
Phase space z=(x,v), x in T^3=[-pi,pi]^3, v in R^3, d=6 (d_x=d_v=3).

Generated by experiment_kinetic.py.

## Config (effective)
{json.dumps(cfg, indent=2)}

## Files
- metrics.csv               : one row per (method, seed, snapshot); see header.
- cloud_d6_seed*.npz        : final-time particle clouds (X,V[,w]) per method.
- reaction_histograms.npz   : per-snapshot reaction-rate histograms per method.
- config_used.json          : exact config used for this run.

## Methods compared (shared transport CRN)
weighted, weighted_resample (nESS<{cfg['ess_thresh']} trigger), poisson, minvar.

## Field solve
Spectral screened-Poisson  -Lap_x c + kappa^2 c = rho - rho_bar  from particle
x-positions, K_x={cfg['K_x']} modes/dim, gauge c_hat_0=0.  See field_kinetic.py.
"""
    with open(os.path.join(rd, "README.md"), "w") as f:
        f.write(txt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    argv = sys.argv[1:]
    cfg = resolve_config(argv)
    rd = cfg["results_dir"]
    os.makedirs(rd, exist_ok=True)
    print(f"=== Field-coupled 6D kinetic Keller-Segel (d={cfg['d']}) ===")
    print("backend:", jax.default_backend(), "| devices:", jax.devices())
    print(f"N0={cfg['N0']} tau={cfg['tau']} T={cfg['T']} steps={int(round(cfg['T']/cfg['tau']))} "
          f"K_x={cfg['K_x']} seeds={cfg['seeds']} scheme={cfg['velocity_scheme']}")
    print("results_dir:", rd)

    with open(os.path.join(rd, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    records = []
    hist_store = []
    overflow_any = False
    t_all = time.time()
    for seed in cfg["seeds"]:
        ov_p, ov_m, rt, final = run_seed(seed, cfg, records, hist_store)
        overflow_any = overflow_any or ov_p or ov_m
        if cfg["save_clouds"]:
            write_clouds(rd, seed, final)
        print(f"seed {seed}: runtime {rt:.2f}s  N_active(poisson final)="
              f"{final['Xp'].shape[0]}  N_active(minvar final)={final['Xm'].shape[0]}  "
              f"overflow_p={ov_p} overflow_m={ov_m}")
        # checkpoint the CSV after each completed seed so a wall-clock overrun on a
        # later seed never loses the seeds already finished.
        write_csv(rd, records)

    csv_path = write_csv(rd, records)
    write_histograms(rd, hist_store)
    write_readme(rd, cfg)
    print("wrote", csv_path, "rows:", len(records))
    if overflow_any:
        print("WARNING: branching buffer overflow occurred. "
              "Increase buffer_mult or strengthen alpha_rho/beta or reduce lambda_g/T.")
    print(f"TOTAL wallclock: {time.time() - t_all:.2f}s")


if __name__ == "__main__":
    main()
