"""
Experiment -- Two-stage switching localized growth: branching vs weighted+resample.
====================================================================================

This is a NEW experiment (a sibling of experiment.py / resample_baseline.py).  It
is designed to expose the STRUCTURAL weakness of GLOBAL resampling: resampling
commits the FIXED particle budget to the currently high-weight region and
discards lineage diversity elsewhere, so when growth later MOVES to a new region
the resampled cloud has almost no surviving ancestors there and cannot recover.
Branching never throws background lineages away, so it keeps particles
everywhere and re-resolves the new growth region.

MODEL (2D torus T^2 = [-pi,pi]^2), u0(x) = 1 (uniform):

    d_t u = D Laplacian(u) + r(t,x) u ,

    G_sigma(x;c) = exp(-d_T(x,c)^2 / (2 sigma^2))      (d_T periodic distance)
    s(t)         = 0.5*(1 + tanh((t - T/2)/delta))      (smooth 0 -> 1 switch)
    r(t,x)       = lambda*[ (1-s(t)) G_sigma(x;cA) + s(t) G_sigma(x;cB) ] - beta

Growth lives near cA for t < T/2, then switches to cB for t > T/2.

METHODS compared (common random numbers: same initial particles, same transport
Brownian increments, same per-step keys):
    1. weighted              (no resampling)
    2. weighted_ess          (systematic resample when global nESS < 0.5)
    3. weighted_always       (systematic resample every step)
    4. poisson               (Poisson equal-weight branching)
    5. minvar                (minimum-variance integer branching)

GROUND TRUTH: deterministic Fourier split-step (Strang) on a grid x grid mesh
with reference substeps so the reference dt = tau_ref, the reaction field
RECOMPUTED each substep from r(t,x) (time-dependent).

KEY DIAGNOSTICS (not just global L2):
  * global relative L2 error at T;
  * LOCAL relative L2 error at T in B_A = {G_sigma(.;cA) >= 0.5} and
    B_B = {G_sigma(.;cB) >= 0.5}.  B_B (second-stage region) is the key number.
  * N_{B_B}^distinct(T/2^-): number of DISTINCT ANCESTORS (initial lineages)
    among active particles inside B_B JUST BEFORE the switch.  Resampling kills
    diversity -> few distinct B-ancestors survive -> cannot recover.  Branching
    retains background lineages -> many distinct B-ancestors -> recovers.
  * global nESS(t); n_resamples (resample methods); active count (branching).

Run:
    python experiment_switch.py                 # full config
    python experiment_switch.py --smoke         # tiny smoke
    python experiment_switch.py --config c.json # override
"""

import os
import sys
import json
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Reuse the shared building blocks (see common_particle.py).
from common_particle import (          # noqa: E402
    generate_density_estimation,
    em_transport,
    wrap_torus,
    reaction_weighted,
    reaction_poisson,
    reaction_minvar,
    nESS,
)
# Reuse geometry / IO patterns from experiment.py and the resampler from
# resample_baseline.py.  We intentionally do NOT modify their behavior.
from experiment import (               # noqa: E402
    PERIOD, L, grid_coords, branch_compact,
)


# Systematic resampling -- byte-identical to resample_baseline.systematic_resample
# (resample_baseline.py runs the full baseline at import time, so we copy the
# small pure function here instead of importing the module).
def systematic_resample(rng, w):
    """Systematic resampling: returns N0 indices drawn proportional to w."""
    assert np.isfinite(w).all() and w.sum() > 0, "invalid weights in resample"
    N = w.shape[0]
    p = w / np.sum(w)
    positions = (rng.random() + np.arange(N)) / N
    cumsum = np.cumsum(p)
    cumsum[-1] = 1.0  # guard against round-off
    return np.clip(np.searchsorted(cumsum, positions, side="right"), 0, N - 1)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG = {
    "D": 0.02,
    "lambda": 10.0,
    "beta": 0.1,
    "sigma": 0.25,
    "cA": [-1.2, 0.0],
    "cB": [1.2, 0.0],
    "T": 1.2,
    "delta": 0.04,
    "tau": 1e-3,            # steps = T/tau = 1200
    "N0": 20000,            # initial particle count
    "buffer_mult": 8,       # branching buffer = buffer_mult*N0; deterministic mass
                            #   grows only ~2.1x over [0,T] (beta=0.1), so 8x gives
                            #   a ~3.8x safety margin over the expected peak count.
                            #   (See run_summary.csv: ps_* / steps ~ mean active.)
    "K": 48,                # Fourier modes per direction for reconstruction
    "grid": 512,            # reference / evaluation grid size
    "tau_ref": 2.5e-4,      # reference time step (=> ref_substeps = tau/tau_ref)
    "eta": 0.5,             # B_A,B_B = {G_sigma(.;c) >= eta}  (FIXED, predefined)
    "n_snapshots": 25,      # number of saved time snapshots over [0,T]
    "ess_threshold": 0.5,   # global nESS resample trigger
    "resample_seed_offset": 10_000,
    "seeds": [0, 1, 2, 3, 4, 5, 6, 7],
    "results_dir": "results/switch",
    "smoke": False,
}

SMOKE_OVERRIDES = {
    "N0": 2000,
    "tau": 4e-3,            # steps = T/tau = 300
    "grid": 128,
    "tau_ref": 1e-3,        # ref_substeps = 4
    "K": 32,
    "buffer_mult": 8,
    "seeds": [0, 1, 2, 3],
    "n_snapshots": 13,
    "results_dir": "results/switch_smoke",
    "smoke": True,
}


# ---------------------------------------------------------------------------
# Geometry: periodic distance, switching reaction
# ---------------------------------------------------------------------------
def periodic_sq_dist(X, c):
    """Squared periodic distance on [-pi,pi]^2 between rows of X and point c."""
    dx = X[:, 0] - c[0]
    dy = X[:, 1] - c[1]
    dx = dx - L * jnp.round(dx / L)
    dy = dy - L * jnp.round(dy / L)
    return dx * dx + dy * dy


def G_at(X, c, sigma):
    d2 = periodic_sq_dist(X, jnp.asarray(c))
    return jnp.exp(-d2 / (2.0 * sigma ** 2))


def s_of_t(t, cfg):
    return 0.5 * (1.0 + np.tanh((t - 0.5 * cfg["T"]) / cfg["delta"]))


def r_of_xt(X, t, cfg):
    """Time-dependent switching reaction r(t,x) on the particle cloud X."""
    s = s_of_t(t, cfg)
    GA = G_at(X, cfg["cA"], cfg["sigma"])
    GB = G_at(X, cfg["cB"], cfg["sigma"])
    return cfg["lambda"] * ((1.0 - s) * GA + s * GB) - cfg["beta"]


# ---------------------------------------------------------------------------
# Grid versions of G (for reference field and region masks)
# ---------------------------------------------------------------------------
def G_grid(XX, YY, c, sigma):
    dx = XX - c[0]
    dy = YY - c[1]
    dx = dx - L * np.round(dx / L)
    dy = dy - L * np.round(dy / L)
    return np.exp(-(dx * dx + dy * dy) / (2.0 * sigma ** 2))


# ---------------------------------------------------------------------------
# Reference: time-dependent Fourier split-step (Strang) on the grid
# ---------------------------------------------------------------------------
def make_reference(cfg):
    """Return (u0, advance_one_tau(u, t_start), xs, XX, YY, GgA, GgB).

    advance_one_tau integrates ONE outer step tau starting from physical time
    t_start, using `sub` Strang substeps of size dt = tau/sub with the reaction
    field recomputed at the substep MIDPOINT time (time-dependent r(t,x)).
    """
    n = cfg["grid"]
    xs, XX, YY = grid_coords(n)
    GgA = G_grid(XX, YY, cfg["cA"], cfg["sigma"])
    GgB = G_grid(XX, YY, cfg["cB"], cfg["sigma"])
    u0 = np.ones((n, n), dtype=np.float64)          # u0 = 1 (uniform)

    k = np.fft.fftfreq(n, d=(L / n)) * 2.0 * np.pi  # integer wavenumbers
    KX, KY = np.meshgrid(k, k, indexing="xy")
    lap = -(KX ** 2 + KY ** 2)

    tau = cfg["tau"]
    sub = max(1, int(round(tau / cfg["tau_ref"])))
    dt = tau / sub
    diff_half = np.exp(cfg["D"] * lap * (dt / 2.0))

    def react_field(t_mid):
        s = s_of_t(t_mid, cfg)
        rg = cfg["lambda"] * ((1.0 - s) * GgA + s * GgB) - cfg["beta"]
        return np.exp(rg * dt)

    def advance_one_tau(u, t_start):
        t = t_start
        for _ in range(sub):
            uh = np.fft.fft2(u) * diff_half
            u = np.real(np.fft.ifft2(uh))
            # reaction at substep midpoint (time-dependent r)
            u = u * react_field(t + 0.5 * dt)
            uh = np.fft.fft2(u) * diff_half
            u = np.real(np.fft.ifft2(uh))
            t += dt
        return u

    return u0, advance_one_tau, xs, XX, YY, GgA, GgB


# ---------------------------------------------------------------------------
# Field reconstruction (mass-scaled physical density) -- same pattern as
# experiment.reconstruct_field, kept local to avoid import coupling on its
# exact signature.
# ---------------------------------------------------------------------------
def reconstruct_field(density_estimation, density_evaluate_grid, X, w, mask,
                      mass_scale, XX, YY):
    coeff = density_estimation(X, weights=w, mask=mask)
    prob = density_evaluate_grid(jnp.asarray(XX), jnp.asarray(YY), coeff)
    return np.asarray(prob) * mass_scale


# ---------------------------------------------------------------------------
# Error metrics: global + local relative L2 over B_A and B_B
# ---------------------------------------------------------------------------
def error_metrics(u, u_ref, BA, BB, cell_area):
    diff = u - u_ref
    refL2 = np.sqrt(np.sum(u_ref ** 2) * cell_area)
    L2_rel = np.sqrt(np.sum(diff ** 2) * cell_area) / refL2 if refL2 > 0 else np.nan

    def local_rel(mask):
        rr = np.sqrt(np.sum((u_ref[mask]) ** 2) * cell_area)
        dd = np.sqrt(np.sum((diff[mask]) ** 2) * cell_area)
        return dd / rr if rr > 0 else np.nan

    return {
        "L2_rel_err": L2_rel,
        "L2_rel_err_BA": local_rel(BA),
        "L2_rel_err_BB": local_rel(BB),
        "total_mass": float(np.sum(u) * cell_area),
        "peak_height": float(np.max(u)),
    }


# ---------------------------------------------------------------------------
# Distinct-ancestor count inside a region
# ---------------------------------------------------------------------------
def distinct_ancestors_in_region(X, anc, active_mask, c, sigma, eta, cfg):
    """Count distinct ancestor indices among ACTIVE particles inside
    B = {G_sigma(.;c) >= eta}."""
    Gp = np.asarray(G_at(jnp.asarray(X), c, sigma))
    inB = (Gp >= eta) & active_mask
    if not np.any(inB):
        return 0, 0
    a = np.asarray(anc)[inB]
    return int(np.unique(a).size), int(np.sum(inB))


# ---------------------------------------------------------------------------
# Per-seed run of all five methods (shared CRN)
# ---------------------------------------------------------------------------
def run_seed(seed, cfg, ref_u0, advance_ref, XX, YY, GgA, GgB,
             density_estimation, density_evaluate_grid,
             records, anc_records, fields_store):
    t0 = time.time()
    n = cfg["grid"]
    cell_area = (L / n) ** 2
    eta = cfg["eta"]
    tau = cfg["tau"]
    T = cfg["T"]
    steps = int(round(T / tau))
    N0 = cfg["N0"]
    buffer_size = cfg["buffer_mult"] * N0
    sigma = cfg["sigma"]
    cA, cB = cfg["cA"], cfg["cB"]
    period = jnp.asarray(PERIOD)
    BA = (GgA >= eta)
    BB = (GgB >= eta)
    M0 = float(np.sum(ref_u0) * cell_area)        # = L^2 since u0 = 1

    # switch step index: the largest step s with t_s = s*tau < T/2 (i.e. T/2^-)
    half = 0.5 * T
    switch_step = int(np.floor((half - 1e-12) / tau))
    switch_step = min(max(switch_step, 0), steps)

    snap_steps = sorted(set(np.linspace(0, steps, cfg["n_snapshots"], dtype=int).tolist()
                            + [switch_step]))

    # ---- initial particles: u0 = 1 is uniform on the torus -> uniform sample ----
    key = jax.random.PRNGKey(seed)
    key, k_init = jax.random.split(key)
    X_init = jax.random.uniform(k_init, (N0, 2), minval=-np.pi, maxval=np.pi,
                                dtype=jnp.float64)

    # ---- reference timeline (time-dependent) ----
    u_ref = ref_u0.copy()
    ref_snapshots = {0: u_ref.copy()}
    t = 0.0
    for s in range(1, steps + 1):
        u_ref = advance_ref(u_ref, t)
        t += tau
        if s in snap_steps:
            ref_snapshots[s] = u_ref.copy()

    # ---- per-method state ----
    # weighted (no resample) -- fixed N0
    Xw = X_init
    ww = jnp.ones((N0,), dtype=jnp.float64)
    aw = np.arange(N0, dtype=np.int64)            # ancestor index
    # weighted_ess / weighted_always -- fixed N0, resampled in place
    Xe = X_init; we = jnp.ones((N0,), dtype=jnp.float64); ae = np.arange(N0, dtype=np.int64)
    Xa = X_init; wa = jnp.ones((N0,), dtype=jnp.float64); aa = np.arange(N0, dtype=np.int64)
    rng_e = np.random.default_rng(cfg["resample_seed_offset"] + 1 + seed)
    rng_a = np.random.default_rng(cfg["resample_seed_offset"] + 2 + seed)
    n_res_e = 0
    n_res_a = 0

    # branching buffers (poisson, minvar) -- ancestors tracked as int arrays
    def init_buffer():
        Xb = np.zeros((buffer_size, 2), dtype=np.float64)
        Xb[:N0] = np.asarray(X_init)
        mb = np.zeros((buffer_size,), dtype=bool)
        mb[:N0] = True
        ab = np.full((buffer_size,), -1, dtype=np.int64)
        ab[:N0] = np.arange(N0, dtype=np.int64)
        return jnp.asarray(Xb), mb, ab

    Xp, maskp, ap = init_buffer()
    Xm, maskm, am = init_buffer()
    ps_p = 0
    ps_m = 0
    overflow_p = False
    overflow_m = False

    maskw = jnp.ones((N0,), dtype=bool)

    # ---------------- snapshot helper ----------------
    def take_snapshot(s):
        t_s = s * tau
        u_ref_s = ref_snapshots[s]

        # weighted
        sum_w = float(jnp.sum(ww))
        uw = reconstruct_field(density_estimation, density_evaluate_grid, Xw, ww,
                               maskw, (sum_w / N0) * M0, XX, YY)
        # weighted_ess
        sum_e = float(jnp.sum(we))
        ue = reconstruct_field(density_estimation, density_evaluate_grid, Xe, we,
                               maskw, (sum_e / N0) * M0, XX, YY)
        # weighted_always
        sum_a = float(jnp.sum(wa))
        ua = reconstruct_field(density_estimation, density_evaluate_grid, Xa, wa,
                               maskw, (sum_a / N0) * M0, XX, YY)
        # branching
        np_act = int(np.sum(maskp))
        up = reconstruct_field(density_estimation, density_evaluate_grid, Xp,
                               jnp.ones((buffer_size,)), jnp.asarray(maskp),
                               (np_act / N0) * M0, XX, YY)
        nm_act = int(np.sum(maskm))
        um = reconstruct_field(density_estimation, density_evaluate_grid, Xm,
                               jnp.ones((buffer_size,)), jnp.asarray(maskm),
                               (nm_act / N0) * M0, XX, YY)

        # global nESS for the weighted family
        nessw = float(nESS(ww))
        nesse = float(nESS(we))
        nessa = float(nESS(wa))

        method_data = [
            ("weighted",        uw, dict(global_nESS=nessw, N_active=N0,        n_resamples=0)),
            ("weighted_ess",    ue, dict(global_nESS=nesse, N_active=N0,        n_resamples=n_res_e)),
            ("weighted_always", ua, dict(global_nESS=nessa, N_active=N0,        n_resamples=n_res_a)),
            ("poisson",         up, dict(global_nESS=np.nan, N_active=np_act,   n_resamples=0)),
            ("minvar",          um, dict(global_nESS=np.nan, N_active=nm_act,   n_resamples=0)),
        ]
        out = {}
        for method, u, extra in method_data:
            m = error_metrics(u, u_ref_s, BA, BB, cell_area)
            rec = dict(seed=seed, method=method, t=t_s)
            rec.update(m); rec.update(extra)
            records.append(rec)
            out[method] = u
        out["reference"] = u_ref_s
        return out

    # distinct-ancestor snapshot at T/2^- (the key diagnostic)
    def take_ancestor_snapshot(s):
        t_s = s * tau
        rows = [
            ("weighted",        np.asarray(Xw), aw, np.asarray(maskw)),
            ("weighted_ess",    np.asarray(Xe), ae, np.asarray(maskw)),
            ("weighted_always", np.asarray(Xa), aa, np.asarray(maskw)),
            ("poisson",         np.asarray(Xp), ap, maskp),
            ("minvar",          np.asarray(Xm), am, maskm),
        ]
        for method, X, anc, mk in rows:
            dB, nB = distinct_ancestors_in_region(X, anc, mk, cB, sigma, eta, cfg)
            dA, nA = distinct_ancestors_in_region(X, anc, mk, cA, sigma, eta, cfg)
            # GLOBAL distinct ancestors among ALL active particles -- the cleanest
            # mechanistic signal: resampling collapses global lineage diversity,
            # which is what later prevents recovery in the new growth region.
            anc_act = np.asarray(anc)[mk]
            global_distinct = int(np.unique(anc_act).size) if anc_act.size else 0
            anc_records.append(dict(
                seed=seed, method=method, t=t_s,
                distinct_anc_BB=dB, N_BB=nB,
                distinct_anc_BA=dA, N_BA=nA,
                global_distinct_anc=global_distinct,
                N_active=int(np.sum(mk)),
            ))

    if 0 in snap_steps:
        take_snapshot(0)

    final_fields = None
    for s in range(1, steps + 1):
        t_pre = (s - 1) * tau           # reaction evaluated at start of step
        key, kT, kp_r, km_r = jax.random.split(key, 4)
        # Shared transport stream over the full buffer; front N0 slots receive
        # identical increments across all methods (normal(kT,(M,2))[:k] ==
        # normal(kT,(k,2)) under JAX threefry).
        dWbuf = jax.random.normal(kT, shape=(buffer_size, 2), dtype=jnp.float64)

        # ---- weighted (no resample) ----
        rW = r_of_xt(Xw, t_pre, cfg)
        Xw = wrap_torus(em_transport(Xw, jnp.zeros_like(Xw), cfg["D"], tau, dWbuf[:N0]), period)
        ww = reaction_weighted(ww, rW, tau)

        # ---- weighted_ess ----
        rE = r_of_xt(Xe, t_pre, cfg)
        Xe = wrap_torus(em_transport(Xe, jnp.zeros_like(Xe), cfg["D"], tau, dWbuf[:N0]), period)
        we = reaction_weighted(we, rE, tau)
        we_np = np.asarray(we)
        g_ness_e = float((we_np.sum() ** 2) / (N0 * np.sum(we_np ** 2)))
        if g_ness_e < cfg["ess_threshold"]:
            idx = systematic_resample(rng_e, we_np)
            Xe = jnp.asarray(np.asarray(Xe)[idx])
            ae = ae[idx]                                  # children inherit ancestor
            we = jnp.full((N0,), we_np.sum() / N0, dtype=jnp.float64)
            n_res_e += 1

        # ---- weighted_always ----
        rA = r_of_xt(Xa, t_pre, cfg)
        Xa = wrap_torus(em_transport(Xa, jnp.zeros_like(Xa), cfg["D"], tau, dWbuf[:N0]), period)
        wa = reaction_weighted(wa, rA, tau)
        wa_np = np.asarray(wa)
        idx = systematic_resample(rng_a, wa_np)
        Xa = jnp.asarray(np.asarray(Xa)[idx])
        aa = aa[idx]
        wa = jnp.full((N0,), wa_np.sum() / N0, dtype=jnp.float64)
        n_res_a += 1

        # ---- poisson branching ----
        rP = r_of_xt(Xp, t_pre, cfg)
        Xp = wrap_torus(em_transport(Xp, jnp.zeros_like(Xp), cfg["D"], tau, dWbuf), period)
        nu_p = jnp.where(jnp.asarray(maskp), reaction_poisson(kp_r, rP, tau), 0)
        Xpb, mpb, ov_p, n_new_p = branch_compact(Xp, nu_p, buffer_size)
        ap = np.repeat(ap, np.asarray(nu_p).astype(np.int64), axis=0)
        ap = _fit_ancestor(ap, n_new_p, buffer_size)
        if ov_p:
            overflow_p = True
            raise RuntimeError(f"[seed {seed}] poisson buffer overflow at step {s} "
                               f"(n_new>{buffer_size}); increase buffer_mult")
        Xp, maskp = jnp.asarray(Xpb), mpb
        ps_p += n_new_p

        # ---- minvar branching ----
        rM = r_of_xt(Xm, t_pre, cfg)
        Xm = wrap_torus(em_transport(Xm, jnp.zeros_like(Xm), cfg["D"], tau, dWbuf), period)
        nu_m = jnp.where(jnp.asarray(maskm), reaction_minvar(km_r, rM, tau), 0)
        Xmb, mmb, ov_m, n_new_m = branch_compact(Xm, nu_m, buffer_size)
        am = np.repeat(am, np.asarray(nu_m).astype(np.int64), axis=0)
        am = _fit_ancestor(am, n_new_m, buffer_size)
        if ov_m:
            overflow_m = True
            raise RuntimeError(f"[seed {seed}] minvar buffer overflow at step {s} "
                               f"(n_new>{buffer_size}); increase buffer_mult")
        Xm, maskm = jnp.asarray(Xmb), mmb
        ps_m += n_new_m

        if s == switch_step:
            take_ancestor_snapshot(s)               # T/2^- distinct ancestors
        if s in snap_steps:
            out = take_snapshot(s)
            if s == steps:
                final_fields = out

    runtime = time.time() - t0
    if final_fields is None:
        final_fields = take_snapshot(steps)
    fields_store[seed] = dict(
        reference=final_fields["reference"],
        weighted=final_fields["weighted"],
        weighted_ess=final_fields["weighted_ess"],
        weighted_always=final_fields["weighted_always"],
        poisson=final_fields["poisson"],
        minvar=final_fields["minvar"],
    )
    # tag runtime onto this seed's records
    for rec in records:
        if rec["seed"] == seed and "runtime_s" not in rec:
            rec["runtime_s"] = runtime
    return overflow_p, overflow_m, runtime, dict(
        n_res_e=n_res_e, n_res_a=n_res_a, ps_p=ps_p, ps_m=ps_m, switch_step=switch_step)


def _fit_ancestor(anc, n_new, buffer_size):
    """branch_compact truncates positions to buffer_size on overflow; mirror that
    on the ancestor array and pad inactive slots with -1 so lengths stay aligned."""
    if anc.shape[0] >= n_new:
        anc = anc[:n_new]
    out = np.full((buffer_size,), -1, dtype=np.int64)
    k = min(n_new, buffer_size, anc.shape[0])
    out[:k] = anc[:k]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def resolve_config(argv):
    cfg = dict(CONFIG)
    if "--smoke" in argv:
        cfg.update(SMOKE_OVERRIDES)
    if "--config" in argv:
        i = argv.index("--config")
        with open(argv[i + 1]) as f:
            cfg.update(json.load(f))
    return cfg


def main():
    argv = sys.argv[1:]
    cfg = resolve_config(argv)
    print("=== Two-stage switching growth: branching vs weighted+resample ===")
    print("backend:", jax.default_backend(), "| devices:", jax.devices())
    rd = cfg["results_dir"]
    os.makedirs(rd, exist_ok=True)
    cfg["_ref_substeps"] = max(1, int(round(cfg["tau"] / cfg["tau_ref"])))
    cfg["jax_version"] = jax.__version__
    cfg["numpy_version"] = np.__version__
    with open(os.path.join(rd, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    density_estimation, _, density_evaluate_grid = generate_density_estimation(
        n_freq=cfg["K"], period=PERIOD)
    ref_u0, advance_ref, xs, XX, YY, GgA, GgB = make_reference(cfg)

    records = []
    anc_records = []
    fields_store = {}
    overflow_any = False
    extra_summ = []
    t_all = time.time()
    for seed in cfg["seeds"]:
        ov_p, ov_m, rt, ex = run_seed(
            seed, cfg, ref_u0, advance_ref, XX, YY, GgA, GgB,
            density_estimation, density_evaluate_grid,
            records, anc_records, fields_store)
        overflow_any = overflow_any or ov_p or ov_m
        ex["seed"] = seed; ex["runtime_s"] = rt
        extra_summ.append(ex)
        print(f"seed {seed}: rt {rt:.2f}s  n_res_ess={ex['n_res_e']} "
              f"n_res_always={ex['n_res_a']}  ps_p={ex['ps_p']:.2e} "
              f"ps_m={ex['ps_m']:.2e}  switch_step={ex['switch_step']}", flush=True)

    # ---- write metrics CSV (time series) ----
    cols = ["seed", "method", "t", "total_mass", "peak_height",
            "L2_rel_err", "L2_rel_err_BA", "L2_rel_err_BB",
            "global_nESS", "N_active", "n_resamples", "runtime_s"]
    with open(os.path.join(rd, "metrics.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for rec in records:
            f.write(",".join(str(rec.get(c, "")) for c in cols) + "\n")

    # ---- write ancestor CSV (the key diagnostic, at T/2^-) ----
    acols = ["seed", "method", "t", "distinct_anc_BB", "N_BB",
             "distinct_anc_BA", "N_BA", "global_distinct_anc", "N_active"]
    with open(os.path.join(rd, "ancestors_at_switch.csv"), "w") as f:
        f.write(",".join(acols) + "\n")
        for rec in anc_records:
            f.write(",".join(str(rec.get(c, "")) for c in acols) + "\n")

    # ---- save final fields per seed ----
    for seed, fld in fields_store.items():
        np.savez(os.path.join(rd, f"fields_seed{seed}.npz"),
                 xs=xs, XX=XX, YY=YY, GgA=GgA, GgB=GgB, **fld)

    # ---- console summary at final time T ----
    import statistics as stt
    finalT = cfg["T"]
    print("\n=== FINAL-TIME (t=T) summary: mean +/- std over seeds ===")
    print(f"{'method':16s} {'L2rel':>16s} {'L2rel_BA':>16s} {'L2rel_BB(KEY)':>18s} "
          f"{'N_active':>10s}")
    for method in ["weighted", "weighted_ess", "weighted_always", "poisson", "minvar"]:
        sub = [r for r in records if r["method"] == method
               and abs(r["t"] - finalT) < 0.5 * cfg["tau"]]
        if not sub:
            continue
        def ms(c):
            v = [r[c] for r in sub if not (isinstance(r[c], float) and np.isnan(r[c]))]
            if not v:
                return (np.nan, 0.0)
            return (stt.mean(v), (stt.pstdev(v) if len(v) > 1 else 0.0))
        l2 = ms("L2_rel_err"); la = ms("L2_rel_err_BA"); lb = ms("L2_rel_err_BB")
        na = ms("N_active")
        print(f"{method:16s} {l2[0]:7.4f}+/-{l2[1]:6.4f} {la[0]:7.4f}+/-{la[1]:6.4f} "
              f"{lb[0]:7.4f}+/-{lb[1]:6.4f} {na[0]:10.0f}")

    print("\n=== ANCESTOR DIVERSITY at T/2^- : mean +/- std over seeds ===")
    print(f"{'method':16s} {'distinct_BB(KEY)':>16s} {'N_BB':>8s} "
          f"{'global_distinct':>16s} {'N_active':>10s}")
    for method in ["weighted", "weighted_ess", "weighted_always", "poisson", "minvar"]:
        sub = [r for r in anc_records if r["method"] == method]
        if not sub:
            continue
        def ms(c):
            v = [r[c] for r in sub]
            return (stt.mean(v), (stt.pstdev(v) if len(v) > 1 else 0.0))
        d = ms("distinct_anc_BB"); nb = ms("N_BB")
        gd = ms("global_distinct_anc"); na = ms("N_active")
        print(f"{method:16s} {d[0]:8.1f}+/-{d[1]:6.1f} {nb[0]:8.0f} "
              f"{gd[0]:10.0f}+/-{gd[1]:6.0f} {na[0]:10.0f}")

    # ---- extra summary CSV ----
    with open(os.path.join(rd, "run_summary.csv"), "w") as f:
        f.write("seed,runtime_s,n_res_ess,n_res_always,ps_poisson,ps_minvar,switch_step\n")
        for ex in extra_summ:
            f.write(f"{ex['seed']},{ex['runtime_s']:.3f},{ex['n_res_e']},{ex['n_res_a']},"
                    f"{ex['ps_p']},{ex['ps_m']},{ex['switch_step']}\n")

    if overflow_any:
        print("\nWARNING: branching buffer overflow occurred. Increase buffer_mult.")
    print(f"\nwrote metrics/ancestors/fields to {rd}")
    print(f"TOTAL wallclock: {time.time() - t_all:.2f}s")


if __name__ == "__main__":
    main()
