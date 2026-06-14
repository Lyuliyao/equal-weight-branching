"""Separated growth islands -- the local-degeneracy rebuttal benchmark (Sec. 5.2).
================================================================================

Scalar reaction-diffusion on the torus T^2 = [-pi,pi]^2 with M WELL-SEPARATED
Gaussian growth islands of mildly varying amplitude:

    d_t u = D Laplacian(u) + r(x) u ,
    r(x) = lambda * G_multi(x) - beta ,
    G_multi(x) = sum_{m=1}^M a_m exp( -d_T(x, c_m)^2 / (2 sigma^2) ) ,
    a_m = 1 + 0.25 sin(2 pi m / M),     u0(x) = 1 .

d_T is the periodic torus distance.  The centers c_m sit on a 4x4 (M=16) or 5x5
(M=25) grid kept away from the periodic seam.  The DIAGNOSTIC regions are the
half-height disks of the UNIT-amplitude Gaussian,

    B_m = { x : exp(-d_T(x,c_m)^2/(2 sigma^2)) >= 0.5 } ,

which we assert are disjoint before running.

WHY THIS EXPERIMENT.  Global ESS-triggered resampling and global relative L2 can
look acceptable while one or more growth islands -- especially the weaker-
amplitude ones -- are represented by only a few effective particles.  Equal-weight
branching converts local growth into local particle count, so per-island mass
error stays small and uniform.  The central diagnostic is therefore the per-island
mass error

    E_m = | M_m^method - M_m^ref | / M_m^ref ,
    M_m^ref    = int_{B_m} u_ref(T,x) dx        (deterministic spectral reference),
    M_m^method = mu_T(B_m)                        (DIRECT particle mass, no field
                                                   reconstruction),

NOT the global relative L2.

METHODS (common initial particles + common Brownian increments per seed; only the
reaction representation differs):
    weighted              : positions diffuse; w_i *= exp(r tau).
    weighted_ess_resample : weighted + systematic resampling when global nESS
                            drops below ess_threshold (standard adaptive SMC).
    minvar_branch         : equal-weight minimum-variance integer branching.
    poisson_branch        : equal-weight Poisson birth-death branching (optional).
    minvar_branch_cap     : minvar branching with an explicit population cap
                            (LABELLED as capped; off by default).

The deterministic reference is a Fourier Strang split-step solve on a 512^2 grid
(>= 4K and >= 512^2 per the plan).  The branching buffer is auto-sized from the
reference total-mass growth so the integer population never overflows.

Run:
    python multi_island.py --smoke
    python multi_island.py --M 16 --N0 20000 --K 64 --T 0.8 --seeds 0 1 2 3 4 5 6 7 \
        --methods weighted weighted_ess_resample minvar_branch poisson_branch \
        --out_dir results/multi_island
"""
import os
import sys
import json
import time
import argparse
import datetime
import platform
import subprocess

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from common_particle import (
    generate_density_estimation,
    em_transport,
    wrap_torus,
    reaction_weighted,
    reaction_minvar,
    reaction_poisson,
    nESS,
)

PERIOD = [[-np.pi, np.pi], [-np.pi, np.pi]]
L = 2.0 * np.pi


# ---------------------------------------------------------------------------
# CONFIG (defaults from the revision plan, Sec. 2)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    M=16,
    sigma=0.16,
    D=0.01,
    lam=12.0,
    beta=0.8,
    T=0.8,
    tau=1e-3,
    N0=20000,
    K=64,                  # reconstruction bandwidth (figures only; E_m is recon-free)
    grid=512,              # deterministic reference resolution (>= 512^2)
    ref_substeps=1,
    eta=0.5,               # B_m = {G_sigma(x;c_m) >= eta}
    ess_threshold=0.5,     # global-nESS resampling trigger (same as resample_baseline)
    n_snapshots=20,
    seeds=[0, 1, 2, 3, 4, 5, 6, 7],
    methods=["weighted", "weighted_ess_resample", "minvar_branch", "poisson_branch"],
    buffer_safety=1.6,     # branching buffer = ceil(safety * M_final/M_0)
    recon_subsample=200000,  # cap particles used for FIELD reconstruction (figures)
    pc_cap_mult=8,         # population cap for minvar_branch_cap = pc_cap_mult * N0
    out_dir="results/multi_island",
)

SMOKE = dict(
    M=16, N0=2000, K=16, T=0.05, tau=1e-3, grid=128, n_snapshots=6,
    seeds=[0, 1], recon_subsample=20000, methods=[
        "weighted", "weighted_ess_resample", "minvar_branch", "poisson_branch"],
    out_dir="results/multi_island_smoke",
)


# ---------------------------------------------------------------------------
# Torus geometry helpers
# ---------------------------------------------------------------------------
def torus_delta(x, c):
    """Periodic displacement x - c wrapped into [-L/2, L/2] per coordinate."""
    d = x - c
    return d - L * jnp.round(d / L)


def torus_dist2(X, c):
    """Squared periodic torus distance between rows of X (N,2) and a point c."""
    dx = torus_delta(X[:, 0], c[0])
    dy = torus_delta(X[:, 1], c[1])
    return dx * dx + dy * dy


def torus_dist2_grid(XX, YY, c):
    dx = XX - c[0]; dy = YY - c[1]
    dx = dx - L * np.round(dx / L)
    dy = dy - L * np.round(dy / L)
    return dx * dx + dy * dy


def make_grid_centers(M):
    """Return (M,2) separated centers on a sqrt(M) x sqrt(M) grid, off the seam.

    Cell-centered on [-pi,pi]: positions -pi + (i+0.5)*(L/n) for i=0..n-1, so the
    nearest center is L/(2n) from the seam (max separation from the boundary).
    """
    n = int(round(np.sqrt(M)))
    assert n * n == M, "M must be a perfect square (16 or 25)."
    axis = -np.pi + (np.arange(n) + 0.5) * (L / n)
    cx, cy = np.meshgrid(axis, axis, indexing="xy")
    return np.stack([cx.ravel(), cy.ravel()], axis=1)


def make_amplitudes(M):
    m = np.arange(1, M + 1)
    return 1.0 + 0.25 * np.sin(2.0 * np.pi * m / M)


def G_multi_grid(XX, YY, centers, amplitudes, sigma):
    G = np.zeros_like(XX)
    for a_m, c in zip(amplitudes, centers):
        G += a_m * np.exp(-torus_dist2_grid(XX, YY, c) / (2.0 * sigma ** 2))
    return G


def r_of_particles(X, centers, amplitudes, sigma, lam, beta):
    """Reaction rate r(X) for particle positions (N,2). centers (M,2) jnp."""
    G = jnp.zeros((X.shape[0],), dtype=jnp.float64)
    for a_m, c in zip(amplitudes, centers):
        G = G + a_m * jnp.exp(-torus_dist2(X, c) / (2.0 * sigma ** 2))
    return lam * G - beta


def island_masks_on_grid(XX, YY, centers, sigma, eta):
    """Boolean half-height masks B_m on the reference grid (unit-amplitude Gaussian)."""
    masks = []
    thr2 = -2.0 * sigma ** 2 * np.log(eta)   # d^2 <= thr2  <=>  exp(-d^2/2sig^2) >= eta
    for c in centers:
        masks.append(torus_dist2_grid(XX, YY, c) <= thr2)
    return np.stack(masks, axis=0)           # (M, Ny, Nx)


def island_index_particles(X, centers, sigma, eta):
    """For each island return a boolean mask over particles inside B_m (unit Gaussian)."""
    thr2 = -2.0 * sigma ** 2 * np.log(eta)
    Xn = np.asarray(X)
    out = []
    for c in centers:
        d2 = torus_dist2_grid(Xn[:, 0], Xn[:, 1], np.asarray(c))
        out.append(d2 <= thr2)
    return out                               # list of (N,) bool


# ---------------------------------------------------------------------------
# Deterministic Fourier split-step reference (512^2 grid)
# ---------------------------------------------------------------------------
def reference_solver(cfg, centers, amplitudes):
    n = cfg["grid"]
    xs = -np.pi + (np.arange(n) + 0.5) * (L / n)
    XX, YY = np.meshgrid(xs, xs, indexing="xy")
    Gg = G_multi_grid(XX, YY, centers, amplitudes, cfg["sigma"])
    rg = cfg["lam"] * Gg - cfg["beta"]
    u0 = np.ones_like(XX)                     # u0 = 1

    k = np.fft.fftfreq(n, d=(L / n)) * 2.0 * np.pi
    KX, KY = np.meshgrid(k, k, indexing="xy")
    lap = -(KX ** 2 + KY ** 2)

    tau = cfg["tau"]
    sub = cfg["ref_substeps"]
    dt = tau / sub
    diff_half = np.exp(cfg["D"] * lap * (dt / 2.0))
    react = np.exp(rg * dt)

    def advance_one_tau(u):
        for _ in range(sub):
            u = np.real(np.fft.ifft2(np.fft.fft2(u) * diff_half))
            u = u * react
            u = np.real(np.fft.ifft2(np.fft.fft2(u) * diff_half))
        return u

    return u0, advance_one_tau, xs, XX, YY, Gg


# ---------------------------------------------------------------------------
# Field reconstruction (figures + global L2 only; E_m is reconstruction-free)
# ---------------------------------------------------------------------------
def reconstruct_field(density_estimation, density_evaluate_grid, X, w, mass_scale,
                      XX, YY, subsample=None, rng=None):
    """Reconstruct P_K(probability density) * mass_scale on the grid.

    For very large equal-weight clouds, optionally use a uniform random subsample
    (unbiased for the probability density) to keep the Fourier einsum in memory.
    """
    Xn = np.asarray(X)
    wn = None if w is None else np.asarray(w)
    if subsample is not None and Xn.shape[0] > subsample:
        idx = (rng or np.random).choice(Xn.shape[0], size=subsample, replace=False)
        Xn = Xn[idx]
        wn = None if wn is None else wn[idx]
    coeff = density_estimation(jnp.asarray(Xn),
                               weights=(None if wn is None else jnp.asarray(wn)))
    prob = density_evaluate_grid(jnp.asarray(XX), jnp.asarray(YY), coeff)
    return np.asarray(prob) * mass_scale


def systematic_resample(rng_np, w):
    """Systematic resampling indices drawn proportional to weights w."""
    N = w.shape[0]
    p = w / np.sum(w)
    positions = (rng_np.random() + np.arange(N)) / N
    cumsum = np.cumsum(p)
    cumsum[-1] = 1.0
    return np.clip(np.searchsorted(cumsum, positions, side="right"), 0, N - 1)


# ---------------------------------------------------------------------------
# Branching compaction into a fixed-size buffer
# ---------------------------------------------------------------------------
def branch_compact(X_active, nu, buffer_size):
    X_np = np.asarray(X_active)
    nu_np = np.asarray(nu).astype(np.int64)
    children = np.repeat(X_np, nu_np, axis=0)
    n_new = children.shape[0]
    overflow = n_new > buffer_size
    if overflow:
        children = children[:buffer_size]
        n_new = buffer_size
    Xbuf = np.zeros((buffer_size, 2), dtype=np.float64)
    Xbuf[:n_new] = children
    mask = np.zeros((buffer_size,), dtype=bool)
    mask[:n_new] = True
    return Xbuf, mask, overflow, n_new


# ---------------------------------------------------------------------------
# Per-island metrics
# ---------------------------------------------------------------------------
def island_masses_ref(u_ref, masks, cell_area):
    return np.array([np.sum(u_ref[mk]) * cell_area for mk in masks])


def weighted_local_ess(w):
    s1 = np.sum(w); s2 = np.sum(w * w)
    return (s1 * s1 / s2) if s2 > 0 else 0.0


def compute_island_metrics(X, w_or_None, mass_per_particle, centers, sigma, eta):
    """Return per-island (M_m, local_eff_count) for one particle method.

    weighted: M_m = mass_per_particle * sum_{i in B_m} w_i ;
              local_eff = (sum w)^2/(sum w^2)  over particles in B_m.
    branching (w_or_None is None): M_m = mass_per_particle * count ;
              local_eff = count.
    """
    idx_list = island_index_particles(X, centers, sigma, eta)
    M = len(centers)
    masses = np.zeros(M)
    local_eff = np.zeros(M)
    if w_or_None is None:
        for m, idx in enumerate(idx_list):
            c = int(np.sum(idx))
            masses[m] = mass_per_particle * c
            local_eff[m] = float(c)
    else:
        wn = np.asarray(w_or_None)
        for m, idx in enumerate(idx_list):
            wm = wn[idx]
            masses[m] = mass_per_particle * np.sum(wm)
            local_eff[m] = weighted_local_ess(wm) if wm.size else 0.0
    return masses, local_eff


# ---------------------------------------------------------------------------
# One method run for one seed
# ---------------------------------------------------------------------------
def run_method(method, seed, cfg, centers_j, amplitudes, ref_snapshots, snap_steps,
               XX, YY, Gg, masks, M0, density_estimation, density_evaluate_grid,
               X_init, dW_streamer, ess_threshold):
    """Run a single method, return (time_series rows, final particle state dict)."""
    n = cfg["grid"]; cell_area = (L / n) ** 2
    sigma = cfg["sigma"]; eta = cfg["eta"]
    tau = cfg["tau"]; steps = int(round(cfg["T"] / tau))
    N0 = cfg["N0"]; D = cfg["D"]; lam = cfg["lam"]; beta = cfg["beta"]
    period = jnp.asarray(PERIOD)
    rng_recon = np.random.default_rng(1000 + seed)
    rng_resample = np.random.default_rng(10_000 + seed)

    M_ref_islands = island_masses_ref(ref_snapshots[steps], masks, cell_area)

    is_branch = method in ("minvar_branch", "poisson_branch", "minvar_branch_cap")
    buffer_size = cfg["buffer_size"] if is_branch else N0
    cap = cfg["pc_cap_mult"] * N0 if method == "minvar_branch_cap" else None

    # initial state
    if is_branch:
        Xb = np.zeros((buffer_size, 2)); Xb[:N0] = np.asarray(X_init)
        mb = np.zeros((buffer_size,), dtype=bool); mb[:N0] = True
        X = jnp.asarray(Xb); mask = jnp.asarray(mb)
        w_c = M0 / N0                      # common per-particle mass (cap rescales it)
    else:
        X = X_init
        w = jnp.ones((N0,), dtype=jnp.float64)

    key = jax.random.PRNGKey(10_000 * seed + 7)
    ps = 0                                 # integrated particle-steps
    n_resamples = 0
    rows = []
    final_state = {}

    def record(s):
        t = s * tau
        u_ref_s = ref_snapshots[s]
        if is_branch:
            nact = int(jnp.sum(mask))
            mass_pp = w_c
            total_mass = mass_pp * nact
            masses, local_eff = compute_island_metrics(
                np.asarray(X)[np.asarray(mask)], None, mass_pp, centers_np, sigma, eta)
            global_nESS = 1.0
            max_mean_w = 1.0
            u_field = reconstruct_field(
                density_estimation, density_evaluate_grid,
                np.asarray(X)[np.asarray(mask)], None, total_mass, XX, YY,
                subsample=cfg["recon_subsample"], rng=rng_recon)
        else:
            sum_w = float(jnp.sum(w))
            mass_pp = M0 / N0
            total_mass = mass_pp * sum_w
            wn = np.asarray(w)
            masses, local_eff = compute_island_metrics(
                np.asarray(X), wn, mass_pp, centers_np, sigma, eta)
            global_nESS = float((sum_w ** 2) / (N0 * np.sum(wn ** 2)))
            max_mean_w = float(np.max(wn) / np.mean(wn))
            nact = N0
            u_field = reconstruct_field(
                density_estimation, density_evaluate_grid, np.asarray(X), wn,
                total_mass, XX, YY, subsample=cfg["recon_subsample"], rng=rng_recon)
        # global relative L2 vs reference
        diff = u_field - u_ref_s
        refL2 = np.sqrt(np.sum(u_ref_s ** 2) * cell_area)
        global_rel_L2 = float(np.sqrt(np.sum(diff ** 2) * cell_area) / refL2) if refL2 > 0 else np.nan
        # per-island errors
        Em = np.abs(masses - M_ref_islands) / np.maximum(M_ref_islands, 1e-300)
        rows.append(dict(
            seed=seed, method=method, t=t, Nact=nact, total_mass=total_mass,
            global_nESS=global_nESS, max_mean_weight=max_mean_w,
            global_rel_L2=global_rel_L2,
            min_local_eff=float(np.min(local_eff)),
            median_local_eff=float(np.median(local_eff)),
            max_local_eff=float(np.max(local_eff)),
            mean_Em=float(np.mean(Em)), median_Em=float(np.median(Em)),
            max_Em=float(np.max(Em)), num_Em_gt_20pct=int(np.sum(Em > 0.20)),
            n_resamples=n_resamples,
        ))
        if s == steps:
            # subsampled final cloud for the resolution-hybrid demo (seed-cheap).
            # IMPORTANT: rescale the per-particle mass after subsampling so that the
            # saved cloud's total mass still equals the method's true total_mass
            # (a uniform subsample of an equal-weight cloud needs mpp *= full/saved).
            if is_branch:
                Xc = np.asarray(X)[np.asarray(mask)]
                wc = None
            else:
                Xc = np.asarray(X); wc = np.asarray(w)
            ncap = cfg["recon_subsample"]
            if Xc.shape[0] > ncap:
                sidx = rng_recon.choice(Xc.shape[0], size=ncap, replace=False)
                Xc = Xc[sidx]; wc = None if wc is None else wc[sidx]
            # effective per-particle mass so that n_saved*mpp (or mpp*sum w) == total_mass
            if wc is None:
                cloud_mpp = total_mass / max(Xc.shape[0], 1)
            else:
                cloud_mpp = total_mass / max(float(np.sum(wc)), 1e-300)
            final_state.update(dict(
                masses=masses, local_eff=local_eff, Em=Em,
                M_ref_islands=M_ref_islands, u_field=u_field,
                Nact=nact, ps=ps,
                cloud_X=Xc, cloud_w=wc, cloud_mass_pp=cloud_mpp,
                cloud_total_mass=total_mass))

    centers_np = np.asarray(centers_j)
    if 0 in snap_steps:
        record(0)

    for s in range(1, steps + 1):
        key, kT, kr = jax.random.split(key, 3)
        dW = dW_streamer(kT, buffer_size if is_branch else N0)
        if is_branch:
            rX = r_of_particles(X, centers_j, amplitudes, sigma, lam, beta)
            X = wrap_torus(em_transport(X, jnp.zeros_like(X), D, tau, dW), period)
            if method == "poisson_branch":
                nu = jnp.where(mask, reaction_poisson(kr, rX, tau), 0)
            else:
                nu = jnp.where(mask, reaction_minvar(kr, rX, tau), 0)
            Xb, mb, ov, n_new = branch_compact(X, nu, buffer_size)
            if ov:
                raise RuntimeError(
                    f"{method} buffer overflow at step {s} (>{buffer_size}); "
                    f"increase buffer_safety")
            # optional explicit population cap (LABELLED variant)
            if cap is not None and n_new > cap:
                keep = cap
                perm = np.asarray(jax.random.permutation(
                    jax.random.fold_in(kr, 99), n_new))[:keep]
                pos = Xb[:n_new][perm]
                w_c = w_c * (n_new / keep)
                Xb = np.zeros((buffer_size, 2)); Xb[:keep] = pos
                mb = np.zeros((buffer_size,), dtype=bool); mb[:keep] = True
                n_new = keep
            X = jnp.asarray(Xb); mask = jnp.asarray(mb)
            ps += n_new
        else:
            rX = r_of_particles(X, centers_j, amplitudes, sigma, lam, beta)
            X = wrap_torus(em_transport(X, jnp.zeros_like(X), D, tau, dW), period)
            w = reaction_weighted(w, rX, tau)
            ps += N0
            if method == "weighted_ess_resample":
                wn = np.asarray(w)
                g_ness = float((wn.sum() ** 2) / (N0 * np.sum(wn ** 2)))
                if g_ness < ess_threshold:
                    idx = systematic_resample(rng_resample, wn)
                    X = jnp.asarray(np.asarray(X)[idx])
                    wbar = wn.sum() / N0
                    w = jnp.full((N0,), wbar, dtype=jnp.float64)
                    n_resamples += 1
        if s in snap_steps:
            record(s)

    return rows, final_state


# ---------------------------------------------------------------------------
# dW streamer: shared front-N0 increments across methods (common random numbers)
# ---------------------------------------------------------------------------
def make_dW_streamer(seed, buffer_size_max):
    """Return a function (key, n) -> (n,2) normals.  Because normal(key,(M,2))[:n]
    == normal(key,(n,2)) for the SAME key, the shared front particles get identical
    Brownian increments across weighted (n=N0) and branching (n=buffer) methods."""
    def streamer(key, n):
        return jax.random.normal(key, shape=(n, 2), dtype=jnp.float64)
    return streamer


# ---------------------------------------------------------------------------
# Reproducibility record
# ---------------------------------------------------------------------------
def git_hash(short=False):
    try:
        args = ["git", "-C", _HERE, "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def write_records(out_dir, cfg, centers, amplitudes, extra):
    os.makedirs(out_dir, exist_ok=True)
    cfg_out = dict(cfg)
    cfg_out["centers"] = np.asarray(centers).tolist()
    cfg_out["amplitudes"] = np.asarray(amplitudes).tolist()
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg_out, f, indent=2)
    manifest = dict(
        experiment="multi_island",
        git_commit=git_hash(False),
        git_commit_short=git_hash(True),
        command_line=" ".join([sys.executable] + sys.argv),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        jax_version=jax.__version__,
        datetime=datetime.datetime.now().isoformat(timespec="seconds"),
        seeds=cfg["seeds"],
        out_dir=os.path.abspath(out_dir),
        population_control_active=("minvar_branch_cap" in cfg["methods"]),
        reference_solver="Fourier Strang split-step, grid %d^2" % cfg["grid"],
        ess_threshold=cfg["ess_threshold"],
        buffer_size=cfg.get("buffer_size"),
        ref_mass_growth=extra.get("ref_mass_growth"),
    )
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def _write_csv(path, rows, cols):
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")


def write_outputs(out_dir, cfg, centers, amplitudes, mass_growth, xs, XX, YY,
                  ref_final, Gg, fields_store, ts_rows, island_rows,
                  island_ess_rows, per_seed_rows, verbose=False):
    """Write all CSVs / fields / records.  Called after EACH seed (so a dev-node
    watchdog kill still leaves usable results for the completed seeds) and at the
    end.  metrics_summary averages over whatever seeds are done so far."""
    np.savez(os.path.join(out_dir, "fields_ref.npz"),
             xs=xs, XX=XX, YY=YY, reference=ref_final, Gg=Gg,
             centers=centers, amplitudes=amplitudes)
    for seed, flds in fields_store.items():
        np.savez(os.path.join(out_dir, f"fields_seed{seed}.npz"), **flds)

    _write_csv(os.path.join(out_dir, "time_series.csv"), ts_rows,
               ["seed", "method", "t", "Nact", "total_mass", "global_nESS",
                "max_mean_weight", "global_rel_L2", "min_local_eff",
                "median_local_eff", "max_local_eff", "mean_Em", "median_Em",
                "max_Em", "num_Em_gt_20pct", "n_resamples"])
    _write_csv(os.path.join(out_dir, "island_masses.csv"), island_rows,
               ["seed", "method", "m", "amplitude", "cx", "cy", "M_ref",
                "M_method", "E_m", "local_eff"])
    _write_csv(os.path.join(out_dir, "island_local_ess.csv"), island_ess_rows,
               ["seed", "method", "m", "amplitude", "local_eff", "E_m"])
    _write_csv(os.path.join(out_dir, "per_seed_metrics.csv"), per_seed_rows,
               ["seed", "method", "N0", "Nact_final", "particle_steps",
                "global_rel_L2", "global_nESS", "max_mean_weight", "mean_Em",
                "median_Em", "max_Em", "num_Em_gt_20pct", "min_local_eff",
                "median_local_eff", "n_resamples"])

    summary_rows = []
    for method in cfg["methods"]:
        sub = [r for r in per_seed_rows if r["method"] == method]
        if not sub:
            continue
        mean = lambda c: float(np.mean([r[c] for r in sub]))   # noqa: E731
        std = lambda c: float(np.std([r[c] for r in sub]))     # noqa: E731
        summary_rows.append(dict(
            method=method, n_seeds=len(sub), N0=cfg["N0"],
            Nact_final=mean("Nact_final"), particle_steps=mean("particle_steps"),
            global_rel_L2=mean("global_rel_L2"), global_nESS=mean("global_nESS"),
            max_mean_weight=mean("max_mean_weight"), mean_Em=mean("mean_Em"),
            mean_Em_std=std("mean_Em"), median_Em=mean("median_Em"),
            max_Em=mean("max_Em"), max_Em_std=std("max_Em"),
            num_Em_gt_20pct=mean("num_Em_gt_20pct"),
            min_local_eff=mean("min_local_eff"),
            median_local_eff=mean("median_local_eff"),
            n_resamples=mean("n_resamples")))
    _write_csv(os.path.join(out_dir, "metrics_summary.csv"), summary_rows,
               ["method", "n_seeds", "N0", "Nact_final", "particle_steps",
                "global_rel_L2", "global_nESS", "max_mean_weight", "mean_Em",
                "mean_Em_std", "median_Em", "max_Em", "max_Em_std",
                "num_Em_gt_20pct", "min_local_eff", "median_local_eff", "n_resamples"])
    write_records(out_dir, cfg, centers, amplitudes,
                  extra=dict(ref_mass_growth=mass_growth))
    if verbose:
        print(f"\n=== summary (mean over {summary_rows[0]['n_seeds'] if summary_rows else 0} seeds) ===")
        for r in summary_rows:
            print(f"{r['method']:24s}: mean E_m={r['mean_Em']:.3f}  "
                  f"max E_m={r['max_Em']:.3f}  #(E_m>20%)={r['num_Em_gt_20pct']:.1f}/{cfg['M']}  "
                  f"global nESS={r['global_nESS']:.3f}  global L2={r['global_rel_L2']:.3f}  "
                  f"min local eff={r['min_local_eff']:.1f}  Nact={r['Nact_final']:.0f}", flush=True)
    return summary_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def resolve_cfg():
    p = argparse.ArgumentParser(description="Separated growth-island benchmark")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--M", type=int)
    p.add_argument("--sigma", type=float)
    p.add_argument("--D", type=float)
    p.add_argument("--lambda", dest="lam", type=float)
    p.add_argument("--beta", type=float)
    p.add_argument("--T", type=float)
    p.add_argument("--tau", type=float)
    p.add_argument("--N0", type=int)
    p.add_argument("--K", type=int)
    p.add_argument("--grid", type=int)
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--methods", type=str, nargs="+")
    p.add_argument("--ess_threshold", type=float)
    p.add_argument("--out_dir", type=str)
    args = p.parse_args()

    cfg = dict(DEFAULTS)
    if args.smoke:
        cfg.update(SMOKE)
    for k in ["M", "sigma", "D", "lam", "beta", "T", "tau", "N0", "K", "grid",
              "seeds", "methods", "ess_threshold", "out_dir"]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    cfg["smoke"] = args.smoke
    return cfg


def main():
    cfg = resolve_cfg()
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    print("=== Multi-island benchmark (Sec. 5.2) ===")
    print("backend:", jax.default_backend())
    print("config:", {k: cfg[k] for k in ["M", "sigma", "D", "lam", "beta", "T",
          "tau", "N0", "K", "grid", "seeds", "methods"]})

    centers = make_grid_centers(cfg["M"])
    amplitudes = make_amplitudes(cfg["M"])
    centers_j = jnp.asarray(centers)
    sigma = cfg["sigma"]; eta = cfg["eta"]

    # reference solve + island masks + disjointness assertion
    ref_u0, advance_ref, xs, XX, YY, Gg = reference_solver(cfg, centers, amplitudes)
    masks = island_masks_on_grid(XX, YY, centers, sigma, eta)
    overlap = masks.sum(axis=0).max()
    assert overlap <= 1, ("Island diagnostic regions B_m overlap "
                          "(max coverage %d); reduce sigma or spread centers." % overlap)
    print(f"island disjointness OK (max coverage {int(overlap)}); "
          f"B_m radius = {sigma*np.sqrt(-2*np.log(eta)):.4f}, "
          f"center spacing = {L/int(round(np.sqrt(cfg['M']))):.4f}")

    tau = cfg["tau"]; steps = int(round(cfg["T"] / tau))
    snap_steps = sorted(set(np.linspace(0, steps, cfg["n_snapshots"], dtype=int).tolist()))
    n = cfg["grid"]; cell_area = (L / n) ** 2

    # reference timeline
    t_ref0 = time.time()
    u = ref_u0.copy()
    ref_snapshots = {0: u.copy()}
    for s in range(1, steps + 1):
        u = advance_ref(u)
        if s in snap_steps:
            ref_snapshots[s] = u.copy()
    M0 = float(np.sum(ref_u0) * cell_area)
    M_final = float(np.sum(ref_snapshots[steps]) * cell_area)
    mass_growth = M_final / M0
    print(f"reference done in {time.time()-t_ref0:.1f}s | M0={M0:.3f} "
          f"M_final={M_final:.3f} growth={mass_growth:.1f}x")

    # auto-size branching buffer from reference mass growth
    buffer_size = int(np.ceil(cfg["buffer_safety"] * mass_growth * cfg["N0"]))
    buffer_size = max(buffer_size, 2 * cfg["N0"])
    cfg["buffer_size"] = buffer_size
    print(f"branching buffer_size = {buffer_size} "
          f"({buffer_size/cfg['N0']:.1f}x N0)")

    density_estimation, _, density_evaluate_grid = generate_density_estimation(
        n_freq=cfg["K"], period=PERIOD)

    ts_rows = []           # time_series rows
    island_rows = []       # island_masses rows (final time, per seed/method/island)
    island_ess_rows = []   # island_local_ess rows
    per_seed_rows = []     # per_seed_metrics (final time)
    fields_store = {}      # seed -> {method: u_field} for the snapshot figure

    for seed in cfg["seeds"]:
        key = jax.random.PRNGKey(seed)
        key, k_init = jax.random.split(key)
        X_init = jax.random.uniform(k_init, (cfg["N0"], 2), minval=-np.pi, maxval=np.pi,
                                    dtype=jnp.float64)
        dW_streamer = make_dW_streamer(seed, buffer_size)
        seed_fields = {}
        t_seed = time.time()
        for method in cfg["methods"]:
            rows, final = run_method(
                method, seed, cfg, centers_j, amplitudes, ref_snapshots, snap_steps,
                XX, YY, Gg, masks, M0, density_estimation, density_evaluate_grid,
                X_init, dW_streamer, cfg["ess_threshold"])
            ts_rows.extend(rows)
            last = rows[-1]
            per_seed_rows.append(dict(
                seed=seed, method=method, N0=cfg["N0"], Nact_final=final["Nact"],
                particle_steps=final["ps"], global_rel_L2=last["global_rel_L2"],
                global_nESS=last["global_nESS"], max_mean_weight=last["max_mean_weight"],
                mean_Em=last["mean_Em"], median_Em=last["median_Em"],
                max_Em=last["max_Em"], num_Em_gt_20pct=last["num_Em_gt_20pct"],
                min_local_eff=last["min_local_eff"],
                median_local_eff=last["median_local_eff"],
                n_resamples=last["n_resamples"]))
            for m in range(cfg["M"]):
                island_rows.append(dict(
                    seed=seed, method=method, m=m, amplitude=float(amplitudes[m]),
                    cx=float(centers[m, 0]), cy=float(centers[m, 1]),
                    M_ref=float(final["M_ref_islands"][m]),
                    M_method=float(final["masses"][m]), E_m=float(final["Em"][m]),
                    local_eff=float(final["local_eff"][m])))
                island_ess_rows.append(dict(
                    seed=seed, method=method, m=m,
                    amplitude=float(amplitudes[m]),
                    local_eff=float(final["local_eff"][m]),
                    E_m=float(final["Em"][m])))
            seed_fields[method] = final["u_field"]
            # save final particle cloud (seed 0 only) for the resolution-hybrid demo
            if seed == cfg["seeds"][0] and method in ("weighted", "minvar_branch"):
                cloud_dir = os.path.join(out_dir, "clouds")
                os.makedirs(cloud_dir, exist_ok=True)
                np.savez(os.path.join(cloud_dir, f"cloud_{method}_seed{seed}.npz"),
                         X=final["cloud_X"],
                         w=(np.ones(final["cloud_X"].shape[0]) if final["cloud_w"] is None
                            else final["cloud_w"]),
                         mass_per_particle=final["cloud_mass_pp"],
                         box=np.array(PERIOD), centers=centers, amplitudes=amplitudes,
                         sigma=cfg["sigma"], method=method)
        fields_store[seed] = seed_fields
        print(f"seed {seed} done in {time.time()-t_seed:.1f}s", flush=True)
        # incremental write after EACH seed (watchdog-kill safe)
        write_outputs(out_dir, cfg, centers, amplitudes, mass_growth, xs, XX, YY,
                      ref_snapshots[steps], Gg, fields_store, ts_rows, island_rows,
                      island_ess_rows, per_seed_rows, verbose=True)

    print("\nwrote outputs to", out_dir, flush=True)


if __name__ == "__main__":
    main()
