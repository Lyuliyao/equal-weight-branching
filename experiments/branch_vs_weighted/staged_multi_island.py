"""Staged separated growth islands -- the paper-strong rebuttal benchmark (Sec. 5.2).
=====================================================================================

A TIME-STAGGERED version of the separated-island benchmark, designed to prove the
paper-level claim (not just a diagnostic):

  When separated growth regions become important at different times, global
  ESS-triggered resampling can keep a healthy global ESS but lose local lineages
  in the later regions.  Equal-weight branching creates particles where the source
  turns on and gives smaller late-island mass / local-field errors at comparable
  particle-step cost.

PDE on the torus T^2 = [-pi,pi]^2:

    d_t u = D Laplacian(u) + r(t,x) u ,
    r(t,x) = lambda * sum_{g=1}^G s_g(t) * sum_{m in G_g} a_m G(x;c_m) - beta ,
    G(x;c_m) = exp(-d_T(x,c_m)^2 / (2 sigma^2)) ,
    s_g(t) = 0.5[ tanh((t-t_on_g)/delta) - tanh((t-t_off_g)/delta) ] .

M=16 islands on a 4x4 grid, split into G=4 spatially-separated activation groups
(a 2x2 checkerboard sub-lattice each, so every stage activates separated regions).
The LATE group (g=G) turns on last and is the discriminating diagnostic.

Two design choices remove the irrelevant initial-island sampling noise that
dominated the static benchmark and isolate the reaction representation:

  * STRATIFIED uniform initial cloud: deterministic per-island quotas so every
    island starts with the SAME number of particles inside B_m (u0 == 1);
  * common stratified initial positions AND common Brownian increments across all
    same-budget methods.

Methods compared:
    weighted                          : raw weighted particles
    weighted_ess_resample             : systematic resampling when global nESS<0.5
    weighted_ess_resample_costmatched : same, with N0 chosen so integrated
                                        particle-steps match branching
    minvar_branch                     : equal-weight minimum-variance branching

Run:
    python staged_multi_island.py --smoke
    python staged_multi_island.py --seeds 0 1 2 3 4 5 6 7 --out_dir results/staged
"""
import os
import sys
import json
import time
import argparse
import datetime
import platform

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from common_particle import (
    generate_density_estimation, em_transport, wrap_torus,
    reaction_weighted, reaction_minvar, reaction_poisson, nESS,
)
from multi_island import (
    PERIOD, L, torus_dist2, torus_dist2_grid, make_grid_centers, make_amplitudes,
    island_masks_on_grid, island_index_particles, reconstruct_field,
    systematic_resample, branch_compact, weighted_local_ess, git_hash,
)

# ---------------------------------------------------------------------------
# CONFIG (initial schedule; tuned in parameter_log.md)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    M=16, G=4, sigma=0.16, D=0.01, lam=9.0, beta=0.8,
    T=1.2, tau=1e-3, N0=20000, K=64, grid=512,
    delta=0.03,
    # amplitude variation across islands: 0.0 = uniform (a_m=1).  The staged
    # mechanism is about activation TIMING, not amplitude; uniform amplitudes
    # remove the amplitude-driven branching-process variance that confounded the
    # static benchmark and let every late island reach a comparable local count.
    amp_var=0.0,
    # activation windows [t_on, t_off] per group (ordered; group G-1 is the late one)
    windows=[[0.00, 0.35], [0.25, 0.60], [0.50, 0.90], [0.80, 1.20]],
    eta=0.5, ess_threshold=0.5, n_snapshots=24,
    seeds=[0, 1, 2, 3, 4, 5, 6, 7],
    methods=["weighted", "weighted_ess_resample",
             "weighted_ess_resample_costmatched", "minvar_branch"],
    buffer_safety=1.8, recon_subsample=200000,
    out_dir="results/staged_multi_island",
)

SMOKE = dict(
    N0=4000, K=16, grid=128, T=0.4, n_snapshots=8, seeds=[0],
    windows=[[0.00, 0.12], [0.08, 0.20], [0.16, 0.30], [0.26, 0.40]],
    out_dir="results/staged_smoke",
)


# ---------------------------------------------------------------------------
# Group assignment: 2x2 checkerboard sub-lattices (each group spatially spread)
# ---------------------------------------------------------------------------
def assign_groups(M, G):
    """Return groups[m] in 0..G-1 for the 4x4 island grid (M=16, G=4).

    Island m sits at (row, col) = (m//n, m%n), n=sqrt(M).  Group =
    (row%2)*2 + (col%2): four interleaved 2x2 sub-lattices, each spread across
    the whole domain so each stage activates separated regions.
    """
    n = int(round(np.sqrt(M)))
    groups = np.zeros(M, dtype=int)
    for m in range(M):
        r, c = m // n, m % n
        groups[m] = (r % 2) * 2 + (c % 2)
    assert groups.max() == G - 1, "group assignment expects G=4 on a 4x4 grid"
    return groups


def staged_amplitudes(M, amp_var):
    """a_m = 1 + amp_var*sin(2 pi m/M); amp_var=0 gives uniform amplitudes."""
    m = np.arange(1, M + 1)
    return 1.0 + amp_var * np.sin(2.0 * np.pi * m / M)


def s_windows(t, windows, delta):
    """s_g(t) for each group (array length G)."""
    out = []
    for (t_on, t_off) in windows:
        out.append(0.5 * (np.tanh((t - t_on) / delta) - np.tanh((t - t_off) / delta)))
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Time-dependent reaction on particles
# ---------------------------------------------------------------------------
def group_G_particles(X, centers_j, amplitudes, groups, sigma):
    """Return list of G arrays G_g(X) = sum_{m in g} a_m exp(-d^2/2sig^2)."""
    Gg = {}
    for g in range(int(groups.max()) + 1):
        acc = jnp.zeros((X.shape[0],), dtype=jnp.float64)
        for m in np.where(groups == g)[0]:
            acc = acc + amplitudes[m] * jnp.exp(
                -torus_dist2(X, centers_j[m]) / (2.0 * sigma ** 2))
        Gg[g] = acc
    return Gg


def r_staged_particles(X, t, centers_j, amplitudes, groups, sigma, lam, beta,
                       windows, delta):
    s = s_windows(t, windows, delta)
    Gg = group_G_particles(X, centers_j, amplitudes, groups, sigma)
    acc = jnp.zeros((X.shape[0],), dtype=jnp.float64)
    for g in Gg:
        acc = acc + s[g] * Gg[g]
    return lam * acc - beta


# ---------------------------------------------------------------------------
# Time-dependent Fourier split-step reference
# ---------------------------------------------------------------------------
def reference_solver(cfg, centers, amplitudes, groups):
    n = cfg["grid"]
    xs = -np.pi + (np.arange(n) + 0.5) * (L / n)
    XX, YY = np.meshgrid(xs, xs, indexing="xy")
    # per-group spatial growth field G_g(x)
    Gg_grid = []
    for g in range(cfg["G"]):
        acc = np.zeros_like(XX)
        for m in np.where(groups == g)[0]:
            acc += amplitudes[m] * np.exp(
                -torus_dist2_grid(XX, YY, centers[m]) / (2.0 * cfg["sigma"] ** 2))
        Gg_grid.append(acc)
    u0 = np.ones_like(XX)

    k = np.fft.fftfreq(n, d=(L / n)) * 2.0 * np.pi
    KX, KY = np.meshgrid(k, k, indexing="xy")
    lap = -(KX ** 2 + KY ** 2)
    tau = cfg["tau"]
    diff_half = np.exp(cfg["D"] * lap * (tau / 2.0))

    def advance_one_tau(u, t_mid):
        s = s_windows(t_mid, cfg["windows"], cfg["delta"])
        rg = cfg["lam"] * sum(s[g] * Gg_grid[g] for g in range(cfg["G"])) - cfg["beta"]
        react = np.exp(rg * tau)
        u = np.real(np.fft.ifft2(np.fft.fft2(u) * diff_half))
        u = u * react
        u = np.real(np.fft.ifft2(np.fft.fft2(u) * diff_half))
        return u

    return u0, advance_one_tau, xs, XX, YY


# ---------------------------------------------------------------------------
# Stratified uniform initial cloud (u0 == 1): equal particles per island B_m
# ---------------------------------------------------------------------------
def stratified_initial_particles(rng, N0, cfg, centers):
    """Return (N0,2) positions: equal deterministic quota inside each B_m disk,
    background filled by stratified per-cell quotas.  Same uniform measure u0==1,
    lower variance than i.i.d. uniform; identical quotas regardless of seed."""
    M = cfg["M"]; n = int(round(np.sqrt(M)))
    sigma = cfg["sigma"]; eta = cfg["eta"]
    R_B = sigma * np.sqrt(-2.0 * np.log(eta))
    half = L / (2 * n)                       # half-cell
    cell_area = (L / n) ** 2
    disk_area = np.pi * R_B ** 2
    q_cell = N0 // M                          # per-cell quota
    q_disk = int(round(q_cell * disk_area / cell_area))
    pts = []
    # distribute the rounding remainder of cells across the first few cells
    rem = N0 - q_cell * M
    for m in range(M):
        c = centers[m]
        qc = q_cell + (1 if m < rem else 0)
        qd = min(q_disk, qc)
        qb = qc - qd
        # disk: polar uniform
        r = R_B * np.sqrt(rng.random(qd))
        th = 2.0 * np.pi * rng.random(qd)
        pts.append(np.stack([c[0] + r * np.cos(th), c[1] + r * np.sin(th)], axis=1))
        # background within the cell box minus the disk (rejection)
        got = 0; acc = []
        while got < qb:
            bx = c[0] + (rng.random(2 * qb + 8) * 2 - 1) * half
            by = c[1] + (rng.random(2 * qb + 8) * 2 - 1) * half
            d2 = (bx - c[0]) ** 2 + (by - c[1]) ** 2
            keep = d2 > R_B ** 2
            xy = np.stack([bx[keep], by[keep]], axis=1)
            acc.append(xy); got += xy.shape[0]
        pts.append(np.concatenate(acc, axis=0)[:qb])
    P = np.concatenate(pts, axis=0)[:N0]
    # wrap into the torus (cells are interior, so this is a no-op except seam guards)
    P[:, 0] = (P[:, 0] + np.pi) % L - np.pi
    P[:, 1] = (P[:, 1] + np.pi) % L - np.pi
    return jnp.asarray(P)


# ---------------------------------------------------------------------------
# Per-island metrics (mass via particle counting; local L2 via reconstruction)
# ---------------------------------------------------------------------------
def island_masses_ref(u_ref, masks, cell_area):
    return np.array([np.sum(u_ref[mk]) * cell_area for mk in masks])


def compute_island_metrics(X, w_or_None, mass_pp, centers, sigma, eta):
    idx_list = island_index_particles(X, centers, sigma, eta)
    M = len(centers)
    masses = np.zeros(M); local_eff = np.zeros(M)
    if w_or_None is None:
        for m, idx in enumerate(idx_list):
            c = int(np.sum(idx)); masses[m] = mass_pp * c; local_eff[m] = float(c)
    else:
        wn = np.asarray(w_or_None)
        for m, idx in enumerate(idx_list):
            wm = wn[idx]
            masses[m] = mass_pp * np.sum(wm)
            local_eff[m] = weighted_local_ess(wm) if wm.size else 0.0
    return masses, local_eff


def island_local_L2(u_field, u_ref, masks, cell_area):
    """Relative L2 error of the reconstructed field restricted to each B_m."""
    out = np.zeros(len(masks))
    for m, mk in enumerate(masks):
        num = np.sqrt(np.sum((u_field[mk] - u_ref[mk]) ** 2) * cell_area)
        den = np.sqrt(np.sum(u_ref[mk] ** 2) * cell_area)
        out[m] = num / den if den > 0 else np.nan
    return out


# ---------------------------------------------------------------------------
# One method for one seed
# ---------------------------------------------------------------------------
def run_method(method, seed, cfg, centers_j, centers_np, amplitudes, groups,
               ref_snapshots, snap_steps, XX, YY, masks, late_idx, M0,
               density_estimation, density_evaluate_grid, X_init, N0_method,
               buffer_size, ess_threshold):
    n = cfg["grid"]; cell_area = (L / n) ** 2
    sigma = cfg["sigma"]; eta = cfg["eta"]; tau = cfg["tau"]
    steps = int(round(cfg["T"] / tau))
    D = cfg["D"]; lam = cfg["lam"]; beta = cfg["beta"]
    windows = cfg["windows"]; delta = cfg["delta"]
    period = jnp.asarray(PERIOD)
    rng_recon = np.random.default_rng(1000 + seed)
    rng_resample = np.random.default_rng(10_000 + seed + (777 if "costmatched" in method else 0))
    M_ref_islands = island_masses_ref(ref_snapshots[steps], masks, cell_area)

    is_branch = method == "minvar_branch"
    bufsz = buffer_size if is_branch else N0_method

    if is_branch:
        Xb = np.zeros((bufsz, 2)); Xb[:N0_method] = np.asarray(X_init)
        mb = np.zeros((bufsz,), dtype=bool); mb[:N0_method] = True
        X = jnp.asarray(Xb); mask = jnp.asarray(mb); w_c = M0 / N0_method
    else:
        X = X_init; w = jnp.ones((N0_method,), dtype=jnp.float64)

    key = jax.random.PRNGKey(10_000 * seed + (3 if "costmatched" in method else 7))
    ps = 0; n_resamples = 0
    rows = []; final = {}

    def record(s):
        t = s * tau; u_ref_s = ref_snapshots[s]
        if is_branch:
            nact = int(jnp.sum(mask)); mass_pp = w_c; total_mass = mass_pp * nact
            Xa = np.asarray(X)[np.asarray(mask)]
            masses, local_eff = compute_island_metrics(Xa, None, mass_pp, centers_np, sigma, eta)
            global_nESS = 1.0; max_mean_w = 1.0
            u_field = reconstruct_field(density_estimation, density_evaluate_grid,
                                        Xa, None, total_mass, XX, YY,
                                        subsample=cfg["recon_subsample"], rng=rng_recon)
        else:
            wn = np.asarray(w); sum_w = float(np.sum(wn)); mass_pp = M0 / N0_method
            total_mass = mass_pp * sum_w
            masses, local_eff = compute_island_metrics(np.asarray(X), wn, mass_pp, centers_np, sigma, eta)
            global_nESS = float((sum_w ** 2) / (N0_method * np.sum(wn ** 2)))
            max_mean_w = float(np.max(wn) / np.mean(wn)); nact = N0_method
            u_field = reconstruct_field(density_estimation, density_evaluate_grid,
                                        np.asarray(X), wn, total_mass, XX, YY,
                                        subsample=cfg["recon_subsample"], rng=rng_recon)
        diff = u_field - u_ref_s
        refL2 = np.sqrt(np.sum(u_ref_s ** 2) * cell_area)
        global_rel_L2 = float(np.sqrt(np.sum(diff ** 2) * cell_area) / refL2) if refL2 > 0 else np.nan
        Em = np.abs(masses - M_ref_islands) / np.maximum(M_ref_islands, 1e-300)
        locL2 = island_local_L2(u_field, u_ref_s, masks, cell_area)
        rows.append(dict(
            seed=seed, method=method, t=t, Nact=nact, total_mass=total_mass,
            global_nESS=global_nESS, max_mean_weight=max_mean_w, global_rel_L2=global_rel_L2,
            mean_Em=float(np.mean(Em)), max_Em=float(np.max(Em)),
            num_Em_gt_20pct=int(np.sum(Em > 0.20)),
            mean_late_Em=float(np.mean(Em[late_idx])), max_late_Em=float(np.max(Em[late_idx])),
            num_late_gt_20pct=int(np.sum(Em[late_idx] > 0.20)),
            max_late_localL2=float(np.nanmax(locL2[late_idx])),
            min_local_eff=float(np.min(local_eff)),
            min_late_local_eff=float(np.min(local_eff[late_idx])),
            median_local_eff=float(np.median(local_eff)), n_resamples=n_resamples))
        if s == steps:
            if is_branch:
                Xc = np.asarray(X)[np.asarray(mask)]; wc = None
            else:
                Xc = np.asarray(X); wc = np.asarray(w)
            ncap = cfg["recon_subsample"]
            if Xc.shape[0] > ncap:
                si = rng_recon.choice(Xc.shape[0], size=ncap, replace=False)
                Xc = Xc[si]; wc = None if wc is None else wc[si]
            cmpp = total_mass / max(Xc.shape[0], 1) if wc is None else total_mass / max(float(np.sum(wc)), 1e-300)
            final.update(dict(masses=masses, local_eff=local_eff, Em=Em, locL2=locL2,
                              M_ref_islands=M_ref_islands, u_field=u_field, Nact=nact, ps=ps,
                              cloud_X=Xc, cloud_w=wc, cloud_mass_pp=cmpp))

    if 0 in snap_steps:
        record(0)
    for s in range(1, steps + 1):
        t_mid = (s - 0.5) * tau
        key, kT, kr = jax.random.split(key, 3)
        dW = jax.random.normal(kT, shape=(bufsz if is_branch else N0_method, 2), dtype=jnp.float64)
        if is_branch:
            X = wrap_torus(em_transport(X, jnp.zeros_like(X), D, tau, dW), period)
            rX = r_staged_particles(X, t_mid, centers_j, amplitudes, groups, sigma, lam, beta, windows, delta)
            nu = jnp.where(mask, reaction_minvar(kr, rX, tau), 0)
            Xb, mb, ov, n_new = branch_compact(X, nu, bufsz)
            if ov:
                raise RuntimeError(f"{method} buffer overflow at step {s} (>{bufsz}); raise buffer_safety")
            X = jnp.asarray(Xb); mask = jnp.asarray(mb); ps += n_new
        else:
            X = wrap_torus(em_transport(X, jnp.zeros_like(X), D, tau, dW), period)
            rX = r_staged_particles(X, t_mid, centers_j, amplitudes, groups, sigma, lam, beta, windows, delta)
            w = reaction_weighted(w, rX, tau); ps += N0_method
            if method in ("weighted_ess_resample", "weighted_ess_resample_costmatched"):
                wn = np.asarray(w)
                g_ness = float((wn.sum() ** 2) / (N0_method * np.sum(wn ** 2)))
                if g_ness < ess_threshold:
                    idx = systematic_resample(rng_resample, wn)
                    X = jnp.asarray(np.asarray(X)[idx])
                    w = jnp.full((N0_method,), wn.sum() / N0_method, dtype=jnp.float64)
                    n_resamples += 1
        if s in snap_steps:
            record(s)
    return rows, final


def _write_csv(path, rows, cols):
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")


def resolve_cfg():
    p = argparse.ArgumentParser(description="Staged separated growth-island benchmark")
    p.add_argument("--smoke", action="store_true")
    for k, t in [("M", int), ("sigma", float), ("D", float), ("beta", float),
                 ("T", float), ("tau", float), ("N0", int), ("K", int), ("grid", int),
                 ("delta", float), ("ess_threshold", float), ("out_dir", str)]:
        p.add_argument(f"--{k}", type=t)
    p.add_argument("--lambda", dest="lam", type=float)
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--methods", type=str, nargs="+")
    p.add_argument("--config", type=str)
    args = p.parse_args()
    cfg = dict(DEFAULTS)
    if args.smoke:
        cfg.update(SMOKE)
    if args.config:
        with open(args.config) as f:
            loaded = json.load(f)
        for k in ["M", "G", "sigma", "D", "lam", "beta", "T", "tau", "N0", "K", "grid",
                  "delta", "windows", "ess_threshold", "seeds", "methods", "buffer_safety"]:
            if k in loaded:
                cfg[k] = loaded[k]
    for k in ["M", "sigma", "D", "lam", "beta", "T", "tau", "N0", "K", "grid",
              "delta", "ess_threshold", "seeds", "methods", "out_dir"]:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    cfg["smoke"] = args.smoke
    return cfg


def main():
    cfg = resolve_cfg()
    out_dir = cfg["out_dir"]; os.makedirs(out_dir, exist_ok=True)
    print("=== Staged multi-island benchmark (Sec. 5.2) ===  backend:", jax.default_backend())
    print("config:", {k: cfg[k] for k in ["M", "G", "sigma", "D", "lam", "beta", "T",
          "tau", "N0", "K", "grid", "delta", "windows", "seeds", "methods"]}, flush=True)

    centers = make_grid_centers(cfg["M"]); amplitudes = staged_amplitudes(cfg["M"], cfg["amp_var"])
    centers_j = jnp.asarray(centers); groups = assign_groups(cfg["M"], cfg["G"])
    late_g = cfg["G"] - 1; late_idx = np.where(groups == late_g)[0]
    sigma = cfg["sigma"]; eta = cfg["eta"]
    print(f"late group {late_g} islands: {late_idx.tolist()} (window {cfg['windows'][late_g]})")

    ref_u0, advance_ref, xs, XX, YY = reference_solver(cfg, centers, amplitudes, groups)
    masks = island_masks_on_grid(XX, YY, centers, sigma, eta)
    assert masks.sum(axis=0).max() <= 1, "island B_m overlap"
    tau = cfg["tau"]; steps = int(round(cfg["T"] / tau))
    snap_steps = sorted(set(np.linspace(0, steps, cfg["n_snapshots"], dtype=int).tolist()))
    n = cfg["grid"]; cell_area = (L / n) ** 2

    t0 = time.time(); u = ref_u0.copy(); ref_snapshots = {0: u.copy()}
    for s in range(1, steps + 1):
        u = advance_ref(u, (s - 0.5) * tau)
        if s in snap_steps:
            ref_snapshots[s] = u.copy()
    M0 = float(np.sum(ref_u0) * cell_area)
    mass_growth = float(np.sum(ref_snapshots[steps]) * cell_area) / M0
    print(f"reference {time.time()-t0:.1f}s | growth={mass_growth:.1f}x", flush=True)
    buffer_size = max(int(np.ceil(cfg["buffer_safety"] * mass_growth * cfg["N0"])), 2 * cfg["N0"])
    cfg["buffer_size"] = buffer_size

    density_estimation, _, density_evaluate_grid = generate_density_estimation(n_freq=cfg["K"], period=PERIOD)

    ts_rows = []; island_rows = []; per_seed_rows = []; fields_store = {}; cost_info = {}
    for seed in cfg["seeds"]:
        rng_init = np.random.default_rng(seed)
        X_init = stratified_initial_particles(rng_init, cfg["N0"], cfg, centers)
        t_seed = time.time(); seed_fields = {}
        # branching first (sets the cost-match budget)
        ordered = [m for m in cfg["methods"] if m == "minvar_branch"] + \
                  [m for m in cfg["methods"] if m != "minvar_branch"]
        N0_cm = None
        for method in ordered:
            if method == "weighted_ess_resample_costmatched":
                C_branch = cost_info.get((seed, "minvar_branch"))
                N0_cm = int(round(C_branch / steps)) if C_branch else cfg["N0"]
                rng_cm = np.random.default_rng(50_000 + seed)
                Xm = stratified_initial_particles(rng_cm, N0_cm, cfg, centers)
                N0_method = N0_cm; X_use = Xm
            else:
                N0_method = cfg["N0"]; X_use = X_init
            rows, final = run_method(method, seed, cfg, centers_j, centers, amplitudes, groups,
                                     ref_snapshots, snap_steps, XX, YY, masks, late_idx, M0,
                                     density_estimation, density_evaluate_grid, X_use,
                                     N0_method, buffer_size, cfg["ess_threshold"])
            ts_rows.extend(rows); last = rows[-1]
            cost_info[(seed, method)] = final["ps"]
            per_seed_rows.append(dict(
                seed=seed, method=method, N0=N0_method, Nact_final=final["Nact"],
                particle_steps=final["ps"], global_rel_L2=last["global_rel_L2"],
                global_nESS=last["global_nESS"], max_mean_weight=last["max_mean_weight"],
                mean_Em=last["mean_Em"], max_Em=last["max_Em"], num_Em_gt_20pct=last["num_Em_gt_20pct"],
                mean_late_Em=last["mean_late_Em"], max_late_Em=last["max_late_Em"],
                num_late_gt_20pct=last["num_late_gt_20pct"], max_late_localL2=last["max_late_localL2"],
                min_local_eff=last["min_local_eff"], min_late_local_eff=last["min_late_local_eff"],
                n_resamples=last["n_resamples"]))
            for m in range(cfg["M"]):
                island_rows.append(dict(seed=seed, method=method, m=m, group=int(groups[m]),
                    is_late=int(m in late_idx), amplitude=float(amplitudes[m]),
                    cx=float(centers[m, 0]), cy=float(centers[m, 1]),
                    M_ref=float(final["M_ref_islands"][m]), M_method=float(final["masses"][m]),
                    E_m=float(final["Em"][m]), local_L2=float(final["locL2"][m]),
                    local_eff=float(final["local_eff"][m])))
            seed_fields[method] = final["u_field"]
            if seed == cfg["seeds"][0] and method in ("weighted_ess_resample_costmatched", "minvar_branch"):
                cd = os.path.join(out_dir, "clouds"); os.makedirs(cd, exist_ok=True)
                np.savez(os.path.join(cd, f"cloud_{method}_seed{seed}.npz"),
                         X=final["cloud_X"], w=(np.ones(final["cloud_X"].shape[0]) if final["cloud_w"] is None else final["cloud_w"]),
                         mass_per_particle=final["cloud_mass_pp"], box=np.array(PERIOD),
                         centers=centers, amplitudes=amplitudes, sigma=cfg["sigma"], method=method)
        fields_store[seed] = seed_fields
        print(f"seed {seed} done in {time.time()-t_seed:.1f}s  (N0_cm={N0_cm})", flush=True)
        write_outputs(out_dir, cfg, centers, amplitudes, groups, late_idx, mass_growth,
                      xs, XX, YY, ref_snapshots[steps], fields_store, ts_rows, island_rows,
                      per_seed_rows, verbose=True)
    print("\nwrote outputs to", out_dir, flush=True)


def write_outputs(out_dir, cfg, centers, amplitudes, groups, late_idx, mass_growth,
                  xs, XX, YY, ref_final, fields_store, ts_rows, island_rows, per_seed_rows,
                  verbose=False):
    np.savez(os.path.join(out_dir, "fields_ref.npz"), xs=xs, XX=XX, YY=YY,
             reference=ref_final, centers=centers, amplitudes=amplitudes,
             groups=groups, late_idx=late_idx)
    for seed, flds in fields_store.items():
        np.savez(os.path.join(out_dir, f"fields_seed{seed}.npz"), **flds)
    _write_csv(os.path.join(out_dir, "time_series.csv"), ts_rows,
               ["seed", "method", "t", "Nact", "total_mass", "global_nESS", "max_mean_weight",
                "global_rel_L2", "mean_Em", "max_Em", "num_Em_gt_20pct", "mean_late_Em",
                "max_late_Em", "num_late_gt_20pct", "max_late_localL2", "min_local_eff",
                "min_late_local_eff", "median_local_eff", "n_resamples"])
    _write_csv(os.path.join(out_dir, "island_masses.csv"), island_rows,
               ["seed", "method", "m", "group", "is_late", "amplitude", "cx", "cy",
                "M_ref", "M_method", "E_m", "local_L2", "local_eff"])
    _write_csv(os.path.join(out_dir, "island_local_ess.csv"), island_rows,
               ["seed", "method", "m", "group", "is_late", "amplitude", "local_eff", "E_m", "local_L2"])
    _write_csv(os.path.join(out_dir, "per_seed_metrics.csv"), per_seed_rows,
               ["seed", "method", "N0", "Nact_final", "particle_steps", "global_rel_L2",
                "global_nESS", "max_mean_weight", "mean_Em", "max_Em", "num_Em_gt_20pct",
                "mean_late_Em", "max_late_Em", "num_late_gt_20pct", "max_late_localL2",
                "min_local_eff", "min_late_local_eff", "n_resamples"])
    # late-group + summary
    summ = []; late_rows = []
    for method in cfg["methods"]:
        sub = [r for r in per_seed_rows if r["method"] == method]
        if not sub:
            continue
        mean = lambda c: float(np.mean([r[c] for r in sub]))
        std = lambda c: float(np.std([r[c] for r in sub]))
        row = dict(method=method, n_seeds=len(sub),
                   N0=int(np.mean([r["N0"] for r in sub])),
                   particle_steps=mean("particle_steps"), Nact_final=mean("Nact_final"),
                   global_rel_L2=mean("global_rel_L2"), global_nESS=mean("global_nESS"),
                   mean_Em=mean("mean_Em"), max_Em=mean("max_Em"),
                   num_Em_gt_20pct=mean("num_Em_gt_20pct"),
                   mean_late_Em=mean("mean_late_Em"), mean_late_Em_std=std("mean_late_Em"),
                   max_late_Em=mean("max_late_Em"), max_late_localL2=mean("max_late_localL2"),
                   num_late_gt_20pct=mean("num_late_gt_20pct"),
                   min_local_eff=mean("min_local_eff"), min_late_local_eff=mean("min_late_local_eff"))
        summ.append(row); late_rows.append(row)
    cols = ["method", "n_seeds", "N0", "particle_steps", "Nact_final", "global_rel_L2",
            "global_nESS", "mean_Em", "max_Em", "num_Em_gt_20pct", "mean_late_Em",
            "mean_late_Em_std", "max_late_Em", "max_late_localL2", "num_late_gt_20pct",
            "min_local_eff", "min_late_local_eff"]
    _write_csv(os.path.join(out_dir, "metrics_summary.csv"), summ, cols)
    _write_csv(os.path.join(out_dir, "late_group_metrics.csv"), late_rows, cols)
    # records
    cfg_out = dict(cfg); cfg_out["centers"] = np.asarray(centers).tolist()
    cfg_out["amplitudes"] = np.asarray(amplitudes).tolist()
    cfg_out["groups"] = np.asarray(groups).tolist(); cfg_out["late_idx"] = late_idx.tolist()
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg_out, f, indent=2)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(dict(experiment="staged_multi_island", git_commit=git_hash(False),
                       git_commit_short=git_hash(True),
                       command_line=" ".join([sys.executable] + sys.argv),
                       python_version=platform.python_version(), numpy_version=np.__version__,
                       jax_version=jax.__version__,
                       datetime=datetime.datetime.now().isoformat(timespec="seconds"),
                       seeds=cfg["seeds"], out_dir=os.path.abspath(out_dir),
                       reference_solver="time-dependent Fourier Strang split-step, grid %d^2" % cfg["grid"],
                       initial_cloud="stratified uniform (equal particles per B_m)",
                       ess_threshold=cfg["ess_threshold"], buffer_size=cfg.get("buffer_size"),
                       ref_mass_growth=mass_growth), f, indent=2)
    if verbose and summ:
        print(f"--- summary ({summ[0]['n_seeds']} seeds) ---", flush=True)
        for r in summ:
            print(f"{r['method']:34s}: late mean E_m={r['mean_late_Em']:.3f} max={r['max_late_Em']:.3f} "
                  f"#late>20%={r['num_late_gt_20pct']:.1f}/{len(late_idx)} lateLocL2={r['max_late_localL2']:.3f} "
                  f"nESS={r['global_nESS']:.3f} minLateEff={r['min_late_local_eff']:.0f} "
                  f"Nact={r['Nact_final']:.0f}({r['Nact_final']/r['N0']:.1f}x) ps={r['particle_steps']:.2e}", flush=True)


if __name__ == "__main__":
    main()
