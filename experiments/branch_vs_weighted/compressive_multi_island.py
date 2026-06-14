"""Compressive-readout staged multi-island (CLAUDE.md §3-7) -- OPTIONAL pilot.
================================================================================

Staged separated growth islands PLUS a deterministic compressive drift that
sharpens the late-island cores AFTER they have grown:

    d_t u = -div(b(t,x) u) + D Laplacian(u) + r(t,x) u ,
    r(t,x) = lambda sum_g s_g(t) sum_{m in G_g} a_m G(x;c_m) - beta ,
    b(t,x) = -kappa sum_g h_g(t) sum_{m in G_g} chi_m(x) (x-c_m)_T .

The compressive drift (only on the late group by default) pulls particles inward
toward each late-island center once the source has grown, so the late cores become
sharp.  The point is to convert "branching keeps a higher local particle count"
into a visible LOCAL FIELD / PEAK accuracy advantage: a method with too few local
particles cannot reconstruct the compressed core.

PRIMARY metrics are local field / shape (CLAUDE.md §5), NOT per-island mass:
  * local window L2 over W_m = {|x-c_m|<=R_W}, R_W=0.25;
  * local peak error in W_m (bandwidth-sensitive);
  * narrow-Gaussian observable <psi_m, mu> with sigma_obs=0.04 (reconstruction-free);
  * local count / local ESS.
Per-island mass E_m is reported as a SANITY CHECK only.

Methods (same initial particles + Brownian increments across same-budget methods):
    weighted, weighted_ess_resample, weighted_ess_resample_costmatched, minvar_branch.

The deterministic reference is a pseudo-spectral split-step on a 512^2 grid:
exact diffusion + RK2 advection-reaction (the drift makes advection pseudo-spectral).

Run:  python compressive_multi_island.py --smoke
      python compressive_multi_island.py --kappa 8 --seeds 0 --out_dir results/comp_pilot
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

from common_particle import (generate_density_estimation, em_transport, wrap_torus,
                             reaction_weighted, reaction_minvar)
from multi_island import (PERIOD, L, torus_dist2, torus_dist2_grid, make_grid_centers,
                          island_masks_on_grid, island_index_particles, reconstruct_field,
                          systematic_resample, branch_compact, weighted_local_ess, git_hash)
from staged_multi_island import (assign_groups, staged_amplitudes, s_windows,
                                 r_staged_particles, stratified_initial_particles,
                                 island_masses_ref, _write_csv)

DEFAULTS = dict(
    M=16, G=4, sigma=0.16, D=0.01, lam=13.0, beta=0.1,
    T=1.2, tau=1e-3, N0=20000, K=64, grid=512, delta=0.03, amp_var=0.0,
    windows=[[0.00, 0.35], [0.25, 0.60], [0.50, 0.90], [0.80, 1.20]],
    # compressive drift
    kappa=8.0, sigma_b=0.20, t_comp=0.95, delta_comp=0.03, compress_late_only=True,
    R_W=0.25, sigma_obs=0.04,
    eta=0.5, ess_threshold=0.5, n_snapshots=24,
    seeds=[0, 1, 2, 3], buffer_safety=1.8, recon_subsample=200000,
    out_dir="results/compressive_multi_island",
)
SMOKE = dict(N0=4000, K=16, grid=128, T=0.4, n_snapshots=6, seeds=[0],
             windows=[[0.00, 0.12], [0.08, 0.20], [0.16, 0.30], [0.26, 0.40]],
             t_comp=0.30, out_dir="results/comp_smoke")


# ---------------------------------------------------------------------------
# Compressive drift
# ---------------------------------------------------------------------------
def h_late(t, cfg):
    return 0.5 * (1.0 + np.tanh((t - cfg["t_comp"]) / cfg["delta_comp"]))


def drift_groups(cfg):
    """Which groups are compressed: late group only, or all."""
    if cfg["compress_late_only"]:
        return [cfg["G"] - 1]
    return list(range(cfg["G"]))


def b_particles(X, t, centers_j, groups, cfg):
    """Compressive drift b(t,X) at particle positions (N,2): inward toward centers."""
    kappa = cfg["kappa"]; sig_b = cfg["sigma_b"]
    h = h_late(t, cfg)
    acc = jnp.zeros_like(X)
    for g in drift_groups(cfg):
        for m in np.where(groups == g)[0]:
            c = centers_j[m]
            dx = X[:, 0] - c[0]; dy = X[:, 1] - c[1]
            dx = dx - L * jnp.round(dx / L); dy = dy - L * jnp.round(dy / L)
            chi = jnp.exp(-(dx * dx + dy * dy) / (2.0 * sig_b ** 2))
            acc = acc + jnp.stack([chi * dx, chi * dy], axis=1)
    return -kappa * h * acc


# ---------------------------------------------------------------------------
# Pseudo-spectral reference: exact diffusion + RK2 advection-reaction.
# ---------------------------------------------------------------------------
def reference_solver(cfg, centers, amplitudes, groups):
    n = cfg["grid"]
    xs = -np.pi + (np.arange(n) + 0.5) * (L / n)
    XX, YY = np.meshgrid(xs, xs, indexing="xy")
    Gg_grid = []
    for g in range(cfg["G"]):
        acc = np.zeros_like(XX)
        for m in np.where(groups == g)[0]:
            acc += amplitudes[m] * np.exp(
                -torus_dist2_grid(XX, YY, centers[m]) / (2.0 * cfg["sigma"] ** 2))
        Gg_grid.append(acc)
    # compressive drift spatial fields per group: B_g = sum_{m in g} chi_m (x-c_m)
    Bx = {}; By = {}
    for g in drift_groups(cfg):
        bx = np.zeros_like(XX); by = np.zeros_like(YY)
        for m in np.where(groups == g)[0]:
            dx = XX - centers[m][0]; dy = YY - centers[m][1]
            dx -= L * np.round(dx / L); dy -= L * np.round(dy / L)
            chi = np.exp(-(dx * dx + dy * dy) / (2.0 * cfg["sigma_b"] ** 2))
            bx += chi * dx; by += chi * dy
        Bx[g] = bx; By[g] = by
    u0 = np.ones_like(XX)

    k = np.fft.fftfreq(n, d=(L / n)) * 2.0 * np.pi
    KX, KY = np.meshgrid(k, k, indexing="xy")
    lap = -(KX ** 2 + KY ** 2)
    iKX = 1j * KX; iKY = 1j * KY
    tau = cfg["tau"]
    diff_half = np.exp(cfg["D"] * lap * (tau / 2.0))

    def b_field(t):
        h = h_late(t, cfg)
        bx = np.zeros_like(XX); by = np.zeros_like(YY)
        for g in drift_groups(cfg):
            bx += Bx[g]; by += By[g]
        return (-cfg["kappa"] * h * bx, -cfg["kappa"] * h * by)

    def rhs_adv_react(u, t):
        s = s_windows(t, cfg["windows"], cfg["delta"])
        rg = cfg["lam"] * sum(s[g] * Gg_grid[g] for g in range(cfg["G"])) - cfg["beta"]
        bx, by = b_field(t)
        # -div(b u) = -(d/dx(bx u) + d/dy(by u)) pseudo-spectrally
        fx = np.fft.fft2(bx * u); fy = np.fft.fft2(by * u)
        divbu = np.real(np.fft.ifft2(iKX * fx + iKY * fy))
        return -divbu + rg * u

    def advance_one_tau(u, t_mid):
        u = np.real(np.fft.ifft2(np.fft.fft2(u) * diff_half))    # half diffusion
        # RK2 (midpoint) for advection+reaction over tau
        k1 = rhs_adv_react(u, t_mid - 0.5 * tau)
        k2 = rhs_adv_react(u + 0.5 * tau * k1, t_mid)
        u = u + tau * k2
        u = np.real(np.fft.ifft2(np.fft.fft2(u) * diff_half))    # half diffusion
        return u

    return u0, advance_one_tau, xs, XX, YY


# ---------------------------------------------------------------------------
# Local-field metrics (primary) + narrow-Gaussian observable + mass sanity.
# ---------------------------------------------------------------------------
def local_window_metrics(u_field, u_ref, XX, YY, centers, R_W, cell_area):
    """Relative local L2 and relative local peak error over W_m={|x-c_m|<=R_W}."""
    M = len(centers); locL2 = np.zeros(M); peak = np.zeros(M)
    for m in range(M):
        d2 = torus_dist2_grid(XX, YY, centers[m]); mk = d2 <= R_W ** 2
        num = np.sqrt(np.sum((u_field[mk] - u_ref[mk]) ** 2) * cell_area)
        den = np.sqrt(np.sum(u_ref[mk] ** 2) * cell_area)
        locL2[m] = num / den if den > 0 else np.nan
        pr = np.max(u_ref[mk]) if mk.any() else np.nan
        pm = np.max(u_field[mk]) if mk.any() else np.nan
        peak[m] = abs(pm - pr) / pr if pr > 0 else np.nan
    return locL2, peak


def narrow_obs(X, w_or_None, mass_pp, centers, sigma_obs, u_ref, XX, YY, cell_area):
    """<psi_m, mu> (reconstruction-free) vs int psi_m u_ref; relative error per island."""
    M = len(centers); err = np.zeros(M)
    Xn = np.asarray(X)
    wn = None if w_or_None is None else np.asarray(w_or_None)
    for m in range(M):
        d2p = torus_dist2_grid(Xn[:, 0], Xn[:, 1], np.asarray(centers[m]))
        psi = np.exp(-d2p / (2.0 * sigma_obs ** 2))
        meas = mass_pp * (np.sum(psi) if wn is None else np.sum(psi * wn))
        d2g = torus_dist2_grid(XX, YY, centers[m])
        psig = np.exp(-d2g / (2.0 * sigma_obs ** 2))
        ref = np.sum(psig * u_ref) * cell_area
        err[m] = abs(meas - ref) / ref if ref > 0 else np.nan
    return err


def island_local_counts(X, w_or_None, centers, R_W):
    M = len(centers); eff = np.zeros(M)
    Xn = np.asarray(X); wn = None if w_or_None is None else np.asarray(w_or_None)
    for m in range(M):
        d2 = torus_dist2_grid(Xn[:, 0], Xn[:, 1], np.asarray(centers[m]))
        inside = d2 <= R_W ** 2
        if wn is None:
            eff[m] = float(np.sum(inside))
        else:
            wi = wn[inside]; eff[m] = weighted_local_ess(wi) if wi.size else 0.0
    return eff


# ---------------------------------------------------------------------------
# One method for one seed
# ---------------------------------------------------------------------------
def run_method(method, seed, cfg, centers_j, centers_np, amplitudes, groups,
               ref_snapshots, snap_steps, XX, YY, masks, late_idx, M0,
               density_estimation, density_evaluate_grid, X_init, N0_method,
               buffer_size, ess_threshold):
    n = cfg["grid"]; cell_area = (L / n) ** 2
    sigma = cfg["sigma"]; eta = cfg["eta"]; tau = cfg["tau"]
    steps = int(round(cfg["T"] / tau)); D = cfg["D"]; lam = cfg["lam"]; beta = cfg["beta"]
    windows = cfg["windows"]; delta = cfg["delta"]; R_W = cfg["R_W"]; sig_obs = cfg["sigma_obs"]
    period = jnp.asarray(PERIOD)
    rng_recon = np.random.default_rng(1000 + seed)
    rng_resample = np.random.default_rng(10_000 + seed + (777 if "costmatched" in method else 0))
    M_ref_islands = island_masses_ref(ref_snapshots[steps], masks, cell_area)
    is_branch = method == "minvar_branch"; bufsz = buffer_size if is_branch else N0_method

    if is_branch:
        Xb = np.zeros((bufsz, 2)); Xb[:N0_method] = np.asarray(X_init)
        mb = np.zeros((bufsz,), dtype=bool); mb[:N0_method] = True
        X = jnp.asarray(Xb); mask = jnp.asarray(mb); w_c = M0 / N0_method
    else:
        X = X_init; w = jnp.ones((N0_method,), dtype=jnp.float64)
    key = jax.random.PRNGKey(10_000 * seed + (3 if "costmatched" in method else 7))
    ps = 0; n_resamples = 0; rows = []; final = {}

    def record(s):
        t = s * tau; u_ref_s = ref_snapshots[s]
        if is_branch:
            nact = int(jnp.sum(mask)); mass_pp = w_c; total_mass = mass_pp * nact
            Xa = np.asarray(X)[np.asarray(mask)]; wa = None
            global_nESS = 1.0
            u_field = reconstruct_field(density_estimation, density_evaluate_grid,
                                        Xa, None, total_mass, XX, YY,
                                        subsample=cfg["recon_subsample"], rng=rng_recon)
        else:
            wn = np.asarray(w); sum_w = float(np.sum(wn)); mass_pp = M0 / N0_method
            total_mass = mass_pp * sum_w; Xa = np.asarray(X); wa = wn; nact = N0_method
            global_nESS = float((sum_w ** 2) / (N0_method * np.sum(wn ** 2)))
            u_field = reconstruct_field(density_estimation, density_evaluate_grid,
                                        Xa, wn, total_mass, XX, YY,
                                        subsample=cfg["recon_subsample"], rng=rng_recon)
        diff = u_field - u_ref_s; refL2 = np.sqrt(np.sum(u_ref_s ** 2) * cell_area)
        global_rel_L2 = float(np.sqrt(np.sum(diff ** 2) * cell_area) / refL2) if refL2 > 0 else np.nan
        locL2, peak = local_window_metrics(u_field, u_ref_s, XX, YY, centers_np, R_W, cell_area)
        obs = narrow_obs(Xa, wa, mass_pp, centers_np, sig_obs, u_ref_s, XX, YY, cell_area)
        lc = island_local_counts(Xa, wa, centers_np, R_W)
        # mass E_m sanity
        idxl = island_index_particles(Xa, centers_np, sigma, eta)
        masses = np.array([mass_pp * (np.sum(ix) if wa is None else np.sum(wa[ix])) for ix in idxl])
        Em = np.abs(masses - M_ref_islands) / np.maximum(M_ref_islands, 1e-300)
        li = late_idx
        rows.append(dict(seed=seed, method=method, t=t, Nact=nact, global_nESS=global_nESS,
                         global_rel_L2=global_rel_L2,
                         mean_late_locL2=float(np.nanmean(locL2[li])), max_late_locL2=float(np.nanmax(locL2[li])),
                         mean_late_peak=float(np.nanmean(peak[li])), max_late_peak=float(np.nanmax(peak[li])),
                         mean_late_obs=float(np.nanmean(obs[li])), max_late_obs=float(np.nanmax(obs[li])),
                         min_late_local_count=float(np.min(lc[li])), median_late_local_count=float(np.median(lc[li])),
                         mean_late_Em=float(np.nanmean(Em[li])), max_late_Em=float(np.nanmax(Em[li]))))
        if s == steps:
            final.update(dict(u_field=u_field, locL2=locL2, peak=peak, obs=obs, lc=lc, Em=Em,
                              masses=masses, M_ref_islands=M_ref_islands, Nact=nact, ps=ps))

    if 0 in snap_steps:
        record(0)
    for s in range(1, steps + 1):
        t_mid = (s - 0.5) * tau
        key, kT, kr = jax.random.split(key, 3)
        dW = jax.random.normal(kT, shape=(bufsz if is_branch else N0_method, 2), dtype=jnp.float64)
        drift = b_particles(X, t_mid, centers_j, groups, cfg)
        if is_branch:
            X = wrap_torus(em_transport(X, drift, D, tau, dW), period)
            rX = r_staged_particles(X, t_mid, centers_j, amplitudes, groups, sigma, lam, beta, windows, delta)
            nu = jnp.where(mask, reaction_minvar(kr, rX, tau), 0)
            Xb, mb, ov, n_new = branch_compact(X, nu, bufsz)
            if ov:
                raise RuntimeError(f"{method} buffer overflow step {s}")
            X = jnp.asarray(Xb); mask = jnp.asarray(mb); ps += n_new
        else:
            X = wrap_torus(em_transport(X, drift, D, tau, dW), period)
            rX = r_staged_particles(X, t_mid, centers_j, amplitudes, groups, sigma, lam, beta, windows, delta)
            w = reaction_weighted(w, rX, tau); ps += N0_method
            if method in ("weighted_ess_resample", "weighted_ess_resample_costmatched"):
                wn = np.asarray(w); g_ness = float((wn.sum() ** 2) / (N0_method * np.sum(wn ** 2)))
                if g_ness < ess_threshold:
                    idx = systematic_resample(rng_resample, wn)
                    X = jnp.asarray(np.asarray(X)[idx])
                    w = jnp.full((N0_method,), wn.sum() / N0_method, dtype=jnp.float64); n_resamples += 1
        if s in snap_steps:
            record(s)
    return rows, final


def resolve_cfg():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    for k, t in [("kappa", float), ("D", float), ("lambda", float), ("beta", float),
                 ("N0", int), ("K", int), ("grid", int), ("T", float), ("t_comp", float),
                 ("out_dir", str)]:
        p.add_argument(f"--{k}", dest=("lam" if k == "lambda" else k), type=t)
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--config", type=str)
    args = p.parse_args()
    cfg = dict(DEFAULTS)
    if args.smoke:
        cfg.update(SMOKE)
    if args.config:
        cfg.update({k: v for k, v in json.load(open(args.config)).items() if k in cfg})
    for k in ["kappa", "D", "lam", "beta", "N0", "K", "grid", "T", "t_comp", "seeds", "out_dir"]:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    cfg["smoke"] = args.smoke
    return cfg


def main():
    cfg = resolve_cfg(); out_dir = cfg["out_dir"]; os.makedirs(out_dir, exist_ok=True)
    print("=== Compressive-readout multi-island (OPTIONAL §3-7) === backend:", jax.default_backend())
    print("config:", {k: cfg[k] for k in ["M", "G", "kappa", "sigma_b", "t_comp", "lam", "beta",
          "D", "T", "N0", "K", "grid", "compress_late_only", "R_W", "sigma_obs", "seeds"]}, flush=True)
    centers = make_grid_centers(cfg["M"]); amplitudes = staged_amplitudes(cfg["M"], cfg["amp_var"])
    centers_j = jnp.asarray(centers); groups = assign_groups(cfg["M"], cfg["G"])
    late_idx = np.where(groups == cfg["G"] - 1)[0]; sigma = cfg["sigma"]; eta = cfg["eta"]
    print(f"late islands {late_idx.tolist()}; compress groups {drift_groups(cfg)}; t_comp={cfg['t_comp']}")

    ref_u0, advance_ref, xs, XX, YY = reference_solver(cfg, centers, amplitudes, groups)
    masks = island_masks_on_grid(XX, YY, centers, sigma, eta)
    tau = cfg["tau"]; steps = int(round(cfg["T"] / tau))
    snap_steps = sorted(set(np.linspace(0, steps, cfg["n_snapshots"], dtype=int).tolist()))
    n = cfg["grid"]; cell_area = (L / n) ** 2
    t0 = time.time(); u = ref_u0.copy(); ref_snapshots = {0: u.copy()}
    for s in range(1, steps + 1):
        u = advance_ref(u, (s - 0.5) * tau)
        if s in snap_steps:
            ref_snapshots[s] = u.copy()
    M0 = float(np.sum(ref_u0) * cell_area)
    growth = float(np.sum(ref_snapshots[steps]) * cell_area) / M0
    umin = float(np.min(ref_snapshots[steps]))
    print(f"reference {time.time()-t0:.1f}s growth={growth:.1f}x umin={umin:.3f}", flush=True)
    if umin < -0.05 * np.max(ref_snapshots[steps]):
        print(f"[WARN] reference has notable negativity (umin={umin:.3f}); advection may be under-resolved", flush=True)
    buffer_size = max(int(np.ceil(cfg["buffer_safety"] * growth * cfg["N0"])), 2 * cfg["N0"])
    cfg["buffer_size"] = buffer_size
    density_estimation, _, density_evaluate_grid = generate_density_estimation(n_freq=cfg["K"], period=PERIOD)

    ts_rows = []; per_seed_rows = []; fields_store = {}; cost_info = {}
    methods = ["weighted", "weighted_ess_resample", "weighted_ess_resample_costmatched", "minvar_branch"]
    for seed in cfg["seeds"]:
        X_init = stratified_initial_particles(np.random.default_rng(seed), cfg["N0"], cfg, centers)
        t_seed = time.time(); seed_fields = {}
        ordered = ["minvar_branch"] + [m for m in methods if m != "minvar_branch"]
        N0_cm = None
        for method in ordered:
            if method == "weighted_ess_resample_costmatched":
                C = cost_info.get((seed, "minvar_branch")); N0_cm = int(round(C / steps)) if C else cfg["N0"]
                Xu = stratified_initial_particles(np.random.default_rng(50_000 + seed), N0_cm, cfg, centers)
                N0m = N0_cm
            else:
                N0m = cfg["N0"]; Xu = X_init
            rows, final = run_method(method, seed, cfg, centers_j, centers, amplitudes, groups,
                                     ref_snapshots, snap_steps, XX, YY, masks, late_idx, M0,
                                     density_estimation, density_evaluate_grid, Xu, N0m, buffer_size,
                                     cfg["ess_threshold"])
            ts_rows.extend(rows); last = rows[-1]; cost_info[(seed, method)] = final["ps"]
            per_seed_rows.append(dict(seed=seed, method=method, N0=N0m, Nact_final=final["Nact"],
                particle_steps=final["ps"], global_rel_L2=last["global_rel_L2"], global_nESS=last["global_nESS"],
                mean_late_locL2=last["mean_late_locL2"], max_late_locL2=last["max_late_locL2"],
                mean_late_peak=last["mean_late_peak"], max_late_peak=last["max_late_peak"],
                mean_late_obs=last["mean_late_obs"], max_late_obs=last["max_late_obs"],
                min_late_local_count=last["min_late_local_count"], mean_late_Em=last["mean_late_Em"]))
            seed_fields[method] = final["u_field"]
        fields_store[seed] = seed_fields
        print(f"seed {seed} done {time.time()-t_seed:.1f}s (N0_cm={N0_cm})", flush=True)

    np.savez(os.path.join(out_dir, "fields_ref.npz"), XX=XX, YY=YY, reference=ref_snapshots[steps],
             centers=centers, late_idx=late_idx)
    for s, fl in fields_store.items():
        np.savez(os.path.join(out_dir, f"fields_seed{s}.npz"), **fl)
    _write_csv(os.path.join(out_dir, "per_seed_metrics.csv"), per_seed_rows, list(per_seed_rows[0]))
    _write_csv(os.path.join(out_dir, "time_series.csv"), ts_rows, list(ts_rows[0]))
    # summary
    summ = []
    for method in methods:
        sub = [r for r in per_seed_rows if r["method"] == method]
        mean = lambda c: float(np.mean([r[c] for r in sub])); std = lambda c: float(np.std([r[c] for r in sub]))
        row = dict(method=method, n_seeds=len(sub), N0=int(np.mean([r["N0"] for r in sub])),
                   particle_steps=mean("particle_steps"), Nact_x=mean("Nact_final") / max(mean("N0"), 1),
                   global_rel_L2=mean("global_rel_L2"), global_nESS=mean("global_nESS"))
        for c in ["mean_late_locL2", "max_late_locL2", "mean_late_peak", "max_late_peak",
                  "mean_late_obs", "max_late_obs", "min_late_local_count", "mean_late_Em"]:
            row[c] = mean(c)
            if c in ("mean_late_locL2", "mean_late_peak", "mean_late_obs"):
                row[c + "_std"] = std(c)
        summ.append(row)
    _write_csv(os.path.join(out_dir, "metrics_summary.csv"), summ, list(summ[0]))
    cfg_out = dict(cfg); cfg_out["centers"] = centers.tolist(); cfg_out["late_idx"] = late_idx.tolist()
    json.dump(cfg_out, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    json.dump(dict(experiment="compressive_multi_island", git_commit=git_hash(False),
                   command_line=" ".join([sys.executable] + sys.argv), numpy_version=np.__version__,
                   datetime=datetime.datetime.now().isoformat(timespec="seconds"),
                   ref_growth=growth, ref_umin=umin, buffer_size=buffer_size,
                   reference="pseudo-spectral split-step (exact diffusion + RK2 advection-reaction)"),
              open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    print(f"\n--- summary ({summ[0]['n_seeds']} seeds): PRIMARY = late local L2 / peak / narrow-obs ---", flush=True)
    for r in summ:
        print(f"{r['method']:34s}: locL2 mean={r['mean_late_locL2']:.3f} max={r['max_late_locL2']:.3f} | "
              f"peak mean={r['mean_late_peak']:.3f} | obs mean={r['mean_late_obs']:.3f} | "
              f"minLateCnt={r['min_late_local_count']:.0f} | massEm(sanity)={r['mean_late_Em']:.3f} | "
              f"nESS={r['global_nESS']:.2f} Nact={r['Nact_x']:.1f}x ps={r['particle_steps']:.2e}", flush=True)
    print("wrote", out_dir, flush=True)


if __name__ == "__main__":
    main()
