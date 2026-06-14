"""
audit_fourier_kde.py -- Fourier-bandwidth & KDE reconstruction audit (CLAUDE.md S7).
====================================================================================

This is an APPENDIX ROBUSTNESS CHECK, not a new dynamics experiment.  The main
tables report the relative L2 error of the Fourier-reconstructed solver output
u_h^{N,K} = P_K mu_t^N.  A legitimate concern is whether the reported method
ordering is an artifact of a single Fourier bandwidth K (too large -> dominated by
Monte-Carlo coefficient noise; too small -> dominated by truncation bias).  This
script re-runs the EXACT production dynamics of S5.2 (localized growth) and S5.3
(switching growth) for a few seeds, captures the FINAL particle clouds, and:

  (1) Fourier K-sweep, for each K computes (CLAUDE.md S7.3)
        E_total(K)    = ||P_K mu^N - u_ref||      / ||u_ref||
        E_particle(K) = ||P_K mu^N - P_K u_ref||   / ||P_K u_ref||
        E_proj(K)     = ||P_K u_ref - u_ref||      / ||u_ref||
      P_K of the deterministic reference field uses the SAME folded cos/sin basis
      as the particle reconstruction (band-limited L2 projection), so E_particle
      isolates particle/representation error at a fixed reconstruction scale.

  (2) KDE h-sweep with a COMMON periodic Gaussian smoothing scale h (CLAUDE.md
      S7.4), via FFT (deposit -> FFT -> multiply by exp(-0.5 h^2 |k|^2) -> iFFT):
        E_KDE_rep(h) = ||u_h^N - u_{ref,h}|| / ||u_{ref,h}||
        E_bias(h)    = ||u_{ref,h} - u_ref|| / ||u_ref||
      The same h is used for every method; no per-method or data-driven bandwidth.

For S5.3 the global error and the local errors over B_A (old growth) and B_B (new
growth) are reported, to check that the Table-6 mechanism (ESS competitive in B_A,
branching better in B_B) is stable across K and h.

The replicated dynamics are validated against the archived production final fields
(reference_results/{branch_vs_weighted,switch}/fields_seed0.npz): the seed-0
reconstruction at the production bandwidth must match the archived field to
floating-point tolerance, proving the rerun is byte-faithful (same CRN, same keys).

Outputs (reference_results/reconstruction_audit/{localized_growth,switching_growth}/):
    fourier_k_sweep.csv   kde_h_sweep.csv   config_used.json   manifest.json
    snapshots/<exp>_seed0.npz   (seed-0 final clouds, for recomputation)
    plot_data/*.npz           (aggregated sweep curves for plotting)

Usage:
    python audit_fourier_kde.py                 # both experiments, seeds 0,1,2
    python audit_fourier_kde.py --exp localized --seeds 0
    python audit_fourier_kde.py --smoke         # tiny configs (validation only)
"""
import os
import sys
import json
import csv
import time
import argparse
import subprocess

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
BVW = os.path.join(HERE, "..", "branch_vs_weighted")
sys.path.insert(0, BVW)

from common_particle import (                       # noqa: E402
    generate_density_estimation, em_transport, wrap_torus,
    reaction_weighted, reaction_minvar, nESS, make_norm,
)
import experiment as e52                            # noqa: E402  (S5.2)
import experiment_switch as e53                     # noqa: E402  (S5.3)
from experiment import PERIOD, L, branch_compact    # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
REFDIR = os.path.join(REPO, "reference_results", "reconstruction_audit")

# Audit grids (CLAUDE.md S7.3 / S7.4)
K_LIST = {"localized": [8, 12, 16, 24], "switching": [32, 48, 64]}
H_LIST = {"localized": [0.10, 0.15, 0.20, 0.25], "switching": [0.06, 0.10, 0.15]}
K_LIST_SMOKE = {"localized": [8, 16], "switching": [16, 32]}
H_LIST_SMOKE = {"localized": [0.15, 0.25], "switching": [0.10, 0.15]}


# ===========================================================================
# Reconstruction / projection / smoothing primitives
# ===========================================================================
def recon_particles(X, w, K, mass):
    """P_K of the empirical measure mu = mass * sum_i w_i delta_{X_i} / sum_i w_i,
    on the grid, using the production folded cos/sin basis.  Returns (Ny,Nx)."""
    de, _, deg = generate_density_estimation(n_freq=K, period=PERIOD)
    coeff = de(jnp.asarray(X), weights=jnp.asarray(w), mask=None)
    prob = deg(jnp.asarray(_XX), jnp.asarray(_YY), coeff)
    return np.asarray(prob) * mass


def band_project(field, K):
    """L2 projection P_K of a grid density `field` (= reference field) onto the
    SAME band-limited folded cos/sin space used by recon_particles.  This is the
    band-limited reference P_K u_ref consistent with the particle P_K."""
    _, _, deg = generate_density_estimation(n_freq=K, period=PERIOD)
    n = field.shape[0]
    cell = (L / n) ** 2
    freq = np.arange(K)
    xv = _XX[0, :]
    yv = _YY[:, 0]
    tx = 2.0 * np.pi * freq[None, :] * (xv[:, None] + np.pi) / L   # (Nx,K)
    ty = 2.0 * np.pi * freq[None, :] * (yv[:, None] + np.pi) / L   # (Ny,K)
    cx, sx = np.cos(tx), np.sin(tx)
    cy, sy = np.cos(ty), np.sin(ty)
    F = field                                                       # (Ny,Nx)
    Ccc = np.einsum('yx,xk,yl->kl', F, cx, cy) * cell
    Ccs = np.einsum('yx,xk,yl->kl', F, cx, sy) * cell
    Scc = np.einsum('yx,xk,yl->kl', F, sx, cy) * cell
    Scs = np.einsum('yx,xk,yl->kl', F, sx, sy) * cell
    nf = np.asarray(make_norm(K)) / (L * L)
    coeff = {"cos-cos": jnp.asarray(nf * Ccc), "cos-sin": jnp.asarray(nf * Ccs),
             "sin-cos": jnp.asarray(nf * Scc), "sin-sin": jnp.asarray(nf * Scs)}
    return np.asarray(deg(jnp.asarray(_XX), jnp.asarray(_YY), coeff))


def cic_deposit(X, m, n):
    """Cloud-in-cell deposit of per-particle mass m onto the cell-centered periodic
    n x n grid.  Returns a mass field (Ny,Nx) with sum == sum(m)."""
    dx = L / n
    x0 = -np.pi + 0.5 * dx
    fx = (np.asarray(X)[:, 0] - x0) / dx
    fy = (np.asarray(X)[:, 1] - x0) / dx
    ix0 = np.floor(fx).astype(np.int64)
    iy0 = np.floor(fy).astype(np.int64)
    tx = fx - ix0
    ty = fy - iy0
    ix0 %= n; iy0 %= n
    ix1 = (ix0 + 1) % n; iy1 = (iy0 + 1) % n
    M = np.zeros((n, n), dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    np.add.at(M, (iy0, ix0), m * (1 - tx) * (1 - ty))
    np.add.at(M, (iy1, ix0), m * (1 - tx) * ty)
    np.add.at(M, (iy0, ix1), m * tx * (1 - ty))
    np.add.at(M, (iy1, ix1), m * tx * ty)
    return M


def fft_gauss(density, h, n):
    """Periodic Gaussian smoothing of a density field with unit-integral kernel
    eta_h (FT exp(-0.5 h^2 |k|^2)); mass-conserving (k=0 multiplier = 1)."""
    k = np.fft.fftfreq(n, d=(L / n)) * 2.0 * np.pi
    KX, KY = np.meshgrid(k, k, indexing="xy")
    G = np.exp(-0.5 * h * h * (KX ** 2 + KY ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(density) * G))


def l2(field, cell, mask=None):
    if mask is None:
        return float(np.sqrt(np.sum(field ** 2) * cell))
    return float(np.sqrt(np.sum(field[mask] ** 2) * cell))


# ===========================================================================
# Exact-CRN dynamics replicas (capture final clouds only)
# ===========================================================================
def run_localized(seed, cfg):
    """Byte-faithful replica of experiment.run_seed (S5.2) for weighted + minvar.
    Same key sequence: PRNGKey(seed) -> k_init; per step split(.,4) -> kT,kp_r,km_r."""
    global _XX, _YY
    ref_u0, advance_ref, xs, XX, YY, Gg = e52.reference_solver(cfg)
    _XX, _YY = XX, YY
    n = cfg["grid"]; cell = (L / n) ** 2
    tau = cfg["tau"]; steps = int(round(cfg["T"] / tau)); N0 = cfg["N0"]
    buffer_size = cfg["buffer_mult"] * N0
    period = jnp.asarray(PERIOD)

    key = jax.random.PRNGKey(seed)
    key, k_init = jax.random.split(key)
    X_init, _ = e52.sample_initial_particles(k_init, N0, cfg)
    M0 = float(np.sum(ref_u0) * cell)

    u_ref = ref_u0.copy()
    for _ in range(steps):
        u_ref = advance_ref(u_ref)

    Xw = X_init
    ww = jnp.ones((N0,), dtype=jnp.float64)
    Xb = np.zeros((buffer_size, 2)); Xb[:N0] = np.asarray(X_init)
    mb = np.zeros((buffer_size,), bool); mb[:N0] = True
    Xm = jnp.asarray(Xb); maskm = jnp.asarray(mb); ps_m = 0

    for s in range(1, steps + 1):
        key, kT, kp_r, km_r = jax.random.split(key, 4)        # kp_r unused (no poisson)
        dWbuf = jax.random.normal(kT, shape=(buffer_size, 2), dtype=jnp.float64)
        rW = e52.r_of(Xw, cfg)
        Xw = wrap_torus(em_transport(Xw, jnp.zeros_like(Xw), cfg["D"], tau, dWbuf[:N0]), period)
        ww = reaction_weighted(ww, rW, tau)
        rM = e52.r_of(Xm, cfg)
        Xm = wrap_torus(em_transport(Xm, jnp.zeros_like(Xm), cfg["D"], tau, dWbuf), period)
        nu_m = jnp.where(maskm, reaction_minvar(km_r, rM, tau), 0)
        Xmb, mmb, ov, n_new = branch_compact(Xm, nu_m, buffer_size)
        if ov:
            raise RuntimeError(f"minvar overflow at step {s}")
        Xm, maskm = jnp.asarray(Xmb), jnp.asarray(mmb)
        ps_m += n_new

    Xw_np = np.asarray(Xw); ww_np = np.asarray(ww)
    mk = np.asarray(maskm); nm = int(mk.sum()); Xm_np = np.asarray(Xm)[mk]
    Gw = np.asarray(e52.G_of(jnp.asarray(Xw_np), cfg))
    Gm = np.asarray(e52.G_of(jnp.asarray(Xm_np), cfg))
    eta = cfg["eta"]
    return dict(
        XX=XX, YY=YY, xs=xs, Gg=Gg, u_ref=u_ref, M0=M0, N0=N0, cell=cell, steps=steps, eta=eta,
        methods={
            "weighted": dict(X=Xw_np, w=ww_np, mass=(ww_np.sum() / N0) * M0, N_active=N0,
                             ps=N0 * steps, weighted=True, Gpart=Gw),
            "minvar": dict(X=Xm_np, w=np.ones(nm), mass=(nm / N0) * M0, N_active=nm,
                           ps=ps_m, weighted=False, Gpart=Gm),
        },
        regions={"global": None, "B": (Gg >= eta)})


def run_switching(seed, cfg):
    """Byte-faithful replica of experiment_switch.run_seed (S5.3) for weighted,
    weighted_ess, minvar.  weighted_always/poisson are skipped but the jax key
    split stays 4-way and rng_e (weighted_ess) is an independent numpy Generator,
    so the captured clouds are identical to production."""
    global _XX, _YY
    ref_u0, advance_ref, xs, XX, YY, GgA, GgB = e53.make_reference(cfg)
    _XX, _YY = XX, YY
    n = cfg["grid"]; cell = (L / n) ** 2
    tau = cfg["tau"]; T = cfg["T"]; steps = int(round(T / tau)); N0 = cfg["N0"]
    buffer_size = cfg["buffer_mult"] * N0
    eta = cfg["eta"]; sigma = cfg["sigma"]; cA, cB = cfg["cA"], cfg["cB"]
    period = jnp.asarray(PERIOD)

    key = jax.random.PRNGKey(seed)
    key, k_init = jax.random.split(key)
    X_init = jax.random.uniform(k_init, (N0, 2), minval=-np.pi, maxval=np.pi, dtype=jnp.float64)
    M0 = float(np.sum(ref_u0) * cell)
    rng_e = np.random.default_rng(cfg["resample_seed_offset"] + 1 + seed)

    u_ref = ref_u0.copy(); t = 0.0
    for _ in range(steps):
        u_ref = advance_ref(u_ref, t); t += tau

    Xw = X_init; ww = jnp.ones((N0,), dtype=jnp.float64)
    Xe = X_init; we = jnp.ones((N0,), dtype=jnp.float64); n_res_e = 0
    Xb = np.zeros((buffer_size, 2)); Xb[:N0] = np.asarray(X_init)
    mb = np.zeros((buffer_size,), bool); mb[:N0] = True
    Xm = jnp.asarray(Xb); maskm = mb; ps_m = 0

    for s in range(1, steps + 1):
        t_pre = (s - 1) * tau
        key, kT, kp_r, km_r = jax.random.split(key, 4)        # kp_r unused
        dWbuf = jax.random.normal(kT, shape=(buffer_size, 2), dtype=jnp.float64)
        # weighted
        rW = e53.r_of_xt(Xw, t_pre, cfg)
        Xw = wrap_torus(em_transport(Xw, jnp.zeros_like(Xw), cfg["D"], tau, dWbuf[:N0]), period)
        ww = reaction_weighted(ww, rW, tau)
        # weighted_ess
        rE = e53.r_of_xt(Xe, t_pre, cfg)
        Xe = wrap_torus(em_transport(Xe, jnp.zeros_like(Xe), cfg["D"], tau, dWbuf[:N0]), period)
        we = reaction_weighted(we, rE, tau)
        we_np = np.asarray(we)
        g_ness_e = float((we_np.sum() ** 2) / (N0 * np.sum(we_np ** 2)))
        if g_ness_e < cfg["ess_threshold"]:
            idx = e53.systematic_resample(rng_e, we_np)
            Xe = jnp.asarray(np.asarray(Xe)[idx])
            we = jnp.full((N0,), we_np.sum() / N0, dtype=jnp.float64)
            n_res_e += 1
        # minvar
        rM = e53.r_of_xt(Xm, t_pre, cfg)
        Xm = wrap_torus(em_transport(Xm, jnp.zeros_like(Xm), cfg["D"], tau, dWbuf), period)
        nu_m = jnp.where(jnp.asarray(maskm), reaction_minvar(km_r, rM, tau), 0)
        Xmb, mmb, ov, n_new = branch_compact(Xm, nu_m, buffer_size)
        if ov:
            raise RuntimeError(f"minvar overflow at step {s}")
        Xm, maskm = jnp.asarray(Xmb), mmb
        ps_m += n_new

    Xw_np = np.asarray(Xw); ww_np = np.asarray(ww)
    Xe_np = np.asarray(Xe); we_np = np.asarray(we)
    mk = np.asarray(maskm); nm = int(mk.sum()); Xm_np = np.asarray(Xm)[mk]

    def Gp(X, c):
        return np.asarray(e53.G_at(jnp.asarray(X), c, sigma))
    BA = (GgA >= eta); BB = (GgB >= eta)
    return dict(
        XX=XX, YY=YY, xs=xs, GgA=GgA, GgB=GgB, u_ref=u_ref, M0=M0, N0=N0, cell=cell,
        steps=steps, eta=eta, n_res_e=n_res_e,
        methods={
            "weighted": dict(X=Xw_np, w=ww_np, mass=(ww_np.sum() / N0) * M0, N_active=N0,
                             ps=N0 * steps, weighted=True,
                             GA=Gp(Xw_np, cA), GB=Gp(Xw_np, cB)),
            "weighted_ess": dict(X=Xe_np, w=we_np, mass=(we_np.sum() / N0) * M0, N_active=N0,
                                 ps=N0 * steps, weighted=True,
                                 GA=Gp(Xe_np, cA), GB=Gp(Xe_np, cB)),
            "minvar": dict(X=Xm_np, w=np.ones(nm), mass=(nm / N0) * M0, N_active=nm,
                           ps=ps_m, weighted=False,
                           GA=Gp(Xm_np, cA), GB=Gp(Xm_np, cB)),
        },
        regions={"global": None, "B_A": BA, "B_B": BB})


# ===========================================================================
# Sweep computation for one seed snapshot
# ===========================================================================
def local_diag(md, region_key, eta):
    """global_nESS and local nESS (weighted) or local count (branching) in a region."""
    if md["weighted"]:
        gness = float(nESS(jnp.asarray(md["w"])))
    else:
        gness = np.nan
    # which particle subset is "in" the region
    if region_key in ("global", "B"):
        if region_key == "B":
            Gp = md.get("Gpart")
            inR = (Gp >= eta) if Gp is not None else np.ones(md["w"].shape, bool)
        else:
            inR = np.ones(md["w"].shape, bool)
    elif region_key == "B_A":
        inR = (md["GA"] >= eta)
    elif region_key == "B_B":
        inR = (md["GB"] >= eta)
    else:
        inR = np.ones(md["w"].shape, bool)
    if md["weighted"]:
        wsub = md["w"][inR]
        loc = float(nESS(jnp.asarray(wsub))) if wsub.size else np.nan
    else:
        loc = float(np.sum(inR))
    return gness, loc


def sweep_seed(exp, seed, snap, K_list, h_list):
    """Return (fourier_rows, kde_rows) for one seed snapshot."""
    u_ref = snap["u_ref"]; cell = snap["cell"]; n = u_ref.shape[0]; eta = snap["eta"]
    regions = snap["regions"]
    frows, krows = [], []

    # ---- Fourier K-sweep ----
    proj_cache = {}
    for K in K_list:
        uref_K = band_project(u_ref, K)
        proj_cache[K] = uref_K
        for mname, md in snap["methods"].items():
            uPK = recon_particles(md["X"], md["w"], K, md["mass"])
            for rkey, rmask in regions.items():
                refn = l2(u_ref, cell, rmask)
                refKn = l2(uref_K, cell, rmask)
                E_total = l2(uPK - u_ref, cell, rmask) / refn if refn > 0 else np.nan
                E_part = l2(uPK - uref_K, cell, rmask) / refKn if refKn > 0 else np.nan
                E_proj = l2(uref_K - u_ref, cell, rmask) / refn if refn > 0 else np.nan
                gness, loc = local_diag(md, rkey, eta)
                frows.append(dict(experiment=exp, method=mname, seed=seed, K=K, h="",
                                  region=rkey, E_total=E_total, E_particle=E_part,
                                  E_proj=E_proj, E_KDE_rep="", E_bias="",
                                  global_nESS=gness, local_nESS_or_count=loc,
                                  N_active=md["N_active"], particle_steps=md["ps"]))

    # ---- KDE h-sweep ----
    for h in h_list:
        uref_h = fft_gauss(u_ref, h, n)
        for mname, md in snap["methods"].items():
            mpart = (md["mass"] / md["w"].sum()) * md["w"]   # per-particle physical mass
            massfield = cic_deposit(md["X"], mpart, n)
            rho = massfield / cell
            uh = fft_gauss(rho, h, n)
            for rkey, rmask in regions.items():
                refhn = l2(uref_h, cell, rmask)
                refn = l2(u_ref, cell, rmask)
                E_rep = l2(uh - uref_h, cell, rmask) / refhn if refhn > 0 else np.nan
                E_bias = l2(uref_h - u_ref, cell, rmask) / refn if refn > 0 else np.nan
                gness, loc = local_diag(md, rkey, eta)
                krows.append(dict(experiment=exp, method=mname, seed=seed, K="", h=h,
                                  region=rkey, E_total="", E_particle="", E_proj="",
                                  E_KDE_rep=E_rep, E_bias=E_bias,
                                  global_nESS=gness, local_nESS_or_count=loc,
                                  N_active=md["N_active"], particle_steps=md["ps"]))
    return frows, krows


# ===========================================================================
# Validation against archived production fields
# ===========================================================================
PROD = {"localized": dict(K=16, csv=("branch_vs_weighted", "metrics.csv"),
                          fields=None, methods=["weighted", "minvar"]),
        "switching": dict(K=48, csv=("switch", "metrics.csv"),
                          fields=("switch", "fields_seed0.npz"),
                          methods=["weighted", "weighted_ess", "minvar"])}


def _archived_final_L2(exp):
    """Per-seed final-time global L2_rel_err from the archived production metrics.csv."""
    sub, name = PROD[exp]["csv"]
    path = os.path.join(REPO, "reference_results", sub, name)
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    tmax = max(float(r["t"]) for r in rows)
    out = {}
    for r in rows:
        if abs(float(r["t"]) - tmax) < 1e-9:
            out.setdefault(r["method"], {})[int(r["seed"])] = float(r["L2_rel_err"])
    return out


def validate(exp, frows, seed0_snap, smoke):
    """Validate the replicated dynamics: per-seed E_total at the production K
    (global) must reproduce the archived production final L2_rel_err; for S5.3,
    also a seed-0 field-level match against fields_seed0.npz."""
    if smoke:
        return {"archived": False, "note": "smoke (config differs from production)"}
    Kp = PROD[exp]["K"]
    arch = _archived_final_L2(exp)
    out = {"archived": arch is not None, "K": Kp}
    if arch is not None:
        present = set(r["region"] for r in frows)
        gregion = "global" if "global" in present else "B"
        worst = 0.0
        per = {}
        for r in frows:
            if str(r["K"]) == str(Kp) and r["region"] == gregion:
                m = r["method"]; sd = int(r["seed"])
                if m in arch and sd in arch[m] and r["E_total"] not in ("", None):
                    diff = abs(float(r["E_total"]) - arch[m][sd])
                    per.setdefault(m, []).append(diff)
                    worst = max(worst, diff)
        out["L2_match_maxabs"] = worst
        out["L2_match_per_method"] = {m: max(v) for m, v in per.items()}
    # field-level seed-0 check where an archived field npz exists
    if PROD[exp]["fields"]:
        sub, name = PROD[exp]["fields"]
        fp = os.path.join(REPO, "reference_results", sub, name)
        if os.path.exists(fp) and seed0_snap is not None:
            d = np.load(fp); cell = seed0_snap["cell"]
            fld = {}
            for mname in PROD[exp]["methods"]:
                md = seed0_snap["methods"][mname]
                u = recon_particles(md["X"], md["w"], Kp, md["mass"])
                ua = np.asarray(d[mname])
                num = float(np.sqrt(np.sum((u - ua) ** 2) * cell))
                den = float(np.sqrt(np.sum(ua ** 2) * cell))
                fld[mname] = num / den if den > 0 else np.nan
            out["field_seed0_relL2"] = fld
    return out


# ===========================================================================
# Driver
# ===========================================================================
def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO).decode().strip()
    except Exception:
        return "unknown"


def run_experiment(exp, seeds, smoke):
    if exp == "localized":
        cfg = dict(e52.CONFIG)
        runner = run_localized
        outdir = os.path.join(REFDIR, "localized_growth")
    else:
        cfg = dict(e53.CONFIG)
        runner = run_switching
        outdir = os.path.join(REFDIR, "switching_growth")
    if smoke:
        if exp == "localized":
            cfg.update(dict(N0=2000, tau=0.05, grid=128, K=16))
        else:
            cfg.update(dict(N0=2000, tau=4e-3, grid=128, tau_ref=1e-3, K=32, buffer_mult=8))
    K_list = (K_LIST_SMOKE if smoke else K_LIST)[exp]
    h_list = (H_LIST_SMOKE if smoke else H_LIST)[exp]

    os.makedirs(os.path.join(outdir, "snapshots"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "plot_data"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "figures"), exist_ok=True)

    all_f, all_k = [], []
    seed0_snap = None
    t0 = time.time()
    for seed in seeds:
        ts = time.time()
        snap = runner(seed, cfg)
        if seed == 0:
            seed0_snap = snap
            # save seed-0 clouds for recomputation (whitelisted snapshots/*.npz).
            # Keep it lean: store the 1-D axis (meshgrid is rebuilt on load) and
            # cast reference/growth fields to float32 (CLAUDE.md: small .npz only).
            save = dict(u_ref=snap["u_ref"].astype(np.float32), xs=snap["xs"],
                        M0=snap["M0"], N0=snap["N0"], cell=snap["cell"],
                        eta=snap["eta"], K_list=np.array(K_list), h_list=np.array(h_list))
            for mname, md in snap["methods"].items():
                save[f"{mname}_X"] = md["X"].astype(np.float32)
                save[f"{mname}_w"] = md["w"].astype(np.float32)
                save[f"{mname}_mass"] = md["mass"]
            if exp == "switching":
                save["GgA"] = snap["GgA"].astype(np.float32)
                save["GgB"] = snap["GgB"].astype(np.float32)
            else:
                save["Gg"] = snap["Gg"].astype(np.float32)
            np.savez_compressed(os.path.join(outdir, "snapshots", f"{exp}_seed0.npz"), **save)
        frows, krows = sweep_seed(exp, seed, snap, K_list, h_list)
        all_f += frows; all_k += krows
        print(f"[{exp}] seed {seed} done in {time.time()-ts:.1f}s "
              f"(N_active minvar={snap['methods']['minvar']['N_active']})", flush=True)

    # ---- validate replicated dynamics against archived production numbers ----
    validations = validate(exp, all_f, seed0_snap, smoke)
    if validations.get("archived"):
        msg = f"[{exp}] validation: E_total(K={validations['K']}) vs archived metrics.csv "
        if "L2_match_maxabs" in validations:
            msg += f"max|diff|={validations['L2_match_maxabs']:.2e}"
        if "field_seed0_relL2" in validations:
            msg += " | seed0 field relL2 " + ", ".join(
                f"{k}={v:.2e}" for k, v in validations["field_seed0_relL2"].items())
        print(msg)
    else:
        print(f"[{exp}] validation skipped ({validations.get('note', 'no archived data')})")

    # ---- write CSVs ----
    schema = ["experiment", "method", "seed", "K", "h", "region", "E_total",
              "E_particle", "E_proj", "E_KDE_rep", "E_bias", "global_nESS",
              "local_nESS_or_count", "N_active", "particle_steps"]
    for rows, name in [(all_f, "fourier_k_sweep.csv"), (all_k, "kde_h_sweep.csv")]:
        with open(os.path.join(outdir, name), "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=schema)
            wtr.writeheader()
            for r in rows:
                wtr.writerow(r)

    # ---- aggregated plot_data (mean +/- std over seeds) ----
    save_plot_data(exp, outdir, all_f, all_k, K_list, h_list, seeds)

    # ---- config + manifest ----
    with open(os.path.join(outdir, "config_used.json"), "w") as f:
        json.dump({k: v for k, v in cfg.items()
                   if isinstance(v, (int, float, str, bool, list))}, f, indent=2)
    manifest = dict(
        experiment=exp, smoke=bool(smoke), seeds=list(seeds),
        K_list=K_list, h_list=h_list, git_hash=git_hash(),
        python=sys.version.split()[0], jax=jax.__version__, numpy=np.__version__,
        command="python " + " ".join([os.path.basename(__file__)] + sys.argv[1:]),
        validation_seed0=validations,
        wallclock_s=round(time.time() - t0, 1),
        note="Appendix reconstruction-bandwidth/KDE audit; reruns exact production "
             "dynamics (CLAUDE.md S7), no algorithm or parameter change.")
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[{exp}] wrote CSVs + plot_data + manifest to {outdir} "
          f"({time.time()-t0:.1f}s total)")
    return validations


def _agg(rows, key_field, key_vals, regions, methods, metric):
    """mean/std over seeds, shape [method][region][key] -> (mean,std)."""
    out = {}
    for m in methods:
        out[m] = {}
        for rg in regions:
            out[m][rg] = {}
            for kv in key_vals:
                vals = [float(r[metric]) for r in rows
                        if r["method"] == m and r["region"] == rg
                        and str(r[key_field]) == str(kv)
                        and r[metric] not in ("", None)
                        and not (isinstance(r[metric], float) and np.isnan(r[metric]))]
                if vals:
                    out[m][rg][kv] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def save_plot_data(exp, outdir, frows, krows, K_list, h_list, seeds):
    regions = sorted(set(r["region"] for r in frows))
    methods = sorted(set(r["method"] for r in frows))
    blob = dict(experiment=exp, K_list=np.array(K_list, float),
                h_list=np.array(h_list, float), regions=np.array(regions),
                methods=np.array(methods), n_seeds=len(seeds))
    for metric in ["E_total", "E_particle", "E_proj"]:
        agg = _agg(frows, "K", K_list, regions, methods, metric)
        for m in methods:
            for rg in regions:
                mean = np.array([agg[m][rg].get(k, (np.nan, np.nan))[0] for k in K_list])
                std = np.array([agg[m][rg].get(k, (np.nan, np.nan))[1] for k in K_list])
                blob[f"fourier_{metric}_{m}_{rg}_mean"] = mean
                blob[f"fourier_{metric}_{m}_{rg}_std"] = std
    for metric in ["E_KDE_rep", "E_bias"]:
        agg = _agg(krows, "h", h_list, regions, methods, metric)
        for m in methods:
            for rg in regions:
                mean = np.array([agg[m][rg].get(h, (np.nan, np.nan))[0] for h in h_list])
                std = np.array([agg[m][rg].get(h, (np.nan, np.nan))[1] for h in h_list])
                blob[f"kde_{metric}_{m}_{rg}_mean"] = mean
                blob[f"kde_{metric}_{m}_{rg}_std"] = std
    np.savez(os.path.join(outdir, "plot_data", f"{exp}_sweeps.npz"), **blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["localized", "switching", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    print("backend:", jax.default_backend(), "| devices:", jax.devices())
    exps = ["localized", "switching"] if args.exp == "both" else [args.exp]
    summary = {}
    for exp in exps:
        summary[exp] = run_experiment(exp, args.seeds, args.smoke)
    print("\n=== VALIDATION SUMMARY (replicated dynamics vs archived production) ===")
    for exp, v in summary.items():
        if v.get("archived"):
            line = f"  {exp}: E_total(K={v['K']}) vs metrics.csv max|diff|={v.get('L2_match_maxabs', float('nan')):.2e}"
            if "field_seed0_relL2" in v:
                line += "; seed0 field relL2=" + ", ".join(
                    f"{k}={vv:.2e}" for k, vv in v["field_seed0_relL2"].items())
            print(line)
        else:
            print(f"  {exp}: {v.get('note', 'no archived data')}")


if __name__ == "__main__":
    main()
