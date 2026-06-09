"""
Experiment 1 -- Branching vs weighted particles in localized growth.
====================================================================

LINEAR reaction-diffusion on the 2D torus T^2 = [-pi,pi]^2:

    d_t u = D Laplacian(u) + r(x) u ,    r(x) = lambda G(x) - beta ,
    G(x) = exp(-|x - x0|_T^2 / (2 sigma^2))      (periodic distance)

Three particle representations of the reaction term are compared under
IDENTICAL initial particles and IDENTICAL Brownian increments (per seed):

    (1) WEIGHTED       : positions diffuse only; w_i *= exp(r(X_i) tau).
    (2) POISSON        : equal-weight integer branching, unbiased.
    (3) MINVAR         : equal-weight minimum-variance integer branching.

A deterministic Fourier split-step solver on a 256x256 grid gives the
ground-truth reference u_ref(t,x).

Run:
    python experiment.py            # full config
    python experiment.py --smoke    # tiny smoke config
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

from common_particle import (
    generate_density_estimation,
    em_transport,
    wrap_torus,
    reaction_weighted,
    reaction_poisson,
    reaction_minvar,
    nESS,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG = {
    "D": 0.05,
    "lambda": 8.0,
    "beta": 1.0,
    "sigma": 0.5,
    "x0": [0.0, 0.0],
    "T": 1.0,
    "tau": 2e-3,            # steps = T/tau = 500
    "N0": 20000,            # initial particle count
    "buffer_mult": 16,      # branching buffer size = buffer_mult * N0 (mass grows ~7x at T=1)
    "K": 16,                # Fourier modes per direction for reconstruction
    "grid": 256,            # reference / evaluation grid size
    "ref_substeps": 1,      # reference split-step substeps per tau (>=1, finer if >1)
    "eta": 0.5,             # B = {x : G(x) >= eta}  (FIXED, predefined)
    "n_snapshots": 20,      # number of saved time snapshots over [0,T]
    "seeds": [0, 1, 2, 3, 4, 5, 6, 7],
    "results_dir": "results/branch_vs_weighted",
    "smoke": False,
}

SMOKE_OVERRIDES = {
    "N0": 2000,
    "tau": 0.05,            # steps = T/tau = 20
    "seeds": [0, 1],
    "n_snapshots": 10,
    "smoke": True,
}

PERIOD = [[-np.pi, np.pi], [-np.pi, np.pi]]
L = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Geometry helpers (periodic distance, growth bump, reaction)
# ---------------------------------------------------------------------------
def periodic_sq_dist(X, x0):
    """Squared periodic distance on [-pi,pi]^2 between rows of X and point x0."""
    dx = X[:, 0] - x0[0]
    dy = X[:, 1] - x0[1]
    dx = dx - L * jnp.round(dx / L)
    dy = dy - L * jnp.round(dy / L)
    return dx * dx + dy * dy


def G_of(X, cfg):
    x0 = jnp.asarray(cfg["x0"])
    d2 = periodic_sq_dist(X, x0)
    return jnp.exp(-d2 / (2.0 * cfg["sigma"] ** 2))


def r_of(X, cfg):
    return cfg["lambda"] * G_of(X, cfg) - cfg["beta"]


def grid_coords(n):
    """Cell-centered periodic grid on [-pi,pi]^2; returns (XX, YY) meshgrid xy."""
    xs = -np.pi + (np.arange(n) + 0.5) * (L / n)
    XX, YY = np.meshgrid(xs, xs, indexing="xy")  # XX varies along columns (x)
    return xs, XX, YY


def G_grid(XX, YY, cfg):
    x0 = cfg["x0"]
    dx = XX - x0[0]
    dy = YY - x0[1]
    dx = dx - L * np.round(dx / L)
    dy = dy - L * np.round(dy / L)
    return np.exp(-(dx * dx + dy * dy) / (2.0 * cfg["sigma"] ** 2))


# ---------------------------------------------------------------------------
# Reference: Fourier split-step on the grid
# ---------------------------------------------------------------------------
def reference_solver(cfg):
    n = cfg["grid"]
    xs, XX, YY = grid_coords(n)
    Gg = G_grid(XX, YY, cfg)
    rg = cfg["lambda"] * Gg - cfg["beta"]           # (n,n) reaction field
    u0 = 1.0 + 0.5 * Gg                              # initial condition

    # Fourier wavenumbers for [-pi,pi]: k = integer (period 2pi)
    k = np.fft.fftfreq(n, d=(L / n)) * 2.0 * np.pi   # = integers * 1
    KX, KY = np.meshgrid(k, k, indexing="xy")
    lap = -(KX ** 2 + KY ** 2)                       # Laplacian symbol

    tau = cfg["tau"]
    sub = cfg["ref_substeps"]
    dt = tau / sub
    diff_half = np.exp(cfg["D"] * lap * (dt / 2.0))  # half diffusion (Strang)
    react = np.exp(rg * dt)                          # full reaction

    def advance_one_tau(u):
        for _ in range(sub):
            uh = np.fft.fft2(u) * diff_half
            u = np.real(np.fft.ifft2(uh))
            u = u * react
            uh = np.fft.fft2(u) * diff_half
            u = np.real(np.fft.ifft2(uh))
        return u

    return u0, advance_one_tau, xs, XX, YY, Gg


# ---------------------------------------------------------------------------
# Initial particle sampling from u0 (rejection sampling on the torus)
# ---------------------------------------------------------------------------
def sample_initial_particles(key, N0, cfg):
    """Sample N0 particles from u0(x) = 1 + 0.5 G(x) on the torus via rejection.
    Returns (N0,2) positions and the total initial mass M0 = integral u0 dx."""
    umax = 1.5  # max of u0
    out = []
    got = 0
    k = key
    while got < N0:
        k, k1, k2, k3 = jax.random.split(k, 4)
        batch = max(N0, 4096)
        x = jax.random.uniform(k1, (batch,), minval=-np.pi, maxval=np.pi)
        y = jax.random.uniform(k2, (batch,), minval=-np.pi, maxval=np.pi)
        u = jax.random.uniform(k3, (batch,), minval=0.0, maxval=umax)
        XY = jnp.stack([x, y], axis=1)
        uval = 1.0 + 0.5 * G_of(XY, cfg)
        keep = u < uval
        acc = XY[keep]
        out.append(np.asarray(acc))
        got += int(acc.shape[0])
    pts = np.concatenate(out, axis=0)[:N0]
    # total mass of u0 over the torus
    M0 = (1.0 + 0.5 * (2.0 * np.pi * cfg["sigma"] ** 2)) if False else None
    # compute M0 analytically: int 1 dx = L^2 ; int 0.5 G dx = 0.5 * 2 pi sigma^2 (approx on torus)
    # On the torus the gaussian bump integrates to ~ 2 pi sigma^2 for small sigma.
    M0 = L * L + 0.5 * (2.0 * np.pi * cfg["sigma"] ** 2)
    return jnp.asarray(pts), M0


# ---------------------------------------------------------------------------
# Field reconstruction from particles (mass-scaled physical density)
# ---------------------------------------------------------------------------
def reconstruct_field(density_estimation, density_evaluate_grid, X, w, mask,
                      mass_scale, XX, YY):
    """Reconstruct u on the grid: P_K (probability density) * mass_scale.

    mass_scale converts the unit-mass probability density into the physical
    measure: mass_scale = (total measure mass).  Because P_K integrates to 1
    over the domain, u_grid integrates to mass_scale.
    """
    coeff = density_estimation(X, weights=w, mask=mask)
    prob = density_evaluate_grid(jnp.asarray(XX), jnp.asarray(YY), coeff)  # (Ny,Nx)
    return np.asarray(prob) * mass_scale


# ---------------------------------------------------------------------------
# Metric computation on the grid
# ---------------------------------------------------------------------------
def grid_metrics(u, u_ref, XX, YY, Gg, eta, cell_area):
    """Compute L1, L2-rel, peak height & loc errors, local mass over B."""
    diff = u - u_ref
    L1 = np.sum(np.abs(diff)) * cell_area
    L2 = np.sqrt(np.sum(diff ** 2) * cell_area)
    refL2 = np.sqrt(np.sum(u_ref ** 2) * cell_area)
    L2_rel = L2 / refL2 if refL2 > 0 else np.nan

    peak = np.max(u)
    peak_ref = np.max(u_ref)
    iy, ix = np.unravel_index(np.argmax(u), u.shape)
    iy_r, ix_r = np.unravel_index(np.argmax(u_ref), u_ref.shape)
    px, py = XX[iy, ix], YY[iy, ix]
    pxr, pyr = XX[iy_r, ix_r], YY[iy_r, ix_r]
    # periodic peak-location distance
    ddx = px - pxr; ddx -= L * np.round(ddx / L)
    ddy = py - pyr; ddy -= L * np.round(ddy / L)
    peak_loc_err = np.sqrt(ddx ** 2 + ddy ** 2)

    Bmask = (Gg >= eta)
    local_mass = np.sum(u[Bmask]) * cell_area
    local_mass_ref = np.sum(u_ref[Bmask]) * cell_area

    total_mass = np.sum(u) * cell_area
    return {
        "L1_err": L1,
        "L2_rel_err": L2_rel,
        "peak_height": peak,
        "peak_height_err": abs(peak - peak_ref),
        "peak_loc_err": peak_loc_err,
        "local_mass_B": local_mass,
        "local_mass_B_err": abs(local_mass - local_mass_ref),
        "total_mass": total_mass,
    }


# ---------------------------------------------------------------------------
# Branching step (host-side compaction into a fixed-size buffer)
# ---------------------------------------------------------------------------
def branch_compact(X_active, nu, buffer_size):
    """Replicate each active particle nu_i times into a fixed buffer.

    Returns (Xbuf, mask, overflow_flag). Compaction done with numpy.repeat.
    """
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
# Run all three methods for one seed
# ---------------------------------------------------------------------------
def run_seed(seed, cfg, ref_u0, advance_ref, XX, YY, Gg, density_estimation,
             density_evaluate_grid, records, fields_store):
    t0 = time.time()
    n = cfg["grid"]
    cell_area = (L / n) ** 2
    eta = cfg["eta"]
    tau = cfg["tau"]
    steps = int(round(cfg["T"] / tau))
    N0 = cfg["N0"]
    buffer_size = cfg["buffer_mult"] * N0
    snap_steps = sorted(set(np.linspace(0, steps, cfg["n_snapshots"], dtype=int).tolist()))

    key = jax.random.PRNGKey(seed)
    key, k_init = jax.random.split(key)
    X_init, _M0_approx = sample_initial_particles(k_init, N0, cfg)
    # Exact initial mass via grid quadrature of u0 (avoids the analytic torus approx).
    M0 = float(np.sum(ref_u0) * cell_area)

    # ----- reference field timeline -----
    u_ref = ref_u0.copy()
    ref_snapshots = {0: u_ref.copy()}
    for s in range(1, steps + 1):
        u_ref = advance_ref(u_ref)
        if s in snap_steps:
            ref_snapshots[s] = u_ref.copy()

    # ----- WEIGHTED -----
    Xw = X_init
    ww = jnp.ones((N0,), dtype=jnp.float64)
    maskw = jnp.ones((N0,), dtype=bool)

    # ----- branching buffers -----
    def init_buffer():
        Xb = np.zeros((buffer_size, 2), dtype=np.float64)
        Xb[:N0] = np.asarray(X_init)
        mb = np.zeros((buffer_size,), dtype=bool)
        mb[:N0] = True
        return jnp.asarray(Xb), jnp.asarray(mb)

    Xp, maskp = init_buffer()  # poisson
    Xm, maskm = init_buffer()  # minvar
    overflow_p = False
    overflow_m = False

    period = jnp.asarray(PERIOD)

    def snapshot(s, record=True):
        t = s * tau
        u_ref_s = ref_snapshots[s]
        # WEIGHTED reconstruction: mass = (sum_w / N0) * M0
        sum_w = float(jnp.sum(ww * maskw))
        mass_w = (sum_w / N0) * M0
        uw = reconstruct_field(density_estimation, density_evaluate_grid,
                               Xw, ww, maskw, mass_w, XX, YY)
        # branching: mass = (N_active / N0) * M0
        np_act = int(jnp.sum(maskp))
        mass_p = (np_act / N0) * M0
        up = reconstruct_field(density_estimation, density_evaluate_grid,
                               Xp, jnp.ones((buffer_size,)), maskp, mass_p, XX, YY)
        nm_act = int(jnp.sum(maskm))
        mass_m = (nm_act / N0) * M0
        um = reconstruct_field(density_estimation, density_evaluate_grid,
                               Xm, jnp.ones((buffer_size,)), maskm, mass_m, XX, YY)

        Bmask = (Gg >= eta)
        # weighted diagnostics
        w_act = np.asarray(ww)[np.asarray(maskw)]
        global_nESS_w = float(nESS(jnp.asarray(w_act)))
        max_over_mean = float(np.max(w_act) / np.mean(w_act))
        # local nESS over particles whose G >= eta
        Gpart = np.asarray(G_of(Xw, cfg))[np.asarray(maskw)]
        loc_idx = Gpart >= eta
        if np.sum(loc_idx) > 0:
            local_nESS_w = float(nESS(jnp.asarray(w_act[loc_idx])))
        else:
            local_nESS_w = np.nan

        # branching local counts
        Gp = np.asarray(G_of(Xp, cfg)) * np.asarray(maskp)
        n_local_p = int(np.sum((np.asarray(G_of(Xp, cfg)) >= eta) & np.asarray(maskp)))
        n_local_m = int(np.sum((np.asarray(G_of(Xm, cfg)) >= eta) & np.asarray(maskm)))

        for method, u, extra in [
            ("weighted", uw, dict(global_nESS=global_nESS_w, local_nESS_B=local_nESS_w,
                                   max_w_over_mean_w=max_over_mean, N_active=N0,
                                   N_local_B=np.nan)),
            ("poisson", up, dict(global_nESS=np.nan, local_nESS_B=np.nan,
                                 max_w_over_mean_w=np.nan, N_active=np_act,
                                 N_local_B=n_local_p)),
            ("minvar", um, dict(global_nESS=np.nan, local_nESS_B=np.nan,
                                max_w_over_mean_w=np.nan, N_active=nm_act,
                                N_local_B=n_local_m)),
        ]:
            m = grid_metrics(u, u_ref_s, XX, YY, Gg, eta, cell_area)
            rec = dict(seed=seed, method=method, t=t, runtime_s=np.nan)
            rec.update(m)
            rec.update(extra)
            if record:
                records.append(rec)

        return uw, up, um, u_ref_s

    # snapshot at t=0
    if 0 in snap_steps:
        snapshot(0)

    final_fields = None
    for s in range(1, steps + 1):
        key, kT, kp_r, km_r = jax.random.split(key, 4)
        # Common random numbers: ONE transport stream kT is shared by all three
        # methods. ALL JAX ops act on FIXED-SIZE arrays (N0 for weighted, the full
        # buffer for branching) so XLA compiles each step once -- dynamic slicing
        # X[:nact] would recompile every step and exhaust memory. Inactive buffer
        # slots are carried along but masked out (their offspring count is forced
        # to 0). Since normal(kT,(M,2))[:k]==normal(kT,(k,2)), the shared front
        # particles receive identical Brownian increments across methods.
        dWbuf = jax.random.normal(kT, shape=(buffer_size, 2), dtype=jnp.float64)

        # ---- WEIGHTED (fixed N0) ----
        rW = r_of(Xw, cfg)
        Xw = wrap_torus(em_transport(Xw, jnp.zeros_like(Xw), cfg["D"], tau, dWbuf[:N0]), period)
        ww = reaction_weighted(ww, rW, tau)

        # ---- POISSON branching (full buffer + mask) ----
        rP = r_of(Xp, cfg)
        Xp = wrap_torus(em_transport(Xp, jnp.zeros_like(Xp), cfg["D"], tau, dWbuf), period)
        nu_p = jnp.where(maskp, reaction_poisson(kp_r, rP, tau), 0)
        Xpb, mpb, ov_p, n_new_p = branch_compact(Xp, nu_p, buffer_size)
        if ov_p:
            raise RuntimeError(f"poisson buffer overflow at step {s} (n_new>{buffer_size}); increase buffer_mult")
        Xp, maskp = jnp.asarray(Xpb), jnp.asarray(mpb)

        # ---- MINVAR branching (full buffer + mask) ----
        rM = r_of(Xm, cfg)
        Xm = wrap_torus(em_transport(Xm, jnp.zeros_like(Xm), cfg["D"], tau, dWbuf), period)
        nu_m = jnp.where(maskm, reaction_minvar(km_r, rM, tau), 0)
        Xmb, mmb, ov_m, n_new_m = branch_compact(Xm, nu_m, buffer_size)
        if ov_m:
            raise RuntimeError(f"minvar buffer overflow at step {s} (n_new>{buffer_size}); increase buffer_mult")
        Xm, maskm = jnp.asarray(Xmb), jnp.asarray(mmb)

        if s in snap_steps:
            out = snapshot(s)              # records metrics once
            if s == steps:
                final_fields = out         # capture final fields here (no re-run)

    runtime = time.time() - t0
    # fill in runtime for this seed's records
    for rec in records:
        if rec["seed"] == seed and np.isnan(rec["runtime_s"]):
            rec["runtime_s"] = runtime

    # store final-time fields for plotting (captured during the loop -> no duplicate rows)
    if final_fields is None:
        final_fields = snapshot(steps, record=False)
    uw_f, up_f, um_f, uref_f = final_fields
    fields_store[seed] = dict(
        reference=uref_f, weighted=uw_f, poisson=up_f, minvar=um_f,
    )
    return overflow_p, overflow_m, runtime


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
    print("=== Experiment 1: branch vs weighted ===")
    print("backend:", jax.default_backend(), "| devices:", jax.devices())
    rd = cfg["results_dir"]
    os.makedirs(rd, exist_ok=True)
    with open(os.path.join(rd, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    density_estimation, _, density_evaluate_grid = generate_density_estimation(
        n_freq=cfg["K"], period=PERIOD)

    ref_u0, advance_ref, xs, XX, YY, Gg = reference_solver(cfg)

    records = []
    fields_store = {}
    overflow_any = False
    t_all = time.time()
    for seed in cfg["seeds"]:
        ov_p, ov_m, rt = run_seed(seed, cfg, ref_u0, advance_ref, XX, YY, Gg,
                                  density_estimation, density_evaluate_grid,
                                  records, fields_store)
        overflow_any = overflow_any or ov_p or ov_m
        print(f"seed {seed}: runtime {rt:.2f}s  overflow_poisson={ov_p} overflow_minvar={ov_m}")

    # write metrics CSV
    cols = ["seed", "method", "t", "total_mass", "L1_err", "L2_rel_err",
            "peak_height", "peak_height_err", "peak_loc_err",
            "local_mass_B", "local_mass_B_err",
            "global_nESS", "local_nESS_B", "max_w_over_mean_w",
            "N_active", "N_local_B", "runtime_s"]
    csv_path = os.path.join(rd, "metrics.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for rec in records:
            f.write(",".join(str(rec.get(c, "")) for c in cols) + "\n")
    print("wrote", csv_path, "rows:", len(records))

    # save fields for each seed
    for seed, fld in fields_store.items():
        np.savez(os.path.join(rd, f"fields_seed{seed}.npz"),
                 xs=xs, XX=XX, YY=YY, **fld)
    print("saved field npz for seeds:", list(fields_store.keys()))

    if overflow_any:
        print("WARNING: branching buffer overflow occurred. "
              "Increase buffer_mult or reduce lambda/T.")
    print(f"TOTAL wallclock: {time.time() - t_all:.2f}s")


if __name__ == "__main__":
    main()
