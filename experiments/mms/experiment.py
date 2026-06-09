"""
Experiment 2 -- Manufactured-solution (MMS) verification.
=========================================================

PDE on T^2 = [-pi,pi]^2 with constant advection b = (b1,b2):

    d_t u = -div(b u) + D Laplacian(u) + r(t,x) u

The exact positive solution is

    u_ex(t,x) = M(t) (1 + a cos x1 + b_ sin x2 + c cos(x1+x2)),   M(t)=exp(gamma t)

with small a,b_,c so u_ex > 0.  The reaction coefficient r(t,x) that makes
u_ex an exact solution is derived analytically (closed form) and verified by
finite differences in the unit test below.

Particle method (Poisson branching kernel, unbiased) is used.  We measure the
relative L2 error of the reconstructed density vs u_ex at the final time, in
three convergence studies:

    (a) errors_vs_N.csv   : vary N (expect slope ~ -1/2)
    (b) errors_vs_tau.csv : vary tau
    (c) errors_vs_K.csv   : vary K (Fourier modes)

Run:
    python experiment.py          # full config
    python experiment.py --smoke  # tiny smoke config
    python experiment.py --test   # run unit tests only (kernels + r FD check)
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
    reaction_poisson,
    reaction_minvar,
    reaction_weighted,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG = {
    "D": 0.05,
    "b": [0.5, 0.3],
    "gamma": 0.3,
    "a": 0.2,
    "b_": 0.15,
    "c": 0.1,
    "T": 0.5,
    "buffer_mult": 8,
    # study (a) errors vs N: fix tau, K
    "N_list": [2000, 4000, 8000, 16000, 32000, 64000],
    "N_tau": 2.5e-3,        # steps = 200
    "N_K": 8,
    # study (b) errors vs tau: fix N, K
    "tau_N": 64000,
    "tau_K": 8,
    "tau_list": [2.0e-2, 1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3],
    # study (c) errors vs K: fix N, small tau
    "K_N": 64000,
    "K_tau": 1.25e-3,
    "K_list": [2, 3, 4, 6, 8, 12],
    "grid": 256,            # evaluation grid for error
    "seeds": [0, 1, 2, 3],
    "results_dir": "results/mms",
    "smoke": False,
}

SMOKE_OVERRIDES = {
    "N_list": [2000, 4000],
    "N_tau": 5.0e-2,        # steps = 10
    "N_K": 6,
    "tau_N": 4000,
    "tau_K": 6,
    "tau_list": [1.0e-1, 5.0e-2],
    "K_N": 4000,
    "K_tau": 5.0e-2,
    "K_list": [4, 6],
    "seeds": [0, 1],
    "smoke": True,
}

PERIOD = [[-np.pi, np.pi], [-np.pi, np.pi]]
L = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Exact solution and analytic reaction coefficient
# ---------------------------------------------------------------------------
def u_ex_grid(t, XX, YY, cfg):
    a, b_, c = cfg["a"], cfg["b_"], cfg["c"]
    M = np.exp(cfg["gamma"] * t)
    P = 1.0 + a * np.cos(XX) + b_ * np.sin(YY) + c * np.cos(XX + YY)
    return M * P


def u_ex_points(t, X, cfg):
    a, b_, c = cfg["a"], cfg["b_"], cfg["c"]
    M = jnp.exp(cfg["gamma"] * t)
    x1, x2 = X[:, 0], X[:, 1]
    P = 1.0 + a * jnp.cos(x1) + b_ * jnp.sin(x2) + c * jnp.cos(x1 + x2)
    return M * P


def r_of(t, X, cfg):
    """Analytic reaction coefficient r(t,x) making u_ex an exact solution.

    With P = 1 + a cos x1 + b_ sin x2 + c cos(x1+x2):
       d_x1 P = -a sin x1 - c sin(x1+x2)
       d_x2 P =  b_ cos x2 - c sin(x1+x2)
       Lap P  = -a cos x1 - b_ sin x2 - 2 c cos(x1+x2)
    r = gamma + (b1 d_x1 P + b2 d_x2 P - D Lap P) / P     (advection conservative,
        constant b => div(b u) = b . grad u)
    """
    a, b_, c = cfg["a"], cfg["b_"], cfg["c"]
    b1, b2 = cfg["b"][0], cfg["b"][1]
    D = cfg["D"]
    x1, x2 = X[:, 0], X[:, 1]
    P = 1.0 + a * jnp.cos(x1) + b_ * jnp.sin(x2) + c * jnp.cos(x1 + x2)
    dP1 = -a * jnp.sin(x1) - c * jnp.sin(x1 + x2)
    dP2 = b_ * jnp.cos(x2) - c * jnp.sin(x1 + x2)
    lapP = -a * jnp.cos(x1) - b_ * jnp.sin(x2) - 2.0 * c * jnp.cos(x1 + x2)
    return cfg["gamma"] + (b1 * dP1 + b2 * dP2 - D * lapP) / P


def grid_coords(n):
    xs = -np.pi + (np.arange(n) + 0.5) * (L / n)
    XX, YY = np.meshgrid(xs, xs, indexing="xy")
    return xs, XX, YY


# ---------------------------------------------------------------------------
# Sample initial particles from u_ex(0,.) = P(x) (rejection sampling)
# ---------------------------------------------------------------------------
def sample_initial(key, N, cfg):
    a, b_, c = cfg["a"], cfg["b_"], cfg["c"]
    Pmax = 1.0 + a + b_ + c + 0.05   # safe upper bound on P
    out, got = [], 0
    k = key
    while got < N:
        k, k1, k2, k3 = jax.random.split(k, 4)
        batch = max(N, 4096)
        x = jax.random.uniform(k1, (batch,), minval=-np.pi, maxval=np.pi)
        y = jax.random.uniform(k2, (batch,), minval=-np.pi, maxval=np.pi)
        u = jax.random.uniform(k3, (batch,), minval=0.0, maxval=Pmax)
        P = 1.0 + a * jnp.cos(x) + b_ * jnp.sin(y) + c * jnp.cos(x + y)
        keep = u < P
        acc = np.asarray(jnp.stack([x, y], axis=1)[keep])
        out.append(acc)
        got += int(acc.shape[0])
    return jnp.asarray(np.concatenate(out, axis=0)[:N])


# initial measure mass = integral of P over torus = L^2 (cos/sin integrate to 0)
def M0_of(cfg):
    return L * L


# ---------------------------------------------------------------------------
# branching compaction
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
    return jnp.asarray(Xbuf), jnp.asarray(mask), overflow, n_new


# ---------------------------------------------------------------------------
# Single particle run -> final relative-L2 error
# ---------------------------------------------------------------------------
def run_one(seed, N, tau, K, cfg, density_estimation, density_evaluate_grid,
            XX, YY, kernel="poisson"):
    steps = int(round(cfg["T"] / tau))
    buffer_size = cfg["buffer_mult"] * N
    period = jnp.asarray(PERIOD)
    b = jnp.asarray(cfg["b"], dtype=jnp.float64)
    M0 = M0_of(cfg)
    n = cfg["grid"]
    cell_area = (L / n) ** 2

    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    X_init = sample_initial(k0, N, cfg)

    if kernel == "weighted":
        X = X_init
        w = jnp.ones((N,), dtype=jnp.float64)
        mask = jnp.ones((N,), dtype=bool)
        for s in range(steps):
            t = s * tau
            key, kw, kr = jax.random.split(key, 3)
            dW = jax.random.normal(kw, shape=(N, 2), dtype=jnp.float64)
            rr = r_of(t, X, cfg)
            # PDE has -div(b u); the Kolmogorov-forward (Fokker-Planck) drift is +b.
            drift = jnp.broadcast_to(b, X.shape)
            X = em_transport(X, drift, cfg["D"], tau, dW)
            X = wrap_torus(X, period)
            w = reaction_weighted(w, rr, tau)
        sum_w = float(jnp.sum(w))
        mass = (sum_w / N) * M0
        coeff = density_estimation(X, weights=w, mask=mask)
        n_act = N
    else:
        # branching
        Xb = np.zeros((buffer_size, 2), dtype=np.float64)
        Xb[:N] = np.asarray(X_init)
        mb = np.zeros((buffer_size,), dtype=bool)
        mb[:N] = True
        X, mask = jnp.asarray(Xb), jnp.asarray(mb)
        overflow = False
        kern_fn = reaction_poisson if kernel == "poisson" else reaction_minvar
        # All JAX ops act on the FIXED-SIZE buffer (masking inactive slots) so XLA
        # compiles the step once; dynamic slicing X[:nact] would recompile every
        # step and exhaust memory. Inactive slots are carried but get offspring 0.
        for s in range(steps):
            t = s * tau
            key, kw, kr = jax.random.split(key, 3)
            dW = jax.random.normal(kw, shape=(buffer_size, 2), dtype=jnp.float64)
            rr = r_of(t, X, cfg)
            # PDE has -div(b u); the Kolmogorov-forward (Fokker-Planck) drift is +b.
            drift = jnp.broadcast_to(b, X.shape)
            X = wrap_torus(em_transport(X, drift, cfg["D"], tau, dW), period)
            nu = jnp.where(mask, kern_fn(kr, rr, tau), 0)
            X, mask, ov, n_new = branch_compact(X, nu, buffer_size)
            if ov:
                raise RuntimeError(f"MMS buffer overflow at step {s} (N={N}); increase buffer_mult")
            overflow = overflow or ov
        n_act = int(jnp.sum(mask))
        mass = (n_act / N) * M0
        coeff = density_estimation(X, weights=jnp.ones((buffer_size,)), mask=mask)

    prob = np.asarray(density_evaluate_grid(jnp.asarray(XX), jnp.asarray(YY), coeff))
    u = prob * mass
    u_ref = u_ex_grid(cfg["T"], XX, YY, cfg)
    diff = u - u_ref
    L2 = np.sqrt(np.sum(diff ** 2) * cell_area)
    refL2 = np.sqrt(np.sum(u_ref ** 2) * cell_area)
    return L2 / refL2, n_act


# ---------------------------------------------------------------------------
# Convergence studies
# ---------------------------------------------------------------------------
def study(cfg, varname, values, fixed, density_cache, XX, YY):
    """Generic convergence study; returns list of dict rows."""
    rows = []
    for v in values:
        N = fixed.get("N", v if varname == "N" else None)
        tau = fixed.get("tau", v if varname == "tau" else None)
        K = fixed.get("K", v if varname == "K" else None)
        if varname == "N":
            N = int(v)
        elif varname == "tau":
            tau = float(v)
        elif varname == "K":
            K = int(v)
        if K not in density_cache:
            density_cache[K] = generate_density_estimation(n_freq=K, period=PERIOD)
        de, _, deg = density_cache[K]
        errs = []
        for seed in cfg["seeds"]:
            err, nact = run_one(seed, int(N), float(tau), int(K), cfg, de, deg,
                                XX, YY, kernel="poisson")
            errs.append(err)
        errs = np.array(errs)
        rows.append(dict(var=varname, value=v, N=N, tau=tau, K=K,
                         mean_L2_rel=float(np.mean(errs)),
                         std_L2_rel=float(np.std(errs)),
                         n_seeds=len(cfg["seeds"])))
        print(f"  {varname}={v}: mean L2_rel={np.mean(errs):.4e} std={np.std(errs):.2e}")
    return rows


def fit_slope(x, y):
    """Fit log y = slope log x + b ; return slope."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    sol, *_ = np.linalg.lstsq(A, ly, rcond=None)
    return float(sol[0])


def write_csv(path, rows, cols):
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
def unit_tests(cfg):
    print("=== UNIT TESTS ===")
    # 1) kernel sample-mean ~ exp(r tau)
    tau = 1e-2
    rng = jax.random.PRNGKey(123)
    ok_kernels = True
    for rval in [-2.0, -0.5, 0.0, 1.0, 3.0]:
        r = jnp.full((100000,), rval, dtype=jnp.float64)
        rng, k1, k2 = jax.random.split(rng, 3)
        nu_p = reaction_poisson(k1, r, tau)
        nu_m = reaction_minvar(k2, r, tau)
        m = float(np.exp(rval * tau))
        mp = float(jnp.mean(nu_p.astype(jnp.float64)))
        mm = float(jnp.mean(nu_m.astype(jnp.float64)))
        op = abs(mp - m) < 0.02 * max(1.0, m)
        om = abs(mm - m) < 0.02 * max(1.0, m)
        ok_kernels = ok_kernels and op and om
        print(f"  r={rval:+.1f} target m={m:.5f} | poisson {mp:.5f} ({'ok' if op else 'BAD'})"
              f" | minvar {mm:.5f} ({'ok' if om else 'BAD'})")
    # also weighted: deterministic exp
    w = reaction_weighted(jnp.ones((5,)), jnp.full((5,), 1.0), tau)
    print(f"  weighted exp check: {float(w[0]):.6f} vs {np.exp(1.0*tau):.6f}")

    # 2) MMS r(t,x): analytic vs finite-difference residual of the PDE
    # Verify that with the analytic r, (d_t u + div(b u) - D Lap u - r u) ~ 0 by FD.
    n = 200
    xs = -np.pi + (np.arange(n)) * (L / n)
    XX, YY = np.meshgrid(xs, xs, indexing="xy")
    h = L / n
    dt = 1e-4
    t0 = 0.137
    u_t = u_ex_grid(t0, XX, YY, cfg)
    u_tp = u_ex_grid(t0 + dt, XX, YY, cfg)
    u_tm = u_ex_grid(t0 - dt, XX, YY, cfg)
    dudt = (u_tp - u_tm) / (2 * dt)
    # periodic FD derivatives
    ux = (np.roll(u_t, -1, axis=1) - np.roll(u_t, 1, axis=1)) / (2 * h)   # d/dx1 (columns)
    uy = (np.roll(u_t, -1, axis=0) - np.roll(u_t, 1, axis=0)) / (2 * h)   # d/dx2 (rows)
    uxx = (np.roll(u_t, -1, axis=1) - 2 * u_t + np.roll(u_t, 1, axis=1)) / h ** 2
    uyy = (np.roll(u_t, -1, axis=0) - 2 * u_t + np.roll(u_t, 1, axis=0)) / h ** 2
    lap = uxx + uyy
    b1, b2 = cfg["b"]
    divbu = b1 * ux + b2 * uy   # constant b
    Xflat = jnp.asarray(np.stack([XX.ravel(), YY.ravel()], axis=1))
    r_an = np.asarray(r_of(t0, Xflat, cfg)).reshape(XX.shape)
    residual = dudt + divbu - cfg["D"] * lap - r_an * u_t
    max_res = float(np.max(np.abs(residual)))
    rel = max_res / float(np.max(np.abs(dudt) + 1e-12))
    ok_r = rel < 1e-3
    print(f"  MMS PDE residual (FD) max={max_res:.3e} rel={rel:.3e} ({'ok' if ok_r else 'BAD'})")
    print("=== UNIT TESTS", "PASSED" if (ok_kernels and ok_r) else "FAILED", "===")
    return ok_kernels and ok_r


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
    print("=== Experiment 2: MMS verification ===")
    print("backend:", jax.default_backend(), "| devices:", jax.devices())

    if "--test" in argv:
        unit_tests(cfg)
        return

    rd = cfg["results_dir"]
    os.makedirs(rd, exist_ok=True)
    with open(os.path.join(rd, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # always run unit tests first (cheap, important sanity)
    unit_tests(cfg)

    _, XX, YY = grid_coords(cfg["grid"]) if False else (None, *grid_coords(cfg["grid"])[1:])
    xs, XX, YY = grid_coords(cfg["grid"])
    density_cache = {}

    t_all = time.time()

    # study (a): errors vs N
    print("--- study (a): errors vs N ---")
    rows_N = study(cfg, "N", cfg["N_list"],
                   fixed={"tau": cfg["N_tau"], "K": cfg["N_K"]},
                   density_cache=density_cache, XX=XX, YY=YY)
    write_csv(os.path.join(rd, "errors_vs_N.csv"), rows_N,
              ["var", "value", "N", "tau", "K", "mean_L2_rel", "std_L2_rel", "n_seeds"])
    slope_N = fit_slope([r["N"] for r in rows_N], [r["mean_L2_rel"] for r in rows_N])

    # study (b): errors vs tau
    print("--- study (b): errors vs tau ---")
    rows_tau = study(cfg, "tau", cfg["tau_list"],
                     fixed={"N": cfg["tau_N"], "K": cfg["tau_K"]},
                     density_cache=density_cache, XX=XX, YY=YY)
    write_csv(os.path.join(rd, "errors_vs_tau.csv"), rows_tau,
              ["var", "value", "N", "tau", "K", "mean_L2_rel", "std_L2_rel", "n_seeds"])
    slope_tau = fit_slope([r["tau"] for r in rows_tau], [r["mean_L2_rel"] for r in rows_tau])

    # study (c): errors vs K
    print("--- study (c): errors vs K ---")
    rows_K = study(cfg, "K", cfg["K_list"],
                   fixed={"N": cfg["K_N"], "tau": cfg["K_tau"]},
                   density_cache=density_cache, XX=XX, YY=YY)
    write_csv(os.path.join(rd, "errors_vs_K.csv"), rows_K,
              ["var", "value", "N", "tau", "K", "mean_L2_rel", "std_L2_rel", "n_seeds"])

    print("=== FITTED SLOPES ===")
    print(f"  errors vs N : slope = {slope_N:.3f}  (expect ~ -0.5)")
    print(f"  errors vs tau: slope = {slope_tau:.3f}")
    print(f"TOTAL wallclock: {time.time() - t_all:.2f}s")


if __name__ == "__main__":
    main()
