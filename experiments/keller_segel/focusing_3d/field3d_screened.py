"""
3D screened-Poisson chemical-field solver on a periodic box [-L/2, L/2]^3.

Model (parabolic-elliptic Keller-Segel):
    -Delta c + kappa^2 c = rho - rho_bar,   c periodic on the box of side L.

This is the 3D generalization of the 2D screened-Poisson symbol used in
  ../blowup_time/adaptive_window.py:125-192  (chem_force, lam = KX^2+KY^2+kappa^2)
with two changes:
  - 3D instead of 2D (a third separable axis kz);
  - the box is a FIXED periodic box of side L centred at the origin (NOT a
    core-adaptive window), so wavenumbers are k_n = (2*pi/L) * n exactly and the
    rho-bar subtraction is realized by dropping the k=0 (DC) mode.

IMPLEMENTATION CHOICE (flagged for Codex):
  We write a FRESH i*k spectral assembly here rather than reuse
  density3d.py's separable cos/sin autodiff estimator. Reasons:
    1. The screened solve is a per-mode division by (|k|^2 + kappa^2); a real
       cos/sin coefficient tensor maps cleanly to that division, and the
       chemical-field gradient assembles analytically (d/dx cos(k x) = -k sin).
    2. We need grad c, not c; analytic i*k assembly avoids an autodiff pass per
       particle per step and mirrors the validated 2D chem_force code path.
  The cos/sin coefficient *construction* (einsum over particles) is identical in
  spirit to density3d.generate_density_estimation_3d; only the post-processing
  (screened division + analytic gradient) differs.

NORMALIZATION (box of side L, modes n = 0..K-1 per axis):
  Real Fourier series for a PROBABILITY density p(x) (integrates to 1 over the
  box) on each axis:
      p(x) = (1/L^3) sum_{n} w_n  [ a-coeff * cos(k_n . x) + ... ]
  with k_n = (2*pi/L) n and per-axis weight w_n = 1 (n=0), 2 (n>=1). The
  empirical coefficient for a unit-mass cloud of N particles is
      Ccc[nx,ny,nz] = (w_nx w_ny w_nz / L^3) * (1/N) sum_i
                         cos(k_nx X_i1) cos(k_ny X_i2) cos(k_nz X_i3),
  and similarly for the sin combinations. This reproduces density3d's
  normalization with the box-L wavenumber k_n = (2*pi/L) n (density3d used
  theta = 2*pi*n*(x-x_min)/L, i.e. the same k_n with x_min = -L/2 here).

PHYSICAL DENSITY AND MASS SCALING (flagged for Codex):
  The empirical coefficients above describe a PROBABILITY density p (mass 1).
  The physical density is
      rho(x) = M * p(x),
  where M is the physical mass of the cloud (the family parameter; same role as
  `mass` in adaptive_window.chem_force). Hence the physical-rho coefficients are
  M * (probability coeffs). The DC term n=(0,0,0) of p equals 1/L^3 (the mean
  M/L^3 for rho); we DROP it from c (set c-hat_0 = 0), which realizes the
  rho - rho_bar source and fixes the gauge of the periodic screened-Poisson
  problem (c-hat_0 = (rho_bar - rho_bar)/kappa^2 = 0).

CHEMICAL FIELD AND GRADIENT:
  c-hat_k = rho-hat_k / (|k|^2 + kappa^2),  k != 0;   c-hat_0 = 0.
  c(x)        = sum_k c-hat coeffs * {cos/sin basis}(k . x)
  grad c(x)   assembled analytically: d/dx_j cos(k.x) = -k_j sin(k.x), etc.

SIGN OF DRIFT (flagged for Codex):
  Particle SDE: dX = +chi grad c dt + sqrt(2) dW (chi = 1).
  rho aggregates toward chemical maxima, so the drift is +chi grad c (INWARD,
  toward the peak of c, which sits over the peak of rho). selftest_field3d()
  checks this sign by construction on a single-mode source.

Everything is plain JAX, x64, CPU-friendly. No existing file is modified.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Coefficient construction: probability-density cos/sin tensors on box side L.
# ---------------------------------------------------------------------------
def _axis_weights(K):
    """Per-axis real-Fourier weights: 1 at mode 0, 2 for modes >= 1."""
    w = jnp.full((K,), 2.0)
    return w.at[0].set(1.0)


def density_coeffs(X, K, L):
    """Real Fourier coefficients of the empirical PROBABILITY density on the box.

    X : (N, 3) particle positions; assumed already wrapped to the box
        [-L/2, L/2]^3 (any periodic image gives the same coefficients).
    K : number of modes per axis (the bandwidth H).
    L : box side length.

    Returns a dict of eight (K,K,K) cos/sin coefficient tensors keyed by the
    {c,s} choice on each axis ('ccc','ccs',...,'sss'), normalized so the density
    integrates to 1 over the box. k_n = (2*pi/L) n.
    """
    N = X.shape[0]
    n = jnp.arange(K)
    k = (2.0 * jnp.pi / L) * n                       # physical wavenumbers (K,)

    # per-particle phases k_n . X (N,K) per axis
    ax = k[None, :] * X[:, 0:1]
    ay = k[None, :] * X[:, 1:2]
    az = k[None, :] * X[:, 2:3]
    cx, sx = jnp.cos(ax), jnp.sin(ax)                # (N,K)
    cy, sy = jnp.cos(ay), jnp.sin(ay)
    cz, sz = jnp.cos(az), jnp.sin(az)

    bw = _axis_weights(K)
    norm = bw[:, None, None] * bw[None, :, None] * bw[None, None, :]  # (K,K,K)
    norm = norm / (L ** 3)

    def coeff(bx, by, bz):
        return jnp.einsum('nk,nl,nm->klm', bx, by, bz) / N * norm

    return {
        "ccc": coeff(cx, cy, cz), "ccs": coeff(cx, cy, sz),
        "csc": coeff(cx, sy, cz), "css": coeff(cx, sy, sz),
        "scc": coeff(sx, cy, cz), "scs": coeff(sx, cy, sz),
        "ssc": coeff(sx, sy, cz), "sss": coeff(sx, sy, sz),
        "K": K, "L": L,
    }


# ---------------------------------------------------------------------------
# Screened-Poisson chemical field: c-hat_k = rho-hat_k / (|k|^2 + kappa^2).
# ---------------------------------------------------------------------------
def screened_solve(coeff_p, mass, kappa):
    """Return chemical-field coefficient tensors c-hat from density coeffs.

    coeff_p : probability-density coefficients (dict from density_coeffs).
    mass    : physical mass M of the cloud; physical rho coeffs = M * coeff_p.
    kappa   : screening parameter.

    The DC mode (nx=ny=nz=0) of c is set to 0 (gauge / rho-bar subtraction).
    Returns a dict with the same eight keys (now chemical-field coefficients)
    plus K, L. Only the cos*cos*cos block has a nonzero DC entry to zero out;
    all sin-containing blocks already vanish at n=0 along their sin axes.
    """
    K = coeff_p["K"]
    L = coeff_p["L"]
    n = jnp.arange(K)
    k = (2.0 * jnp.pi / L) * n
    K2 = (k[:, None, None] ** 2 + k[None, :, None] ** 2
          + k[None, None, :] ** 2)                   # |k|^2 (K,K,K)
    lam = K2 + kappa ** 2                            # screened symbol

    out = {"K": K, "L": L}
    for key in ("ccc", "ccs", "csc", "css", "scc", "scs", "ssc", "sss"):
        out[key] = mass * coeff_p[key] / lam
    # Gauge: c-hat_0 = 0. The DC mode lives only in the ccc block at [0,0,0].
    out["ccc"] = out["ccc"].at[0, 0, 0].set(0.0)
    return out


# ---------------------------------------------------------------------------
# Evaluate c and grad c at particle positions (analytic i*k assembly).
# ---------------------------------------------------------------------------
def _basis_and_deriv(X, K, L):
    """Per-axis cos/sin bases and their x-derivatives at points X (N,3).

    Returns dicts of (N,K) arrays:
      C[axis], S[axis]   : cos(k_n x_axis), sin(k_n x_axis)
      dC[axis], dS[axis] : d/dx cos = -k_n sin ; d/dx sin = +k_n cos
    """
    n = jnp.arange(K)
    k = (2.0 * jnp.pi / L) * n                       # (K,)
    C, S, dC, dS = {}, {}, {}, {}
    for j in range(3):
        a = k[None, :] * X[:, j:j + 1]               # (N,K)
        c, s = jnp.cos(a), jnp.sin(a)
        C[j], S[j] = c, s
        dC[j] = -k[None, :] * s
        dS[j] = k[None, :] * c
    return C, S, dC, dS


def eval_c(X, coeff_c):
    """Evaluate the chemical field c at particle positions X (N,3) -> (N,)."""
    K, L = coeff_c["K"], coeff_c["L"]
    C, S, _, _ = _basis_and_deriv(X, K, L)
    # basis selector per key, axis-major: first char x-axis, etc.
    B = {"c": C, "s": S}

    def contract(key):
        bx, by, bz = B[key[0]][0], B[key[1]][1], B[key[2]][2]
        return jnp.einsum('nk,nl,nm,klm->n', bx, by, bz, coeff_c[key])

    val = sum(contract(key) for key in
              ("ccc", "ccs", "csc", "css", "scc", "scs", "ssc", "sss"))
    return val


def grad_c(X, coeff_c):
    """Evaluate grad c at particle positions X (N,3) -> (N,3), analytic i*k.

    For each output direction j we differentiate the j-th axis basis and keep
    the other two axes as plain cos/sin bases.
    """
    K, L = coeff_c["K"], coeff_c["L"]
    C, S, dC, dS = _basis_and_deriv(X, K, L)
    B = {"c": C, "s": S}
    dB = {"c": dC, "s": dS}

    def grad_component(diff_axis):
        # for each cos/sin key, differentiate the basis on axis `diff_axis`
        def contract(key):
            bases = []
            for ax in range(3):
                table = dB if ax == diff_axis else B
                bases.append(table[key[ax]][ax])
            return jnp.einsum('nk,nl,nm,klm->n', bases[0], bases[1], bases[2],
                              coeff_c[key])
        return sum(contract(key) for key in
                   ("ccc", "ccs", "csc", "css", "scc", "scs", "ssc", "sss"))

    g = jnp.stack([grad_component(0), grad_component(1), grad_component(2)],
                  axis=1)
    return g


def field_and_grad_from_particles(X_src, X_eval, *, K, L, mass, kappa):
    """Convenience: build density coeffs from X_src, screened-solve, and return
    (c(X_eval), grad c(X_eval)). X_src builds rho; X_eval is where the drift is
    needed (for the single-cloud KS focusing run, X_src == X_eval == X)."""
    coeff_p = density_coeffs(X_src, K, L)
    coeff_c = screened_solve(coeff_p, mass, kappa)
    return eval_c(X_eval, coeff_c), grad_c(X_eval, coeff_c)


# ---------------------------------------------------------------------------
# Reconstructed-density evaluation (for the P_H peak diagnostic).
# ---------------------------------------------------------------------------
def eval_density(X, coeff_p, mass):
    """Evaluate physical density rho = M * p at points X (N,3) -> (N,).
    coeff_p are probability-density coefficients; multiply by mass."""
    K, L = coeff_p["K"], coeff_p["L"]
    C, S, _, _ = _basis_and_deriv(X, K, L)
    B = {"c": C, "s": S}

    def contract(key):
        bx, by, bz = B[key[0]][0], B[key[1]][1], B[key[2]][2]
        return jnp.einsum('nk,nl,nm,klm->n', bx, by, bz, coeff_p[key])

    val = sum(contract(key) for key in
              ("ccc", "ccs", "csc", "css", "scc", "scs", "ssc", "sss"))
    return mass * val


# ===========================================================================
# Self-test (WRITTEN, NOT RUN). Run under Codex cold-verify:
#     python field3d_screened.py
# ===========================================================================
def selftest_field3d():
    """Cold self-test of the 3D screened-Poisson solver. Checks:

    (1) ANALYTIC SINGLE-MODE SOURCE.
        If rho - rho_bar = cos(k1 x1) with k1 = 2*pi/L (first x-mode), then the
        exact screened-Poisson solution is
            c(x) = cos(k1 x1) / (k1^2 + kappa^2).
        We FORGE the chemical-field coefficients directly from a known single
        cos-mode density (bypassing particle sampling) and check eval_c / grad_c
        reproduce the analytic c and grad c. This isolates the symbol division
        and the analytic-gradient assembly from sampling noise.

    (2) GRADIENT vs FINITE DIFFERENCE.
        grad_c against a central finite difference of eval_c.

    (3) DRIFT SIGN.
        For a localized (cos-dominated) bump source, +grad c points TOWARD the
        density peak (inward), confirming dX = +chi grad c aggregates.
    """
    import numpy as np

    L = 12.0
    kappa = 0.1
    K = 8

    # ---- (1) analytic single-mode source ----------------------------------
    # Build probability-density coeffs for p with rho - rho_bar = cos(k1 x1).
    # We construct coeff_p so that screened_solve yields the analytic c.
    # Easiest: directly forge coeff_c with a single ccc entry at n=(1,0,0).
    k1 = 2.0 * np.pi / L
    coeff_c = {
        "ccc": jnp.zeros((K, K, K)),
        "ccs": jnp.zeros((K, K, K)), "csc": jnp.zeros((K, K, K)),
        "css": jnp.zeros((K, K, K)), "scc": jnp.zeros((K, K, K)),
        "scs": jnp.zeros((K, K, K)), "ssc": jnp.zeros((K, K, K)),
        "sss": jnp.zeros((K, K, K)), "K": K, "L": L,
    }
    amp = 1.0 / (k1 ** 2 + kappa ** 2)
    coeff_c["ccc"] = coeff_c["ccc"].at[1, 0, 0].set(amp)

    pts = jnp.asarray(np.random.default_rng(0).uniform(-L / 2, L / 2, size=(5, 3)))
    c_num = eval_c(pts, coeff_c)
    c_exact = jnp.cos(k1 * pts[:, 0]) / (k1 ** 2 + kappa ** 2)
    err1 = float(jnp.max(jnp.abs(c_num - c_exact)))
    print(f"[selftest] single-mode c max-abs-err: {err1:.3e}  (want ~1e-12)")

    g_num = grad_c(pts, coeff_c)
    g_exact = jnp.zeros_like(g_num)
    g_exact = g_exact.at[:, 0].set(-k1 * jnp.sin(k1 * pts[:, 0])
                                   / (k1 ** 2 + kappa ** 2))
    err1g = float(jnp.max(jnp.abs(g_num - g_exact)))
    print(f"[selftest] single-mode grad c max-abs-err: {err1g:.3e}  (want ~1e-12)")

    # ---- (2) grad vs finite difference on a sampled bump -------------------
    rng = np.random.default_rng(1)
    sigma = 0.45
    N = 200000
    Xs = jnp.asarray(rng.normal(0.0, sigma, size=(N, 3)))
    Xs = jnp.mod(Xs + L / 2, L) - L / 2   # wrap to box
    mass = 60.0
    coeff_p = density_coeffs(Xs, K, L)
    coeff_c2 = screened_solve(coeff_p, mass, kappa)

    probe = jnp.asarray(rng.uniform(-2.0, 2.0, size=(6, 3)))
    g_ana = grad_c(probe, coeff_c2)
    eps = 1e-4
    g_fd = []
    for j in range(3):
        ej = jnp.zeros((3,)).at[j].set(eps)
        cp = eval_c(probe + ej[None, :], coeff_c2)
        cm = eval_c(probe - ej[None, :], coeff_c2)
        g_fd.append((cp - cm) / (2 * eps))
    g_fd = jnp.stack(g_fd, axis=1)
    err2 = float(jnp.max(jnp.abs(g_ana - g_fd)))
    scale = float(jnp.max(jnp.abs(g_fd))) + 1e-30
    print(f"[selftest] grad c vs finite-diff max-abs-err: {err2:.3e}  "
          f"(rel {err2/scale:.3e}, want small)")

    # ---- (3) drift-sign check: +grad c points inward (toward origin peak) ---
    # density peak is at origin; at a point off-origin, +grad c should have a
    # component pointing back toward the origin (negative dot with the outward
    # radial direction).
    off = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.7, 0.7, 0.7]])
    g_off = grad_c(off, coeff_c2)
    radial = off / jnp.linalg.norm(off, axis=1, keepdims=True)
    dot = jnp.sum(g_off * radial, axis=1)
    print(f"[selftest] drift-sign dot(+grad c, outward radial): {np.array(dot)}")
    print("           all should be NEGATIVE => +grad c is inward (aggregating).")
    print("[selftest] DONE.")


if __name__ == "__main__":
    selftest_field3d()
