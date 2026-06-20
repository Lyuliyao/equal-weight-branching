"""field3d_fourier.py -- 3D periodic Fourier reconstruction of the chemical field
v and its gradient, for the FULLY PARABOLIC-PARABOLIC Keller-Segel particle solver.

FULLY PARABOLIC: v is a dynamic field carried by the v-particle cloud, so the
physical chemical field is simply the v-cloud DENSITY,
    v(x) = M_v * rho_v(x),   rho_v = probability density of the v-cloud (int = 1).
There is NO screened-Poisson solve and NO kappa here (that belonged to the old
parabolic-elliptic experiment).  v_coeffs = M_v * (probability-density coeffs);
grad v assembles analytically (d/dx_j cos(k.x) = -k_j sin(k.x)).

The generic cos/sin coefficient + analytic-gradient routines are copied verbatim
(normalization checked) from focusing_3d/field3d_screened.py so this experiment is
self-contained and unit-tested in place.  Convention: modes n=0..K (so K_dyn is the
LARGEST retained integer mode), k_n = (2*pi/L) n; coeffs normalized so the
probability density integrates to 1 over the box [-L/2,L/2]^3.
"""
from functools import partial
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

KEYS = ("ccc", "ccs", "csc", "css", "scc", "scs", "ssc", "sss")


def _axis_weights(Kmodes):
    """Per-axis real-Fourier weights: 1 at mode 0, 2 for modes >= 1.
    Kmodes = K_dyn+1 = number of retained integer modes 0..K_dyn."""
    w = jnp.full((Kmodes,), 2.0)
    return w.at[0].set(1.0)


def density_coeffs(X, K, L):
    """Real Fourier coeffs of the empirical PROBABILITY density on [-L/2,L/2]^3.

    K = LARGEST retained integer mode (K_dyn); modes n=0..K (Kmodes=K+1).
    Returns 8 (Kmodes,Kmodes,Kmodes) cos/sin tensors, normalized so int rho = 1.
    """
    N = X.shape[0]
    Km = K + 1
    n = jnp.arange(Km)
    k = (2.0 * jnp.pi / L) * n
    ax = k[None, :] * X[:, 0:1]; ay = k[None, :] * X[:, 1:2]; az = k[None, :] * X[:, 2:3]
    cx, sx = jnp.cos(ax), jnp.sin(ax)
    cy, sy = jnp.cos(ay), jnp.sin(ay)
    cz, sz = jnp.cos(az), jnp.sin(az)
    bw = _axis_weights(Km)
    norm = bw[:, None, None] * bw[None, :, None] * bw[None, None, :] / (L ** 3)

    def coeff(bx, by, bz):
        return jnp.einsum('nk,nl,nm->klm', bx, by, bz) / N * norm

    return {"ccc": coeff(cx, cy, cz), "ccs": coeff(cx, cy, sz),
            "csc": coeff(cx, sy, cz), "css": coeff(cx, sy, sz),
            "scc": coeff(sx, cy, cz), "scs": coeff(sx, cy, sz),
            "ssc": coeff(sx, sy, cz), "sss": coeff(sx, sy, sz),
            "K": K, "L": L}


def _basis_and_deriv(X, K, L):
    """Per-axis cos/sin bases and x-derivatives at X (N,3); modes 0..K."""
    Km = K + 1
    k = (2.0 * jnp.pi / L) * jnp.arange(Km)
    C, S, dC, dS = {}, {}, {}, {}
    for j in range(3):
        a = k[None, :] * X[:, j:j + 1]
        c, s = jnp.cos(a), jnp.sin(a)
        C[j], S[j] = c, s
        dC[j] = -k[None, :] * s          # d/dx cos = -k sin
        dS[j] = k[None, :] * c           # d/dx sin = +k cos
    return C, S, dC, dS


def eval_field(X, coeff):
    """Evaluate the field with coefficient tensors `coeff` at X (N,3) -> (N,)."""
    K, L = coeff["K"], coeff["L"]
    C, S, _, _ = _basis_and_deriv(X, K, L)
    B = {"c": C, "s": S}
    return sum(jnp.einsum('nk,nl,nm,klm->n',
                          B[key[0]][0], B[key[1]][1], B[key[2]][2], coeff[key])
               for key in KEYS)


def grad_field(X, coeff):
    """Evaluate grad(field) at X (N,3) -> (N,3), analytic i*k assembly."""
    K, L = coeff["K"], coeff["L"]
    C, S, dC, dS = _basis_and_deriv(X, K, L)
    B = {"c": C, "s": S}; dB = {"c": dC, "s": dS}

    def comp(diff_axis):
        def contract(key):
            tabs = [(dB if ax == diff_axis else B)[key[ax]][ax] for ax in range(3)]
            return jnp.einsum('nk,nl,nm,klm->n', tabs[0], tabs[1], tabs[2], coeff[key])
        return sum(contract(key) for key in KEYS)

    return jnp.stack([comp(0), comp(1), comp(2)], axis=1)


# ---------------------------------------------------------------------------
# Fully-parabolic chemical field from the v-cloud: v = M_v * rho_v (NO screen).
# ---------------------------------------------------------------------------
def v_coeffs_from_cloud(Yv, K, L, mass_v):
    """Physical-v Fourier coeffs = mass_v * (probability-density coeffs of Y_v).
    No screened division, no DC zeroing (DC is the physical mean v=M_v/L^3 and is
    irrelevant to grad v)."""
    cp = density_coeffs(Yv, K, L)
    out = {"K": K, "L": L}
    for key in KEYS:
        out[key] = mass_v * cp[key]
    return out


def grad_v_from_cloud(X_eval, Yv, K, L, mass_v):
    """+grad_x v at X_eval (N,3) from the v-cloud Yv (Nv,3). Empty cloud -> 0."""
    if int(Yv.shape[0]) == 0:
        return jnp.zeros((X_eval.shape[0], 3))
    return grad_field(X_eval, v_coeffs_from_cloud(Yv, K, L, mass_v))


def v_field_from_cloud(X_eval, Yv, K, L, mass_v):
    """Evaluate the physical v field at X_eval (for output/diagnostics)."""
    if int(Yv.shape[0]) == 0:
        return jnp.zeros((X_eval.shape[0],))
    return eval_field(X_eval, v_coeffs_from_cloud(Yv, K, L, mass_v))


@partial(jax.jit, static_argnums=(3,))
def grad_v_buffer(X, Ybuf, mask, K, L, omega):
    """JITTED +grad v at X (Nu,3) from a FIXED-CAPACITY v-buffer Ybuf (Ncap,3) with
    active weights `mask` (1.0 active, 0.0 padding).  Identical math to
    grad_v_from_cloud over the active subset:  v = omega * sum_{active} delta, so
    v_coeffs = omega * norm * sum_i mask_i * basis_i  (the per-particle mass omega
    replaces the 1/N_v * mass_v = omega; no division by the active count).  Fixed
    shapes -> compiles once (unlike the eager dynamic-cloud path).
    """
    Km = K + 1
    k = (2.0 * jnp.pi / L) * jnp.arange(Km)
    ax = k[None, :] * Ybuf[:, 0:1]; ay = k[None, :] * Ybuf[:, 1:2]; az = k[None, :] * Ybuf[:, 2:3]
    cx, sx = jnp.cos(ax), jnp.sin(ax)
    cy, sy = jnp.cos(ay), jnp.sin(ay)
    cz, sz = jnp.cos(az), jnp.sin(az)
    bw = jnp.where(jnp.arange(Km) >= 1, 2.0, 1.0)
    norm = (bw[:, None, None] * bw[None, :, None] * bw[None, None, :]) / L ** 3 * omega
    w = mask

    def coeff(bx, by, bz):
        return jnp.einsum('n,nk,nl,nm->klm', w, bx, by, bz) * norm

    cf = {"ccc": coeff(cx, cy, cz), "ccs": coeff(cx, cy, sz),
          "csc": coeff(cx, sy, cz), "css": coeff(cx, sy, sz),
          "scc": coeff(sx, cy, cz), "scs": coeff(sx, cy, sz),
          "ssc": coeff(sx, sy, cz), "sss": coeff(sx, sy, sz), "K": K, "L": L}
    return grad_field(X, cf)


def empirical_mode(X, mass, K, L):
    """Complex continuum Fourier coeffs  hat_f_k = (1/L^3) int f e^{-i q_k.x} dx
    of the empirical field f = mass * rho (density of an equal-weight cloud), for
    integer modes k in {0..K}^3 (one octant; real field -> conjugate symmetry).
    hat_f_k = (mass / (N L^3)) sum_i exp(-i q_k . X_i).  Returns (Km,Km,Km) complex.
    Empty cloud (N=0, e.g. v0=0) -> all-zero coefficients (the field is identically 0).
    """
    N = int(X.shape[0]); Km = K + 1
    if N == 0:
        return jnp.zeros((Km, Km, Km), dtype=jnp.complex128)
    q = (2.0 * jnp.pi / L) * jnp.arange(Km)
    px = jnp.exp(-1j * q[None, :] * X[:, 0:1])     # (N,Km)
    py = jnp.exp(-1j * q[None, :] * X[:, 1:2])
    pz = jnp.exp(-1j * q[None, :] * X[:, 2:3])
    return jnp.einsum('nk,nl,nm->klm', px, py, pz) * (mass / (N * L ** 3))
