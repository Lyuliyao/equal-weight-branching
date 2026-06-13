"""
common_highdim.py
=================

Self-contained building blocks for the HIGH-DIMENSIONAL (4D / 6D) particle
experiment for the particle-method paper.

Benchmark: kinetic localized growth in phase space z = (x, v) on the torus
[-pi, pi]^d with d = d_x + d_v, d_x = d_v.

    d_t f + v . grad_x f = D_v Lap_v f + r[f](z) f ,
    r[f](z) = lambda * G_d(z) * (1 - alpha * m[f]) - beta ,
    m[f]    = integral G_d f dz ,
    G_d(z)  = prod_j (1 + cos z_j) / 2 .

The whole point of the method: the reaction needs ONLY the scalar moment m[f],
which is estimated as a grid-free Monte-Carlo average over active particles

    m_hat = (mass_factor / N0) * sum_{i active} w_i * G_d(Z_i)

with mass_factor = 1 (the equal-weight / equal-mass convention used here, so
that at t=0 with all N0 particles drawn from f0 and unit weights, the empirical
average reproduces integral G_d f0 dz).  NO dense grid, NO field solve in any
dimension.

Particle update per step tau:
    X <- X + V * tau                       (mod 2pi)
    V <- V + sqrt(2 D_v tau) * xi          (mod 2pi),  xi ~ N(0, I_{d_v})
then an equal-weight branching (or weighted) reaction with rate
    r_i = r[f](Z_i) = lambda * G_d(Z_i) * (1 - alpha * m_hat) - beta
using the single shared scalar m_hat.

This module reimplements the three reaction kernels (weighted, Poisson
branching, minimum-variance branching) self-contained -- it does NOT import
from any other experiment directory -- and provides a thin wrapper around the
vendored Functional-Hierarchical-Tensor (FHT) Fourier-sketching reconstruction
for the final-time diagnostic.

jax x64 is enabled by the importing script.
"""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Localized growth kernel  G_d(z) = prod_j (1 + cos z_j) / 2
# ---------------------------------------------------------------------------
def G_d(Z):
    """G_d(z) = prod_j (1 + cos z_j)/2 for rows of Z (shape (N, d)).

    Returns (N,) array in [0, 1], =1 at the origin, =0 when any z_j = +/- pi.
    """
    return jnp.prod((1.0 + jnp.cos(Z)) / 2.0, axis=1)


def G_d_np(Z):
    """NumPy version of G_d for host-side diagnostics."""
    return np.prod((1.0 + np.cos(Z)) / 2.0, axis=1)


# ---------------------------------------------------------------------------
# Grid-free scalar moment estimator  m_hat = integral G_d f dz  (Monte Carlo)
# ---------------------------------------------------------------------------
def moment_estimate(Z, w, mask, N0):
    """Grid-free Monte-Carlo estimate of m[f] = integral G_d f dz.

    Z    : (N, d) particle phase-space positions (fixed-size buffer ok)
    w    : (N,)   per-particle weights (1 for branching)
    mask : (N,)   boolean active flags
    N0   : initial particle count (sets the per-particle mass = 1/N0 * domain
           mass; with f0 sampled as a probability density times M0, the unit
           per-particle contribution G_d is averaged with weight w_i / N0).

    m_hat = (1/N0) * sum_i mask_i w_i G_d(Z_i).

    With f normalized so that integral f dz = N_active/N0 (branching) or
    sum_w/N0 (weighted), this is the correct MC estimator of integral G_d f dz.
    """
    g = G_d(Z)
    wm = w * mask.astype(w.dtype)
    return jnp.sum(wm * g) / N0


# ---------------------------------------------------------------------------
# Reaction rate r_i using the shared scalar moment m_hat (logistic factor)
# ---------------------------------------------------------------------------
def reaction_rate(Z, m_hat, lam, alpha, beta):
    """r_i = lambda * G_d(Z_i) * (1 - alpha * m_hat) - beta."""
    return lam * G_d(Z) * (1.0 - alpha * m_hat) - beta


# ---------------------------------------------------------------------------
# Phase-space Euler-Maruyama step with shared transport/noise increments
# ---------------------------------------------------------------------------
def phase_step(Z, d_x, d_v, D_v, tau, xi):
    """One kinetic transport + velocity-diffusion step on the torus.

    Z   : (N, d) phase-space positions, columns [x (d_x) | v (d_v)]
    xi  : (N, d_v) standard-normal velocity-noise increments (shared)
    Returns wrapped Z after  X += V tau ;  V += sqrt(2 D_v tau) xi.
    """
    X = Z[:, :d_x]
    V = Z[:, d_x:]
    X_new = X + V * tau
    V_new = V + jnp.sqrt(2.0 * D_v * tau) * xi
    Z_new = jnp.concatenate([X_new, V_new], axis=1)
    return wrap_torus(Z_new)


def wrap_torus(Z):
    """Wrap phase-space coordinates back into [-pi, pi]^d."""
    return jnp.mod(Z + np.pi, TWO_PI) - np.pi


def wrap_torus_np(Z):
    return np.mod(Z + np.pi, TWO_PI) - np.pi


# ---------------------------------------------------------------------------
# Reaction kernels  (E[multiplier] = exp(r tau) = m in every case)
# ---------------------------------------------------------------------------
def reaction_weighted(w, r, tau):
    """Weighted representation: w_i *= exp(r_i tau)."""
    return w * jnp.exp(r * tau)


def reaction_poisson(key, r, tau):
    """Poisson (unbiased) integer branching, E[nu] = exp(r tau) = m.

    m >= 1 (r >= 0):  nu = 1 + Poisson(m - 1)   (always >= 1)
    m <  1 (r <  0):  nu = Bernoulli(m)          (0 or 1)
    """
    m = jnp.exp(r * tau)
    k_pois, k_bern = jax.random.split(key)
    lam = jnp.clip(m - 1.0, min=0.0)
    pois = jax.random.poisson(k_pois, lam, shape=r.shape)
    grow = 1 + pois
    u = jax.random.uniform(k_bern, shape=r.shape)
    bern = (u < jnp.clip(m, min=0.0, max=1.0)).astype(jnp.int32)
    nu = jnp.where(m >= 1.0, grow, bern)
    return nu.astype(jnp.int32)


def reaction_minvar(key, r, tau):
    """Minimum-variance integer branching, E[nu] = m, Var = theta(1-theta).

    nu = floor(m) + Bernoulli(theta),  theta = m - floor(m).
    """
    m = jnp.exp(r * tau)
    fl = jnp.floor(m)
    theta = m - fl
    u = jax.random.uniform(key, shape=r.shape)
    nu = fl + (u < theta).astype(jnp.float64)
    return nu.astype(jnp.int32)


# ---------------------------------------------------------------------------
# Branching compaction into a fixed-size buffer (host side)
# ---------------------------------------------------------------------------
def branch_compact(Z_active, nu, buffer_size, d):
    """Replicate each active particle nu_i times into a fixed (buffer_size, d) buffer.

    Returns (Zbuf, mask, overflow_flag, n_new).
    """
    Z_np = np.asarray(Z_active)
    nu_np = np.asarray(nu).astype(np.int64)
    # Check the total BEFORE allocating, so a large burst cannot over-allocate.
    n_total = int(nu_np.sum())
    if n_total > buffer_size:
        # Do not allocate; the caller raises on the overflow flag.
        return (np.zeros((buffer_size, d), dtype=np.float64),
                np.zeros((buffer_size,), dtype=bool), True, n_total)
    children = np.repeat(Z_np, nu_np, axis=0)
    n_new = children.shape[0]
    overflow = False
    Zbuf = np.zeros((buffer_size, d), dtype=np.float64)
    Zbuf[:n_new] = children
    mask = np.zeros((buffer_size,), dtype=bool)
    mask[:n_new] = True
    return Zbuf, mask, overflow, n_new


# ---------------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------------
def nESS(w):
    """Normalized ESS  (sum w)^2 / (n sum w^2) over the given weights."""
    w = jnp.asarray(w, dtype=jnp.float64)
    n = w.shape[0]
    s1 = jnp.sum(w)
    s2 = jnp.sum(w * w)
    return jnp.where(s2 > 0, (s1 * s1) / (n * s2), jnp.nan)


# ---------------------------------------------------------------------------
# Initial sampling: f0(z) = M0 * prod_j N_wrapped(z_j; 0, sigma0)  (product)
# We sample directly from a wrapped product distribution so the moment and FHT
# diagnostics are well defined.  Returns particle positions and the analytic
# initial mass M0 (here we use a product of mild bumps; total mass M0 = 1 by
# choosing f0 a probability density, so initial m_hat ~ integral G_d f0).
# ---------------------------------------------------------------------------
def sample_initial(key, N0, d, sigma0):
    """Sample N0 particles i.i.d. from a wrapped-normal product on [-pi,pi]^d.

    f0 is a PROBABILITY density (mass 1); initial weights are 1 and total mass
    M0 = 1.  Returns (N0, d) positions (wrapped to [-pi,pi]).
    """
    z = jax.random.normal(key, shape=(N0, d), dtype=jnp.float64) * sigma0
    return wrap_torus(z)


# ===========================================================================
# FHT / TT low-rank reconstruction wrapper (final-time diagnostic)
# ===========================================================================
# We reuse the vendored allen_cahn/case2 FHT Fourier-sketching reconstruction
# (functional_hierarchical_tensor_{fourier,sketch}.py + fht_utils.py), patched
# only so ghost dims (d=6 -> pad 8) are marginalized correctly.
#
# The reconstruction takes the final particle cloud (rescaled to [-1,1]^d via
# y = z / pi) and returns a low-rank FHT density object plus extracted 1D/2D
# marginals and the diagonal profile f(s,s,...,s).  Dense K^d reconstruction is
# NEVER attempted -- the FHT low-rank model is the only full-density object.
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _fht_levels(L, r_val=6, s_val=5):
    """Build rank (r) and sketch-size (s) dicts for the FHT, constant per level."""
    deg = None  # filled by caller; placeholder
    return r_val, s_val


def build_fht(y, d_true, deg=8, r_val=6, s_val=5, verbose=False):
    """Build a low-rank FHT Fourier model from samples y in [-1,1]^{d_true}.

    Pads d_true up to dpad = 2**L with ghost dims if needed (e.g. d_true=6 ->
    dpad=8, L=3).  Returns the FunctionalHierarchicalTensorFourier object and a
    metadata dict (L, dpad, ghost_pt, deg).
    """
    from functional_hierarchical_tensor_fourier import FunctionalHierarchicalTensorFourier
    from functional_hierarchical_tensor_sketch import hier_tensor_sketch

    # choose L so that 2**L >= d_true (smallest power of two)
    L = int(np.ceil(np.log2(d_true)))
    dpad = 2 ** L
    ghost_pt = list(range(d_true + 1, dpad + 1))

    N = y.shape[0]
    if dpad > d_true:
        ypad = np.zeros((N, dpad), dtype=np.float64)
        ypad[:, :d_true] = np.asarray(y)
    else:
        ypad = np.asarray(y)

    # per-level rank / sketch-size dicts (constant), mirroring sketching.py
    r = dict()
    s = dict()
    r_level = r_val + 0 * np.arange(L, 0, -1)
    s_level = s_val + 0 * np.arange(L, 0, -1)
    for l in reversed(range(0, L + 1)):
        for k in range(1, 2 ** l + 1):
            if l == L:
                r[(k, l)] = [2 * deg + 1, r_level[L - 1]]
                s[(k, l)] = [2 * deg + 1, r_level[L - 1] + s_level[L - 1]]
            elif l == 0:
                r[(k, l)] = [r_level[0], r_level[0]]
                s[(k, l)] = [r_level[0] + s_level[0], r_level[0] + s_level[0]]
            else:
                r[(k, l)] = [r_level[l - 1], r_level[l], r_level[l]]
                s[(k, l)] = [r_level[l - 1] + s_level[l - 1],
                             r_level[l] + s_level[l],
                             r_level[l] + s_level[l]]

    c = hier_tensor_sketch(ypad, L, dpad, deg, r=r, s=s, debug=False)
    htn = FunctionalHierarchicalTensorFourier(d=d_true, L=L, c=c, deg=deg,
                                              ghost_pt=ghost_pt)
    meta = dict(L=L, dpad=dpad, ghost_pt=ghost_pt, deg=deg, d_true=d_true)
    if verbose:
        print(f"[FHT] d_true={d_true} L={L} dpad={dpad} ghost={ghost_pt} deg={deg}")
    return htn, meta


def _all_leaves(L):
    return list(range(1, 2 ** L + 1))


def fht_marginal_1d(htn, meta, coord, grid_y):
    """1D marginal density of coordinate `coord` (0-indexed real dim) over grid_y
    (points in [-1,1]).  Marginalizes every other real leaf and all ghost leaves.

    Returns the (normalized) marginal in z = pi*y coordinates (integrates to 1).
    """
    L = meta["L"]
    d_true = meta["d_true"]
    leaf_keep = coord + 1  # leaf k is 1-indexed; real dim j -> leaf j+1
    mask = [k for k in _all_leaves(L) if k != leaf_keep]  # ghosts auto-handled
    xq = np.zeros((grid_y.shape[0], d_true), dtype=np.float64)
    xq[:, coord] = grid_y
    m = np.asarray(htn.evaluate_marginal(xq, mask))
    # The low-rank FHT reconstruction can have small negative lobes; clip to a
    # nonnegative density before normalizing so the displayed marginal is a valid
    # density (integrates to 1 over z = pi*y).
    zz = np.pi * grid_y
    m = np.clip(m, 0.0, None)
    area = np.trapezoid(m, zz)
    if area > 0:
        m = m / area
    return m


def fht_marginal_2d(htn, meta, c0, c1, grid_y):
    """2D marginal density of coords (c0, c1) on the grid_y x grid_y mesh
    (in [-1,1]).  Marginalizes all other real + ghost leaves.

    Returns (G, ZZ0, ZZ1) where G has shape (n, n), normalized to integrate to 1
    over z-coordinates.
    """
    L = meta["L"]
    d_true = meta["d_true"]
    keep = {c0 + 1, c1 + 1}
    mask = [k for k in _all_leaves(L) if k not in keep]
    n = grid_y.shape[0]
    Y0, Y1 = np.meshgrid(grid_y, grid_y, indexing="ij")
    flat0 = Y0.ravel()
    flat1 = Y1.ravel()
    xq = np.zeros((flat0.shape[0], d_true), dtype=np.float64)
    xq[:, c0] = flat0
    xq[:, c1] = flat1
    vals = np.asarray(htn.evaluate_marginal(xq, mask)).reshape(n, n)
    vals = np.clip(vals, 0.0, None)
    zz = np.pi * grid_y
    # normalize
    dz = zz[1] - zz[0]
    area = np.sum(vals) * dz * dz
    if area > 0:
        vals = vals / area
    ZZ0, ZZ1 = np.meshgrid(zz, zz, indexing="ij")
    return vals, ZZ0, ZZ1


def fht_diagonal(htn, meta, grid_y):
    """Diagonal profile f(s, s, ..., s) for s = pi*grid_y, normalized so that
    max = 1 is NOT enforced; returns the raw low-rank value (ghosts marginalized).
    """
    L = meta["L"]
    d_true = meta["d_true"]
    ghost_leaves = [k for k in _all_leaves(L) if k in meta["ghost_pt"]]
    xd = np.repeat(grid_y[:, None], d_true, axis=1)
    if ghost_leaves:
        diag = np.asarray(htn.evaluate_marginal(xd, ghost_leaves))
    else:
        diag = np.asarray(htn.evaluate(xd))
    return diag


# ---------------------------------------------------------------------------
# FALLBACK diagnostic: empirical product-Fourier coefficient tensor (rank-1 sum)
# ---------------------------------------------------------------------------
def empirical_fourier_coeffs(Z, w, mask, n_modes=8):
    """Per-coordinate empirical Fourier coefficients (the rank-one-sum diagnostic).

    For each dimension j and mode k in 0..n_modes, returns the weighted means
        a_jk = <cos(k z_j)>,  b_jk = <sin(k z_j)>
    over active particles.  This is a cheap, always-available low-rank-style
    Fourier summary used as the documented fallback if full FHT integration is
    unavailable.  Shapes: a, b each (d, n_modes+1).
    """
    Z = np.asarray(Z)
    w = np.asarray(w) * np.asarray(mask).astype(np.float64)
    den = np.sum(w)
    d = Z.shape[1]
    ks = np.arange(n_modes + 1)
    a = np.zeros((d, n_modes + 1))
    b = np.zeros((d, n_modes + 1))
    for j in range(d):
        theta = np.outer(Z[:, j], ks)  # (N, n_modes+1)
        a[j] = (w @ np.cos(theta)) / den
        b[j] = (w @ np.sin(theta)) / den
    return a, b
