"""
common_kinetic.py
=================

Building blocks for the field-coupled 6D kinetic Keller-Segel particle method.

Phase space z = (x, v), x in T^3 = [-pi,pi]^3 (period L = 2pi), v in R^3.
d = 6, d_x = d_v = 3.

Kinetic PDE (corrected plan section 5.3):

    d_t f + v . grad_x f
        = gamma_v grad_v . ((v - chi grad_x c) f) + D_v Lap_v f + r[rho,c](x) f ,

    - Lap_x c + kappa^2 c = rho - rho_bar ,   rho(t,x) = integral f dv ,
    r[rho,c](x) = lambda_g S_c(c(x)) - alpha_rho S_rho(rho(x)) - beta ,
    S_c(c)   = (1/2)(1 + tanh((c - c0)/delta_c)) ,
    S_rho(r) = r / (r + rho0) .

Particle SDE:
    dX = V dt ,
    dV = -gamma_v (V - chi grad_x c(t,X)) dt + sqrt(2 D_v) dW .

The velocity OU drift toward chi grad_x c is integrated EXACTLY over a step tau
(see ou_velocity_step); X advances with the OLD V (X_{n+1} = X_n + V_n tau).

The three reaction kernels (weighted, Poisson branching, minimum-variance
branching) and the compaction / ESS helpers are imported UNCHANGED from the
validated common_highdim module; we do NOT re-implement them here.

The chemical field c and spatial marginal rho are evaluated SPECTRALLY at the
particle x-positions via field_kinetic (a small Fourier sketch, no dense grid).

jax x64 is enabled by the importing script.
"""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp

TWO_PI = 2.0 * np.pi

# ---------------------------------------------------------------------------
# Reuse the VALIDATED high-dimensional infrastructure unchanged.
# We import the three reaction kernels, the compaction, nESS, and wrap.
# common_highdim.py is VENDORED into this experiment directory (the repo
# convention is that every experiment directory is self-contained, with no
# cross-directory imports), so it is a plain local import -- no sys.path
# manipulation needed.
# ---------------------------------------------------------------------------
from common_highdim import (
    reaction_weighted,
    reaction_poisson,
    reaction_minvar,
    branch_compact,
    nESS,
    wrap_torus,
    wrap_torus_np,
)

from field_kinetic import (
    build_half_spectrum,
    density_coeffs,
    eval_field,
    eval_rho,
    VOL3,
)


# ---------------------------------------------------------------------------
# Reaction saturation factors and rate r[rho,c](x)
# ---------------------------------------------------------------------------
def S_c(c, c0, delta_c):
    """Chemo-activation: S_c(c) = (1/2)(1 + tanh((c - c0)/delta_c)) in (0,1)."""
    return 0.5 * (1.0 + jnp.tanh((c - c0) / delta_c))


def S_rho(rho, rho0):
    """Crowding saturation using the positive part of the density:

        S_rho(rho) = rho_+/(rho_+ + rho0),   rho_+ = max(rho, 0).

    The reconstructed spatial marginal is a truncated-Fourier proxy that can
    have small negative lobes; using the positive part keeps S_rho in [0,1)
    and avoids the singularity at rho = -rho0.  This positive-part form is the
    model as implemented (state it as such in the manuscript)."""
    rp = jnp.clip(rho, min=0.0)
    return rp / (rp + rho0)


def reaction_rate_field(c, rho, lam_g, alpha_rho, beta, c0, delta_c, rho0):
    """Field-coupled reaction rate per particle:

        r = lambda_g S_c(c) - alpha_rho S_rho(rho) - beta .

    c, rho are the chemical / spatial-marginal values at each particle.
    """
    return (lam_g * S_c(c, c0, delta_c)
            - alpha_rho * S_rho(rho, rho0)
            - beta)


# ---------------------------------------------------------------------------
# Velocity update: EXACT Ornstein-Uhlenbeck integrator for the linear drift
#   dV = -gamma_v (V - mu) dt + sqrt(2 D_v) dW,   mu = chi grad_x c(X_n)
# Exact solution over a step tau (mu frozen at X_n):
#   V_{n+1} = mu + e^{-gamma_v tau} (V_n - mu)
#             + sqrt( (D_v/gamma_v)(1 - e^{-2 gamma_v tau}) ) xi ,   xi ~ N(0,I).
# Position uses the OLD velocity (symplectic-Euler / kick-drift convention here):
#   X_{n+1} = X_n + V_n tau  (mod 2pi).
# ---------------------------------------------------------------------------
def ou_velocity_step(X, V, grad_c, gamma_v, chi, D_v, tau, xi):
    """One exact-OU velocity + transport step on the phase-space torus.

    X      : (N, 3) spatial positions (will be wrapped).
    V      : (N, 3) velocities (R^3, NOT wrapped -- velocity is unbounded).
    grad_c : (N, 3) chemical gradient at X (from the spectral field solve).
    xi     : (N, 3) standard-normal velocity-noise increments (shared CRN).
    Returns (X_new, V_new).  X is wrapped to [-pi,pi]^3; V is left in R^3.

    NOTE: only the SPATIAL coordinate lives on the torus.  Velocity is a real
    Euclidean variable with a stationary distribution N(mu, (D_v/gamma_v) I)
    (temperature T_v = D_v/gamma_v), so it must NOT be wrapped.
    """
    mu = chi * grad_c                                   # drift center
    e1 = jnp.exp(-gamma_v * tau)
    sig = jnp.sqrt((D_v / gamma_v) * (1.0 - jnp.exp(-2.0 * gamma_v * tau)))
    X_new = wrap_torus_x(X + V * tau)                   # drift with OLD V
    V_new = mu + e1 * (V - mu) + sig * xi               # exact OU
    return X_new, V_new


def euler_velocity_step(X, V, grad_c, gamma_v, chi, D_v, tau, xi):
    """Euler-Maruyama velocity + transport step (cross-check alternative to OU).

        V_{n+1} = V_n - gamma_v (V_n - chi grad_c) tau + sqrt(2 D_v tau) xi ,
        X_{n+1} = X_n + V_n tau (mod 2pi).
    """
    mu = chi * grad_c
    X_new = wrap_torus_x(X + V * tau)
    V_new = V - gamma_v * (V - mu) * tau + jnp.sqrt(2.0 * D_v * tau) * xi
    return X_new, V_new


def wrap_torus_x(X):
    """Wrap ONLY spatial coordinates back into [-pi,pi]^3 (velocity untouched)."""
    return jnp.mod(X + np.pi, TWO_PI) - np.pi


def wrap_torus_x_np(X):
    return np.mod(X + np.pi, TWO_PI) - np.pi


# ---------------------------------------------------------------------------
# Initial conditions
#   single  : one Gaussian blob in x at origin (wrapped), v ~ N(0, T_v I_3).
#   four    : 4 clusters in x, v ~ N(0, T_v I_3).
# Returns (Z (N0,6) with columns [x(3)|v(3)], where x wrapped, v in R^3).
# ---------------------------------------------------------------------------
def sample_initial_single(key, N0, sigma_x, T_v):
    """Single Gaussian x-blob at origin (wrapped) + Maxwellian velocities."""
    kx, kv = jax.random.split(key)
    X = jax.random.normal(kx, shape=(N0, 3), dtype=jnp.float64) * sigma_x
    X = wrap_torus_x(X)
    V = jax.random.normal(kv, shape=(N0, 3), dtype=jnp.float64) * jnp.sqrt(T_v)
    return jnp.concatenate([X, V], axis=1)


def sample_initial_four(key, N0, sigma_x, T_v):
    """Four x-clusters (centers inside [-pi,pi]^3) + Maxwellian velocities.

    Centers (within [-pi,pi]^3, well-separated):
        (+1, 0, 0), (-1, 0, 0), (0, +1, +0.5), (0, -1, -0.5).
    Particles are split as evenly as possible across the four clusters.
    """
    centers = jnp.asarray([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.5],
        [0.0, -1.0, -0.5],
    ], dtype=jnp.float64)
    kx, kv, kc = jax.random.split(key, 3)
    # assign each particle to a cluster (roughly even)
    cluster = jax.random.randint(kc, shape=(N0,), minval=0, maxval=4)
    base = centers[cluster]                              # (N0,3)
    X = base + jax.random.normal(kx, shape=(N0, 3), dtype=jnp.float64) * sigma_x
    X = wrap_torus_x(X)
    V = jax.random.normal(kv, shape=(N0, 3), dtype=jnp.float64) * jnp.sqrt(T_v)
    return jnp.concatenate([X, V], axis=1)


def sample_initial(key, N0, sigma_x, T_v, kind="single"):
    if kind == "four":
        return sample_initial_four(key, N0, sigma_x, T_v)
    return sample_initial_single(key, N0, sigma_x, T_v)


# ---------------------------------------------------------------------------
# Torus-aware spatial displacement and mass centroid (reconstruction-free)
# ---------------------------------------------------------------------------
def torus_disp(X, center):
    """Minimal-image displacement X - center on [-pi,pi]^3 (per coordinate)."""
    d = X - center
    return np.mod(d + np.pi, TWO_PI) - np.pi


def mass_centroid_x(X, w):
    """Weighted circular mean of x on the torus (per coordinate), in [-pi,pi].

    Uses the standard circular-mean estimator atan2(<sin>, <cos>) so that the
    centroid is well-defined under periodicity.  X (N,3), w (N,).
    """
    W = np.sum(w)
    xc = np.zeros(3)
    for j in range(3):
        cb = np.sum(w * np.cos(X[:, j])) / W
        sb = np.sum(w * np.sin(X[:, j])) / W
        xc[j] = np.arctan2(sb, cb)
    return xc


def quantile_core_radii(X, w, xc, qs=(0.5, 0.9)):
    """Quantile radii R_q of the spatial cloud about centroid xc (torus-aware).

    Returns a tuple of radii, one per q in qs: R_q is the q-quantile of the
    per-particle torus distance |X_i - xc| weighted by w_i.  Reconstruction-free
    core-size diagnostic (R_0.5 = median radius, R_0.9 = 90% radius).
    """
    disp = torus_disp(X, xc)                             # (N,3)
    r = np.sqrt(np.sum(disp ** 2, axis=1))              # (N,)
    order = np.argsort(r)
    r_sorted = r[order]
    w_sorted = w[order]
    cw = np.cumsum(w_sorted)
    if cw[-1] <= 0:
        return tuple(np.nan for _ in qs)
    cw = cw / cw[-1]
    out = []
    for q in qs:
        idx = np.searchsorted(cw, q)
        idx = min(idx, len(r_sorted) - 1)
        out.append(float(r_sorted[idx]))
    return tuple(out)


# ---------------------------------------------------------------------------
# Local phase-space region B(t) = { |x - x_c| <= r_x  AND  |v - v_c| <= r_v }
# ---------------------------------------------------------------------------
def local_region_mask(X, V, xc, vc, r_x, r_v):
    """Boolean mask of particles inside the local phase-space ball B(t).

    x-distance is torus-aware; v-distance is Euclidean (velocity in R^3).
    """
    dx = torus_disp(X, xc)
    rx = np.sqrt(np.sum(dx ** 2, axis=1))
    dv = V - vc
    rv = np.sqrt(np.sum(dv ** 2, axis=1))
    return (rx <= r_x) & (rv <= r_v)


# ---------------------------------------------------------------------------
# Correlation and histogram diagnostics for reaction-coupling evidence
# ---------------------------------------------------------------------------
def safe_corr(a, b):
    """Pearson correlation, returns nan if either side is (near) constant."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return np.nan
    sa = a.std()
    sb = b.std()
    if sa < 1e-14 or sb < 1e-14:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def weighted_corr(a, b, w):
    """Mass-weighted Pearson correlation of a,b with particle weights w.

    For branching methods all weights are equal, so this reduces to the
    unweighted Pearson correlation; for the weighted method it correctly
    measures the correlation of the represented measure (not the bare cloud).
    Returns nan if either weighted std is (near) zero.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if a.size < 2:
        return np.nan
    W = np.sum(w)
    if W <= 0:
        return np.nan
    ma = np.sum(w * a) / W
    mb = np.sum(w * b) / W
    da = a - ma
    db = b - mb
    va = np.sum(w * da * da) / W
    vb = np.sum(w * db * db) / W
    if va < 1e-28 or vb < 1e-28:
        return np.nan
    cov = np.sum(w * da * db) / W
    return float(cov / np.sqrt(va * vb))


def reaction_histogram(r, bins=None):
    """Histogram of per-particle reaction rate r; returns (counts, edges).

    Default bins span a fixed, pre-declared range [-2, 6] so histograms are
    comparable across snapshots/seeds (NOT data-dependent edges).
    """
    if bins is None:
        bins = np.linspace(-2.0, 6.0, 41)
    counts, edges = np.histogram(np.asarray(r), bins=bins)
    return counts.astype(np.int64), edges


def mass_fraction_pos_neg(r, w, N0):
    """Mass fraction with r > 0 (growth) vs r < 0 (decay), normalized by total.

    Returns (frac_pos, frac_neg).  Weighted by w (1 for branching).
    """
    w = np.asarray(w, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    W = np.sum(w)
    if W <= 0:
        return (np.nan, np.nan)
    fp = float(np.sum(w[r > 0]) / W)
    fn = float(np.sum(w[r < 0]) / W)
    return fp, fn


# ---------------------------------------------------------------------------
# ESS-resample with mass preservation (for the weighted+resample method)
# ---------------------------------------------------------------------------
def ess_resample(key, X, V, w, N0):
    """Multinomial resample of (X,V) to a COMMON mass-preserving weight.

    Triggered by the caller when global nESS < threshold.  Draws N_resampled
    = (number of currently-active particles) indices ~ w/sum(w), then resets
    every weight to the COMMON value  w_common = (sum_i w_i)/N_resampled
    (= M_f*N0/N_resampled), so that the total mass sum_i w_i / N0 = M_f is
    preserved EXACTLY (sum of new weights = N_resampled * w_common = M_f * N0
    ... see note).  Returns (X_new, V_new, w_new, mass_before, mass_after).

    Mass bookkeeping: with per-particle mass 1/N0 implicit elsewhere, the total
    measure mass is M_f = (sum_i w_i)/N0.  After resampling to N_r equal-weight
    particles we set w_common = (sum_i w_i)/N_r so that
        (sum_j w_common)/N0 = (N_r w_common)/N0 = (sum_i w_i)/N0 = M_f .
    The empirical measure mass is preserved exactly; only Monte-Carlo support
    is rejuvenated.  We log mass_before/mass_after (should match to round-off).
    """
    n = X.shape[0]
    Wsum = float(np.sum(w))
    mass_before = Wsum / N0
    if Wsum <= 0 or n == 0:
        return X, V, w, mass_before, mass_before
    p = np.asarray(w, dtype=np.float64) / Wsum
    idx = np.asarray(jax.random.choice(key, n, shape=(n,), replace=True,
                                       p=jnp.asarray(p)))
    X_new = X[idx]
    V_new = V[idx]
    w_common = Wsum / n
    w_new = np.full((n,), w_common, dtype=np.float64)
    mass_after = float(np.sum(w_new)) / N0
    return X_new, V_new, w_new, mass_before, mass_after


# ---------------------------------------------------------------------------
# CFL proxy: max |V| tau / dx  (dx a chosen spatial resolution scale)
# ---------------------------------------------------------------------------
def cfl_proxy(V, tau, dx):
    """max_i |V_i|_inf * tau / dx -- transport CFL proxy for the X += V tau step."""
    vmax = float(np.max(np.abs(np.asarray(V))))
    return vmax * tau / dx
