"""
common_particle.py
==================

Shared building blocks for the particle-method numerical experiments.

This module reuses the SAME Fourier-density-reconstruction conventions as
`Keller_Segel/case2_test3/density.py` / `simulation.py`:

  * The reconstruction operator P_K maps an (optionally weighted, masked)
    empirical particle cloud on a 2D torus to a band-limited density via
    cos/sin Fourier coefficients.  Each coefficient is the (weighted) MEAN of
    the corresponding basis function over the active particles, normalized by
    the `norm` array (1 for the constant mode, 2 for a single non-zero index,
    4 for both non-zero) divided by the domain area Lx*Ly.
  * Reconstructed this way the (0,0) mode integrates to 1 over the domain, so
    P_K returns a PROBABILITY density (mass-1) of the active cloud.  The
    finite-measure mass is tracked separately by multiplying by
    (current_count / N0) for branching, or (sum_w / N0) for weighted.

It also provides:
  * Euler-Maruyama transport with externally supplied (shared) Brownian
    increments, so different reaction representations can be compared under
    identical noise.
  * The three reaction kernels: weighted, Poisson branching, and
    minimum-variance integer branching.

Everything keeps jax x64 enabled by the importing script.
"""

import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Fourier density reconstruction  P_K   (density.py style)
# ---------------------------------------------------------------------------
def make_norm(K):
    """norm[k,l] = 1/2/4 depending on how many of k,l are zero (cos/sin folding)."""
    norm = np.zeros((K, K))
    norm[0, 0] = 1.0
    norm[0, 1:] = 2.0
    norm[1:, 0] = 2.0
    norm[1:, 1:] = 4.0
    return jnp.asarray(norm)


def generate_density_estimation(n_freq=10, period=None):
    """
    Build (density_estimation, density_evaluate, density_evaluate_grid) for a
    PERIODIC domain.  `period` is a 2x2 array [[xmin,xmax],[ymin,ymax]].

    density_estimation(data, weights=None, mask=None) -> coeff dict
        data    : (N, 2) particle positions (fixed-size buffer ok)
        weights : (N,) per-particle weights (default all ones)
        mask    : (N,) boolean active flags (default all active)
        Coefficients are the weighted mean of the basis over ACTIVE particles,
        i.e. sum_i (mask_i w_i basis_i) / sum_i (mask_i w_i).  Hence P_K is a
        probability density (integrates to 1 over the domain).  Multiply the
        evaluated density by the measure mass externally.

    density_evaluate(point, coeff) -> scalar density at a single (x,y).
    density_evaluate_grid(XX, YY, coeff) -> density on a meshgrid (vectorized).
    """
    K = n_freq
    norm = make_norm(K)
    if period is None:
        raise ValueError("period must be provided for periodic boundary conditions.")
    period = jnp.asarray(period, dtype=jnp.float64)
    x_min = period[0, 0]
    x_max = period[0, 1]
    y_min = period[1, 0]
    y_max = period[1, 1]
    Lx = x_max - x_min
    Ly = y_max - y_min
    freq = jnp.arange(K)

    def density_estimation(data, weights=None, mask=None):
        x_data, y_data = data[:, 0], data[:, 1]
        theta_x = 2.0 * jnp.pi * freq[None, :] * (x_data[:, None] - x_min) / Lx
        theta_y = 2.0 * jnp.pi * freq[None, :] * (y_data[:, None] - y_min) / Ly
        cx = jnp.cos(theta_x)   # (n,K)
        sx = jnp.sin(theta_x)
        cy = jnp.cos(theta_y)
        sy = jnp.sin(theta_y)

        if weights is None:
            w = jnp.ones((data.shape[0],), dtype=jnp.float64)
        else:
            w = jnp.asarray(weights, dtype=jnp.float64)
        if mask is not None:
            w = w * jnp.asarray(mask, dtype=jnp.float64)
        # Guard against extinction (all weights zero): fall back to den=1 so the
        # zero numerators give a zero density rather than NaN/inf.
        den = jnp.where(jnp.sum(w) > 0.0, jnp.sum(w), 1.0)
        wc = w[:, None]

        norm_factor = norm / (Lx * Ly)
        Ccc = jnp.einsum('nk,nl->kl', wc * cx, cy) / den
        Ccs = jnp.einsum('nk,nl->kl', wc * cx, sy) / den
        Scc = jnp.einsum('nk,nl->kl', wc * sx, cy) / den
        Scs = jnp.einsum('nk,nl->kl', wc * sx, sy) / den

        coeffs = {
            "cos-cos": norm_factor * Ccc,
            "cos-sin": norm_factor * Ccs,
            "sin-cos": norm_factor * Scc,
            "sin-sin": norm_factor * Scs,
            "Lx": Lx, "Ly": Ly,
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "K": K,
        }
        return coeffs

    def density_evaluate(point, coeff):
        x_data, y_data = point[0], point[1]
        x_data = jnp.mod(x_data - x_min, Lx) + x_min
        y_data = jnp.mod(y_data - y_min, Ly) + y_min
        theta_x = 2.0 * jnp.pi * freq * (x_data - x_min) / Lx
        theta_y = 2.0 * jnp.pi * freq * (y_data - y_min) / Ly
        Z = jnp.zeros((K, K))
        Z += coeff["cos-cos"] * jnp.cos(theta_x[..., None]) * jnp.cos(theta_y[None, :])
        Z += coeff["cos-sin"] * jnp.cos(theta_x[..., None]) * jnp.sin(theta_y[None, :])
        Z += coeff["sin-cos"] * jnp.sin(theta_x[..., None]) * jnp.cos(theta_y[None, :])
        Z += coeff["sin-sin"] * jnp.sin(theta_x[..., None]) * jnp.sin(theta_y[None, :])
        return jnp.sum(Z)

    def density_evaluate_grid(XX, YY, coeff):
        """Evaluate density on meshgrid arrays XX, YY (same shape). Returns same shape.

        Uses separable structure: build (Ngx,K) and (Ngy,K) basis on the axes
        from the meshgrid, then contract with coefficient matrices.
        """
        xv = jnp.mod(XX[0, :] - x_min, Lx) + x_min        # x varies along columns
        yv = jnp.mod(YY[:, 0] - y_min, Ly) + y_min        # y varies along rows
        tx = 2.0 * jnp.pi * freq[None, :] * (xv[:, None] - x_min) / Lx   # (Nx,K)
        ty = 2.0 * jnp.pi * freq[None, :] * (yv[:, None] - y_min) / Ly   # (Ny,K)
        cxg, sxg = jnp.cos(tx), jnp.sin(tx)
        cyg, syg = jnp.cos(ty), jnp.sin(ty)
        # Z[iy,ix] = sum_kl [ cc cx[ix,k] cy[iy,l] + ... ]
        Z = (cyg @ coeff["cos-cos"].T @ cxg.T
             + syg @ coeff["cos-sin"].T @ cxg.T
             + cyg @ coeff["sin-cos"].T @ sxg.T
             + syg @ coeff["sin-sin"].T @ sxg.T)
        return Z   # shape (Ny, Nx) matching meshgrid(indexing='xy')

    return density_estimation, density_evaluate, density_evaluate_grid


# ---------------------------------------------------------------------------
# Euler-Maruyama transport with shared Brownian increments
# ---------------------------------------------------------------------------
def em_transport(X, drift, D, tau, dW):
    """One Euler-Maruyama step.

    X      : (N,2) positions
    drift  : (N,2) drift b(X)  (zero for pure diffusion)
    D      : scalar diffusion coefficient (so noise amplitude sqrt(2 D tau))
    tau    : time step
    dW     : (N,2) standard normal increments (shared across methods)
    """
    return X + drift * tau + jnp.sqrt(2.0 * D * tau) * dW


def wrap_torus(X, period):
    """Wrap positions back into the periodic box [xmin,xmax]x[ymin,ymax]."""
    period = jnp.asarray(period, dtype=jnp.float64)
    x_min, x_max = period[0, 0], period[0, 1]
    y_min, y_max = period[1, 0], period[1, 1]
    Lx, Ly = x_max - x_min, y_max - y_min
    x = jnp.mod(X[:, 0] - x_min, Lx) + x_min
    y = jnp.mod(X[:, 1] - y_min, Ly) + y_min
    return jnp.stack([x, y], axis=1)


# ---------------------------------------------------------------------------
# Reaction kernels
# ---------------------------------------------------------------------------
def reaction_weighted(w, r, tau):
    """Weighted representation: w_i *= exp(r_i tau). Returns updated weights."""
    return w * jnp.exp(r * tau)


def reaction_poisson(key, r, tau):
    """Poisson branching offspring counts. E[nu] = exp(r tau) = m.

    If m >= 1 (r>=0):  nu = 1 + Poisson(m - 1)   (always >=1)
    If m < 1  (r<0) :  nu = Bernoulli(m)          (0 or 1)
    Returns integer offspring counts nu (same shape as r).
    """
    m = jnp.exp(r * tau)
    k_pois, k_bern = jax.random.split(key)
    # Poisson with rate (m-1) for the grow branch; lam must be >=0
    lam = jnp.clip(m - 1.0, min=0.0)
    pois = jax.random.poisson(k_pois, lam, shape=r.shape)
    grow = 1 + pois
    u = jax.random.uniform(k_bern, shape=r.shape)
    bern = (u < jnp.clip(m, min=0.0, max=1.0)).astype(jnp.int32)
    nu = jnp.where(m >= 1.0, grow, bern)
    return nu.astype(jnp.int32)


def reaction_minvar(key, r, tau):
    """Minimum-variance integer branching. E[nu]=m, Var=theta(1-theta).

    nu = floor(m) + Bernoulli(theta),  theta = m - floor(m).
    Returns integer offspring counts nu.
    """
    m = jnp.exp(r * tau)
    fl = jnp.floor(m)
    theta = m - fl
    u = jax.random.uniform(key, shape=r.shape)
    nu = fl + (u < theta).astype(jnp.float64)
    return nu.astype(jnp.int32)


# ---------------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------------
def nESS(w):
    """Normalized effective sample size (1/N)(sum w)^2 / sum w^2, over the given w."""
    w = jnp.asarray(w, dtype=jnp.float64)
    n = w.shape[0]
    s1 = jnp.sum(w)
    s2 = jnp.sum(w * w)
    return jnp.where(s2 > 0, (s1 * s1) / (n * s2), jnp.nan)
