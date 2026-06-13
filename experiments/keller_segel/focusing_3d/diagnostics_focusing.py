"""
Grid-free (reconstruction-light) diagnostics for the 3D KS focusing study.

All spatial quantities are torus-aware on the box [-L/2, L/2]^3 (minimal-image
displacement). The single particle cloud carries rho (conservative chemotaxis +
diffusion; no branching), so the particle COUNT is constant and total mass is
conserved exactly -- the mass-drift diagnostic is a sanity check on bookkeeping.

Diagnostics:
  - x_c(t)        : torus-aware mass centroid of the cloud.
  - R_0.5, R_0.9  : quantile radii of particles about x_c (core radii).
  - rho_core(t)   : 0.5 M / ((4/3) pi R_0.5^3) -- mean physical density of the
                    inner-half-mass core (a grid-free peak-density surrogate).
  - P_H(t)        : reconstructed physical-density peak ||P_H rho||_inf at
                    bandwidth H, max over a coarse eval grid AND particle points.
  - C_H(t)        : chemical-field peak ||c_H||_inf at bandwidth H.
  - Q_c(t)        : self-convergence ratio of C_H at two bandwidths.
  - mass drift    : |N_t/N_0 - 1| (should be ~0; count is constant).
  - tetra extras  : per-cluster centroids, per-cluster R_0.5/R_0.9, and the
                    minimum inter-cluster center distance (torus-aware).

The mass M multiplies the PROBABILITY density to give physical rho, so all
density-valued diagnostics (rho_core, P_H, C_H) scale with M as expected.
"""
import numpy as np
import jax
import jax.numpy as jnp

import field3d_screened as fld


# ---------------------------------------------------------------------------
# Torus geometry on [-L/2, L/2]^3.
# ---------------------------------------------------------------------------
def torus_disp(points, center, L):
    """Minimal-image displacement points - center on the box of side L.
    points (n,3), center (3,) -> (n,3) in [-L/2, L/2]^3."""
    d = jnp.asarray(points) - jnp.asarray(center)[None, :]
    return (d + L / 2.0) % L - L / 2.0


def torus_centroid(X, L, n_iter=3):
    """Torus-aware centroid via iterative minimal-image re-centering.

    Start from particle 0 (any reference); compute mean of minimal-image
    displacements; shift the reference; repeat. Converges for a localized cloud.
    Returns a (3,) centroid wrapped to the box.
    """
    X = jnp.asarray(X)
    c = X[0]
    for _ in range(n_iter):
        d = torus_disp(X, c, L)
        c = c + jnp.mean(d, axis=0)
    c = (c + L / 2.0) % L - L / 2.0
    return c


def core_radii(X, x_c, L, quantiles=(0.5, 0.9)):
    """Quantile radii of particles about x_c (torus-aware). Returns a dict."""
    d = torus_disp(X, x_c, L)
    r = jnp.sqrt(jnp.sum(d ** 2, axis=1))
    out = {}
    for q in quantiles:
        out[q] = float(jnp.quantile(r, q))
    return out


def rho_core(R_half, M):
    """Mean physical density of the inner-half-mass core:
        rho_core = 0.5 M / ((4/3) pi R_0.5^3).
    A grid-free peak-density surrogate that scales with M and ~ R_0.5^{-3}."""
    vol = (4.0 / 3.0) * np.pi * (R_half ** 3)
    return float(0.5 * M / max(vol, 1e-30))


def mass_drift(n_active, n0):
    """Relative drift of total (probability) mass: |n_active/n0 - 1|.
    For the conservative single cloud this should be ~0 (count constant)."""
    return float(abs(n_active / n0 - 1.0))


# ---------------------------------------------------------------------------
# Reconstructed peaks P_H (density) and C_H (chemical), self-convergence Q_c.
# ---------------------------------------------------------------------------
def _eval_grid(x_c, half, n_side, L):
    """Coarse evaluation grid (n_side^3) centred at x_c, extent +-half, wrapped."""
    g = jnp.linspace(-half, half, n_side)
    GX, GY, GZ = jnp.meshgrid(g, g, g, indexing="ij")
    pts = jnp.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=1) + jnp.asarray(x_c)[None, :]
    pts = (pts + L / 2.0) % L - L / 2.0
    return pts


def peak_density_PH(X, M, H, L, x_c, grid_half=2.0, n_side=41):
    """P_H(t) = ||P_H rho||_inf at bandwidth H, evaluated ON THE EVAL GRID.

    Returns sup over a coarse grid (centred at x_c) of |rho_H|.  The band-limited
    reconstruction is smooth, so the grid (spacing 2*grid_half/(n_side-1)) resolves
    its peak; we report the L-infinity norm of the reconstructed field (grid only,
    matching the stated diagnostic, so it is comparable across bandwidths H)."""
    coeff_p = fld.density_coeffs(X, H, L)
    pts = _eval_grid(x_c, grid_half, n_side, L)
    vals_grid = fld.eval_density(pts, coeff_p, M)
    return float(jnp.max(jnp.abs(vals_grid)))


def peak_chem_CH(X, M, H, L, kappa, x_c, grid_half=2.0, n_side=41):
    """C_H(t) = ||c_H||_inf at bandwidth H, evaluated ON THE EVAL GRID."""
    coeff_p = fld.density_coeffs(X, H, L)
    coeff_c = fld.screened_solve(coeff_p, M, kappa)
    pts = _eval_grid(x_c, grid_half, n_side, L)
    vals_grid = fld.eval_c(pts, coeff_c)
    return float(jnp.max(jnp.abs(vals_grid)))


def self_convergence_Qc(X, M, L, kappa, x_c, H_lo=12, H_hi=24,
                        grid_half=2.0, n_side=41):
    """Self-convergence ratio Q_c = ||c_{H_hi}||_inf / ||c_{H_lo}||_inf from the
    SAME particle cloud, re-solving the field at two bandwidths. Q_c -> 1 as the
    reconstruction converges; growth with H signals under-resolution."""
    c_lo = peak_chem_CH(X, M, H_lo, L, kappa, x_c, grid_half, n_side)
    c_hi = peak_chem_CH(X, M, H_hi, L, kappa, x_c, grid_half, n_side)
    return float(c_hi / max(c_lo, 1e-30)), c_lo, c_hi


def drift_cfl(X, M, H, L, kappa, chi, tau):
    """Stability monitor: ratio of the chemotactic drift step to the diffusion
    step, max_i |chi grad c(X_i)| * tau / sqrt(2 tau).  Should stay well below 1;
    if it grows as the core tightens, reduce tau (Codex-requested monitor)."""
    coeff_p = fld.density_coeffs(X, H, L)
    coeff_c = fld.screened_solve(coeff_p, M, kappa)
    gc = fld.grad_c(X, coeff_c)
    gmax = float(jnp.max(jnp.sqrt(jnp.sum(gc ** 2, axis=1))))
    return chi * gmax * tau / float(jnp.sqrt(2.0 * tau))


# ---------------------------------------------------------------------------
# Tetrahedral (4-cluster) extras.
# ---------------------------------------------------------------------------
def cluster_centroids(X, labels, n_clusters, L):
    """Per-cluster torus-aware centroids. Returns (n_clusters, 3) array."""
    cents = []
    labels = np.asarray(labels)
    for m in range(n_clusters):
        idx = np.where(labels == m)[0]
        cents.append(np.asarray(torus_centroid(X[idx], L)))
    return np.stack(cents, axis=0)


def cluster_core_radii(X, labels, cents, n_clusters, L):
    """Per-cluster R_0.5 / R_0.9 about each cluster centroid."""
    labels = np.asarray(labels)
    rad = []
    for m in range(n_clusters):
        idx = np.where(labels == m)[0]
        rr = core_radii(X[idx], cents[m], L)
        rad.append((rr[0.5], rr[0.9]))
    return np.asarray(rad)   # (n_clusters, 2)


def min_intercluster_distance(cents, L):
    """Minimum pairwise torus-aware distance between cluster centroids."""
    n = cents.shape[0]
    dmin = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = (cents[i] - cents[j] + L / 2.0) % L - L / 2.0
            dist = float(np.sqrt(np.sum(d ** 2)))
            dmin = min(dmin, dist)
    return dmin
