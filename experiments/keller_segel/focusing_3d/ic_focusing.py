"""
Initial conditions for the 3D Keller-Segel focusing / self-convergence study.

Two families on the periodic box [-L/2, L/2]^3 (L = 12 by default):

5.5(b) radial Gaussian mass family:
    rho_0^M(x) = M (2 pi sigma^2)^{-3/2} exp(-|x|^2 / (2 sigma^2)),  sigma = 0.45,
    M in {20, 40, 60, 80, 100}.
  The particle cloud SHAPE is fixed: positions are drawn x ~ N(0, sigma^2 I_3)
  (so the empirical PROBABILITY density approximates the unit-mass Gaussian).
  The physical mass M does NOT change the number of particles; it enters only as
  the field-solve scale (physical rho = M * p), exactly like `mass` in
  ../blowup_time/adaptive_window.py. Hence N controls Monte-Carlo resolution and
  M controls the chemotactic forcing strength independently.

5.5(c) nonradial 4-cluster (tetrahedral):
    centers c1=(1,1,1), c2=(1,-1,-1), c3=(-1,1,-1), c4=(-1,-1,1);
    rho_0(x) = sum_m (M/4) (2 pi sigma_c^2)^{-3/2} exp(-|x - c_m|^2/(2 sigma_c^2)),
    M = 80, sigma_c = 0.25.
  Particles are split equally among the four clusters (each cluster is a narrow
  Gaussian); the physical mass M again enters only via the field scale.

All sampling uses numpy Generators (seeded) so positions are reproducible and
identical across methods/bandwidths sharing a seed. Positions are wrapped into
the box [-L/2, L/2]^3 (periodic). sigma << L so wrap-around is negligible.
"""
import numpy as np
import jax.numpy as jnp


TETRA_CENTERS = np.array([
    [1.0, 1.0, 1.0],
    [1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
])


def _wrap(X, L):
    """Wrap positions into [-L/2, L/2]^3 (periodic)."""
    return jnp.mod(jnp.asarray(X) + L / 2.0, L) - L / 2.0


def gaussian_ic(rng, N, M, sigma, L):
    """Radial Gaussian IC (5.5(b)).

    rng   : numpy.random.Generator (seeded).
    N     : number of particles (Monte-Carlo resolution; independent of M).
    M     : physical mass (field-solve scale; returned for the caller to pass on).
    sigma : Gaussian width.
    L     : box side.

    Returns (X, M) with X a (N,3) jnp array wrapped to the box. The empirical
    cloud has unit probability mass; the physical mass M is carried separately.
    """
    X = rng.normal(0.0, sigma, size=(N, 3))
    return _wrap(X, L), float(M)


def tetra_clusters_ic(rng, N, M, sigma_c, centers=TETRA_CENTERS, L=12.0):
    """Nonradial 4-cluster tetrahedral IC (5.5(c)).

    Particles are split as evenly as possible among the clusters; each cluster
    is a narrow Gaussian of width sigma_c about its center. Returns (X, M, labels)
    where labels (N,) gives the cluster index of each particle (for per-cluster
    diagnostics). The physical mass M is split M/4 per cluster but, since each
    cluster carries an equal particle share, the empirical probability density is
    the equal-weight average; physical rho = M * p as usual.
    """
    centers = np.asarray(centers, dtype=float)
    n_cl = centers.shape[0]
    base = N // n_cl
    counts = [base] * n_cl
    for j in range(N - base * n_cl):       # distribute remainder
        counts[j] += 1

    parts, labels = [], []
    for m in range(n_cl):
        nm = counts[m]
        Xm = rng.normal(0.0, sigma_c, size=(nm, 3)) + centers[m][None, :]
        parts.append(Xm)
        labels.append(np.full((nm,), m, dtype=np.int32))
    X = np.concatenate(parts, axis=0)
    labels = np.concatenate(labels, axis=0)
    return _wrap(X, L), float(M), jnp.asarray(labels)
