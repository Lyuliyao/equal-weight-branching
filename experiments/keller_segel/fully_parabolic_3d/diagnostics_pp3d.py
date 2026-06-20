"""diagnostics_pp3d.py -- torus-aware 3D diagnostics for the fully parabolic run.

All spatial quantities use the minimal-image displacement on [-L/2,L/2]^3.
Torus centroid + quantile core radii are the same logic as
focusing_3d/diagnostics_focusing.py (copied here, self-contained, unit-tested in 3D).
"""
import numpy as np


def torus_disp(X, center, L):
    """Minimal-image displacement X - center on the torus. (N,3)."""
    d = np.asarray(X, float) - np.asarray(center, float)
    return (d + L / 2.0) % L - L / 2.0


def torus_centroid(X, L, n_iter=4):
    """Torus-aware mass centroid via iterative minimal-image re-centering."""
    X = np.asarray(X, float)
    c = X[0].copy() if X.shape[0] else np.zeros(3)
    for _ in range(n_iter):
        c = (c + torus_disp(X, c, L).mean(axis=0))
        c = (c + L / 2.0) % L - L / 2.0
    return c


def core_radii(X, x_c, L, quantiles=(0.2, 0.5, 0.8)):
    """Quantile radii of particles about x_c (torus-aware). dict q->R_q."""
    r = np.linalg.norm(torus_disp(X, x_c, L), axis=1)
    return {q: float(np.quantile(r, q)) for q in quantiles}


def covariance_eigs(X, x_c, L):
    """Eigenvalues of the torus-aware position covariance (cloud shape)."""
    d = torus_disp(X, x_c, L)
    return np.sort(np.linalg.eigvalsh(np.cov(d.T)))[::-1]


def mass_in_ball(X, center, r_c, L, mass_per_particle):
    """Mass inside B(center, r_c) (torus-aware)."""
    r = np.linalg.norm(torus_disp(X, center, L), axis=1)
    return float(np.sum(r <= r_c) * mass_per_particle)


# ---- tetra cluster diagnostics ----
def cluster_centroids(X, labels, n_clusters, L):
    return np.stack([torus_centroid(X[labels == m], L) for m in range(n_clusters)])


def torus_dist(a, b, L):
    d = (np.asarray(a) - np.asarray(b) + L / 2.0) % L - L / 2.0
    return float(np.linalg.norm(d))


def pairwise_dists(centroids, L):
    n = len(centroids)
    return [torus_dist(centroids[i], centroids[j], L)
            for i in range(n) for j in range(i + 1, n)]


def d_min(centroids, L):
    return float(min(pairwise_dists(centroids, L)))


def overlap_indicator(centroids, R05, L):
    """min_{m!=n} d(c_m,c_n) / (R05_m + R05_n).  <~1 => clusters overlap."""
    n = len(centroids)
    return float(min(torus_dist(centroids[i], centroids[j], L) / (R05[i] + R05[j])
                     for i in range(n) for j in range(i + 1, n)))


def symmetry_residual(centroids, L):
    """Coefficient of variation of the six pairwise centroid distances."""
    d = np.asarray(pairwise_dists(centroids, L))
    return float(d.std() / d.mean()) if d.mean() > 0 else np.nan
