"""initial_conditions.py -- particle initial conditions on the torus [-L/2,L/2]^3.

Radial: u0 = M * wrapped-Gaussian(sigma) at origin (sample N(0,sigma^2 I3), wrap).
Tetra : four equal clusters at the tetrahedral vertices (a,a,a),(a,-a,-a),
        (-a,a,-a),(-a,-a,a) with per-cluster width sigma_c; returns cluster labels.
v-cloud starts EMPTY when v0=0 (N_v(0)=0).
"""
import numpy as np


def wrap_to_box(X, L):
    """Wrap positions into [-L/2, L/2]^3."""
    return (np.asarray(X) + L / 2.0) % L - L / 2.0


def sample_wrapped_gaussian(rng, N, center, sigma, L):
    """N points ~ N(center, sigma^2 I3) wrapped onto the torus. (N,3)."""
    X = rng.normal(np.asarray(center, float), sigma, size=(N, 3))
    return wrap_to_box(X, L)


TETRA = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64)


def sample_tetra(rng, N, a, sigma_c, L):
    """Four equal Gaussian clusters at a*TETRA. Returns (X (N,3), labels (N,)).
    N is split as evenly as possible across the 4 clusters."""
    counts = [N // 4] * 4
    for i in range(N - sum(counts)):
        counts[i] += 1
    Xs, labs = [], []
    for m, (c, nm) in enumerate(zip(a * TETRA, counts)):
        Xs.append(rng.normal(c, sigma_c, size=(nm, 3)))
        labs.append(np.full(nm, m, dtype=np.int64))
    X = wrap_to_box(np.concatenate(Xs, axis=0), L)
    labels = np.concatenate(labs, axis=0)
    return X, labels


def empty_v_cloud():
    """v0 = 0  ->  the chemical cloud starts as the empty measure."""
    return np.zeros((0, 3), dtype=np.float64)
