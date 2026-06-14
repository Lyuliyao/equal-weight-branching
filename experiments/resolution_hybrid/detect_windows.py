"""detect_windows.py -- choose local reconstruction windows FROM PARTICLES.
==========================================================================

The local windows where the reconstruction needs enrichment must be inferred
from the particle cloud, NOT hand-picked from the final image (CLAUDE.md Sec. 6).
This module provides:

  * reconstruction-FREE core diagnostics: weighted centroid, quantile radii
    R_q (R_0.5, R_0.8, R_0.9), weighted covariance / principal axes, core mass
    inside a ball -- all computed straight from the particle positions/weights;
  * single-core window detection (KS concentration core);
  * multi-core window detection via a coarse particle-histogram + connected
    components (for the separated growth islands), with a known-B_m fallback that
    keeps the island table reproducible.

These are deliberately simple, dependency-free (numpy only) estimators.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Window object used by the residual-particle acceptance machinery.
# ---------------------------------------------------------------------------
class Window:
    """A local reconstruction window: square of half-size `half` at `center`,
    with a padded box (pad*half) and a smooth radial taper chi supported inside.

    chi = 1 for r <= taper_frac*half, smoothly -> 0 by r = pad*half.
    """
    def __init__(self, center, half, pad=1.5, taper_frac=0.85):
        self.center = np.asarray(center, dtype=np.float64)
        self.half = float(half)
        self.pad = float(pad)
        self.taper_frac = float(taper_frac)
        self.r_in = self.taper_frac * self.half
        self.r_out = self.pad * self.half

    def padded_box(self):
        c, p = self.center, self.pad * self.half
        return [[c[0] - p, c[0] + p], [c[1] - p, c[1] + p]]

    def in_padded(self, X):
        X = np.asarray(X)
        p = self.pad * self.half
        return ((np.abs(X[:, 0] - self.center[0]) <= p)
                & (np.abs(X[:, 1] - self.center[1]) <= p))

    def taper(self, pts):
        pts = np.asarray(pts)
        r = np.sqrt((pts[:, 0] - self.center[0]) ** 2
                    + (pts[:, 1] - self.center[1]) ** 2)
        chi = np.ones_like(r)
        band = (r > self.r_in) & (r < self.r_out)
        chi[band] = 0.5 * (1.0 + np.cos(np.pi * (r[band] - self.r_in)
                                        / (self.r_out - self.r_in)))
        chi[r >= self.r_out] = 0.0
        return chi


def particles_in_padded_window(X, window):
    """Indices of particles inside the padded window (used by the acceptance rule)."""
    return np.where(window.in_padded(X))[0]


# ---------------------------------------------------------------------------
# Reconstruction-free core diagnostics
# ---------------------------------------------------------------------------
def weighted_centroid(X, w=None):
    X = np.asarray(X, dtype=np.float64)
    if w is None:
        return X.mean(axis=0)
    w = np.asarray(w, dtype=np.float64)
    return (w[:, None] * X).sum(axis=0) / w.sum()


def weighted_quantile(values, weights, q):
    """Weighted q-quantile of a 1D array."""
    values = np.asarray(values); weights = np.asarray(weights)
    o = np.argsort(values)
    v, wq = values[o], weights[o]
    cw = np.cumsum(wq)
    cw /= cw[-1]
    return float(np.interp(q, cw, v))


def quantile_radii(X, w, center, qs=(0.5, 0.8, 0.9)):
    """R_q : weighted q-quantile of the radius from `center`. Reconstruction-free."""
    X = np.asarray(X)
    r = np.sqrt((X[:, 0] - center[0]) ** 2 + (X[:, 1] - center[1]) ** 2)
    w = np.ones(X.shape[0]) if w is None else np.asarray(w)
    return {q: weighted_quantile(r, w, q) for q in qs}


def core_mass(X, w, center, radius, mass_per_particle):
    """mu(B(center, radius)) = mass_pp * (weighted) count inside the ball. Recon-free."""
    X = np.asarray(X)
    r = np.sqrt((X[:, 0] - center[0]) ** 2 + (X[:, 1] - center[1]) ** 2)
    w = np.ones(X.shape[0]) if w is None else np.asarray(w)
    return float(mass_per_particle * w[r <= radius].sum())


def weighted_covariance(X, w, center):
    X = np.asarray(X)
    w = np.ones(X.shape[0]) if w is None else np.asarray(w)
    d = X - np.asarray(center)
    C = (w[:, None, None] * np.einsum('ni,nj->nij', d, d)).sum(axis=0) / w.sum()
    evals, evecs = np.linalg.eigh(C)
    return C, evals, evecs


def local_effective_count(X, w, center, radius):
    """Local effective sample size inside the ball (weighted) or count (equal-weight)."""
    X = np.asarray(X)
    r = np.sqrt((X[:, 0] - center[0]) ** 2 + (X[:, 1] - center[1]) ** 2)
    inside = r <= radius
    if w is None:
        return float(np.sum(inside))
    wi = np.asarray(w)[inside]
    s1 = wi.sum(); s2 = (wi * wi).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0


# ---------------------------------------------------------------------------
# Single-core window (KS concentration core)
# ---------------------------------------------------------------------------
def detect_core_window(X, w, mass_per_particle, c_window=3.0, qs=(0.5, 0.8, 0.9)):
    """Return a dict describing the particle-detected core window.

    center = weighted centroid; radii R_q; window half-size = c_window * R_0.8.
    All quantities are reconstruction-free particle statistics.
    """
    center = weighted_centroid(X, w)
    radii = quantile_radii(X, w, center, qs=qs)
    half = c_window * radii[0.8]
    C, evals, evecs = weighted_covariance(X, w, center)
    return dict(center=center.tolist(), R05=radii[0.5], R08=radii[0.8],
                R09=radii.get(0.9, np.nan), half=float(half),
                cov_evals=evals.tolist(), cov_axes=evecs.tolist(),
                local_count=local_effective_count(X, w, center, radii[0.8]),
                local_mass_R08=core_mass(X, w, center, radii[0.8], mass_per_particle))


# ---------------------------------------------------------------------------
# Multi-core windows via a coarse histogram + connected components.
# Used to SHOW the islands can be found WITHOUT the reference solution.
# ---------------------------------------------------------------------------
def _connected_components(binary):
    """Label 4-connected components of a 2D boolean array (numpy-only flood fill)."""
    labels = np.zeros(binary.shape, dtype=np.int32)
    cur = 0
    H, W = binary.shape
    for i in range(H):
        for j in range(W):
            if binary[i, j] and labels[i, j] == 0:
                cur += 1
                stack = [(i, j)]
                labels[i, j] = cur
                while stack:
                    a, b = stack.pop()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < H and 0 <= nb < W and binary[na, nb] \
                                and labels[na, nb] == 0:
                            labels[na, nb] = cur
                            stack.append((na, nb))
    return labels, cur


def detect_islands_from_particles(X, w, box, mass_per_particle, nbins=40,
                                  mass_frac_thresh=0.02, c_window=3.0):
    """Particle-derived island windows (no reference solution).

    Histogram the (weighted) particles on a coarse nbins x nbins grid, threshold
    bins whose local density exceeds `mass_frac_thresh` of the peak bin, label
    connected components, and return one window per component (weighted centroid +
    R_0.8 -> half = c_window*R_0.8).  Returns a list of window dicts.
    """
    X = np.asarray(X)
    (x0, x1), (y0, y1) = box
    w = np.ones(X.shape[0]) if w is None else np.asarray(w)
    H, xe, ye = np.histogram2d(X[:, 0], X[:, 1], bins=nbins,
                               range=[[x0, x1], [y0, y1]], weights=w)
    peak = H.max()
    binary = H >= mass_frac_thresh * peak
    labels, ncomp = _connected_components(binary.T)   # transpose: rows=y
    windows = []
    cx_centers = 0.5 * (xe[:-1] + xe[1:])
    cy_centers = 0.5 * (ye[:-1] + ye[1:])
    for c in range(1, ncomp + 1):
        rows, cols = np.where(labels == c)
        # particles whose bin is in this component
        bx = np.clip(np.searchsorted(xe, X[:, 0]) - 1, 0, nbins - 1)
        by = np.clip(np.searchsorted(ye, X[:, 1]) - 1, 0, nbins - 1)
        member = np.isin(by * nbins + bx, rows * nbins + cols)
        if member.sum() < 5:
            continue
        Xc = X[member]; wc = w[member]
        win = detect_core_window(Xc, wc, mass_per_particle, c_window=c_window)
        win["n_particles"] = int(member.sum())
        windows.append(win)
    return windows


# ---------------------------------------------------------------------------
# Known-B_m island windows (keep the island table reproducible).
# ---------------------------------------------------------------------------
def island_windows_known(centers, sigma, c_window=3.0):
    """Windows from the KNOWN island centers (B_m radius = sigma*sqrt(2 ln 2))."""
    R_B = sigma * np.sqrt(2.0 * np.log(2.0))
    return [dict(center=list(c), R08=R_B, half=float(c_window * R_B),
                 known=True) for c in np.asarray(centers)]
