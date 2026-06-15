"""
Mass-quantile core radii R_q(t) -- method-independent (core_collapse plan §2,4,5).
==================================================================================

R_q(t) = inf{ r : mu(B(x_c(t), r)) >= q M }, the smallest radius around the
mass center containing a fraction q of the total mass.  Computed identically from:
  * LDG P1-DG cells via sub-cell Gauss quadrature samples (`ldg_quad_samples`);
  * equal-weight particle clouds via ordered distances (`particle_radii`).

This is (almost) reconstruction-free: no Fourier/KDE/peak bandwidth enters R_q.
"""
import numpy as np


def radii_from_samples(x, y, mass, qs, center=None, interpolate=True):
    """Mass-quantile radii from weighted point samples.

    x, y, mass : 1D arrays of sample locations and (nonnegative or signed) masses.
    qs         : iterable of quantiles in (0,1).
    center     : (cx,cy) to use; if None, the mass-weighted centroid.
    Returns (cx, cy, M, {q: R_q}).  R_q = NaN if M<=0.
    """
    x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel()
    m = np.asarray(mass, float).ravel()
    M = float(m.sum())
    out = {float(q): np.nan for q in qs}
    if not np.isfinite(M) or M <= 0:
        cx = float(np.average(x)) if center is None else float(center[0])
        cy = float(np.average(y)) if center is None else float(center[1])
        return cx, cy, M, out
    if center is None:
        cx = float((m * x).sum() / M); cy = float((m * y).sum() / M)
    else:
        cx, cy = float(center[0]), float(center[1])
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    order = np.argsort(d, kind="mergesort")
    d_s = d[order]; m_s = m[order]
    cum = np.cumsum(m_s)
    for q in qs:
        target = float(q) * M
        k = int(np.searchsorted(cum, target, side="left"))
        if k >= len(d_s):
            out[float(q)] = float(d_s[-1])
            continue
        if interpolate and k > 0:
            c0, c1 = cum[k - 1], cum[k]
            if c1 > c0:
                frac = (target - c0) / (c1 - c0)
                out[float(q)] = float(d_s[k - 1] + frac * (d_s[k] - d_s[k - 1]))
            else:
                out[float(q)] = float(d_s[k])
        else:
            out[float(q)] = float(d_s[k])
    return cx, cy, M, out


def inner_center(x, y, mass, q0=0.2, n_iter=1):
    """Robust inner-mass center (plan §2.1): start from the global centroid, then
    recompute using only mass inside B(center, R_q0).  n_iter refinements."""
    cx, cy, M, R = radii_from_samples(x, y, mass, [q0])
    if not np.isfinite(M) or M <= 0:
        return cx, cy
    x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel()
    m = np.asarray(mass, float).ravel()
    for _ in range(max(0, n_iter)):
        r0 = R[float(q0)]
        if not np.isfinite(r0) or r0 <= 0:
            break
        inb = (x - cx) ** 2 + (y - cy) ** 2 <= r0 * r0
        Mi = m[inb].sum()
        if Mi <= 0:
            break
        cx = float((m[inb] * x[inb]).sum() / Mi)
        cy = float((m[inb] * y[inb]).sum() / Mi)
        _, _, _, R = radii_from_samples(x, y, m, [q0], center=(cx, cy))
    return cx, cy


def ldg_quad_samples(U, mesh, quad_order=3, clip=False):
    """Sub-cell Gauss-quadrature mass samples of a P1-DG field U (shape (Ny,Nx,3),
    local field c0 + c1*xi + c2*eta on [-1,1]^2; cell (j,i) centered at
    (xc[i], yc[j])).  Returns (x, y, mass) flat arrays.  sum(mass) == total u-mass.

    clip=False -> raw mass w*u; clip=True -> w*max(u,0).  Linear modes integrate to
    zero so cell mass == c0*A for any quad order >= 1 (mass-conservative)."""
    gp, gw = np.polynomial.legendre.leggauss(quad_order)
    XI, ETA = np.meshgrid(gp, gp, indexing="ij")
    xi = XI.ravel(); eta = ETA.ravel()
    w = np.outer(gw, gw).ravel()                          # (Q,), sums to 4
    c0 = U[..., 0]; c1 = U[..., 1]; c2 = U[..., 2]        # (Ny,Nx)
    Ny, Nx = c0.shape
    uq = (c0[..., None] + c1[..., None] * xi[None, None, :]
          + c2[..., None] * eta[None, None, :])           # (Ny,Nx,Q)
    if clip:
        uq = np.maximum(uq, 0.0)
    massq = uq * w[None, None, :] * (mesh.A / 4.0)         # (Ny,Nx,Q)
    xq = np.broadcast_to(mesh.xc[None, :, None] + 0.5 * mesh.dx * xi[None, None, :],
                         (Ny, Nx, len(xi)))
    yq = np.broadcast_to(mesh.yc[:, None, None] + 0.5 * mesh.dy * eta[None, None, :],
                         (Ny, Nx, len(xi)))
    return xq.ravel(), yq.ravel(), massq.ravel()


def ldg_core_radii(U, mesh, qs, quad_order=3, center_mode="global", q0=0.2):
    """Convenience: raw and clipped R_q + centers + masses for one LDG field."""
    out = {}
    for tag, clip in (("raw", False), ("clip", True)):
        x, y, m = ldg_quad_samples(U, mesh, quad_order=quad_order, clip=clip)
        if center_mode == "inner":
            cx, cy = inner_center(x, y, m, q0=q0, n_iter=1)
            cx, cy, M, R = radii_from_samples(x, y, m, qs, center=(cx, cy))
        else:
            cx, cy, M, R = radii_from_samples(x, y, m, qs)
        out[tag] = dict(cx=cx, cy=cy, M=M, R=R)
    return out


def particle_radii(X, qs, center=None):
    """R_q from an equal-weight particle cloud X (N,2): the ceil(q N)-th ordered
    distance from the cloud centroid (or `center`)."""
    X = np.asarray(X, float)
    N = X.shape[0]
    out = {float(q): np.nan for q in qs}
    if N == 0:
        return np.nan, np.nan, out
    cx, cy = (X.mean(0) if center is None else np.asarray(center, float))
    d = np.sort(np.sqrt((X[:, 0] - cx) ** 2 + (X[:, 1] - cy) ** 2))
    for q in qs:
        k = int(np.ceil(float(q) * N))
        k = min(max(k, 1), N)
        out[float(q)] = float(d[k - 1])
    return float(cx), float(cy), out
