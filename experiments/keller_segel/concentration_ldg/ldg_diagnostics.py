"""LDG-style blow-up/concentration diagnostics for the 2D Keller-Segel particle
method.

These mirror the diagnostics used in local discontinuous Galerkin (LDG) blow-up
studies of Keller-Segel (cf. Li-Yang-Zhou 2017, `li2017local`):
  - the reconstructed L2 norm  S_{K,N}(t) = ||P_K mu||_{L2}  of the physical-u
    field on the adaptive window, and
  - the core mass  M_core(r,t) = mu(B(x_c, r))  inside small balls about the
    cluster centre.
A focusing/blow-up signature is a sharp rise of S_{K,N}(t) together with mass
M_core concentrating into ever-smaller balls.

REUSE.  Everything here is built on the VALIDATED reconstruction in
`../blowup_time/adaptive_window.py`:
  * window geometry  compute_window
  * probability density on the window  density_coeffs_y / eval_density_y
  * the physical-u rescaling convention  u_phys = mass * (pi/L)^2 * rho_y
    (identical to `adaptive_window.peak_density`)
We DO NOT re-derive the rescaling: we import that module and copy ONLY the two
scalar conventions it uses (the (pi/L)^2 density factor and the [-pi,pi]^2 grid).
No file in ../blowup_time is modified.

KEY DIMENSIONAL FACTS (taken verbatim from adaptive_window.py):
  map  x = x_c + (L/pi) y,   y in [-pi,pi]^2   (window has physical side 2L)
  u_phys(x) = mass * (pi/L)^2 * rho_y(y)       (rho_y is a probability density)
  L2 quadrature on the endpoint mesh linspace(-pi,pi,n_grid): composite
  trapezoid, physical node spacing hx = 2L/(n_grid-1) (see recon_L2_norm)
"""
import os
import sys

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Use the validated reconstruction vendored into THIS directory (self-contained).
_BLOWUP_DIR = os.path.dirname(os.path.abspath(__file__))
if _BLOWUP_DIR not in sys.path:
    sys.path.insert(0, _BLOWUP_DIR)

from adaptive_window import eval_density_y  # noqa: E402  (validated module)


# ---------------------------------------------------------------------------
# Reconstructed L2 norm of the physical-u field on the adaptive window.
# ---------------------------------------------------------------------------
def recon_L2_norm(coeff_y, x_c, L, mass, n_grid=129):
    """Reconstructed L2 norm  S_{K,N}(t) = ||P_K mu||_{L2}  of the physical-u
    field on the adaptive window.

    Mirrors `adaptive_window.peak_density` EXACTLY for the rescaling: evaluate
    the probability density rho_y on an n_grid x n_grid mesh of y in [-pi,pi]^2,
    rescale to the physical density u_phys = mass*(pi/L)^2 * rho_y, and integrate
    u_phys^2 over the window with composite-trapezoid quadrature.

    QUADRATURE (Codex-verified).  The mesh is linspace(-pi,pi,n_grid), an
    ENDPOINT grid with spacing hy = 2*pi/(n_grid-1).  The correct quadrature is
    composite trapezoid: weights w=[0.5,1,...,1,0.5] on each axis.  In physical
    units hx = (L/pi)*hy = 2L/(n_grid-1), so the per-node area weight is
    outer(w,w) * hx^2.  (An unweighted dx=(2L/n_grid)^2 would mis-handle the
    periodic boundary; (2L/(n_grid-1))^2 would double-count it.)  Then
    S = sqrt( sum_{ij} W_ij * u_phys_ij^2 ).
    """
    g = jnp.linspace(-jnp.pi, jnp.pi, n_grid)
    GX, GY = jnp.meshgrid(g, g, indexing="ij")
    pts = jnp.stack([GX.ravel(), GY.ravel()], axis=1)
    rho_y = jax.vmap(eval_density_y, in_axes=(0, None))(pts, coeff_y)
    u_phys = mass * (jnp.pi / L) ** 2 * rho_y          # physical density
    # composite-trapezoid area weights on the endpoint mesh, in physical units
    w1d = jnp.ones(n_grid).at[0].set(0.5).at[-1].set(0.5)
    W = (w1d[:, None] * w1d[None, :]).ravel()          # (n_grid^2,)
    hx = 2.0 * L / (n_grid - 1)                         # physical node spacing
    return jnp.sqrt(jnp.sum(W * u_phys ** 2) * hx ** 2)


# ---------------------------------------------------------------------------
# Core mass inside small balls about the cluster centre.
# ---------------------------------------------------------------------------
def core_mass(X, x_c, radii, N, mass):
    """M_core(r,t) = mu(B(x_c, r)) for each r in `radii`.

    Each of the N equal-weight particles carries physical mass `mass/N`, so the
    measure of a ball is (count of particles inside)/N * mass.

    Returns dict {r: M_core(r)} with python floats.  `radii` defaults to
    [0.01, 0.02, 0.04] (small physical core radii used by the LDG diagnostics).
    """
    if radii is None:
        radii = [0.01, 0.02, 0.04]
    r = jnp.sqrt(jnp.sum((X - x_c) ** 2, axis=1))
    out = {}
    for rad in radii:
        cnt = jnp.sum(r <= rad)
        out[float(rad)] = float(cnt) / float(N) * float(mass)
    return out


# ---------------------------------------------------------------------------
# Self-test on a known Gaussian (run as a script; uses the same IC as
# blowup_time so the rescaling can be cross-checked against peak_density).
# ---------------------------------------------------------------------------
def _selftest(seed=0, N=400000, a=84.0, mass=10.0 * jnp.pi, K=12):
    """Sanity check: for u0 = (mass*a/pi) exp(-a|x|^2) on R^2 the exact L2 norm
    is ||u0||_{L2} = (mass*a/pi) * sqrt(pi/(2a)) = mass*sqrt(a/(2pi)).  Compare to
    recon_L2_norm.  Also print core masses.  (No assertion; prints ratios.)
    """
    import numpy as np
    from adaptive_window import compute_window, density_coeffs_y
    rng = np.random.default_rng(seed)
    std = 1.0 / np.sqrt(2 * a)
    X = jnp.asarray(rng.normal(0.0, std, size=(N, 2)))
    x_c, L = compute_window(X, tau=1e-7, L_min=1e-3, q_window=0.99)
    Y = (X - x_c) * (jnp.pi / L)
    coeff = density_coeffs_y(Y, K)
    S = float(recon_L2_norm(coeff, x_c, L, mass))
    S_true = float(mass) * float(np.sqrt(a / (2 * np.pi)))
    cm = core_mass(X, x_c, [0.01, 0.02, 0.04], N, mass)
    print(f"[selftest] N={N} K={K} L={float(L):.4f}")
    print(f"[selftest] recon L2 = {S:.4f}  analytic L2 = {S_true:.4f}  "
          f"ratio = {S / S_true:.4f}")
    print(f"[selftest] core mass = {cm}")
    return S, S_true, cm


if __name__ == "__main__":
    _selftest()
