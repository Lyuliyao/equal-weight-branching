"""Reconstruct grad v DIRECTLY from the v-particle cloud (parabolic-parabolic KS).

In the parabolic-ELLIPTIC model (adaptive_window.chem_force), the chemical v is
SLAVED to u by the screened Poisson solve  -Delta v + v = u, i.e. each Fourier
mode is divided by  lam = |kx|^2 + 1.  In the parabolic-PARABOLIC model used
here, v is a GENUINE dynamic field carried by its own particle cloud; the
chemotactic drift needs  grad_x v  reconstructed straight from the v-cloud
density, with NO elliptic divide.

`grad_v_from_cloud` is a copy of the analytic gradient assembly inside
`adaptive_window.chem_force` with two changes:
  (1) lam = 1            (NO per-mode division by |kx|^2 + 1; v is reconstructed
                          directly as the physical v-density, not screened-solved)
  (2) sign / return       returns +grad_x v   (chem_force returns -chi * grad v;
                          here the caller multiplies by chi to form the drift).

DIMENSIONAL CONSISTENCY (identical to chem_force, verified there as
`verify_gaussian`):
  map  x = x_c + (L/pi) y,  y in [-pi,pi]^2.
  The v-cloud Fourier coeffs (density_coeffs_y) are for a PROBABILITY density on
  y, integrating to 1.  The physical v-field is
        v(x)        = mass_v * (pi/L)^2 * rho_y(y(x)),
        grad_x v(x) = mass_v * (pi/L)^2 * (pi/L) * grad_y rho_y(y(x)).
  We carry  scale = mass_v * (pi/L)^2  on the coefficients and let the analytic
  derivative supply the extra (pi/L) per spatial direction via
        kx = (pi/L) * fk,    d/dx cos(k(y+pi)) = -kx sin(k(y+pi)),
  exactly as chem_force does.  So setting lam=1 and scale=mass_v*(pi/L)^2 yields
  the true  grad_x v  of the reconstructed physical v-field.

PHASE-SHIFT CONSISTENCY.  density_coeffs_y builds the basis cos(k(y+pi)),
sin(k(y+pi)) (the +pi shift at adaptive_window.py:78-79).  The assembly below
reproduces that SAME +pi phase shift (phx = fk*pi; Cx,Sx,Cy,Sy rotations) so the
reconstructed gradient is consistent with the stored coefficients.

NOTE (high-mode noise).  grad_v_from_cloud differentiates a RAW particle Fourier
reconstruction with NO 1/lam smoothing (unlike the elliptic solve, which damps
mode k by 1/(|kx|^2+1)).  Differentiation amplifies high modes by |kx| ~ k, so a
large K with a finite particle count injects Monte-Carlo high-mode noise into the
drift.  To regularize this we apply a SMOOTH separable Gaussian low-pass taper
w(k) = exp(-(k/(K-1))^2 / (2 s^2)) to the v-coefficients before assembling the
gradient (see `lowpass_taper`; default width s=0.5, plumbed as --filter_s in the
driver; s>>1 disables it).  Keep K modest (e.g. K<=8-10) for the
parabolic-parabolic drift.  This is flagged for Codex review.

This file imports only the LOCAL vendored adaptive_window.py (jax/numpy only).
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from adaptive_window import (
    compute_window,
    density_coeffs_y,
    eval_density_y,
    _freqs,
)


# ---------------------------------------------------------------------------
# Smooth spectral low-pass taper (high-mode-noise regularization).
# ---------------------------------------------------------------------------
def lowpass_taper(K, s):
    """Separable Gaussian low-pass taper weight w(kx)*w(ky) on a (K,K) tensor.

    Per-axis weight for integer mode k = 0..K-1:
        w(k) = exp( -(k/(K-1))^2 / (2 s^2) ),
    so the lowest mode (k=0) is undamped (w=1) and the highest resolved mode
    (k=K-1) is damped by exp(-1/(2 s^2)).  A SMALL s gives an aggressive taper;
    a LARGE s (e.g. s=1e3) makes w(k) ~ 1 for all k, effectively DISABLING the
    filter (used for the filtered-vs-unfiltered comparison).

    This is a DELIBERATE regularization that replaces the 1/(|kx|^2 + 1)
    smoothing absent from the lam=1 direct v-reconstruction: differentiating a
    raw particle Fourier reconstruction amplifies mode k by |kx| ~ k and injects
    Monte-Carlo high-mode noise into the chemotactic drift; this taper damps it.

    Returns a (K,K) tensor w(kx)*w(ky) to multiply the v-coefficients by.
    """
    fk = _freqs(K)
    denom = jnp.maximum(K - 1, 1)                    # avoid /0 if K==1
    w1 = jnp.exp(-((fk / denom) ** 2) / (2.0 * s ** 2))   # (K,)
    return w1[:, None] * w1[None, :]                 # (K,K) separable


# ---------------------------------------------------------------------------
# grad of v reconstructed DIRECTLY from the v-cloud (no elliptic divide).
# ---------------------------------------------------------------------------
def grad_v_from_cloud(X_eval, coeff_v, x_c, L, mass_v, taper_s=0.5):
    """Return +grad_x v  at each evaluation point X_eval (shape (N,2)).

    v is the physical chemical field reconstructed DIRECTLY from the v-cloud:
        v(x) = mass_v * (pi/L)^2 * rho_y(y),   y = (x - x_c) * (pi/L),
    with rho_y the PROBABILITY density whose Fourier coeffs are `coeff_v`
    (from density_coeffs_y on the v-cloud mapped into the SAME window built from
    the u-cloud).  This mirrors adaptive_window.chem_force EXACTLY except:
        lam = 1            (no screened-Poisson per-mode division), and
        returns +grad v    (no -chi factor; caller forms chi * grad v).

    `taper_s` (default 0.5) sets the width of a SMOOTH separable Gaussian
    low-pass taper applied to the v-coefficients BEFORE the gradient is
    assembled (see `lowpass_taper`); it damps Monte-Carlo high-mode noise that
    differentiation would otherwise amplify.  Set taper_s large (e.g. 1e3) to
    effectively disable the filter.
    """
    K = coeff_v["K"]
    fk = _freqs(K)                                   # integer modes 0..K-1
    kx = (jnp.pi / L) * fk                           # physical wavenumbers (1D)
    # lam = 1  -> NO division by (|kx|^2 + 1).  v is reconstructed directly.
    lam = 1.0

    # smooth spectral low-pass taper (deliberate high-mode-noise regularization)
    taper = lowpass_taper(K, taper_s)                # (K,K), separable w(kx)w(ky)

    # physical-v Fourier coefficients = mass_v * (pi/L)^2 * (probability coeffs)
    scale_v = mass_v * (jnp.pi / L) ** 2
    Vcc = scale_v * taper * coeff_v["cos-cos"] / lam
    Vcs = scale_v * taper * coeff_v["cos-sin"] / lam
    Vsc = scale_v * taper * coeff_v["sin-cos"] / lam
    Vss = scale_v * taper * coeff_v["sin-sin"] / lam

    # Map evaluation points into y and assemble grad_x v analytically.
    Y = (X_eval - x_c) * (jnp.pi / L)                # (N,2)
    ax = fk[None, :] * Y[:, 0:1]                      # (N,K)
    ay = fk[None, :] * Y[:, 1:2]                      # (N,K)
    cax, sax = jnp.cos(ax), jnp.sin(ax)
    cay, say = jnp.cos(ay), jnp.sin(ay)
    # +pi phase shift to match density_coeffs_y's cos(k(y+pi)), sin(k(y+pi)).
    phx = fk * jnp.pi
    cph, sph = jnp.cos(phx), jnp.sin(phx)
    Cx = cax * cph[None, :] - sax * sph[None, :]      # cos(k(y_x+pi))
    Sx = sax * cph[None, :] + cax * sph[None, :]      # sin(k(y_x+pi))
    Cy = cay * cph[None, :] - say * sph[None, :]
    Sy = say * cph[None, :] + cay * sph[None, :]
    # d/dx cos(k(y+pi)) (physical x) = -kx sin(k(y+pi)); the (pi/L) per-derivative
    # is carried by kx = (pi/L) fk.
    dCx = -kx[None, :] * Sx
    dSx = kx[None, :] * Cx
    dCy = -kx[None, :] * Sy
    dSy = kx[None, :] * Cy

    # v   = sum Vcc Cx Cy + Vcs Cx Sy + Vsc Sx Cy + Vss Sx Sy
    # dv/dx differentiates the x-basis; dv/dy differentiates the y-basis.
    gx = (jnp.einsum('nk,nl,kl->n', dCx, Cy, Vcc)
          + jnp.einsum('nk,nl,kl->n', dCx, Sy, Vcs)
          + jnp.einsum('nk,nl,kl->n', dSx, Cy, Vsc)
          + jnp.einsum('nk,nl,kl->n', dSx, Sy, Vss))
    gy = (jnp.einsum('nk,nl,kl->n', Cx, dCy, Vcc)
          + jnp.einsum('nk,nl,kl->n', Cx, dSy, Vcs)
          + jnp.einsum('nk,nl,kl->n', Sx, dCy, Vsc)
          + jnp.einsum('nk,nl,kl->n', Sx, dSy, Vss))
    grad_v = jnp.stack([gx, gy], axis=1)             # (N,2)  = +grad_x v
    return grad_v


# ---------------------------------------------------------------------------
# Reconstructed physical-field helpers (peak / L2) on the adaptive window.
# Used for u-diagnostics (u-cloud) and for the gradv self-test.
# ---------------------------------------------------------------------------
def recon_peak(coeff_y, x_c, L, mass, n_grid=129):
    """Reconstructed physical-field peak  ||P_K rho||_inf  on the window.

    rho_y is the probability density (coeff_y); physical field = mass*(pi/L)^2*rho_y.
    Identical convention to adaptive_window.peak_density (DIAGNOSTIC only).
    """
    g = jnp.linspace(-jnp.pi, jnp.pi, n_grid)
    GX, GY = jnp.meshgrid(g, g, indexing="ij")
    pts = jnp.stack([GX.ravel(), GY.ravel()], axis=1)
    rho_y = jax.vmap(eval_density_y, in_axes=(0, None))(pts, coeff_y)
    field = mass * (jnp.pi / L) ** 2 * rho_y
    return jnp.max(field)


def recon_l2(coeff_y, x_c, L, mass, n_grid=129):
    """Reconstructed physical-field L2 norm  ||P_K field||_{L2(window)}.

    L2 over the physical window of side 2L:  ( integral field(x)^2 dx )^{1/2}.
    field(x) = mass*(pi/L)^2*rho_y(y),  dx = (L/pi)^2 dy,  y-grid on [-pi,pi]^2.
    """
    g = jnp.linspace(-jnp.pi, jnp.pi, n_grid)
    GX, GY = jnp.meshgrid(g, g, indexing="ij")
    pts = jnp.stack([GX.ravel(), GY.ravel()], axis=1)
    rho_y = jax.vmap(eval_density_y, in_axes=(0, None))(pts, coeff_y)
    field = mass * (jnp.pi / L) ** 2 * rho_y
    dy = (g[1] - g[0]) ** 2                           # dy area element on y-grid
    dx_area = (L / jnp.pi) ** 2                        # |det dx/dy|
    return jnp.sqrt(jnp.sum(field ** 2) * dy * dx_area)


def recon_field_grid(coeff_y, x_c, L, mass, n_grid=129):
    """Return (x_grid_1d, field_2d) physical reconstruction for snapshot saving.

    x_grid_1d are PHYSICAL coordinates along one axis (relative to x_c offset
    added back), field_2d = mass*(pi/L)^2*rho_y on the (n_grid,n_grid) window.
    """
    g = jnp.linspace(-jnp.pi, jnp.pi, n_grid)
    GX, GY = jnp.meshgrid(g, g, indexing="ij")
    pts = jnp.stack([GX.ravel(), GY.ravel()], axis=1)
    rho_y = jax.vmap(eval_density_y, in_axes=(0, None))(pts, coeff_y)
    field = (mass * (jnp.pi / L) ** 2 * rho_y).reshape(n_grid, n_grid)
    x_phys = x_c[0] + (L / jnp.pi) * g                # physical x along axis 0
    y_phys = x_c[1] + (L / jnp.pi) * g                # physical y along axis 1
    return x_phys, y_phys, field


# ---------------------------------------------------------------------------
# Self-test (WRITTEN, NOT RUN): grad_v_from_cloud vs an analytic single-mode v,
# plus a finite-difference cross-check.  Run only behind Codex verification.
# ---------------------------------------------------------------------------
def selftest_gradv(seed=0, N=2000000, K=8, k_mode=2):
    """Check grad_v_from_cloud against an analytic single-mode v.

    Construct a v-cloud whose density is  rho(x) ~ (1 + cos(k_mode * x1)) / Z  on
    the window so that the reconstructed physical v has a known analytic gradient.
    Two checks:
      (A) ANALYTIC single-mode: sample points y in [-pi,pi]^2 from the probability
          density proportional to (1 + cos(k_mode (y1+pi)))/((2pi)^2) (separable,
          uniform in y2).  Its physical v on the window is
              v(x) = mass_v * (pi/L)^2 * (1 + cos(k_mode (y1+pi)))/(2pi)^2,
          so  dv/dx1 = -mass_v*(pi/L)^2 * (pi/L) * k_mode sin(k_mode(y1+pi))/(2pi)^2,
          dv/dx2 = 0.  Compare grad_v_from_cloud to this analytic gradient.
      (B) FINITE-DIFFERENCE: compare grad_v_from_cloud to central differences of a
          scalar v-evaluator built from the SAME coeffs (lam=1, scale=mass_v).

    Returns a dict of max relative errors; ALL reconstruction noise scales like
    1/sqrt(N), so use large N.  NOT executed here.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    mass_v = 10.0 * np.pi

    # ----- build a v-cloud on a FIXED window (x_c=0, L=pi so y == x) -----------
    x_c = jnp.zeros(2)
    L = jnp.pi                                         # then y = (x-0)*(pi/L)=x
    # sample y1 from p(y1) ~ (1 + cos(k_mode (y1+pi)))/(2pi) on [-pi,pi] via
    # rejection; y2 uniform on [-pi,pi].
    def sample_y1(n):
        out = []
        got = 0
        while got < n:
            cand = rng.uniform(-np.pi, np.pi, size=4 * n)
            dens = (1.0 + np.cos(k_mode * (cand + np.pi))) / 2.0   # in [0,1]
            keep = rng.uniform(size=4 * n) < dens
            out.append(cand[keep]); got += int(keep.sum())
        return np.concatenate(out)[:n]
    y1 = sample_y1(N)
    y2 = rng.uniform(-np.pi, np.pi, size=N)
    Yc = jnp.asarray(np.stack([y1, y2], axis=1))      # v-cloud already in y==x
    X_v = x_c + (L / jnp.pi) * Yc                      # physical v-cloud (== Yc)
    coeff_v = density_coeffs_y(Yc, K)

    # ----- evaluation points (a few interior points away from window edge) -----
    Xe = jnp.asarray(rng.uniform(-2.0, 2.0, size=(64, 2)))
    g_recon = grad_v_from_cloud(Xe, coeff_v, x_c, L, mass_v)

    # (A) analytic single-mode gradient
    Ye = (Xe - x_c) * (jnp.pi / L)
    pref = mass_v * (jnp.pi / L) ** 2 * (jnp.pi / L)  # = mass_v*(pi/L)^3
    # rho_y = (1 + cos(k(y1+pi)))/(2 (2pi))  [normalized prob density: y2 uniform]
    # -> d rho_y/dy1 = -k sin(k(y1+pi)) / (2 (2pi)); grad_x = (pi/L) * grad_y * scale
    # combine: dv/dx1 = mass_v*(pi/L)^2 * (pi/L) * d rho_y/dy1
    norm_y = 2.0 * (2.0 * jnp.pi)                      # so integral rho_y dy = 1
    gA_x = pref * (-(k_mode) * jnp.sin(k_mode * (Ye[:, 0] + jnp.pi)) / norm_y)
    gA_y = jnp.zeros_like(gA_x)
    g_analytic = jnp.stack([gA_x, gA_y], axis=1)
    relA = float(jnp.linalg.norm(g_recon - g_analytic)
                 / (jnp.linalg.norm(g_analytic) + 1e-30))

    # (B) finite-difference of a scalar v-evaluator with lam=1
    def v_scalar(point):
        rho = eval_density_y((point - x_c) * (jnp.pi / L), coeff_v)
        return mass_v * (jnp.pi / L) ** 2 * rho
    h = 1e-4
    ex = jnp.array([h, 0.0]); ey = jnp.array([0.0, h])
    gfd = []
    for p in np.asarray(Xe):
        p = jnp.asarray(p)
        dvx = (v_scalar(p + ex) - v_scalar(p - ex)) / (2 * h)
        dvy = (v_scalar(p + ey) - v_scalar(p - ey)) / (2 * h)
        gfd.append([float(dvx), float(dvy)])
    gfd = jnp.asarray(gfd)
    relB = float(jnp.linalg.norm(g_recon - gfd)
                 / (jnp.linalg.norm(gfd) + 1e-30))

    print(f"[selftest_gradv] N={N} K={K} k_mode={k_mode}")
    print(f"[selftest_gradv] (A) vs analytic single-mode: rel-err = {relA:.3e}")
    print(f"[selftest_gradv] (B) vs finite-difference    : rel-err = {relB:.3e}")
    print("[selftest_gradv] (A) ~ 1/sqrt(N) sampling floor; (B) should be ~1e-6.")
    return dict(rel_analytic=relA, rel_fd=relB)


if __name__ == "__main__":
    # NOT auto-run in production; behind Codex verification only.
    selftest_gradv()
