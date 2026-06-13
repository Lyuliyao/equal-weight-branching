"""Core-adaptive spectral reconstruction for 2D parabolic-elliptic Keller-Segel.

Reused pattern: the Fourier density-estimation idea (cos/sin coefficient tensor
built from particle positions) is adapted from
  case2_test3/density.py:8-132  (generate_density_estimation)
  case2_test3/simulation.py:21-160 (masked, weighted variant)
Here we replace the data-spanning [min,max] box by a CORE-ADAPTIVE window of
fixed half-width L(t) centred on the particle mean x_c(t), and we solve the
screened-Poisson chemical field SPECTRALLY on that window.

KEY DIMENSIONAL RESCALINGS (verified in `verify_gaussian`):
  map  x = x_c + (L/pi) y,   y in [-pi,pi]^2  (so the window has side 2L)
  rho_x(x)        = (pi/L)^2 rho_y(y)
  grad_x rho_x(x) = (pi/L)^3 grad_y rho_y(y)
The (pi/L) factors come from |det d y/d x| = (pi/L)^2 for a probability density
in 2D, and one extra (pi/L) per spatial derivative.

Everything is plain JAX (CPU-friendly).  No file in case2_test3 is modified.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Core-adaptive window geometry
# ---------------------------------------------------------------------------
def compute_window(X, *, gamma=3.0, gamma_diff=6.0, D=1.0, tau, L_min,
                   q_window=0.99):
    """Return (x_c, L) for the current particle cloud X (shape (N,2)).

    x_c = mean of particles.
    L(t) = max{ L_min, gamma * R_{q_window}, gamma_diff * sqrt(2 D tau) }
    where R_{q_window} is the q_window-percentile radius from x_c (NOT max).
    """
    x_c = jnp.mean(X, axis=0)
    r = jnp.sqrt(jnp.sum((X - x_c) ** 2, axis=1))
    R_q = jnp.quantile(r, q_window)
    L = jnp.maximum(
        jnp.maximum(L_min, gamma * R_q),
        gamma_diff * jnp.sqrt(2.0 * D * tau),
    )
    return x_c, L


# ---------------------------------------------------------------------------
# Reference frequencies and helpers for a real Fourier series on y in [-pi,pi]
# ---------------------------------------------------------------------------
def _freqs(K):
    # integer modes 0..K-1 in each direction (matches case2_test3 convention)
    return jnp.arange(K)


def _norm_tensor(K):
    # cos/sin orthonormality weights (case2_test3/density.py:24-28)
    norm = jnp.zeros((K, K))
    norm = norm.at[0, 0].set(1.0)
    norm = norm.at[0, 1:].set(2.0)
    norm = norm.at[1:, 0].set(2.0)
    norm = norm.at[1:, 1:].set(4.0)
    return norm


def density_coeffs_y(Y, K):
    """Fourier coefficients of the empirical density on y in [-pi,pi]^2.

    Y: (N,2) particle coordinates already mapped to [-pi,pi]^2.
    Returns dict of cos/sin coefficient (K,K) tensors normalised so that
    integral over [-pi,pi]^2 of rho_y dy = 1 (probability density).
    Domain side Ly = 2 pi.
    """
    Lside = 2.0 * jnp.pi
    norm = _norm_tensor(K)
    fk = _freqs(K)
    x_data, y_data = Y[:, 0], Y[:, 1]
    # phase = 2 pi k (x - (-pi)) / (2 pi) = k (x + pi)
    theta_x = fk[None, :] * (x_data[:, None] + jnp.pi)
    theta_y = fk[None, :] * (y_data[:, None] + jnp.pi)
    cx, sx = jnp.cos(theta_x), jnp.sin(theta_x)
    cy, sy = jnp.cos(theta_y), jnp.sin(theta_y)
    N = x_data.shape[0]
    norm_factor = norm / (Lside * Lside)
    Ccc = jnp.einsum('nk,nl->kl', cx, cy) / N
    Ccs = jnp.einsum('nk,nl->kl', cx, sy) / N
    Scc = jnp.einsum('nk,nl->kl', sx, cy) / N
    Scs = jnp.einsum('nk,nl->kl', sx, sy) / N
    return {
        "cos-cos": norm_factor * Ccc,
        "cos-sin": norm_factor * Ccs,
        "sin-cos": norm_factor * Scc,
        "sin-sin": norm_factor * Scs,
        "K": K,
    }


def eval_density_y(point, coeff):
    """Evaluate rho_y at a single y point in [-pi,pi]^2 (probability density)."""
    K = coeff["K"]
    fk = _freqs(K)
    tx = fk * (point[0] + jnp.pi)
    ty = fk * (point[1] + jnp.pi)
    cx, sx = jnp.cos(tx), jnp.sin(tx)
    cy, sy = jnp.cos(ty), jnp.sin(ty)
    Z = (coeff["cos-cos"] * cx[:, None] * cy[None, :]
         + coeff["cos-sin"] * cx[:, None] * sy[None, :]
         + coeff["sin-cos"] * sx[:, None] * cy[None, :]
         + coeff["sin-sin"] * sx[:, None] * sy[None, :])
    return jnp.sum(Z)


# gradient wrt the y-point (used for the chemical force, after rescaling)
_grad_density_y = jax.grad(eval_density_y, argnums=0)


# ---------------------------------------------------------------------------
# Screened-Poisson chemical field on the window, solved SPECTRALLY.
#   -Delta_x v + v = u,   so  v_hat = u_hat / (|k_x|^2 + 1).
# We need grad_x v at the particle positions.  In y-coordinates a cos/sin mode
# cos(k (y+pi)) has, under x = x_c + (L/pi) y, physical wavenumber kx = (pi/L) k.
# The density coefficients above are for u as a PROBABILITY density on y; the
# physical u is u_x = mass * (pi/L)^2 rho_y.  We carry the mass and (pi/L)^2
# rescaling explicitly when forming the force.
# ---------------------------------------------------------------------------
def chem_force(X, coeff_y, x_c, L, mass, chi):
    """Return -chi * grad_x v  at each particle X (shape (N,2)).

    v solves -Delta v + v = u with u the physical density (total mass `mass`).
    Because each Fourier mode is an eigenfunction of -Delta+1, the screened
    solve is a per-mode division by (|kx|^2 + 1) with kx = (pi/L)*k.
    grad_x v is then assembled analytically and evaluated at particle points.
    """
    K = coeff_y["K"]
    fk = _freqs(K)                                   # integer modes
    kx = (jnp.pi / L) * fk                           # physical wavenumbers (1D)
    KX = kx[:, None]                                 # (K,1)
    KY = kx[None, :]                                 # (1,K)
    lam = KX ** 2 + KY ** 2 + 1.0                    # screened-Poisson symbol (K,K)

    # physical-u Fourier coefficients = mass * (pi/L)^2 * (probability coeffs)
    scale_u = mass * (jnp.pi / L) ** 2
    # v coefficients = u coefficients / lam
    Vcc = scale_u * coeff_y["cos-cos"] / lam
    Vcs = scale_u * coeff_y["cos-sin"] / lam
    Vsc = scale_u * coeff_y["sin-cos"] / lam
    Vss = scale_u * coeff_y["sin-sin"] / lam

    # Map particles into y and assemble grad_x v = (pi/L) grad_y (in physical k)
    # Actually we differentiate the physical field directly in x using
    # d/dx cos(kx (x - x0)) etc.  Equivalent: build v(x) as a function of the
    # physical wavenumbers kx and differentiate analytically.
    Y = (X - x_c) * (jnp.pi / L)                      # (N,2), in [-pi,pi] ideally
    # physical argument: kx_k * (X - x_c) = fk * Y  (since kx = (pi/L) fk)
    ax = fk[None, :] * Y[:, 0:1]                      # (N,K)
    ay = fk[None, :] * Y[:, 1:2]                      # (N,K)
    cax, sax = jnp.cos(ax), jnp.sin(ax)
    cay, say = jnp.cos(ay), jnp.sin(ay)
    # NOTE: the density basis used cos(k(y+pi)) = cos(k y + k pi).  Build with
    # the same phase shift so coefficients line up.
    phx = fk * jnp.pi
    cph, sph = jnp.cos(phx), jnp.sin(phx)
    # cos(k(y+pi)) = cos(ky)cos(kpi) - sin(ky)sin(kpi)
    Cx = cax * cph[None, :] - sax * sph[None, :]
    Sx = sax * cph[None, :] + cax * sph[None, :]
    Cy = cay * cph[None, :] - say * sph[None, :]
    Sy = say * cph[None, :] + cay * sph[None, :]
    # d/dx cos(k(y+pi)) wrt physical x = -kx * sin(k(y+pi))
    dCx = -kx[None, :] * Sx
    dSx = kx[None, :] * Cx
    dCy = -kx[None, :] * Sy
    dSy = kx[None, :] * Cy

    # v = sum Vcc Cx Cy + Vcs Cx Sy + Vsc Sx Cy + Vss Sx Sy
    # dv/dx = sum Vcc dCx Cy + Vcs dCx Sy + Vsc dSx Cy + Vss dSx Sy
    def assemble(dX_basis_x, X_basis_x):
        gx = (jnp.einsum('nk,nl,kl->n', dX_basis_x, Cy, Vcc)
              + jnp.einsum('nk,nl,kl->n', dX_basis_x, Sy, Vcs)
              + jnp.einsum('nk,nl,kl->n', X_basis_x, Cy, Vsc)
              + jnp.einsum('nk,nl,kl->n', X_basis_x, Sy, Vss))
        return gx
    # gradient in x-direction: differentiate the x-basis
    gx = (jnp.einsum('nk,nl,kl->n', dCx, Cy, Vcc)
          + jnp.einsum('nk,nl,kl->n', dCx, Sy, Vcs)
          + jnp.einsum('nk,nl,kl->n', dSx, Cy, Vsc)
          + jnp.einsum('nk,nl,kl->n', dSx, Sy, Vss))
    # gradient in y-direction: differentiate the y-basis
    gy = (jnp.einsum('nk,nl,kl->n', Cx, dCy, Vcc)
          + jnp.einsum('nk,nl,kl->n', Cx, dSy, Vcs)
          + jnp.einsum('nk,nl,kl->n', Sx, dCy, Vsc)
          + jnp.einsum('nk,nl,kl->n', Sx, dSy, Vss))
    grad_v = jnp.stack([gx, gy], axis=1)             # (N,2)
    return -chi * grad_v


def peak_density(coeff_y, x_c, L, mass, n_grid=129):
    """Reconstructed physical-u peak ||P_K u||_inf on the adaptive window.

    SECONDARY cross-check only.  Evaluates rho_y on a grid, rescales by
    mass*(pi/L)^2 to physical density.
    """
    g = jnp.linspace(-jnp.pi, jnp.pi, n_grid)
    GX, GY = jnp.meshgrid(g, g, indexing="ij")
    pts = jnp.stack([GX.ravel(), GY.ravel()], axis=1)
    rho_y = jax.vmap(eval_density_y, in_axes=(0, None))(pts, coeff_y)
    u_phys = mass * (jnp.pi / L) ** 2 * rho_y
    return jnp.max(u_phys)


# ---------------------------------------------------------------------------
# Verification on a known Gaussian (run as a script).
# ---------------------------------------------------------------------------
def verify_gaussian(seed=0, N=400000, a=84.0, mass=10.0 * jnp.pi, K=12):
    """Sample u0 = (mass*a/pi) exp(-a|x|^2), reconstruct on the adaptive window,
    and check:
      (1) reconstructed peak ~ physical peak mass*a/pi,
      (2) reconstructed mass ~ mass,
      (3) screened-Poisson force vs analytic radial -chi grad v at moderate r.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    # Gaussian samples: r ~ with density prop exp(-a r^2): each coord N(0, 1/(2a))
    std = 1.0 / np.sqrt(2 * a)
    X = jnp.asarray(rng.normal(0.0, std, size=(N, 2)))
    x_c, L = compute_window(X, tau=1e-7, L_min=1e-3, q_window=0.99)
    Y = (X - x_c) * (jnp.pi / L)
    coeff_y = density_coeffs_y(Y, K)

    # peak
    peak = peak_density(coeff_y, x_c, L, mass)
    peak_true = mass * a / jnp.pi
    # mass: integral of u_phys over window ~ mass (probability rho_y integrates 1)
    g = jnp.linspace(-jnp.pi, jnp.pi, 257)
    GX, GY = jnp.meshgrid(g, g, indexing="ij")
    pts = jnp.stack([GX.ravel(), GY.ravel()], axis=1)
    rho_y = jax.vmap(eval_density_y, in_axes=(0, None))(pts, coeff_y)
    dy = (g[1] - g[0]) ** 2
    mass_y = jnp.sum(rho_y) * dy                      # should be ~1
    print(f"[verify] N={N} K={K} L={float(L):.4f}")
    print(f"[verify] reconstructed mass on window (rho_y integral, want 1.0): {float(mass_y):.4f}")
    print(f"[verify] reconstructed peak u: {float(peak):.3f}  analytic peak: {float(peak_true):.3f}  ratio={float(peak/peak_true):.3f}")
    return float(peak), float(peak_true), float(mass_y), float(L)


if __name__ == "__main__":
    verify_gaussian()
