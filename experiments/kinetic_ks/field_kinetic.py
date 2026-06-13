"""
field_kinetic.py
================

Spectral, grid-free (in the x-marginal) screened-Poisson solver for the
field-coupled 6D kinetic Keller-Segel particle method.

Phase space z = (x, v), x in T^3 = [-pi,pi]^3 (period L = 2pi), v in R^3.
Spatial marginal density  rho(t,x) = integral f dv.

Chemical field on the spatial torus T^3:

    - Lap_x c + kappa^2 c = rho - rho_bar ,        rho_bar = (1/(2pi)^3) integral rho dx .

We solve this SPECTRALLY directly from the particle x-positions using a small
number K_x of Fourier modes per spatial dimension (default K_x = 8, i.e. integer
wavenumbers n_j in {-K_x, ..., K_x}).  Nothing dense in v is ever assembled; the
field solve lives entirely in the 3D spatial marginal and is grid-free in the
sense that the density estimator is a direct empirical Fourier sketch over
particle x-positions (no spatial histogram / no FFT grid).

------------------------------------------------------------------------------
EXACT MATH IMPLEMENTED
------------------------------------------------------------------------------
Convention: rho(x) = sum_k rho_hat_k e^{i k . x},  k = (n1,n2,n3),  n_j integer.

Empirical density Fourier coefficients over the ACTIVE particles
(w_i = 1 for branching; general w_i for weighted), normalized so that
  integral_{T^3} rho dx = (2pi)^3 * rho_hat_0 = M_f = (sum_i w_i)/N0 :

    rho_hat_k = (1 / ((2pi)^3 N0)) * sum_{i active} w_i e^{-i k . X_i} .

(Indeed rho_hat_0 = (sum w_i)/( (2pi)^3 N0 ), so (2pi)^3 rho_hat_0 = M_f.)

Screened-Poisson solution coefficients:

    c_hat_k = rho_hat_k / (|k|^2 + kappa^2)   for k != 0 ,
    c_hat_0 = 0                                (GAUGE FIX) .

The k=0 mode of (rho - rho_bar) is exactly zero by construction, so dropping
c_hat_0 (equivalently subtracting the mean of rho) is consistent; we additionally
fix c_hat_0 = 0 so that c has zero spatial mean and the fixed activation
threshold c0 in r[rho,c] is well-defined.

Evaluation at a query point x (real fields, since rho is real => rho_hat_{-k}
= conj(rho_hat_k)):

    c(x)      = Re sum_k c_hat_k e^{i k . x} ,
    grad c(x) = Re sum_k (i k) c_hat_k e^{i k . x} .

We assemble these with REAL cos/sin sums over the half-spectrum to guarantee a
real output and to avoid complex arithmetic over every particle.  Writing
rho_hat_k = (A_k - i B_k) with
    A_k = (1/((2pi)^3 N0)) sum_i w_i cos(k . X_i) ,
    B_k = (1/((2pi)^3 N0)) sum_i w_i sin(k . X_i) ,
and  c_hat_k = rho_hat_k / D_k  with D_k = |k|^2 + kappa^2, we have for a query x

    c(x)      = sum_{k in H} (2 / D_k) [ A_k cos(k.x) + B_k sin(k.x) ] ,
    d c / dx_j(x) = sum_{k in H} (2 k_j / D_k) [ -A_k sin(k.x) + B_k cos(k.x) ] ,

where H is a half-spectrum: one representative of each {k, -k} pair with k != 0
(the factor 2 accounts for the conjugate partner; the k=0 term is dropped by the
gauge fix).  This is the real assembly actually implemented below.

Cost: O(N * |H|) with |H| = ((2K_x+1)^3 - 1)/2 .  Keep K_x small.

jax x64 is enabled by the importing script (experiment_kinetic.py).
"""

import numpy as np
import jax
import jax.numpy as jnp

TWO_PI = 2.0 * np.pi
VOL3 = (2.0 * np.pi) ** 3  # |T^3| spatial-torus volume


# ---------------------------------------------------------------------------
# Half-spectrum of integer wavevectors k = (n1,n2,n3), n_j in {-K..K},
# keeping exactly one of each {k, -k} pair and dropping k = 0.
# ---------------------------------------------------------------------------
def build_half_spectrum(K_x):
    """Return (Kvecs, Dinv) for the spatial screened-Poisson solve.

    Kvecs : (H, 3) int  -- half-spectrum wavevectors (one per +/- pair, no 0).
    Dinv  : (H,)  float -- 1 / (|k|^2 + kappa^2) is applied later; here we return
            |k|^2 so the caller can fold in kappa^2 (kept separate for clarity).

    Selection rule for "one of each +/- pair": keep k if it is lexicographically
    positive, i.e. the first nonzero component is > 0.
    """
    ns = np.arange(-K_x, K_x + 1)
    N1, N2, N3 = np.meshgrid(ns, ns, ns, indexing="ij")
    allk = np.stack([N1.ravel(), N2.ravel(), N3.ravel()], axis=1)  # ((2K+1)^3,3)
    # drop k = 0
    nonzero = np.any(allk != 0, axis=1)
    allk = allk[nonzero]
    # keep lexicographically-positive representative of each {k,-k} pair
    pos = np.zeros(allk.shape[0], dtype=bool)
    for i, k in enumerate(allk):
        for c in k:
            if c > 0:
                pos[i] = True
                break
            if c < 0:
                pos[i] = False
                break
    Kvecs = allk[pos].astype(np.int64)
    ksq = np.sum(Kvecs.astype(np.float64) ** 2, axis=1)  # |k|^2
    return Kvecs, ksq


# ---------------------------------------------------------------------------
# Empirical (half-spectrum) density Fourier coefficients A_k, B_k.
#   A_k = (1/((2pi)^3 N0)) sum_i w_i cos(k.X_i)
#   B_k = (1/((2pi)^3 N0)) sum_i w_i sin(k.X_i)
# rho_hat_k = A_k - i B_k.  Inputs are the FULL buffer + mask (XLA-friendly).
# ---------------------------------------------------------------------------
def density_coeffs(Xbuf, w, mask, N0, Kvecs):
    """Half-spectrum empirical density coefficients from particle x-positions.

    Xbuf  : (M, 3)  spatial coordinates (full buffer; inactive rows ignored).
    w     : (M,)    per-particle weights (1 for branching).
    mask  : (M,)    boolean active flags.
    N0    : initial particle count (mass normalization 1/N0).
    Kvecs : (H, 3)  half-spectrum wavevectors.

    Returns (A, B) each shape (H,), with the (2pi)^3 N0 normalization folded in.
    """
    wm = (w * mask.astype(w.dtype))[:, None]          # (M,1)
    phase = Xbuf @ Kvecs.T                             # (M, H) = k . X_i
    A = jnp.sum(wm * jnp.cos(phase), axis=0) / (VOL3 * N0)
    B = jnp.sum(wm * jnp.sin(phase), axis=0) / (VOL3 * N0)
    return A, B


# ---------------------------------------------------------------------------
# Evaluate c and grad_x c at the SAME particle x-positions.
#   c(x)        = sum_{k in H} (2/D_k)[ A_k cos(k.x) + B_k sin(k.x) ]
#   dc/dx_j(x)  = sum_{k in H} (2 k_j/D_k)[ -A_k sin(k.x) + B_k cos(k.x) ]
# Returns c (M,) and grad_c (M,3), real.
# ---------------------------------------------------------------------------
def eval_field(Xq, A, B, Kvecs, ksq, kappa):
    """Evaluate chemical c and grad_x c at query x-positions Xq (Nq,3).

    A, B  : (H,) half-spectrum density coefficients (from density_coeffs).
    Kvecs : (H,3) wavevectors; ksq : (H,) = |k|^2 ; kappa : screening constant.
    Returns (c, grad_c) with c (Nq,), grad_c (Nq,3).  Real assembly.
    """
    Dk = ksq + kappa * kappa                           # (H,)
    cA = 2.0 * A / Dk                                   # weight of cos(k.x) in c
    cB = 2.0 * B / Dk                                   # weight of sin(k.x) in c
    phase = Xq @ Kvecs.T                                # (Nq, H)
    cosp = jnp.cos(phase)
    sinp = jnp.sin(phase)
    c = cosp @ cA + sinp @ cB                           # (Nq,)
    # gradient: per component j,  sum_k (2 k_j / D_k)[ -A_k sin + B_k cos ]
    Kf = Kvecs.astype(A.dtype)                          # (H,3)
    # coefficient (per mode) of cos and sin in the gradient, before the k_j factor
    gcos = 2.0 * B / Dk                                 # multiplies cos(k.x)
    gsin = -2.0 * A / Dk                                # multiplies sin(k.x)
    # grad_c[:, j] = sum_k k_j ( gcos_k cos + gsin_k sin )
    cos_term = cosp * gcos[None, :]                     # (Nq,H)
    sin_term = sinp * gsin[None, :]                     # (Nq,H)
    mode_term = cos_term + sin_term                     # (Nq,H)
    grad_c = mode_term @ Kf                             # (Nq,3)
    return c, grad_c


# ---------------------------------------------------------------------------
# Spatial marginal density rho at query x (for ||rho||_inf diagnostics and the
# reaction saturation S_rho).  Same half-spectrum assembly:
#   rho(x) = (2pi)^3 rho_hat_0 ... no: rho_hat_0 contributes the mean.
# We keep the mean term M_f / (2pi)^3 explicitly plus the oscillatory half-sum.
#   rho(x) = M_f/(2pi)^3 + sum_{k in H} 2 [ A_k cos(k.x) + B_k sin(k.x) ] .
# ---------------------------------------------------------------------------
def eval_rho(Xq, A, B, Kvecs, M_f):
    """Evaluate the spatial marginal density rho at query x-positions Xq (Nq,3).

    A, B : half-spectrum density coefficients (already (2pi)^3 N0 normalized).
    M_f  : total mass = (sum_i w_i)/N0 (sets the rho_hat_0 mean term).
    Returns rho (Nq,).  NOTE this is a truncated-Fourier reconstruction of a
    sum of Diracs and is only meant as a smooth density proxy; it can be slightly
    negative for under-resolved clouds.  Used for diagnostics and S_rho.
    """
    phase = Xq @ Kvecs.T
    osc = 2.0 * (jnp.cos(phase) @ A + jnp.sin(phase) @ B)
    return M_f / VOL3 + osc


# ===========================================================================
# SELF-TESTS  (WRITTEN, NOT RUN -- execution gated behind Codex cold-verify)
# ===========================================================================
def selftest_field(verbose=True):
    """Analytic single-mode and finite-difference checks of the field solver.

    Test 1 (single-mode solution).  Place particles so that the empirical density
    matches  rho - rho_bar = cos(x1).  The screened-Poisson solution is then
        c(x) = cos(x1) / (1 + kappa^2)
    and  dc/dx1 = -sin(x1)/(1+kappa^2),  dc/dx2 = dc/dx3 = 0.
    We synthesize the half-spectrum coefficients (A,B) DIRECTLY for the single
    mode k = (1,0,0) [A_{100} = 1/2, B = 0 in the rho = cos x1 convention] and
    check eval_field against the closed form.

    Test 2 (gradient sign / chemotaxis).  The chemotactic drift is +chi grad c.
    For a positive density bump centered at x* (rho - rho_bar > 0 near x*),
    c has a maximum near x*, so grad c points TOWARD x* on the near side, and
    +chi grad c moves particles up the chemical gradient (aggregation).  We
    verify the sign of dc/dx1 for rho = cos(x1): c = cos(x1)/(1+kappa^2) has its
    max at x1 = 0, and dc/dx1 = -sin(x1)/(1+kappa^2) < 0 for x1 in (0,pi),
    i.e. grad c points back toward x1 = 0 (toward the density peak).

    Test 3 (finite-difference gradient).  Compare grad c from eval_field to a
    central finite difference of c from eval_field at random points.

    Returns a dict of max-abs errors.  RAISES AssertionError on failure so a
    later (Codex-approved) run surfaces problems loudly.
    """
    kappa = 0.5

    # ---- single mode k = (1,0,0).  rho - rho_bar = cos(x1).
    # In rho(x) = M_f/(2pi)^3 + sum_{H} 2[A_k cos + B_k sin], the cos(x1) mode
    # corresponds to the half-spectrum representative k=(1,0,0) with 2 A_k = 1,
    # i.e. A_{100} = 1/2, all other A,B = 0, and mean term 0 (rho_bar removed).
    Kvecs = np.array([[1, 0, 0]], dtype=np.int64)
    ksq = np.sum(Kvecs.astype(np.float64) ** 2, axis=1)  # [1.0]
    A = jnp.asarray([0.5], dtype=jnp.float64)
    B = jnp.asarray([0.0], dtype=jnp.float64)

    nq = 17
    s = np.linspace(-np.pi, np.pi, nq)
    Xq = np.zeros((nq, 3))
    Xq[:, 0] = s
    Xq = jnp.asarray(Xq)

    c, grad_c = eval_field(Xq, A, B, jnp.asarray(Kvecs), jnp.asarray(ksq), kappa)
    c = np.asarray(c)
    grad_c = np.asarray(grad_c)

    c_exact = np.cos(s) / (1.0 + kappa ** 2)
    g1_exact = -np.sin(s) / (1.0 + kappa ** 2)

    err_c = float(np.max(np.abs(c - c_exact)))
    err_g1 = float(np.max(np.abs(grad_c[:, 0] - g1_exact)))
    err_g23 = float(np.max(np.abs(grad_c[:, 1:])))

    # ---- gradient-sign / chemotaxis check (drift = +chi grad c)
    # at x1 in (0, pi): dc/dx1 < 0 (points toward the peak at x1=0).
    mid = (s > 0.1) & (s < np.pi - 0.1)
    sign_ok = bool(np.all(grad_c[mid, 0] < 0.0))

    # ---- finite-difference gradient check at random points
    rng = np.random.default_rng(0)
    Xr = jnp.asarray(rng.uniform(-np.pi, np.pi, size=(11, 3)))
    h = 1e-4
    cr, gr = eval_field(Xr, A, B, jnp.asarray(Kvecs), jnp.asarray(ksq), kappa)
    gr = np.asarray(gr)
    fd = np.zeros((11, 3))
    for j in range(3):
        ej = np.zeros((1, 3)); ej[0, j] = h
        cp, _ = eval_field(Xr + jnp.asarray(ej), A, B, jnp.asarray(Kvecs),
                           jnp.asarray(ksq), kappa)
        cm, _ = eval_field(Xr - jnp.asarray(ej), A, B, jnp.asarray(Kvecs),
                           jnp.asarray(ksq), kappa)
        fd[:, j] = (np.asarray(cp) - np.asarray(cm)) / (2 * h)
    err_fd = float(np.max(np.abs(fd - gr)))

    out = dict(err_c=err_c, err_grad_x1=err_g1, err_grad_x2x3=err_g23,
               chemotaxis_sign_ok=sign_ok, err_fd_grad=err_fd)
    if verbose:
        print("[selftest_field]", out)

    assert err_c < 1e-10, f"single-mode c error too large: {err_c}"
    assert err_g1 < 1e-10, f"single-mode grad_x1 error too large: {err_g1}"
    assert err_g23 < 1e-12, f"spurious transverse gradient: {err_g23}"
    assert sign_ok, "chemotactic gradient sign wrong (does not point to peak)"
    assert err_fd < 1e-6, f"finite-difference gradient mismatch: {err_fd}"
    return out


def selftest_density_coeffs(verbose=True):
    """Check density_coeffs mass normalization and the empirical->c pipeline.

    Build N particles at known x-positions, all weight 1, and verify that
    (2pi)^3 * rho_hat_0-equivalent = M_f.  Since the half-spectrum drops k=0, we
    check the mean term directly: M_f = N/N0 (here N0 = N => M_f = 1), and that
    eval_rho integrates (via a coarse quadrature) to approximately M_f.
    """
    K_x = 6
    Kvecs, ksq = build_half_spectrum(K_x)
    rng = np.random.default_rng(1)
    N = 4000
    X = rng.normal(scale=0.7, size=(N, 3))
    X = np.mod(X + np.pi, TWO_PI) - np.pi
    Xbuf = jnp.asarray(X)
    w = jnp.ones((N,), dtype=jnp.float64)
    mask = jnp.ones((N,), dtype=bool)
    N0 = N
    A, B = density_coeffs(Xbuf, w, mask, N0, jnp.asarray(Kvecs))
    M_f = float(jnp.sum(w * mask)) / N0  # = 1.0

    # integrate eval_rho over a coarse 16^3 grid to check total mass ~ M_f
    ng = 16
    gs = (np.arange(ng) + 0.5) / ng * TWO_PI - np.pi
    G1, G2, G3 = np.meshgrid(gs, gs, gs, indexing="ij")
    Xg = jnp.asarray(np.stack([G1.ravel(), G2.ravel(), G3.ravel()], axis=1))
    rho = np.asarray(eval_rho(Xg, A, B, jnp.asarray(Kvecs), M_f))
    dvol = (TWO_PI / ng) ** 3
    mass_quad = float(np.sum(rho) * dvol)
    out = dict(M_f=M_f, mass_quad=mass_quad, rel_err=abs(mass_quad - M_f) / M_f)
    if verbose:
        print("[selftest_density_coeffs]", out)
    assert out["rel_err"] < 1e-6, f"mass normalization off: {out}"
    return out


if __name__ == "__main__":
    # NOTE: this block is provided for the Codex-gated verification step only.
    # Do NOT run directly without cold verification.
    jax.config.update("jax_enable_x64", True)
    selftest_field()
    selftest_density_coeffs()
    print("field_kinetic self-tests passed.")
