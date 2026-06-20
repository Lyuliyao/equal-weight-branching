"""exact_linear_modes.py -- analytic solution of the LINEAR verification case
(Experiment A): chi=0, D_u=D_v=D, v0=0, smooth wrapped-Gaussian u0.

Then u_t = D Delta u,  v_t = D Delta v + alpha u - beta v, and with q_k = 2 pi k / L,
continuum Fourier coeffs  hat_f_k = (1/L^3) int_box f e^{-i q_k.x} dx :

    hat_u_k(t) = e^{-D|q_k|^2 t} hat_u_k(0),
    hat_v_k(t) = (alpha/beta)(1 - e^{-beta t}) e^{-D|q_k|^2 t} hat_u_k(0).

Wrapped-Gaussian IC of mass M, std sigma, centered at 0:
    hat_u_k(0) = (M / L^3) exp(-|q_k|^2 sigma^2 / 2).

Mass laws (k=0 mode):  M_u(t)=M_u(0),  M_v(t)=(alpha/beta)(1-e^{-beta t})M_u(0).
"""
import numpy as np


def q_vec(k, L):
    return (2.0 * np.pi / L) * np.asarray(k, dtype=np.float64)


def u_hat0_gaussian(k, M, sigma, L):
    """Analytic hat_u_k(0) for u0 = M * wrapped-Gaussian(sigma) at origin."""
    q = q_vec(k, L)
    return (M / L ** 3) * np.exp(-0.5 * np.dot(q, q) * sigma ** 2)


def u_hat(k, t, M, sigma, L, D):
    q = q_vec(k, L)
    return u_hat0_gaussian(k, M, sigma, L) * np.exp(-D * np.dot(q, q) * t)


def v_hat(k, t, M, sigma, L, D, alpha, beta):
    return (alpha / beta) * (1.0 - np.exp(-beta * t)) * u_hat(k, t, M, sigma, L, D)


def mass_u(t, M0):
    return M0


def mass_v(t, M0, alpha, beta, Mv0=0.0):
    """Exact chemical mass: M_v(t) = e^{-beta t} Mv0 + (alpha/beta)(1-e^{-beta t}) M0."""
    return np.exp(-beta * t) * Mv0 + (alpha / beta) * (1.0 - np.exp(-beta * t)) * M0


def analytic_u_hat_grid(t, M, sigma, L, D, K):
    """(Km,Km,Km) complex analytic hat_u_k(t) over integer modes 0..K (one octant)."""
    Km = K + 1
    out = np.zeros((Km, Km, Km), dtype=np.complex128)
    for kx in range(Km):
        for ky in range(Km):
            for kz in range(Km):
                out[kx, ky, kz] = u_hat((kx, ky, kz), t, M, sigma, L, D)
    return out


def analytic_v_hat_grid(t, M, sigma, L, D, alpha, beta, K):
    Km = K + 1
    out = np.zeros((Km, Km, Km), dtype=np.complex128)
    for kx in range(Km):
        for ky in range(Km):
            for kz in range(Km):
                out[kx, ky, kz] = v_hat((kx, ky, kz), t, M, sigma, L, D, alpha, beta)
    return out


def gaussian_real_coeffs(amplitude, sigma_t, L, K):
    """Real cos/sin coeff dict (field3d_fourier convention) of the field
    amplitude * wrapped-Gaussian(sigma_t) centered at 0.  The field is even/real,
    so only the 'ccc' (all-cosine) tensor is nonzero:
        ccc[n] = amplitude * (bw_a bw_b bw_c / L^3) * exp(-|q_n|^2 sigma_t^2 / 2),
    bw = 2 for n>=1 else 1 (the same +/-k folding as density_coeffs).  Passing this
    to field3d_fourier.eval_field / grad_field gives the analytic v / grad v.
    """
    import jax.numpy as jnp
    Km = K + 1
    q = (2.0 * np.pi / L) * np.arange(Km)
    bw = np.full(Km, 2.0); bw[0] = 1.0
    ea = np.exp(-0.5 * q ** 2 * sigma_t ** 2)                 # per-axis factor
    ccc = (amplitude * (bw[:, None, None] * bw[None, :, None] * bw[None, None, :]) / L ** 3
           * (ea[:, None, None] * ea[None, :, None] * ea[None, None, :]))
    zero = jnp.zeros((Km, Km, Km))
    out = {k: zero for k in ("ccc", "ccs", "csc", "css", "scc", "scs", "ssc", "sss")}
    out["ccc"] = jnp.asarray(ccc)
    out["K"] = K; out["L"] = L
    return out


def analytic_u_real_coeffs(t, M, sigma, L, D, K):
    return gaussian_real_coeffs(M, np.sqrt(sigma ** 2 + 2 * D * t), L, K)


def analytic_v_real_coeffs(t, M, sigma, L, D, alpha, beta, K):
    amp = (alpha / beta) * (1.0 - np.exp(-beta * t)) * M
    return gaussian_real_coeffs(amp, np.sqrt(sigma ** 2 + 2 * D * t), L, K)


def mode_l2_error(emp_grid, ana_grid, Ktest):
    """sqrt(sum_{k in 0..Ktest}^3 |emp - ana|^2) over the low-mode octant."""
    Kt = Ktest + 1
    e = emp_grid[:Kt, :Kt, :Kt] - ana_grid[:Kt, :Kt, :Kt]
    return float(np.sqrt(np.sum(np.abs(e) ** 2)))
