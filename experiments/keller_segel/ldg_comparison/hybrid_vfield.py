"""
Solver-level residual reconstruction of the chemical field v for the
parabolic-parabolic Keller-Segel particle drift (solver-hybrid design note, Form I).
====================================================================================

This is NOT a diagnostic readout: the field built here is used INSIDE the time step
to evaluate the chemotactic drift  b_u(x) = chi * grad v_hat(x)  on the u-particles.

Form I -- two-level spectral residual on the core-adaptive window (x_c, L):

    v_hat(x) = v_lo(x) + chi(x) [ v_hi(x) - v_lo(x) ],

with  v_lo = P_{Kg} mu_v  and  v_hi = P_{Kl} mu_v  (Kl > Kg) both reconstructed from
the SAME v-cloud on the SAME window, and chi a smooth radial cutoff that is 1 in the
core and 0 toward the window edge.  Inside the core chi=1 so v_hat = v_hi (high
bandwidth); near the window edge chi=0 so v_hat = v_lo (low bandwidth, less
Monte-Carlo high-mode noise).  This is exactly the signed residual
  v_lo + chi (P_Kl mu - P_Kg mu)  =  v_lo + chi residual,
since P_Kl(v_lo restricted) = v_lo for Kl>Kg (the Kg modes embed in the Kl basis),
so no global low mass is double counted.

The drift needs the GRADIENT, including the cutoff-derivative term:

    grad v_hat = grad v_lo + chi (grad v_hi - grad v_lo) + (v_hi - v_lo) grad chi.

The (v_hi - v_lo) grad chi term must NOT be dropped.  Verified by finite difference.
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from adaptive_window import density_coeffs_y, _freqs
from field_pp import grad_v_from_cloud, lowpass_taper


def eval_v_from_cloud(X_eval, coeff_v, x_c, L, mass_v, taper_s=0.5):
    """Value of the physical chemical field v(x) at X_eval from the v-cloud Fourier
    coefficients on the window (x_c, L).  Same conventions as grad_v_from_cloud
    (mass_v*(pi/L)^2 scaling, +pi phase shift, low-pass taper), value only."""
    K = coeff_v["K"]
    fk = _freqs(K)
    taper = lowpass_taper(K, taper_s)
    scale_v = mass_v * (jnp.pi / L) ** 2
    Vcc = scale_v * taper * coeff_v["cos-cos"]
    Vcs = scale_v * taper * coeff_v["cos-sin"]
    Vsc = scale_v * taper * coeff_v["sin-cos"]
    Vss = scale_v * taper * coeff_v["sin-sin"]
    Y = (X_eval - x_c) * (jnp.pi / L)
    ax = fk[None, :] * Y[:, 0:1]
    ay = fk[None, :] * Y[:, 1:2]
    cax, sax = jnp.cos(ax), jnp.sin(ax)
    cay, say = jnp.cos(ay), jnp.sin(ay)
    phx = fk * jnp.pi
    cph, sph = jnp.cos(phx), jnp.sin(phx)
    Cx = cax * cph[None, :] - sax * sph[None, :]
    Sx = sax * cph[None, :] + cax * sph[None, :]
    Cy = cay * cph[None, :] - say * sph[None, :]
    Sy = say * cph[None, :] + cay * sph[None, :]
    v = (jnp.einsum('nk,nl,kl->n', Cx, Cy, Vcc)
         + jnp.einsum('nk,nl,kl->n', Cx, Sy, Vcs)
         + jnp.einsum('nk,nl,kl->n', Sx, Cy, Vsc)
         + jnp.einsum('nk,nl,kl->n', Sx, Sy, Vss))
    return v


def radial_taper(X_eval, x_c, L, frac_in=0.5, frac_out=0.85):
    """Smooth radial raised-cosine cutoff chi and its gradient grad chi at X_eval.
    chi = 1 for r <= r_in = frac_in*L ; 0 for r >= r_out = frac_out*L ; smooth between.
    Returns (chi, grad_chi) with grad_chi shape (N,2)."""
    r_in = frac_in * L
    r_out = frac_out * L
    d = X_eval - x_c
    r = jnp.sqrt(jnp.sum(d * d, axis=1)) + 1e-30
    t = jnp.clip((r - r_in) / (r_out - r_in), 0.0, 1.0)
    chi = 0.5 * (1.0 + jnp.cos(jnp.pi * t))                 # 1 at t=0, 0 at t=1
    # dchi/dr = -0.5 pi sin(pi t) / (r_out - r_in), zero outside [r_in, r_out]
    in_band = (r > r_in) & (r < r_out)
    dchi_dr = jnp.where(in_band,
                        -0.5 * jnp.pi * jnp.sin(jnp.pi * t) / (r_out - r_in), 0.0)
    grad_chi = (dchi_dr / r)[:, None] * d                   # (N,2)
    return chi, grad_chi


class HybridVField:
    """Form I two-level spectral residual solver field for v on the window (x_c,L)."""

    def __init__(self, X_v, x_c, L, mass_v, Kg, Kl, taper_s=0.5,
                 frac_in=0.5, frac_out=0.85):
        self.x_c = jnp.asarray(x_c)
        self.L = float(L)
        self.mass_v = float(mass_v)
        self.taper_s = taper_s
        self.frac_in = frac_in
        self.frac_out = frac_out
        Yv = (jnp.asarray(X_v) - self.x_c) * (jnp.pi / self.L)
        self.coeff_lo = density_coeffs_y(Yv, Kg)
        self.coeff_hi = density_coeffs_y(Yv, Kl)

    def grad(self, X_eval):
        X = jnp.asarray(X_eval)
        glo = grad_v_from_cloud(X, self.coeff_lo, self.x_c, self.L, self.mass_v, self.taper_s)
        ghi = grad_v_from_cloud(X, self.coeff_hi, self.x_c, self.L, self.mass_v, self.taper_s)
        vlo = eval_v_from_cloud(X, self.coeff_lo, self.x_c, self.L, self.mass_v, self.taper_s)
        vhi = eval_v_from_cloud(X, self.coeff_hi, self.x_c, self.L, self.mass_v, self.taper_s)
        chi, gchi = radial_taper(X, self.x_c, self.L, self.frac_in, self.frac_out)
        return (glo + chi[:, None] * (ghi - glo) + (vhi - vlo)[:, None] * gchi)

    def eval(self, X_eval):
        X = jnp.asarray(X_eval)
        vlo = eval_v_from_cloud(X, self.coeff_lo, self.x_c, self.L, self.mass_v, self.taper_s)
        vhi = eval_v_from_cloud(X, self.coeff_hi, self.x_c, self.L, self.mass_v, self.taper_s)
        chi, _ = radial_taper(X, self.x_c, self.L, self.frac_in, self.frac_out)
        return vlo + chi * (vhi - vlo)


if __name__ == "__main__":
    # finite-difference verification of grad(v_hat) vs eval(v_hat)
    rng = np.random.default_rng(0)
    a = 42.0
    sig = 1.0 / np.sqrt(2 * a)
    Xv = rng.normal(0.0, sig, size=(200000, 2))
    x_c = np.array([0.002, -0.003]); L = 0.08; mass_v = 10 * np.pi
    fld = HybridVField(Xv, x_c, L, mass_v, Kg=8, Kl=24, taper_s=0.5)
    # evaluate at random points inside the window
    Xe = x_c + (rng.uniform(-0.6, 0.6, size=(2000, 2))) * L
    g = np.asarray(fld.grad(Xe))
    eps = 1e-6
    gfd = np.zeros_like(g)
    for d in range(2):
        e = np.zeros((1, 2)); e[0, d] = eps
        vp = np.asarray(fld.eval(Xe + e)); vm = np.asarray(fld.eval(Xe - e))
        gfd[:, d] = (vp - vm) / (2 * eps)
    rel = np.linalg.norm(g - gfd) / np.linalg.norm(gfd)
    print(f"FD gradient check (Kg=8,Kl=24): rel err = {rel:.3e}  "
          f"(max|g|={np.max(np.abs(g)):.2e})")
    print("PASS" if rel < 1e-4 else "FAIL -- taper-gradient term likely wrong")
