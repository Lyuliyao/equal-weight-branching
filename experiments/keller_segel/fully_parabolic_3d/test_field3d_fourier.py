"""test_field3d_fourier.py -- unit tests for the 3D Fourier field (plan section 9.1).
Run:  python test_field3d_fourier.py
"""
import numpy as np
import jax.numpy as jnp
import field3d_fourier as F

L = 12.0
KEYS = F.KEYS


def _zero_coeffs(K, L):
    Km = K + 1
    return {**{k: jnp.zeros((Km, Km, Km)) for k in KEYS}, "K": K, "L": L}


def test_single_mode_gradient():
    """(1) v(x)=cos(q_k . x) with k=(1,0,0) -> grad = (-q1 sin(q1 x1), 0, 0)."""
    K = 4; q1 = 2 * np.pi / L
    coeff = _zero_coeffs(K, L)
    coeff["ccc"] = coeff["ccc"].at[1, 0, 0].set(1.0)        # field = cos(q1 x1)
    X = np.random.default_rng(0).uniform(-L / 2, L / 2, size=(50, 3))
    g = np.asarray(F.grad_field(jnp.asarray(X), coeff))
    ga = np.zeros((50, 3)); ga[:, 0] = -q1 * np.sin(q1 * X[:, 0])
    err = np.abs(g - ga).max()
    assert err < 1e-9, f"single-mode grad err {err:.2e}"
    # field value too
    v = np.asarray(F.eval_field(jnp.asarray(X), coeff))
    assert np.abs(v - np.cos(q1 * X[:, 0])).max() < 1e-9
    print(f"[ok] single-mode gradient (err {err:.1e})")


def test_constant_zero_gradient():
    """(2) constant v -> zero gradient (only ccc[0,0,0] nonzero)."""
    K = 4
    coeff = _zero_coeffs(K, L)
    coeff["ccc"] = coeff["ccc"].at[0, 0, 0].set(3.7)
    X = np.random.default_rng(1).uniform(-L / 2, L / 2, size=(40, 3))
    g = np.abs(np.asarray(F.grad_field(jnp.asarray(X), coeff))).max()
    assert g < 1e-12, f"constant grad not zero: {g:.2e}"
    print(f"[ok] constant -> zero gradient ({g:.1e})")


def test_translation_phase_shift():
    """(3) translating the cloud multiplies hat_f_k by e^{-i q_k.delta}; |hat| unchanged."""
    K = 5; rng = np.random.default_rng(2)
    X = rng.uniform(-L / 2, L / 2, size=(2000, 3)); mass = 7.0
    delta = np.array([0.3, -0.7, 1.1])
    Xs = (X + delta + L / 2) % L - L / 2          # translated cloud, wrapped
    h0 = np.asarray(F.empirical_mode(jnp.asarray(X), mass, K, L))
    h1 = np.asarray(F.empirical_mode(jnp.asarray(Xs), mass, K, L))
    # magnitude preserved
    assert np.abs(np.abs(h0) - np.abs(h1)).max() < 1e-9, "translation changed |hat_f_k|"
    # phase: h1 = h0 * exp(-i q_k . delta)
    Km = K + 1; q = (2 * np.pi / L) * np.arange(Km)
    phase = np.exp(-1j * (q[:, None, None] * delta[0] + q[None, :, None] * delta[1]
                          + q[None, None, :] * delta[2]))
    err = np.abs(h1 - h0 * phase).max()
    assert err < 1e-9, f"phase-shift err {err:.2e}"
    print(f"[ok] translation phase shift (err {err:.1e})")


def test_real_vs_complex():
    """(4) real cos/sin density reconstruction == direct complex-mode reconstruction."""
    K = 4; rng = np.random.default_rng(3)
    X = rng.uniform(-L / 2, L / 2, size=(3000, 3)); mass = 5.0
    Xe = rng.uniform(-L / 2, L / 2, size=(20, 3))
    # real cos/sin reconstruction
    cp = F.density_coeffs(jnp.asarray(X), K, L)
    vr = np.asarray(F.eval_field(jnp.asarray(Xe), {**cp})) * mass
    # complex: hat over full +/-K cube, rho(x) = sum_k hat_k e^{i q_k.x}
    ks = np.arange(-K, K + 1)
    val = np.zeros(Xe.shape[0], dtype=np.complex128)
    for kx in ks:
        for ky in ks:
            for kz in ks:
                q = (2 * np.pi / L) * np.array([kx, ky, kz])
                hk = (mass / (X.shape[0] * L ** 3)) * np.sum(
                    np.exp(-1j * (X @ q)))
                val += hk * np.exp(1j * (Xe @ q))
    err = np.abs(vr - val.real).max()
    assert err < 1e-8, f"real-vs-complex err {err:.2e}"
    print(f"[ok] real cos/sin == complex Fourier (err {err:.1e})")


def test_fd_gradient():
    """(5) analytic spectral grad == central finite difference of the field."""
    K = 6; rng = np.random.default_rng(4)
    X = rng.uniform(-L / 2, L / 2, size=(4000, 3)); mass = 3.3
    cp = F.density_coeffs(jnp.asarray(X), K, L)
    coeff = {**cp}
    Xe = rng.uniform(-L / 4, L / 4, size=(15, 3))
    g = np.asarray(F.grad_field(jnp.asarray(Xe), coeff))
    h = 1e-4; gfd = np.zeros_like(g)
    for j in range(3):
        ep = np.zeros((1, 3)); ep[0, j] = h
        fp = np.asarray(F.eval_field(jnp.asarray(Xe + ep), coeff))
        fm = np.asarray(F.eval_field(jnp.asarray(Xe - ep), coeff))
        gfd[:, j] = (fp - fm) / (2 * h)
    err = np.abs(g - gfd).max()
    assert err < 1e-4, f"FD-vs-analytic grad err {err:.2e}"
    print(f"[ok] analytic grad == finite difference (err {err:.1e})")


if __name__ == "__main__":
    test_single_mode_gradient()
    test_constant_zero_gradient()
    test_translation_phase_shift()
    test_real_vs_complex()
    test_fd_gradient()
    print("ALL field3d_fourier tests passed.")
