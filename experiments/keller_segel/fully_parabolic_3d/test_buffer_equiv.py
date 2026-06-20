"""test_buffer_equiv.py -- verify the JITTED fixed-capacity buffer.

(1) masked normalization: grad_v_buffer(X, Ybuf, mask) == grad_v_from_cloud over the
    ACTIVE subset (physical v = omega * sum_active delta);
(2) fast=True reproduces fast=False on the exact LINEAR case (chi=0);
(3) fast=True reproduces fast=False on a short chi>0 radial trajectory.
Run:  python test_buffer_equiv.py
"""
import numpy as np
import jax.numpy as jnp
import field3d_fourier as F
import simulation_pp3d as S

L = 12.0


def test_masked_normalization():
    rng = np.random.default_rng(0)
    K = 6; n = 5000; Ncap = 8000; omega = 0.37
    Yact = rng.uniform(-L / 2, L / 2, size=(n, 3))
    X = rng.uniform(-L / 2, L / 2, size=(40, 3))
    Ybuf = np.zeros((Ncap, 3)); Ybuf[:n] = Yact
    mask = (np.arange(Ncap) < n).astype(np.float64)
    gb = np.asarray(F.grad_v_buffer(jnp.asarray(X), jnp.asarray(Ybuf), jnp.asarray(mask),
                                    K, L, omega))
    gc = np.asarray(F.grad_v_from_cloud(jnp.asarray(X), jnp.asarray(Yact), K, L, omega * n))
    err = np.abs(gb - gc).max()
    assert err < 1e-9, f"masked buffer grad != cloud grad: {err:.2e}"
    # padding rows must not contribute: change them, result unchanged
    Ybuf2 = Ybuf.copy(); Ybuf2[n:] = rng.uniform(-L / 2, L / 2, size=(Ncap - n, 3))
    gb2 = np.asarray(F.grad_v_buffer(jnp.asarray(X), jnp.asarray(Ybuf2), jnp.asarray(mask),
                                     K, L, omega))
    err2 = np.abs(gb2 - gb).max()
    assert err2 < 1e-12, f"padding rows leaked into grad: {err2:.2e}"
    print(f"[ok] masked normalization: buffer==cloud (err {err:.1e}), padding inert ({err2:.1e})")


def _records_close(a, b, keys, tol):
    bad = []
    for ra, rb in zip(a, b):
        for k in keys:
            if k in ra and k in rb and abs(ra[k] - rb[k]) > tol * (1 + abs(ra[k])):
                bad.append((ra["t"], k, ra[k], rb[k]))
    return bad


def test_linear_fast_equals_slow():
    cfg = dict(experiment="linear", ic="radial", L=L, Du=1.0, Dv=1.0, alpha=1.0, beta=1.0,
               chi=0.0, M=10.0, sigma=0.45, v0=0.0, K_dyn=6, K_test=3, Nu=20000,
               tau=1e-3, n_steps=30, kernel="minvar")
    rs, _, _ = S.simulate(cfg, seed=3, diag_every=10, record_linear=True, fast=False)
    rf, _, _ = S.simulate(cfg, seed=3, diag_every=10, record_linear=True, fast=True)
    keys = ["M_u", "M_v", "E_u_modes", "E_v_modes", "E_grad_v_particles", "N_v"]
    bad = _records_close(rs, rf, keys, 1e-8)
    assert not bad, f"linear fast!=slow: {bad[:3]}"
    print(f"[ok] linear: fast == slow (all {keys} match to 1e-8; N_v {rs[-1]['N_v']})")


def test_radial_fast_equals_slow():
    cfg = dict(experiment="radial", ic="radial", L=L, Du=1.0, Dv=1.0, alpha=1.0, beta=1.0,
               chi=1.0, M=64.0, sigma=0.45, v0=0.0, K_dyn=8, K_test=4, Nu=20000,
               tau=1e-3, n_steps=60, kernel="minvar")
    rs, _, _ = S.simulate(cfg, seed=5, diag_every=20, fast=False)
    rf, _, _ = S.simulate(cfg, seed=5, diag_every=20, fast=True)
    keys = ["M_v", "N_v", "R_0_5", "R_0_8", "G_v", "drift_resolution_number"]
    bad = _records_close(rs, rf, keys, 1e-7)
    assert not bad, f"radial chi>0 fast!=slow: {bad[:3]}"
    print(f"[ok] radial chi=1,M=64: fast == slow (R_0.5 {rs[-1]['R_0_5']:.4f}, "
          f"N_v {rs[-1]['N_v']}, all to 1e-7)")


if __name__ == "__main__":
    test_masked_normalization()
    test_linear_fast_equals_slow()
    test_radial_fast_equals_slow()
    print("ALL buffer-equivalence tests passed.")
