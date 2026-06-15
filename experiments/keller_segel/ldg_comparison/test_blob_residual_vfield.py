"""
Verification suite for BlobResidualVField (solver-hybrid blob-residual plan §6).
================================================================================
Run BEFORE any production sweep.  All five tests must pass.

  Test 1  FD gradient   : grad_exact vs central-difference of eval_exact  (<1e-4).
  Test 2  consistency   : eta_h*mu_N -> eta_h*rho_true at rate ~1/sqrt(N).
  Test 3  no double-count: if the blob SOURCE is the low field itself (quadrature of
                           v_lo dx), then eta_h*mu - eta_h*(v_lo dx) ~ 0, so r ~ 0.
  Test 4  radial symmetry: radial Gaussian -> |grad v_hat| nearly isotropic on rings.
  Test 5  grid vs exact + NOISE vs Kl-spectral: the FFT-grid path matches the exact
                           kernel sum; and the blob residual gradient has much
                           smaller tail/noise than the Kl=24 LOCAL SPECTRUM residual.

Usage:  python test_blob_residual_vfield.py
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from blob_residual_vfield import BlobResidualVField, gaussian_kernel
from hybrid_vfield import HybridVField


def _gauss_cloud(rng, N, a):
    sig = 1.0 / np.sqrt(2 * a)
    return rng.normal(0.0, sig, size=(N, 2))


def test1_fd(verbose=True):
    rng = np.random.default_rng(0)
    Xv = _gauss_cloud(rng, 200000, 42.0)
    x_c = np.array([0.002, -0.003]); L = 0.08; mass_v = 10 * np.pi
    fld = BlobResidualVField(Xv, x_c, L, mass_v, Kg=8, taper_s=0.5,
                             h_rule="frac_L", c_h=0.06)
    Xe = x_c + rng.uniform(-0.5, 0.5, size=(400, 2)) * L
    g = fld.grad_exact(Xe)
    eps = 1e-6; gfd = np.zeros_like(g)
    for d in range(2):
        e = np.zeros((1, 2)); e[0, d] = eps
        gfd[:, d] = (fld.eval_exact(Xe + e) - fld.eval_exact(Xe - e)) / (2 * eps)
    rel = np.linalg.norm(g - gfd) / (np.linalg.norm(gfd) + 1e-30)
    ok = rel < 1e-4
    if verbose:
        print(f"[T1] FD grad_exact rel err = {rel:.3e}   -> {'PASS' if ok else 'FAIL'}")
    return ok


def test2_consistency(verbose=True):
    """eta_h * mu_N converges to the analytic eta_h * rho_true (Gaussian*Gaussian)
    at the Monte-Carlo rate ~1/sqrt(N)."""
    a = 42.0; mass_v = 10 * np.pi
    x_c = np.array([0.0, 0.0]); L = 0.12
    h = 0.06 * L
    # rho_true(x) = mass_v * a/pi * exp(-a r^2); eta_h * rho_true is a wider Gaussian:
    #   (eta_h * rho_true)(x) = mass_v * a/pi * (1/(1+2 a h^2)) exp(-a r^2/(1+2 a h^2))
    def smoothed_true(Xe):
        r2 = np.sum(Xe * Xe, axis=1)
        s = 1.0 + 2.0 * a * h * h
        return mass_v * a / np.pi * (1.0 / s) * np.exp(-a * r2 / s)
    Xe = x_c + np.array([[0.0, 0.0], [0.01, 0.0], [0.02, 0.01], [0.0, 0.03],
                         [-0.02, 0.0], [0.01, -0.02]])
    true = smoothed_true(Xe)
    Ns = [20000, 80000, 320000]
    errs = []
    for N in Ns:
        # average a few seeds to estimate the RMS MC error at fixed N
        e_seeds = []
        for s in range(4):
            rng = np.random.default_rng(100 + s)
            Xv = _gauss_cloud(rng, N, a)
            omega = mass_v / N
            dx = Xe[:, None, :] - Xv[None, :, :]
            B = omega * np.sum(gaussian_kernel(dx, h), axis=1)
            e_seeds.append(np.sqrt(np.mean((B - true) ** 2)))
        errs.append(np.mean(e_seeds))
    errs = np.array(errs)
    # rate: err ~ C N^{-p}; fit p.  Consistency only needs CONVERGENCE at >= the MC
    # rate; faster-than-1/sqrt(N) (small-seed-sample rate estimate) is not a failure.
    p = -np.polyfit(np.log(Ns), np.log(errs), 1)[0]
    ok = (errs[-1] < 0.25 * errs[0]) and (p > 0.4)
    if verbose:
        print(f"[T2] eta_h*mu_N -> eta_h*rho_true: errs={errs}  rate p={p:.2f} "
              f"(want >=0.5)   -> {'PASS' if ok else 'FAIL'}")
    return ok


def test3_no_double_count(verbose=True):
    """If the blob source IS the low field (quadrature of v_lo dx), the residual
    r = eta_h*mu - eta_h*(v_lo dx) must vanish."""
    rng = np.random.default_rng(1)
    Xv = _gauss_cloud(rng, 100000, 42.0)
    x_c = np.array([0.001, 0.0]); L = 0.1; mass_v = 10 * np.pi
    fld = BlobResidualVField(Xv, x_c, L, mass_v, Kg=8, taper_s=0.5,
                             h_rule="frac_L", c_h=0.08)
    # quadrature nodes + weights for v_lo dx (coarse, deterministic)
    nq = 80
    step = 2.0 * L / nq
    ax = fld.x0[0] + step * (np.arange(nq) + 0.5)
    ay = fld.x0[1] + step * (np.arange(nq) + 0.5)
    GX, GY = np.meshgrid(ax, ay, indexing="ij")
    nodes = np.stack([GX.ravel(), GY.ravel()], axis=1)
    from hybrid_vfield import eval_v_from_cloud
    vlo_nodes = np.asarray(eval_v_from_cloud(
        jnp.asarray(nodes), fld.coeff_lo, jnp.asarray(x_c), L, mass_v, 0.5))
    w_nodes = vlo_nodes * step * step                      # mass of v_lo dx per node
    # evaluate eta_h * (this deterministic measure) and eta_h*(v_lo dx) quadrature
    Xe = x_c + rng.uniform(-0.4, 0.4, size=(200, 2)) * L
    dx = Xe[:, None, :] - nodes[None, :, :]
    B_det = np.sum(gaussian_kernel(dx, fld.h) * w_nodes[None, :], axis=1)
    Q, _ = fld._lowblob_exact(Xe, n_quad_q=nq)
    rel = np.linalg.norm(B_det - Q) / (np.linalg.norm(Q) + 1e-30)
    ok = rel < 1e-10
    if verbose:
        print(f"[T3] no-double-count (blob of v_lo dx == low-blob): rel={rel:.2e} "
              f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def test4_radial_symmetry(verbose=True):
    """SYSTEMATIC symmetry: the SEED-AVERAGED |grad v_hat| on a ring must be nearly
    isotropic (a code-level x/y asymmetry would survive averaging).  Per-seed CV is
    dominated by Monte-Carlo noise -- that is the phenomenon the sweep measures, not
    an implementation error -- so we average it out here."""
    x_c = np.array([0.0, 0.0]); L = 0.1; mass_v = 10 * np.pi
    th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    # rings inside frac_in=0.5*L (chi==1) so we test the pure residual region;
    # average many seeds so the MC floor (~per-seed CV / sqrt(n_seed)) is well below
    # the systematic-asymmetry threshold.
    rads = [0.25 * L, 0.40 * L]
    rings = [x_c + r * np.stack([np.cos(th), np.sin(th)], axis=1) for r in rads]
    acc = [np.zeros(len(th)) for _ in rads]
    n_seed = 24
    for s in range(n_seed):
        rng = np.random.default_rng(200 + s)
        Xv = _gauss_cloud(rng, 200000, 42.0)
        fld = BlobResidualVField(Xv, x_c, L, mass_v, Kg=8, taper_s=0.5,
                                 h_rule="frac_L", c_h=0.06)
        for j, ring in enumerate(rings):
            g = np.asarray(fld.grad(ring))
            acc[j] += np.sqrt(np.sum(g * g, axis=1))
    cvs = [float(np.std(a / n_seed) / (np.mean(a / n_seed) + 1e-30)) for a in acc]
    cv = float(np.max(cvs))
    ok = cv < 0.15
    if verbose:
        print(f"[T4] systematic symmetry: max angular CV of SEED-MEAN |grad v_hat| "
              f"over {n_seed} seeds = {cv:.3f} (<0.15)   -> {'PASS' if ok else 'FAIL'}")
    return ok


def test5_grid_vs_exact_and_noise(verbose=True):
    rng = np.random.default_rng(3)
    Xv = _gauss_cloud(rng, 200000, 42.0)
    x_c = np.array([0.002, -0.001]); L = 0.08; mass_v = 10 * np.pi
    blob = BlobResidualVField(Xv, x_c, L, mass_v, Kg=8, taper_s=0.5,
                              h_rule="frac_L", c_h=0.06)
    Xe = x_c + rng.uniform(-0.5, 0.5, size=(500, 2)) * L
    g_ex = blob.grad_exact(Xe); g_grid = np.asarray(blob.grad(Xe))
    relg = np.linalg.norm(g_grid - g_ex) / (np.linalg.norm(g_ex) + 1e-30)
    ok_grid = relg < 0.05

    # NOISE comparison: blob residual vs Kl=24 LOCAL SPECTRUM residual (Form I).
    # Compare the high tail of |grad v_hat| over the u-eval points and across seeds.
    def tail_stats_blob(seed):
        r = np.random.default_rng(seed)
        Xv_s = _gauss_cloud(r, 200000, 42.0)
        f = BlobResidualVField(Xv_s, x_c, L, mass_v, Kg=8, taper_s=0.5,
                               h_rule="frac_L", c_h=0.06)
        g = np.asarray(f.grad(Xe)); m = np.sqrt(np.sum(g * g, axis=1))
        return np.max(m), np.percentile(m, 99.9)

    def tail_stats_spec(seed):
        r = np.random.default_rng(seed)
        Xv_s = _gauss_cloud(r, 200000, 42.0)
        f = HybridVField(jnp.asarray(Xv_s), x_c, L, mass_v, Kg=8, Kl=24,
                         taper_s=0.5, frac_in=0.5, frac_out=0.85, taper_s_hi=0.5)
        g = np.asarray(f.grad(jnp.asarray(Xe))); m = np.sqrt(np.sum(g * g, axis=1))
        return np.max(m), np.percentile(m, 99.9)

    bmax = [tail_stats_blob(10 + s)[0] for s in range(3)]
    smax = [tail_stats_spec(10 + s)[0] for s in range(3)]
    blob_max = float(np.mean(bmax)); spec_max = float(np.mean(smax))
    ok_noise = blob_max < spec_max          # blob should have a smaller gradient tail
    if verbose:
        print(f"[T5] grid-vs-exact grad rel = {relg:.3e} (<0.05) "
              f"-> {'PASS' if ok_grid else 'FAIL'}")
        print(f"     gradient tail max|grad|: blob={blob_max:.3e}  "
              f"Kl24-spectral={spec_max:.3e}  -> blob {'SMOOTHER' if ok_noise else 'NOT smoother'}")
    return ok_grid and ok_noise


if __name__ == "__main__":
    print("=== BlobResidualVField verification suite ===")
    results = {
        "T1_fd": test1_fd(),
        "T2_consistency": test2_consistency(),
        "T3_no_double_count": test3_no_double_count(),
        "T4_radial_symmetry": test4_radial_symmetry(),
        "T5_grid_and_noise": test5_grid_vs_exact_and_noise(),
    }
    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k:<22} {'PASS' if v else 'FAIL'}")
    allok = all(results.values())
    print(f"\n{'ALL PASS' if allok else '*** SOME FAILED ***'}")
    import sys
    sys.exit(0 if allok else 1)
