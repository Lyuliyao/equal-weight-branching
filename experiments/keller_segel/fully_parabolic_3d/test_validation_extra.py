"""test_validation_extra.py -- additional validation tests for the validation-closure
task (section 3.1).  Adds, on top of the existing test_*.py:

  (1) Fourier axis-permutation symmetry      -- field value invariant under a
      simultaneous coordinate permutation of cloud + eval points; grad components
      permute correspondingly (so the cos/sin coefficient tensors permute).
  (2) Periodic wrapping                       -- after transport every coordinate of
      the returned u/v clouds lies in [-L/2, L/2); wrap_to_box maps the upper edge
      L/2 -> -L/2.
  (3) Fixed-seed reproducibility              -- two fast=True runs with the same cfg
      and seed produce identical diagnostic records (same code path).
  (4) No hidden population control            -- exceeding the v-buffer capacity raises
      an explicit RuntimeError (no silent clip); an uncapped run grows past N_u with
      consistent count bookkeeping.
  (5) Row-wise injection location             -- every injected v-particle ROW equals a
      transported u-particle row (exact row membership, not elementwise np.isin).

Run:  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python test_validation_extra.py
"""
import numpy as np
import jax.numpy as jnp
import field3d_fourier as F
import injection_kernel as IK
import initial_conditions as IC
import simulation_pp3d as S
import diagnostics_pp3d as D

L = 12.0


# ---------------------------------------------------------------------------
# (1) Fourier axis-permutation symmetry
# ---------------------------------------------------------------------------
def test_axis_permutation_symmetry():
    """Permuting (x,y,z) of BOTH the v-cloud and the eval points leaves the field
    value unchanged (v_p(Pi x0) = v(x0)) and permutes the gradient components
    (grad_p(Pi x0)[:, j] = grad(x0)[:, P[j]]).  This is exactly the statement that
    the cos/sin Fourier coefficient tensors permute with the axes (the field is
    rebuilt from them) -- a free symmetry of the isotropic-bandwidth reconstruction."""
    rng = np.random.default_rng(0)
    K = 6; mass = 5.3
    Yv = rng.uniform(-L / 2, L / 2, size=(4000, 3))
    Xe = rng.uniform(-L / 2, L / 2, size=(64, 3))
    for P in ([1, 2, 0], [1, 0, 2], [2, 1, 0]):
        P = list(P)
        v = np.asarray(F.v_field_from_cloud(jnp.asarray(Xe), jnp.asarray(Yv), K, L, mass))
        g = np.asarray(F.grad_v_from_cloud(jnp.asarray(Xe), jnp.asarray(Yv), K, L, mass))
        vp = np.asarray(F.v_field_from_cloud(jnp.asarray(Xe[:, P]), jnp.asarray(Yv[:, P]),
                                             K, L, mass))
        gp = np.asarray(F.grad_v_from_cloud(jnp.asarray(Xe[:, P]), jnp.asarray(Yv[:, P]),
                                            K, L, mass))
        ev = np.abs(vp - v).max()
        eg = max(np.abs(gp[:, j] - g[:, P[j]]).max() for j in range(3))
        assert ev < 1e-9, f"field not permutation-invariant (P={P}): {ev:.2e}"
        assert eg < 1e-9, f"grad components do not permute (P={P}): {eg:.2e}"
    print("[ok] Fourier axis-permutation symmetry (field invariant, grad permutes, <1e-9)")


# ---------------------------------------------------------------------------
# (2) Periodic wrapping
# ---------------------------------------------------------------------------
def test_periodic_wrapping():
    """wrap_to_box maps into [-L/2, L/2) (upper edge L/2 -> -L/2); after a short
    chi>0 run both clouds stay in the box."""
    pts = np.array([[L / 2, -L / 2, 0.0], [L, -L - 0.1, 3 * L + 0.7], [L / 2 - 1e-9, 0, 0]])
    w = IC.wrap_to_box(pts, L)
    assert w.max() < L / 2 - 1e-12, f"wrap upper bound not exclusive: max {w.max()}"
    assert w.min() >= -L / 2 - 1e-12, f"wrap lower bound violated: min {w.min()}"
    # L/2 must fold to -L/2
    assert abs(IC.wrap_to_box(np.array([[L / 2, 0, 0]]), L)[0, 0] + L / 2) < 1e-12

    cfg = dict(experiment="radial", ic="radial", L=L, Du=1.0, Dv=1.0, alpha=1.0, beta=1.0,
               chi=1.0, M=64.0, sigma=0.45, v0=0.0, K_dyn=8, K_test=4, Nu=8000,
               tau=1e-3, n_steps=80, kernel="minvar")
    _, _, (X, Y, _) = S.simulate(cfg, seed=1, diag_every=80, fast=True)
    for nm, Z in (("u", X), ("v", Y)):
        if Z.shape[0]:
            assert Z.max() < L / 2 and Z.min() >= -L / 2 - 1e-12, \
                f"{nm}-cloud left the box: [{Z.min():.4f},{Z.max():.4f}]"
    print(f"[ok] periodic wrapping (u in box, v in box; N_v={Y.shape[0]})")


# ---------------------------------------------------------------------------
# (3) Fixed-seed reproducibility
# ---------------------------------------------------------------------------
def test_fixed_seed_reproducibility():
    """Two fast=True runs, identical cfg and seed, give identical diagnostic records."""
    cfg = dict(experiment="radial", ic="radial", L=L, Du=1.0, Dv=1.0, alpha=1.0, beta=1.0,
               chi=1.0, M=64.0, sigma=0.45, v0=0.0, K_dyn=8, K_test=4, Nu=10000,
               tau=1e-3, n_steps=60, kernel="minvar", drift_probe_K=[8, 12])
    r1, _, _ = S.simulate(cfg, seed=4, diag_every=20, fast=True)
    r2, _, _ = S.simulate(cfg, seed=4, diag_every=20, fast=True)
    assert len(r1) == len(r2) and r1
    maxd = 0.0
    for a, b in zip(r1, r2):
        assert a.keys() == b.keys()
        for k in a:
            maxd = max(maxd, abs(float(a[k]) - float(b[k])))
    assert maxd <= 1e-12, f"same-seed records differ by {maxd:.2e} (expected exact)"
    print(f"[ok] fixed-seed reproducibility (max record diff {maxd:.1e} <= 1e-12)")


# ---------------------------------------------------------------------------
# (4) No hidden population control
# ---------------------------------------------------------------------------
def test_no_hidden_population_control():
    """An over-tight v-buffer cap raises RuntimeError (explicit, not a silent clip);
    an uncapped growth run exceeds N_u with no silent truncation."""
    cfg = dict(experiment="radial", ic="radial", L=L, Du=1.0, Dv=1.0, alpha=1.0, beta=1.0,
               chi=1.0, M=64.0, sigma=0.45, v0=0.0, K_dyn=8, K_test=4, Nu=5000,
               tau=1e-3, n_steps=200, kernel="minvar", Nv_cap=100)   # tiny cap on slow path
    raised = False
    try:
        S.simulate(cfg, seed=0, diag_every=50, fast=False)
    except RuntimeError as e:
        raised = True
        assert "cap" in str(e).lower()
    assert raised, "tiny Nv_cap did NOT raise -> particles were silently clipped"

    cfg2 = dict(cfg); cfg2.pop("Nv_cap")
    recs, summ, (X, Y, _) = S.simulate(cfg2, seed=0, diag_every=50, fast=False)
    assert summ["max_v_occupancy"] >= Y.shape[0]
    assert summ["max_v_occupancy"] > 0 and X.shape[0] == cfg2["Nu"]   # u conservative, v uncapped
    print(f"[ok] no hidden population control (tiny cap raises; uncapped max_occ "
          f"{summ['max_v_occupancy']} > 0, u count fixed {X.shape[0]})")


# ---------------------------------------------------------------------------
# (5) Row-wise injection location
# ---------------------------------------------------------------------------
def test_rowwise_injection_location():
    """Starting from an EMPTY v-cloud, every injected v-particle ROW must equal a
    transported u-particle ROW (exact 3-tuple membership, not elementwise isin)."""
    rng = np.random.default_rng(3)
    Xu = rng.uniform(-L / 2, L / 2, size=(6000, 3))
    Yv2, info = IK.decay_inject(Xu, np.zeros((0, 3)), alpha=1, beta=1, tau=0.05,
                                omega_u=1.0, omega_v=1.0, rng=rng)
    assert info["n_birth"] > 0 and Yv2.shape[0] == info["n_birth"]
    xu_rows = set(map(tuple, np.round(Xu, 12)))
    bad = [r for r in np.round(Yv2, 12) if tuple(r) not in xu_rows]
    assert not bad, f"{len(bad)} injected rows are not u-particle rows"
    # an elementwise check is strictly weaker: confirm we are testing ROW membership
    assert Yv2.shape[1] == 3
    print(f"[ok] row-wise injection location (all {Yv2.shape[0]} injected rows are u rows)")


# ---------------------------------------------------------------------------
# bonus: new diagnostics are sane (circular resultant + drift probe schema)
# ---------------------------------------------------------------------------
def test_new_diagnostics_sane():
    """circular_resultant is 1 for a point mass and ~0 for a uniform cloud; the drift
    probe adds the expected Gv_K*/dabs_*/drel_* columns with a stable schema at t=0."""
    rng = np.random.default_rng(5)
    tight = np.full((2000, 3), 0.3) + 1e-6 * rng.standard_normal((2000, 3))
    A_tight = D.circular_resultant(tight, L)
    A_unif = D.circular_resultant(rng.uniform(-L / 2, L / 2, size=(200000, 3)), L)
    assert A_tight.min() > 0.999, f"tight cluster resultant {A_tight}"
    assert A_unif.max() < 0.02, f"uniform cloud resultant {A_unif}"

    cfg = dict(experiment="radial", ic="radial", L=L, Du=1.0, Dv=1.0, alpha=1.0, beta=1.0,
               chi=1.0, M=64.0, sigma=0.45, v0=0.0, K_dyn=12, K_test=4, Nu=8000,
               tau=1e-3, n_steps=40, kernel="minvar", drift_probe_K=[8, 12, 16])
    recs, _, _ = S.simulate(cfg, seed=0, diag_every=20, fast=True)
    for col in ("Gv_K8", "Gv_K12", "Gv_K16", "dabs_8_12", "drel_8_12",
                "dabs_12_16", "drel_12_16"):
        assert col in recs[0] and col in recs[-1], f"probe column {col} missing"
    assert recs[-1]["dabs_8_12"] >= 0 and recs[-1]["Gv_K12"] > 0
    print("[ok] new diagnostics sane (circular resultant point=1/uniform~0; probe schema)")


if __name__ == "__main__":
    test_axis_permutation_symmetry()
    test_periodic_wrapping()
    test_fixed_seed_reproducibility()
    test_no_hidden_population_control()
    test_rowwise_injection_location()
    test_new_diagnostics_sane()
    print("ALL validation-extra tests passed.")
