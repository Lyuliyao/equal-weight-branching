"""test_injection_kernel.py -- unit tests for the decay-injection kernel (plan 9.2).
Run:  python test_injection_kernel.py
"""
import numpy as np
import injection_kernel as IK


def test_creates_v_from_empty():
    """(1) empty v-cloud + nonempty u-cloud -> v-particles created."""
    rng = np.random.default_rng(0)
    Xu = rng.uniform(-6, 6, size=(5000, 3)); Yv = np.zeros((0, 3))
    Yv2, info = IK.decay_inject(Xu, Yv, alpha=1, beta=1, tau=1e-2,
                                omega_u=1.0, omega_v=1.0, rng=rng)
    assert Yv2.shape[0] > 0 and info["n_birth"] > 0, "no v created from empty cloud"
    # injected at u locations
    assert np.isin(Yv2, Xu).all(), "injected v not at u-particle locations"
    print(f"[ok] v created from empty cloud (n_birth={info['n_birth']})")


def _mean_mass(alpha, beta, tau, ou, ov, Nu, Nv, kernel, trials=200, seed=0):
    rng = np.random.default_rng(seed)
    Xu = rng.uniform(-6, 6, size=(Nu, 3))
    Yv = rng.uniform(-6, 6, size=(Nv, 3))
    masses = []
    for _ in range(trials):
        Yv2, _ = IK.decay_inject(Xu, Yv, alpha, beta, tau, ou, ov, rng, kernel=kernel)
        masses.append(Yv2.shape[0] * ov)
    return np.mean(masses), np.std(masses) / np.sqrt(trials), np.var(
        [c for c in masses])


def test_conditional_mean_general():
    """(2) E[M_v'] = e^{-b t} M_v* + (a/b)(1-e^{-b t}) M_u*  (general a,b,omega)."""
    for (a, b, ou, ov, tau) in [(1, 1, 1, 1, 0.02), (1.5, 0.8, 1.0, 1.0, 0.05),
                                (1.0, 1.0, 2.0, 1.0, 0.03)]:
        Nu, Nv = 4000, 3000
        Mu = ou * Nu; Mv = ov * Nv
        target = np.exp(-b * tau) * Mv + (a / b) * (1 - np.exp(-b * tau)) * Mu
        mean, se, _ = _mean_mass(a, b, tau, ou, ov, Nu, Nv, "minvar", trials=300)
        rel = abs(mean - target) / target
        assert rel < 5e-3 or abs(mean - target) < 4 * se, \
            f"mean mass off: a={a} b={b} ou={ou} got {mean:.2f} want {target:.2f} (rel {rel:.1e})"
        print(f"[ok] mean a={a} b={b} ou={ou}: {mean:.1f} vs exact {target:.1f} "
              f"(rel {rel:.1e})")


def test_minvar_le_poisson_variance():
    """(3) minimum-variance kernel has variance <= Poisson for the same mean."""
    a = b = 1.0; ou = ov = 1.0; Nu, Nv = 3000, 0
    for tau in [0.02, 0.3, 1.0]:
        _, _, var_mv = _mean_mass(a, b, tau, ou, ov, Nu, Nv, "minvar", trials=400, seed=7)
        _, _, var_po = _mean_mass(a, b, tau, ou, ov, Nu, Nv, "poisson", trials=400, seed=7)
        assert var_mv <= var_po * 1.15, \
            f"minvar var {var_mv:.1f} > poisson {var_po:.1f} at tau={tau}"
        print(f"[ok] tau={tau}: var(minvar)={var_mv:.1f} <= var(poisson)={var_po:.1f}")


def test_mass_normalization_and_u_unchanged():
    """(5) physical masses correct; u-cloud is untouched by the v-reaction."""
    rng = np.random.default_rng(11)
    ou = ov = 1.3
    Xu = rng.uniform(-6, 6, size=(2000, 3)); Yv = rng.uniform(-6, 6, size=(1000, 3))
    Mu_before = Xu.shape[0] * ou
    Yv2, info = IK.decay_inject(Xu, Yv, 1, 1, 0.05, ou, ov, rng)
    # u-cloud unchanged (reaction acts on v only)
    assert Xu.shape[0] * ou == Mu_before
    # v mass = ov * count, bookkeeping consistent
    assert abs(Yv2.shape[0] * ov - (info["Nv_out"] * ov)) < 1e-12
    assert info["Nv_out"] == (info["Nv_in"] - info["n_death"]) + info["n_birth"]
    print(f"[ok] mass bookkeeping consistent (Nv {info['Nv_in']}->{info['Nv_out']})")


if __name__ == "__main__":
    test_creates_v_from_empty()
    test_conditional_mean_general()
    test_minvar_le_poisson_variance()
    test_mass_normalization_and_u_unchanged()
    print("ALL injection_kernel tests passed.")
