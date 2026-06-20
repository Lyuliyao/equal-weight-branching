"""test_exact_linear_case.py -- end-to-end exact linear verification (plan 9.3).

chi=0, Du=Dv=D, v0=0, wrapped-Gaussian u0.  The Lie split is exact-in-expectation
at grid times (equal diffusivities), so the ensemble-mean diagnostics must match the
analytic solution to Monte-Carlo error, with NO systematic time-step bias.
Run:  python test_exact_linear_case.py
"""
import numpy as np
import simulation_pp3d as S
import exact_linear_modes as EX

BASE = dict(ic="radial", L=12.0, Du=1.0, Dv=1.0, alpha=1.0, beta=1.0, chi=0.0,
            M=10.0, sigma=0.45, v0=0.0, K_dyn=6, K_test=3, Nu=20000,
            tau=1e-3, n_steps=40, kernel="minvar")


def _ensemble(cfg, seeds):
    """Mean and standard-error-of-mean over seeds of the last record (t=T)."""
    last = []
    for s in seeds:
        recs, _, _ = S.simulate(cfg, seed=s, diag_every=cfg["n_steps"], record_linear=True)
        last.append(recs[-1])
    keys = last[0].keys()
    mean = {k: float(np.mean([r[k] for r in last])) for k in keys}
    sem = {k: float(np.std([r[k] for r in last]) / np.sqrt(len(seeds))) for k in keys}
    return mean, sem


SEEDS = list(range(8))


def test_mass_law_zero_mode():
    """(1) ensemble-mean M_v(T) matches the exact chemical mass law within 4 SEM
    (MC-calibrated, not a hard tolerance); M_u exactly conserved."""
    m, sem = _ensemble(BASE, SEEDS)
    T = BASE["tau"] * BASE["n_steps"]
    exact = EX.mass_v(T, BASE["M"], BASE["alpha"], BASE["beta"])
    dev = abs(m["M_v"] - exact)
    tol = max(4 * sem["M_v"], 1e-3 * exact)         # 4-sigma MC band
    assert dev < tol, f"M_v(T) {m['M_v']:.4f} vs exact {exact:.4f} dev {dev:.2e} > 4SEM {tol:.2e}"
    assert m["abs_Mu_drift"] < 1e-9, f"M_u drift {m['abs_Mu_drift']:.1e}"
    print(f"[ok] mass law: M_v(T) {m['M_v']:.4f}+-{sem['M_v']:.4f} vs exact {exact:.4f} "
          f"(dev {dev:.1e} < 4SEM {tol:.1e}); M_u drift {m['abs_Mu_drift']:.1e}")


def test_modes_and_gradv():
    """(2,3) low-mode u/v errors and grad-v-at-particle error fall toward the MC
    ~N^{-1/2} rate (4x N -> ~2x smaller) -- require a MEANINGFUL reduction, not just <."""
    lo, _ = _ensemble({**BASE, "Nu": 20000}, SEEDS)
    hi, _ = _ensemble({**BASE, "Nu": 80000}, SEEDS)
    for key in ("E_u_modes", "E_v_modes", "E_grad_v_particles"):
        ratio = lo[key] / max(hi[key], 1e-300)
        assert ratio > 1.4, f"{key} reduction x{ratio:.2f} too weak (expect ~2 for 4x N)"
        print(f"[ok] {key}: N=2e4 {lo[key]:.2e} -> N=8e4 {hi[key]:.2e}  (x{ratio:.2f}, ~2 ideal)")


def test_no_timestep_bias():
    """(4) halving tau (exact split) introduces no bias: mean M_v stays within 4 SEM
    of exact for tau and tau/2."""
    T = BASE["tau"] * BASE["n_steps"]
    exact = EX.mass_v(T, BASE["M"], BASE["alpha"], BASE["beta"])
    a, sa = _ensemble({**BASE, "tau": 1e-3, "n_steps": 40}, SEEDS)
    b, sb = _ensemble({**BASE, "tau": 5e-4, "n_steps": 80}, SEEDS)
    da = abs(a["M_v"] - exact); db = abs(b["M_v"] - exact)
    assert da < 4 * sa["M_v"] + 1e-3 * exact and db < 4 * sb["M_v"] + 1e-3 * exact, \
        f"tau bias: dev(tau)={da:.2e} (4SEM {4*sa['M_v']:.2e}), dev(tau/2)={db:.2e}"
    print(f"[ok] no time-step bias: M_v dev  tau {da:.1e}(4SEM {4*sa['M_v']:.1e}),  "
          f"tau/2 {db:.1e}(4SEM {4*sb['M_v']:.1e})")


def test_v_from_empty():
    """v is genuinely created from the empty initial cloud."""
    recs, summ, _ = S.simulate(BASE, seed=0, diag_every=10, record_linear=True)
    assert recs[0]["N_v"] == 0, "v-cloud should start empty"
    assert recs[-1]["N_v"] > 0, "v never created"
    assert summ["max_v_occupancy"] > 0
    print(f"[ok] v created from empty cloud: N_v 0 -> {recs[-1]['N_v']} "
          f"(max occ {summ['max_v_occupancy']}, cap-free)")


if __name__ == "__main__":
    test_v_from_empty()
    test_mass_law_zero_mode()
    test_modes_and_gradv()
    test_no_timestep_bias()
    print("ALL exact-linear-case tests passed.")
