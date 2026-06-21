"""simulation_pp3d.py -- 3D fully parabolic-parabolic Keller-Segel particle solver.

  u_t = D_u Lap u - chi div(u grad v),   v_t = D_v Lap v + alpha u - beta v,  on T_L^3.

u-cloud conservative (N_u fixed, mass omega each); v-cloud non-conservative, starts
EMPTY when v0=0 and grows by cross-species injection.  First-order Lie split per step
(plan section 3):
  1. reconstruct grad v at the u-particles from the CURRENT v-cloud (M_v = omega*N_v);
  2. transport u:  X += chi grad v tau + sqrt(2 D_u tau) xi,  wrap;
  3. transport v:  Y += sqrt(2 D_v tau) zeta,  wrap;
  4. exact decay-injection:  mu_v <- e^{-beta tau} mu_v* + (alpha/beta)(1-e^{-beta tau}) mu_u*.

Dynamic numpy v-cloud (no fixed buffer); max occupancy logged and an optional Nv_cap
aborts rather than clips.  No population control / resampling.
"""
import numpy as np
import jax.numpy as jnp

import field3d_fourier as F
import injection_kernel as IK
import initial_conditions as IC
import exact_linear_modes as EX
import diagnostics_pp3d as D


def _init_u(cfg, rng):
    if cfg["ic"] == "radial":
        return IC.sample_wrapped_gaussian(rng, cfg["Nu"], (0, 0, 0), cfg["sigma"], cfg["L"])
    if cfg["ic"] == "tetra":
        X, lab = IC.sample_tetra(rng, cfg["Nu"], cfg["a"], cfg["sigma_c"], cfg["L"])
        return X, lab
    raise ValueError(cfg["ic"])


def _linear_diag(t, X, Y, omega, gv, cfg, Mu0):
    """Experiment-A exact-mode diagnostics (chi=0, Du=Dv=D, v0=0)."""
    L, K, Kt = cfg["L"], cfg["K_dyn"], cfg["K_test"]
    M, sigma, D_, a, b = cfg["M"], cfg["sigma"], cfg["Du"], cfg["alpha"], cfg["beta"]
    Mu = omega * X.shape[0]; Mv = omega * Y.shape[0]
    Mv_ex = EX.mass_v(t, Mu0, a, b)
    # low-mode complex errors vs analytic
    emp_u = np.asarray(F.empirical_mode(jnp.asarray(X), Mu, K, L))
    emp_v = np.asarray(F.empirical_mode(jnp.asarray(Y), Mv, K, L))
    ana_u = EX.analytic_u_hat_grid(t, M, sigma, L, D_, K)
    ana_v = EX.analytic_v_hat_grid(t, M, sigma, L, D_, a, b, K)
    Eu = EX.mode_l2_error(emp_u, ana_u, Kt)
    Ev = EX.mode_l2_error(emp_v, ana_v, Kt)
    # grad v at u-particles vs analytic (same K-truncation & basis)
    gva = np.asarray(F.grad_field(jnp.asarray(X),
                                  EX.analytic_v_real_coeffs(t, M, sigma, L, D_, a, b, K)))
    Egv = float(np.sqrt(np.mean(np.sum((gv - gva) ** 2, axis=1))))
    return dict(M_v_exact=Mv_ex, abs_Mv_error=abs(Mv - Mv_ex),
                rel_Mv_error=abs(Mv - Mv_ex) / max(Mv_ex, 1e-300),
                E_u_modes=Eu, E_v_modes=Ev, E_grad_v_particles=Egv)


def _radial_diag(t, X, Y, gv, cfg):
    """Experiment-B reconstruction-free radial diagnostics.

    If cfg['drift_probe_K'] is a list of bandwidths (e.g. [8,12,16]), additionally
    evaluate grad v at the CURRENT u-positions from the CURRENT v-cloud at each probe
    bandwidth and record the rms magnitude Gv_K{K} plus the pairwise reconstruction
    discrepancy dabs/drel between consecutive bandwidths (validation-closure 4.4).
    This is a same-cloud, diagnostic-only readout: it draws no RNG and does NOT enter
    the transport drift, so the trajectory is identical to a run without the probe."""
    L, K, chi, tau = cfg["L"], cfg["K_dyn"], cfg["chi"], cfg["tau"]
    xc = D.torus_centroid(X, L)
    R = D.core_radii(X, xc, L, (0.2, 0.5, 0.8))
    eig = D.covariance_eigs(X, xc, L)
    hK = L / (2 * K + 1)
    gmax = float(np.max(np.linalg.norm(gv, axis=1))) if gv.shape[0] else 0.0
    out = dict(R_0_2=R[0.2], R_0_5=R[0.5], R_0_8=R[0.8],
               cov_eig0=float(eig[0]), cov_eig1=float(eig[1]), cov_eig2=float(eig[2]),
               drift_resolution_number=float(tau * chi * gmax / hK))
    probe = cfg.get("drift_probe_K")
    if probe:
        ks = sorted(int(kp) for kp in probe)
        omega = cfg["M"] / X.shape[0]                 # omega_u = omega_v
        Mv = omega * Y.shape[0]
        gvs = {}
        for kp in ks:
            if Y.shape[0]:
                g = np.asarray(F.grad_v_from_cloud(jnp.asarray(X), jnp.asarray(Y), kp, L, Mv))
            else:
                g = np.zeros((X.shape[0], 3))
            gvs[kp] = g
            out[f"Gv_K{kp}"] = float(np.sqrt(np.mean(np.sum(g ** 2, axis=1))))
        for klo, khi in zip(ks[:-1], ks[1:]):
            d = gvs[khi] - gvs[klo]
            dabs = float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))
            denom = float(np.sqrt(np.mean(np.sum(gvs[khi] ** 2, axis=1))))   # higher-K ref
            out[f"dabs_{klo}_{khi}"] = dabs
            out[f"drel_{klo}_{khi}"] = dabs / (denom + 1e-30)
    return out


def _tetra_diag(t, X, gv, cfg, labels):
    """Experiment-C tetra cluster diagnostics (4 labelled clusters)."""
    L, K, chi, tau = cfg["L"], cfg["K_dyn"], cfg["chi"], cfg["tau"]
    cen = D.cluster_centroids(X, labels, 4, L)
    R05 = [D.core_radii(X[labels == m], cen[m], L, (0.5,))[0.5] for m in range(4)]
    R09 = [D.core_radii(X[labels == m], cen[m], L, (0.9,))[0.9] for m in range(4)]
    rc = cfg.get("r_center", 1.0)
    omega = cfg["M"] / X.shape[0]
    mcen = D.mass_in_ball(X, (0, 0, 0), rc, L, omega)
    hK = L / (2 * K + 1)
    gmax = float(np.max(np.linalg.norm(gv, axis=1))) if gv.shape[0] else 0.0
    out = dict(d_min=D.d_min(cen, L), overlap=D.overlap_indicator(cen, R05, L),
               m_center=mcen, E_sym=D.symmetry_residual(cen, L),
               drift_resolution_number=float(tau * chi * gmax / hK))
    for m in range(4):
        out[f"R05_c{m}"] = R05[m]; out[f"R09_c{m}"] = R09[m]
        out[f"c{m}_x"] = float(cen[m, 0]); out[f"c{m}_y"] = float(cen[m, 1])
        out[f"c{m}_z"] = float(cen[m, 2])
        # centroid-reliability score: per-axis circular resultant, conservative min_j
        A = D.circular_resultant(X[labels == m], L)
        out[f"A_c{m}_x"] = float(A[0]); out[f"A_c{m}_y"] = float(A[1])
        out[f"A_c{m}_z"] = float(A[2]); out[f"A_c{m}"] = float(A.min())
    # the six pairwise centroid distances (i<j): d01,d02,d03,d12,d13,d23
    for (i, j), dij in zip([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
                           D.pairwise_dists(cen, L)):
        out[f"d{i}{j}"] = float(dij)
    return out


def simulate(cfg, seed, diag_every=None, record_linear=False, fast=False):
    """Run the coupled solver. Returns (records list of dicts, summary dict).

    fast=True swaps ONLY the grad v reconstruction to the JITTED fixed-capacity
    buffer (field3d_fourier.grad_v_buffer): the dynamic v-cloud Y is padded to
    Ncap=ceil(buffer_factor*Nu) each step.  All RNG draws, transport order, and
    the decay-injection (injection_kernel) are identical to fast=False; the only
    difference is the floating-point op order of the grad reduction (eager
    density_coeffs vs fused jit einsum), which is algebraically the same but not
    bitwise.  The two paths therefore agree to FP-roundoff, not byte-for-byte:
    for chi=0 the grad never enters transport so trajectories match to ~1e-12;
    for chi>0 the grad roundoff propagates through the deterministic drift and is
    bounded by the test tolerances (verified in test_buffer_equiv.py)."""
    rng = np.random.default_rng(seed)
    L, Du, Dv = cfg["L"], cfg["Du"], cfg["Dv"]
    alpha, beta, chi, tau, K = cfg["alpha"], cfg["beta"], cfg["chi"], cfg["tau"], cfg["K_dyn"]
    n_steps = cfg["n_steps"]
    diag_every = diag_every or cfg.get("diag_every", max(1, n_steps // 50))

    init = _init_u(cfg, rng)
    labels = None
    if isinstance(init, tuple):
        X, labels = init
    else:
        X = init
    Nu = X.shape[0]
    omega = cfg["M"] / Nu                      # omega_u = omega_v = omega
    Mu0 = omega * Nu
    Y = IC.empty_v_cloud() if cfg.get("v0", 0.0) == 0 else \
        IC.sample_wrapped_gaussian(rng, Nu, (0, 0, 0), cfg["sigma"], L)
    sd_u = np.sqrt(2 * Du * tau); sd_v = np.sqrt(2 * Dv * tau)
    Ncap = int(np.ceil(cfg.get("buffer_factor", 1.6) * Nu)) if fast else None
    Nv_cap = Ncap if fast else cfg.get("Nv_cap", None)
    _arange_cap = np.arange(Ncap) if fast else None

    need_drift = (chi != 0.0)                  # chi=0 (Expt A): grad v only at diag steps
    save_times = sorted(cfg.get("save_times", []) or [])   # times to snapshot raw clouds
    clouds, saved_sv = [], set()
    records, max_occ = [], 0
    for n in range(n_steps + 1):
        t = n * tau
        Mv = omega * Y.shape[0]
        is_diag = (n % diag_every == 0) or (n == n_steps)
        # 1. grad v at u-particles from the current v-cloud (skip if not driving & not diag)
        if (need_drift or is_diag) and Y.shape[0]:
            if fast:
                nv = Y.shape[0]
                Ybuf = np.zeros((Ncap, 3)); Ybuf[:nv] = Y
                mask = (_arange_cap < nv).astype(np.float64)
                gv = np.asarray(F.grad_v_buffer(jnp.asarray(X), jnp.asarray(Ybuf),
                                                jnp.asarray(mask), K, L, omega))
            else:
                gv = np.asarray(F.grad_v_from_cloud(jnp.asarray(X), jnp.asarray(Y), K, L, Mv))
        else:
            gv = np.zeros((Nu, 3))
        if is_diag:
            rec = dict(t=t, seed=seed, N_u=int(Nu), N_v=int(Y.shape[0]),
                       M_u=float(omega * Nu), M_v=float(Mv),
                       M_v_exact=float(EX.mass_v(t, Mu0, alpha, beta)),
                       abs_Mu_drift=float(abs(omega * Nu - Mu0)),
                       G_v=float(np.sqrt(np.mean(np.sum(gv ** 2, axis=1)))),
                       max_v_occupancy=int(max(max_occ, Y.shape[0])))
            exp = cfg.get("experiment", "generic")
            if record_linear or exp == "linear":
                rec.update(_linear_diag(t, X, Y, omega, gv, cfg, Mu0))
            elif exp == "radial":
                rec.update(_radial_diag(t, X, Y, gv, cfg))
            elif exp == "tetra":
                rec.update(_tetra_diag(t, X, gv, cfg, labels))
            records.append(rec)
        # snapshot raw clouds at requested times (for u/v slices; reconstruction-free)
        due = [sv for sv in save_times if sv not in saved_sv and t >= sv - 0.5 * tau]
        if due:
            for sv in due:
                saved_sv.add(sv)
            clouds.append((float(t), X.copy(), Y.copy()))
        if n == n_steps:
            break
        # 2. transport u
        X = IC.wrap_to_box(X + chi * gv * tau + sd_u * rng.standard_normal((Nu, 3)), L)
        # 3. transport v
        if Y.shape[0]:
            Y = IC.wrap_to_box(Y + sd_v * rng.standard_normal((Y.shape[0], 3)), L)
        # 4. exact decay-injection
        Y, info = IK.decay_inject(X, Y, alpha, beta, tau, omega, omega, rng,
                                  kernel=cfg.get("kernel", "minvar"))
        max_occ = max(max_occ, Y.shape[0])
        if Nv_cap is not None and Y.shape[0] > Nv_cap:
            raise RuntimeError(f"v-buffer cap {Nv_cap} exceeded at step {n} "
                               f"(N_v={Y.shape[0]}); rerun with larger capacity.")

    summary = dict(omega=omega, Nu=int(Nu), max_v_occupancy=int(max_occ),
                   Nv_cap=Nv_cap, final_Nv=int(Y.shape[0]), clouds=clouds)
    return records, summary, (X, Y, labels)
