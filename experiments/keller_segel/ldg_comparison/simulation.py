"""2D parabolic-PARABOLIC Keller-Segel particle solver -- LDG-style comparison.

Ported (algorithm IDENTICAL) from the working prototype
  Numerical_experiment/Keller_Segel/ks_parabolic_parabolic/simulation_pp2d.py
into the equal-weight-branching repo so this directory is SELF-CONTAINED (it
imports only the locally vendored adaptive_window.py / field_pp.py).

Model (the FULL benchmark used by Li-Shu-Yang LDG / Chertock-Kurganov), on the
plane via a core-adaptive window:
    u_t = Delta u - chi div(u grad v)          (cells;   chi = 1)
    v_t = Delta v + u - v                       (chemical; DYNAMIC, not slaved)
Both diffusions are unit.  This is the parabolic-PARABOLIC model: v is a genuine
dynamic field carried by its OWN particle cloud, evolved by diffusion + the
cross-species INJECTION reaction realizing the EXACT integrator of  v_t = u - v
over tau,  v_{n+1} = e^{-tau} v* + (1-e^{-tau}) u*:
    DECAY: each transported v-particle dies with prob  p = 1 - e^{-tau};
    BIRTH: each transported u-particle spawns a v-particle at its OWN
           post-transport location with prob  p = 1 - e^{-tau}
(with EQUAL per-particle mass for both clouds, q = (1-e^{-tau}) omega_u/omega_v
reduces to a Bernoulli injection with probability 1 - e^{-tau}).

Li-Shu-Yang super-critical Gaussian initial data:
    u0(x) = 840 exp(-84 |x|^2)   (cell mass    int u0 = 840*pi/84 = 10*pi > 8*pi)
    v0(x) = 420 exp(-42 |x|^2)   (chemical mass int v0 = 420*pi/42 = 10*pi)

LDG-STYLE COMPARISON / BOUNDARY DISCLOSURE.  The reference here is a PARTICLE-
FIELD method on a core-adaptive periodic Fourier window, NOT a strict Neumann
method-to-method LDG benchmark.  We use the SAME initial condition and the SAME
reporting times (6e-5, 1.2e-4, 2.0e-4) as the Li-Shu-Yang LDG study.  Boundary
effects are negligible over the very short reporting times because the Gaussian
stays far from the window edge (the window half-width L(t) = gamma * R_q tracks
the collapsing core and the concentrated mass never reaches |y| ~ pi).  We do
NOT claim the singular blow-up time; t_gap is a RESOLUTION-GAP indicator, the
reconstructed peak is BANDWIDTH-SENSITIVE, and the core radii R_0.5, R_0.8 are
RECONSTRUCTION-FREE particle-quantile diagnostics.

DIAGNOSTICS / CSV ALIGNMENT.  diag_*.csv carries, in addition to the prototype
columns, the post-processing columns that tgap.py needs UNCHANGED:
    N        initial per-cloud population (so the (N,K)->(4N,4N? no: 4N,2K) check works)
    K        Fourier modes/axis
    S_L2     alias of S_L2_u (reconstructed physical-u L2 norm S_{K,N}(t))
so tgap.py forms the ratio  S_{2K,4N}(t) / S_{K,N}(t)  with NO edits.

Imports only LOCAL vendored modules (adaptive_window.py, field_pp.py).
"""
import os
import sys
import csv
import json
import time
import shutil
import argparse
import datetime
import subprocess

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from adaptive_window import compute_window, density_coeffs_y  # noqa: E402
from particle_dg_readout import (  # noqa: E402  LDG-matched P1 DG readout (Version A)
    project_particles_to_p1dg, dg_l2_norm, dg_inner_product, dg_peak)
from field_pp import (  # noqa: E402
    grad_v_from_cloud, recon_peak, recon_l2, recon_field_grid,
)


MASS = 10.0 * np.pi   # per-cloud physical mass; per-particle mass = MASS/N0 (BOTH)


# ---------------------------------------------------------------------------
# Gaussian initial-data samplers.  u0 ~ exp(-a |x|^2) => each coord N(0,1/(2a)).
# Mass is tracked by PARTICLE COUNT (count/N0 * MASS), not by the sampler.
# ---------------------------------------------------------------------------
def sample_gauss(rng_np, N, a):
    std = 1.0 / np.sqrt(2.0 * a)
    return jnp.asarray(rng_np.normal(0.0, std, size=(N, 2)))


def quantile_radii(X, x_c, qs):
    r = jnp.sqrt(jnp.sum((X - x_c) ** 2, axis=1))
    return {q: float(jnp.quantile(r, q)) for q in qs}


def core_counts(X, x_c, radii):
    r = jnp.sqrt(jnp.sum((X - x_c) ** 2, axis=1))
    return {q: int(jnp.sum(r <= radii[q])) for q in radii}


# ---------------------------------------------------------------------------
# Transport.  Reconstructs grad v from the V-cloud on the U-window.  Eager
# (coeff_v carries the Python int K; the chemotactic gradient is analytic).
# ---------------------------------------------------------------------------
def make_transport(chi, taper_s):
    def _transport_u(X1, coeff_v, x_c, L, mass_v, tau, xi1):
        gradv = grad_v_from_cloud(X1, coeff_v, x_c, L, mass_v,
                                  taper_s=taper_s)        # +grad v (low-pass tapered)
        return X1 + chi * gradv * tau + jnp.sqrt(2.0 * tau) * xi1

    def _transport_v(X2, tau, xi2):
        return X2 + jnp.sqrt(2.0 * tau) * xi2

    return _transport_u, _transport_v


# ---------------------------------------------------------------------------
# Aliasing guard helpers for the v-cloud reconstruction on the u-window.
# ---------------------------------------------------------------------------
def v_outside_fraction(X2, x_c, L):
    """Fraction of v-particles whose mapped Yv=(X2-x_c)*(pi/L) has max|Yv|>pi."""
    if X2.shape[0] == 0:
        return 0.0, np.zeros(0, dtype=bool)
    Yv = (np.asarray(X2) - np.asarray(x_c)) * (np.pi / float(L))
    outside = np.max(np.abs(Yv), axis=1) > np.pi          # (N2,)
    frac = float(outside.mean())
    return frac, outside


def coeffs_v_on_window(X2, x_c, L, K, N0, mask_outside):
    """Build v-coeffs on the u-window and the effective in-window v-mass."""
    outside_frac, outside = v_outside_fraction(X2, x_c, L)
    N2 = int(X2.shape[0])
    if N2 == 0:
        zero = jnp.zeros((K, K))
        coeff_v = {"cos-cos": zero, "cos-sin": zero,
                   "sin-cos": zero, "sin-sin": zero, "K": K}
        return coeff_v, 0.0, 0.0
    if mask_outside and outside.any():
        in_idx = ~outside
        if not in_idx.any():
            zero = jnp.zeros((K, K))
            coeff_v = {"cos-cos": zero, "cos-sin": zero,
                       "sin-cos": zero, "sin-sin": zero, "K": K}
            return coeff_v, 0.0, outside_frac
        Xin = jnp.asarray(np.asarray(X2)[in_idx])
        Yv = (Xin - x_c) * (jnp.pi / L)
        coeff_v = density_coeffs_y(Yv, K)
        in_frac = 1.0 - outside_frac
        mass_v_eff = (N2 / N0 * MASS) * in_frac           # in-window v mass
    else:
        Yv = (X2 - x_c) * (jnp.pi / L)
        coeff_v = density_coeffs_y(Yv, K)
        mass_v_eff = N2 / N0 * MASS
    return coeff_v, float(mass_v_eff), outside_frac


# ---------------------------------------------------------------------------
# Reproducibility records.
# ---------------------------------------------------------------------------
def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "-C", _HERE, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _pkg_versions():
    vers = {}
    for mod in ("numpy", "jax", "matplotlib"):
        try:
            vers[mod] = __import__(mod).__version__
        except Exception:
            vers[mod] = "unavailable"
    return vers


def write_repro_records(args, outdir, tag, p):
    """Write config.json and manifest.json with full reproducibility metadata."""
    cfg = vars(args).copy()
    cfg.update(dict(
        mass_per_cloud=float(MASS),
        per_particle_mass=float(MASS / args.N),
        p_inject=float(p),
        model="2D parabolic-parabolic KS (v dynamic particle cloud)",
        v_reaction="cross-species injection: v_{n+1}=e^{-tau}v*+(1-e^{-tau})u*",
        drift_sign="+chi grad v (inward, aggregation-driving)",
        gradv_source="v-cloud reconstruction (lam=1, NO elliptic solve)",
        filter_note=("Gaussian spectral low-pass w(k)=exp(-(k/(K-1))^2/(2 s^2)); "
                     "s>>1 disables"),
        ic="Li-Shu-Yang: u0=840 exp(-84|x|^2), v0=420 exp(-42|x|^2), mass 10*pi each",
        reference_note=(
            "PARTICLE-FIELD LDG-STYLE comparison: core-adaptive PERIODIC/WINDOW "
            "Fourier reconstruction, NOT a strict Neumann method-to-method LDG "
            "benchmark. Same initial condition and reporting times as the "
            "Li-Shu-Yang LDG study. Boundary effects are negligible over the very "
            "short reporting times because the Gaussian stays far from the window "
            "edge (concentrated mass never reaches |y| ~ pi)."),
    ))
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    manifest = dict(
        git_commit=_git_hash(),
        command_line=" ".join(sys.argv),
        argv=sys.argv,
        python_version=sys.version.split()[0],
        package_versions=_pkg_versions(),
        resolved_args=vars(args).copy(),
        seed=int(args.seed),
        outdir=os.path.abspath(outdir),
        datetime=datetime.datetime.now().isoformat(),
        population_control="none (equal-weight clouds; v-cloud births/deaths "
                           "from the unbiased injection kernel, no cap)",
        deterministic_reference=(
            "particle-field LDG-STYLE: core-adaptive window Fourier "
            "reconstruction (periodic/window), NOT FD/LDG/Neumann solver"),
        tag=tag,
        report_times=list(args.report_times) if args.report_times else [],
    )
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "snapshots"), exist_ok=True)
    rng_np = np.random.default_rng(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    N0 = args.N                      # initial population of EACH cloud (equal mass)
    chi = float(args.chi)
    tau = float(args.tau)
    n_steps = int(args.n_steps)
    taper_s = float(args.filter_s)   # spectral low-pass width (s>>1 disables it)
    mask_v_outside = bool(args.mask_v_outside)
    cfl_abort = float(args.cfl_abort)
    p = 1.0 - np.exp(-tau)           # per-step death (v) / birth (u->v) probability
    qs = [0.5, 0.8, 0.9, 0.99]

    report_times = sorted(float(t) for t in args.report_times) if args.report_times else []

    # ---- initial clouds (EQUAL per-particle mass = MASS/N0 for BOTH) ---------
    X1 = sample_gauss(rng_np, N0, args.a_u)        # u (cells),   mass MASS = 10 pi
    X2 = sample_gauss(rng_np, N0, args.a_v)        # v (chemical), mass MASS = 10 pi

    transport_u, transport_v = make_transport(chi, taper_s)

    tag = f"N{N0}_K{args.K}_tau{tau:.0e}_q{args.q_window}_seed{args.seed}"

    # ---- reproducibility records (config.json + manifest.json) --------------
    write_repro_records(args, args.outdir, tag, p)
    _readme = os.path.join(_HERE, "README.md")
    if os.path.exists(_readme):
        shutil.copy(_readme, os.path.join(args.outdir, "README_run.md"))

    # diag CSV columns.  In ADDITION to the prototype columns we write:
    #   N, K  -> tgap.py reads these for the (N,K)->(4N,2K) pairing check;
    #   S_L2  -> alias of S_L2_u so tgap.py reads its default column unchanged.
    header = (["step", "t", "N", "K", "M_u", "M_v", "xc_x", "xc_y", "L", "h_eff",
               "S_u", "R2"]
              + [f"R_{q}" for q in qs]                       # R_0.5 R_0.8 R_0.9 R_0.99
              + ["N_0.5", "N_0.8", "N_0.9", "R50_over_heff",
                 "peak_PK_u", "peak_PK_v", "S_L2_u", "S_L2",
                 "outside_v_frac", "drift_cfl", "n_birth", "n_death"])
    dg_ns = list(args.dg_readout_n)
    for nd in dg_ns:                  # Version A: LDG-matched P1 DG readout columns
        header += [f"S_dg_raw_{nd}", f"S_dg_cross_{nd}", f"peak_dg_{nd}",
                   f"ppc_{nd}", f"empty_{nd}"]
    out_csv = os.path.join(args.outdir, f"diag_{tag}.csv")
    rows = []
    saved_report = set()
    csv_fh = open(out_csv, "w", newline="")
    csv_writer = csv.writer(csv_fh)
    csv_writer.writerow(header)
    csv_fh.flush()

    def diagnostics(i, n_birth, n_death):
        N1 = int(X1.shape[0]); N2 = int(X2.shape[0])
        M_u = N1 / N0 * MASS         # conserved (no u-reaction)
        M_v = N2 / N0 * MASS
        x_c, L = compute_window(X1, gamma=args.gamma, gamma_diff=args.gamma_diff,
                                D=args.D, tau=tau, L_min=args.L_min,
                                q_window=args.q_window)
        L = float(L)
        h_eff = L / args.K
        S_u = float(jnp.mean(jnp.sum((X1 - x_c) ** 2, axis=1)))
        R2 = float(np.sqrt(S_u))
        radii = quantile_radii(X1, x_c, qs)          # RECONSTRUCTION-FREE core radii
        counts = core_counts(X1, x_c, radii)
        R50_over_heff = radii[0.5] / h_eff if h_eff > 0 else np.nan
        Yu = (X1 - x_c) * (jnp.pi / L)
        coeff_u = density_coeffs_y(Yu, args.K)
        peak = float(recon_peak(coeff_u, x_c, L, M_u, n_grid=args.diag_grid))   # BANDWIDTH-SENSITIVE
        S_L2_u = float(recon_l2(coeff_u, x_c, L, M_u, n_grid=args.diag_grid))   # = S_{K,N}(t)
        coeff_v, M_v_eff, outside_v_frac = coeffs_v_on_window(
            X2, x_c, L, args.K, N0, mask_v_outside)
        peak_v = float(recon_peak(coeff_v, x_c, L, M_v_eff, n_grid=args.diag_grid))
        gradv_u = grad_v_from_cloud(X1, coeff_v, x_c, L, M_v_eff, taper_s=taper_s)
        max_gv = float(jnp.max(jnp.sqrt(jnp.sum(gradv_u ** 2, axis=1)))) \
            if N1 > 0 else np.nan
        drift_cfl = chi * max_gv * tau / np.sqrt(2.0 * tau)
        row = ([i, i * tau, N0, args.K, M_u, M_v, float(x_c[0]), float(x_c[1]),
                L, h_eff, S_u, R2]
               + [radii[q] for q in qs]
               + [counts[0.5], counts[0.8], counts[0.9], R50_over_heff,
                  peak, peak_v, S_L2_u, S_L2_u,        # S_L2 == S_L2_u alias
                  outside_v_frac, drift_cfl, int(n_birth), int(n_death)])
        # ---- Version A: LDG-matched P1 DG readout from the u-cloud ----
        if dg_ns:
            Xu = np.asarray(X1); wpp = MASS / N0       # equal per-particle u mass
            wu = np.full(Xu.shape[0], wpp)
            half = Xu.shape[0] // 2
            for nd in dg_ns:
                cf, dgd = project_particles_to_p1dg(Xu, wu, nd)
                ca, da = project_particles_to_p1dg(Xu[:half], wu[:half] * 2, nd)
                cb, _ = project_particles_to_p1dg(Xu[half:], wu[half:] * 2, nd)
                s2x = dg_inner_product(ca, cb, da["dx"], da["dy"])
                row += [dg_l2_norm(cf, dgd["dx"], dgd["dy"]),
                        float(np.sqrt(max(s2x, 0.0))), dg_peak(cf),
                        dgd["ppc_mean"], dgd["empty_cell_fraction"]]
        rows.append(row)
        csv_writer.writerow(row)          # incremental flush (walltime-kill safe)
        csv_fh.flush()
        if args.verbose:
            print(f"step {i:6d} t={i*tau:.3e} M_u={M_u:.4f} M_v={M_v:.4f} "
                  f"R50/h_eff={R50_over_heff:5.2f} N50={counts[0.5]:7d} "
                  f"peak_u={peak:.3e} peak_v={peak_v:.3e} L={L:.4f} "
                  f"cfl={drift_cfl:.2e} v_out={outside_v_frac:.3f} "
                  f"b={n_birth} d={n_death}", flush=True)
        return x_c, L, coeff_u, M_u, coeff_v, M_v_eff

    def maybe_save_snapshot(i, x_c, L, coeff_u, M_u, coeff_v, M_v_eff):
        t = i * tau
        for rt in report_times:
            if rt in saved_report:
                continue
            if t + 0.5 * tau >= rt:           # first step at/after report time
                xg, yg, field = recon_field_grid(coeff_u, x_c, L, M_u,
                                                  n_grid=args.snap_grid)
                xgv, ygv, field_v = recon_field_grid(coeff_v, x_c, L, M_v_eff,
                                                      n_grid=args.snap_grid)
                peak_v = float(np.max(np.asarray(field_v)))
                fname = os.path.join(args.outdir, "snapshots",
                                     f"snap_u_t{rt:.4e}_seed{args.seed}.npz")
                extra = {}
                if getattr(args, "save_cloud_snapshots", False):
                    # raw particle clouds for residual / adaptive reconstruction
                    # (CLAUDE.md solver-hybrid plan §8.3); start-of-step clouds,
                    # consistent with the reconstructed field saved alongside.
                    Xu = np.asarray(X1); Xv = np.asarray(X2)
                    extra = dict(
                        X_u=Xu.astype(np.float32), X_v=Xv.astype(np.float32),
                        N_u=int(Xu.shape[0]), N_v=int(Xv.shape[0]),
                        mass_u_total=float(MASS), mass_v_total=float(M_v_eff),
                        mass_per_particle_u=float(MASS / N0),
                        mass_per_particle_v=(float(M_v_eff / Xv.shape[0])
                                             if Xv.shape[0] else 0.0))
                np.savez(fname,
                         t=t, report_time=rt, seed=args.seed, K=args.K, N=N0,
                         x_c=np.asarray(x_c), L=float(L), M_u=float(M_u),
                         M_v=float(M_v_eff),
                         x_grid=np.asarray(xg), y_grid=np.asarray(yg),
                         u_field=np.asarray(field),
                         v_field=np.asarray(field_v),
                         peak_PK_u=float(np.max(np.asarray(field))),
                         peak_PK_v=peak_v, **extra)
                saved_report.add(rt)
                if args.verbose:
                    print(f"  [snapshot] wrote {fname} (t={t:.4e} ~ report {rt})",
                          flush=True)

    t0 = time.time()
    n_birth = n_death = 0
    aborted = False
    for i in range(n_steps + 1):
        if i % args.diag_every == 0 or i == n_steps:
            x_c, L, coeff_u, M_u, coeff_v_d, M_v_d = diagnostics(i, n_birth, n_death)
            maybe_save_snapshot(i, x_c, L, coeff_u, M_u, coeff_v_d, M_v_d)

        if i == n_steps:
            break

        # ---- 0. GUARD: empty v-cloud (no chemical -> no chemotactic drift)--
        if int(X2.shape[0]) == 0:
            print(f"[WARN] step {i}: v-cloud X2 is EMPTY; stopping loop gracefully "
                  f"and writing diagnostics computed so far.", flush=True)
            aborted = True
            break

        # ---- 1. window from the CELL cloud --------------------------------
        x_c, L = compute_window(X1, gamma=args.gamma, gamma_diff=args.gamma_diff,
                                D=args.D, tau=tau, L_min=args.L_min,
                                q_window=args.q_window)
        L = float(L)
        # ---- 2. reconstruct v on the u-window (mask out-of-window v-parts) -
        coeff_v, M_v_eff, _ = coeffs_v_on_window(
            X2, x_c, L, args.K, N0, mask_v_outside)

        # ---- 2b. GUARD: drift CFL blow-out --------------------------------
        gradv_chk = grad_v_from_cloud(X1, coeff_v, x_c, L, M_v_eff, taper_s=taper_s)
        if not bool(jnp.all(jnp.isfinite(gradv_chk))):
            print(f"[WARN] step {i}: non-finite grad v detected; stopping loop "
                  f"gracefully and writing diagnostics computed so far.", flush=True)
            aborted = True
            break
        if int(X1.shape[0]) > 0:
            max_gv = float(jnp.max(jnp.sqrt(jnp.sum(gradv_chk ** 2, axis=1))))
            drift_cfl = chi * max_gv * tau / np.sqrt(2.0 * tau)
            if drift_cfl > cfl_abort:
                print(f"[WARN] step {i}: drift_cfl={drift_cfl:.3e} exceeds "
                      f"--cfl_abort={cfl_abort:.3e}; stopping loop gracefully and "
                      f"writing diagnostics computed so far.", flush=True)
                aborted = True
                break

        rng, k1, k2, kd, kb = jax.random.split(rng, 5)
        xi1 = jax.random.normal(k1, X1.shape, dtype=X1.dtype)
        xi2 = jax.random.normal(k2, X2.shape, dtype=X2.dtype)
        # ---- 3. transport u (chemotaxis from v-cloud + diffusion) ---------
        X1 = transport_u(X1, coeff_v, x_c, L, M_v_eff, tau, xi1)
        # ---- 4. transport v (diffusion only) ------------------------------
        X2 = transport_v(X2, tau, xi2)
        X1.block_until_ready(); X2.block_until_ready()
        # ---- 4b. GUARD: non-finite particle positions ---------------------
        if not (bool(jnp.all(jnp.isfinite(X1))) and bool(jnp.all(jnp.isfinite(X2)))):
            print(f"[WARN] step {i}: non-finite particle positions after transport; "
                  f"stopping loop gracefully and writing diagnostics so far.",
                  flush=True)
            aborted = True
            break
        # ---- 5. cross-species injection reaction on v ---------------------
        death_v = np.asarray(jax.random.uniform(kd, (X2.shape[0],)) < p)  # v decays
        birth_u = np.asarray(jax.random.uniform(kb, (X1.shape[0],)) < p)  # u spawns v
        n_death = int(death_v.sum()); n_birth = int(birth_u.sum())
        X2 = jnp.asarray(np.concatenate(
            [np.asarray(X2)[~death_v], np.asarray(X1)[birth_u]], axis=0))

    t1 = time.time()

    csv_fh.close()                        # rows already flushed incrementally
    Mu0 = rows[0][header.index("M_u")]; MuN = rows[-1][header.index("M_u")]
    print(f"\n[{tag}] wrote {out_csv}")
    if aborted:
        print(f"[{tag}] *** LOOP ABORTED EARLY (guard tripped); "
              f"diagnostics up to step {rows[-1][0]} written. ***")
    print(f"[{tag}] runtime {t1-t0:.1f}s  M_u: {Mu0:.4f}->{MuN:.4f} (should be ~const) "
          f"M_v(T)={rows[-1][header.index('M_v')]:.4f}  N_v(T)={int(X2.shape[0])}")
    return out_csv


def build_parser():
    p = argparse.ArgumentParser(
        description="2D parabolic-parabolic KS particle solver (LDG-style comparison)")
    p.add_argument("--N", type=int, default=80000,
                   help="initial population of EACH cloud (equal mass)")
    p.add_argument("--K", type=int, default=8,
                   help="Fourier modes/axis (keep modest: grad noise)")
    p.add_argument("--tau", type=float, default=1e-6, help="time step")
    p.add_argument("--n_steps", type=int, default=200)
    p.add_argument("--diag_every", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chi", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=3.0)
    p.add_argument("--gamma_diff", type=float, default=6.0)
    p.add_argument("--D", type=float, default=1.0)
    p.add_argument("--L_min", type=float, default=1e-3)
    p.add_argument("--q_window", type=float, default=0.8)
    p.add_argument("--dg_readout_n", type=int, nargs="*", default=[],
                   help="LDG-matched P1 DG readout (Version A) at these diagnostic "
                        "resolutions n on [-0.5,0.5]^2 (e.g. 40 80 160); empty = off")
    p.add_argument("--a_u", type=float, default=84.0, help="u0 ~ exp(-a_u |x|^2)")
    p.add_argument("--a_v", type=float, default=42.0, help="v0 ~ exp(-a_v |x|^2)")
    p.add_argument("--diag_grid", type=int, default=65,
                   help="grid for peak/L2 reconstruction")
    p.add_argument("--snap_grid", type=int, default=129,
                   help="grid for saved u/v snapshots (keep modest so npz stays small)")
    p.add_argument("--report_times", type=float, nargs="*",
                   default=[6e-5, 1.2e-4, 2.0e-4],
                   help="times at which to save reconstructed-u/v .npz snapshots "
                        "(default = Li-Shu-Yang LDG reporting times)")
    p.add_argument("--filter_s", type=float, default=0.5,
                   help="spectral low-pass width s in w(k)=exp(-(k/(K-1))^2/(2 s^2)) "
                        "on the v-coeffs before grad assembly (default 0.5; "
                        "use a large s like 1e3 to DISABLE the filter)")
    p.add_argument("--mask_v_outside", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="drop v-particles with max|Yv|>pi from the v RECONSTRUCTION "
                        "(they still diffuse) and rescale mass_v by the in-window v "
                        "fraction (default on; --no-mask_v_outside to disable)")
    p.add_argument("--cfl_abort", type=float, default=5.0,
                   help="stop the loop gracefully if drift_cfl exceeds this")
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--save_cloud_snapshots", action="store_true",
                   help="also save raw particle clouds X_u,X_v at report times "
                        "(for residual/adaptive reconstruction post-processing)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="tiny smoke run: small N, few steps, still hits >=1 report "
                        "time. Overrides N/n_steps/diag_every/report_times unless "
                        "they were explicitly set smaller.")
    return p


def apply_smoke(args):
    """Tiny smoke configuration: small N, few steps, still reaches >=1 report time.

    Uses tau=1e-6 (the LDG-report-time scale): the super-critical IC has a large
    chemotactic drift, so tau=1e-5 trips the drift-CFL guard at step 0 (cfl scales
    like max|grad v| * sqrt(tau/2)).  At tau=1e-6 the smoke run advances and still
    reaches the first report time 6e-5 (>= 60 steps needed).
    """
    args.N = min(args.N, 4000)
    args.tau = 1e-6
    args.n_steps = 80          # reaches t = 8e-5 > first report time 6e-5
    args.diag_every = 4
    args.report_times = [6e-5]
    args.verbose = True
    return args


if __name__ == "__main__":
    a = build_parser().parse_args()
    if a.smoke:
        a = apply_smoke(a)
    run(a)
