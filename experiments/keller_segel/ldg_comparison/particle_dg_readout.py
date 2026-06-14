"""
LDG-matched P1 DG readout of a particle cloud (blow-up-proxy note, Version A).
============================================================================

Projects a particle empirical measure mu = sum_i omega_i delta_{X_i} onto the SAME
P1 modal DG space and mass-matrix L2 norm as the fixed-flux LDG reference
(experiments/keller_segel/ldg_reference/ldg_solver.py), so the particle and LDG
S_L2(t) live in a common finite-dimensional norm -- no Fourier bandwidth tuning.

On a uniform n x n Cartesian mesh of box=[[x0,x1],[y0,y1]], cell C with area
A=dx*dy and local coords xi=2(x-xc)/dx, eta=2(y-yc)/dy in [-1,1], the L2 projection
of mu is  R mu|_C = c0 + c1 xi + c2 eta  with

    c0 = (1/A) sum_{Xi in C} omega_i
    c1 = (3/A) sum_{Xi in C} omega_i xi_i
    c2 = (3/A) sum_{Xi in C} omega_i eta_i

and the P1 mass-matrix L2 norm is

    ||R mu||_L2^2 = sum_C A ( c0^2 + (c1^2 + c2^2)/3 ).

The cross/split estimator removes the Monte-Carlo self-term using two independent
clouds at the same (N_p, n):  S_cross^2 = <R mu^(a), R mu^(b)>.
"""
import numpy as np


def project_particles_to_p1dg(X, weights, n, box=((-0.5, 0.5), (-0.5, 0.5))):
    """Return P1 DG coefficients (n, n, 3) [c0, c1, c2] of mu = sum w_i delta_{X_i}.

    Particles outside the box are dropped (reported via the returned diag).
    """
    (x0, x1), (y0, y1) = box
    dx = (x1 - x0) / n
    dy = (y1 - y0) / n
    A = dx * dy
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    ix = np.floor((X[:, 0] - x0) / dx).astype(np.int64)
    iy = np.floor((X[:, 1] - y0) / dy).astype(np.int64)
    inside = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    ixx, iyy, ww = ix[inside], iy[inside], w[inside]
    xc = x0 + (ixx + 0.5) * dx
    yc = y0 + (iyy + 0.5) * dy
    xi = 2.0 * (X[inside, 0] - xc) / dx
    eta = 2.0 * (X[inside, 1] - yc) / dy
    c0 = np.zeros((n, n)); c1 = np.zeros((n, n)); c2 = np.zeros((n, n))
    np.add.at(c0, (iyy, ixx), ww)
    np.add.at(c1, (iyy, ixx), ww * xi)
    np.add.at(c2, (iyy, ixx), ww * eta)
    coeffs = np.stack([c0 / A, 3.0 * c1 / A, 3.0 * c2 / A], axis=-1)
    counts = np.zeros((n, n))
    np.add.at(counts, (iyy, ixx), 1.0)
    diag = dict(dx=dx, dy=dy, A=A,
                outside_fraction=float(1.0 - inside.mean()),
                ppc_mean=float(counts.mean()),
                ppc_median=float(np.median(counts)),
                empty_cell_fraction=float((counts == 0).mean()),
                ppc_min_nonzero=float(counts[counts > 0].min()) if np.any(counts > 0) else 0.0)
    return coeffs, diag


def dg_l2_norm(coeffs, dx, dy):
    A = dx * dy
    c0, c1, c2 = coeffs[..., 0], coeffs[..., 1], coeffs[..., 2]
    return float(np.sqrt(np.sum(A * (c0 ** 2 + (c1 ** 2 + c2 ** 2) / 3.0))))


def dg_inner_product(coeffs_a, coeffs_b, dx, dy):
    A = dx * dy
    a, b = coeffs_a, coeffs_b
    return float(np.sum(A * (a[..., 0] * b[..., 0]
                             + (a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]) / 3.0)))


def dg_peak(coeffs):
    """Max over P1 corner values (P1 max is at a cell corner)."""
    c0, c1, c2 = coeffs[..., 0], coeffs[..., 1], coeffs[..., 2]
    return float(np.max(c0[..., None]
                        + c1[..., None] * np.array([1, 1, -1, -1])
                        + c2[..., None] * np.array([1, -1, 1, -1])))


def dg_min(coeffs):
    c0, c1, c2 = coeffs[..., 0], coeffs[..., 1], coeffs[..., 2]
    return float(np.min(c0[..., None]
                        + c1[..., None] * np.array([1, 1, -1, -1])
                        + c2[..., None] * np.array([1, -1, 1, -1])))


def dg_mass(coeffs, dx, dy):
    return float(np.sum(coeffs[..., 0]) * dx * dy)


if __name__ == "__main__":
    # verification: project a particle SAMPLE of the LDG IC onto P1 DG at n=80 and
    # compare to the LDG solver's field_L2(project_ic(u0)) on the same mesh.
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "ldg_reference"))
    from ldg_solver import LDGMesh, project_ic, field_L2, total_mass
    np.random.seed(0)
    a = 84.0
    M = 10.0 * np.pi
    n = 80
    box = ((-0.5, 0.5), (-0.5, 0.5))
    # LDG L2 projection of u0 = 840 exp(-84 r^2) on the n=80 mesh
    m = LDGMesh(-0.5, 0.5, -0.5, 0.5, n, n)
    Ug = project_ic(m, lambda x, y: 840.0 * np.exp(-a * (x ** 2 + y ** 2)))
    S_ldg = field_L2(Ug, m)
    print(f"LDG field_L2(project_ic(u0)) at n={n}: {S_ldg:.4f}  (mass {total_mass(Ug,m):.4f})")
    for Np in [2e4, 8e4, 3.2e5, 1.28e6]:
        Np = int(Np)
        # sample particles ~ exp(-a r^2): Gaussian with sigma^2 = 1/(2a)
        sig = 1.0 / np.sqrt(2 * a)
        X = np.random.normal(0.0, sig, size=(Np, 2))
        w = np.full(Np, M / Np)
        coeffs, dg = project_particles_to_p1dg(X, w, n, box)
        S = dg_l2_norm(coeffs, dg["dx"], dg["dy"])
        # cross estimator with two halves
        h = Np // 2
        ca, da = project_particles_to_p1dg(X[:h], w[:h] * 2, n, box)
        cb, db = project_particles_to_p1dg(X[h:], w[h:] * 2, n, box)
        S2x = dg_inner_product(ca, cb, da["dx"], da["dy"])
        Sx = np.sqrt(max(S2x, 0.0))
        print(f"  Np={Np:>8}: S_raw={S:8.2f}  S_cross={Sx:8.2f}  "
              f"ppc_mean={dg['ppc_mean']:.1f} empty={dg['empty_cell_fraction']:.2f} "
              f"mass={dg_mass(coeffs, dg['dx'], dg['dy']):.3f}")
    print("Expected: S_cross -> S_ldg as Np grows (raw is biased high by the MC self-term).")
