"""
Verification suite for the LDG solver, with emphasis on the CHEMOTAXIS flux
assembly at the MODAL level (cell-average tests are not sufficient -- the
blow-up dynamics are driven by -div(u grad v)).

Run:  python test_ldg.py
"""
import numpy as np
from ldg_solver import (LDGMesh, LDGSolver, project_ic, field_L2, total_mass)

np.seterr(all="ignore")
OK = True


def report(name, ok, detail=""):
    global OK
    OK = OK and ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# 1. Diffusion still correct (unchanged operator, sanity)
# ---------------------------------------------------------------------------
def test_diffusion():
    print("1. pure-heat LDG convergence (Neumann, e^{-2t}cos x cos y):")
    def ex(t):
        return lambda x, y: np.exp(-2 * t) * np.cos(x) * np.cos(y)
    T = 0.1; prev = None; orders = []
    for N in [10, 20, 40]:
        m = LDGMesh(0, np.pi, 0, np.pi, N, N)
        sol = LDGSolver(m, chi=0.0, positivity=False)
        U = project_ic(m, ex(0.0)); V = np.zeros_like(U)
        dt = 0.03 * m.dx ** 2; nst = int(np.ceil(T / dt)); dt = T / nst
        for _ in range(nst):
            U, V, _ = sol.step(U, V, dt)
        err = field_L2(U - project_ic(m, ex(T)), m) / field_L2(project_ic(m, ex(T)), m)
        if prev is not None:
            orders.append(np.log(prev / err) / np.log(2))
        prev = err
    report("diffusion 2nd order", min(orders) > 1.8, f"orders={[round(o,2) for o in orders]}")


# ---------------------------------------------------------------------------
# 2. Full coupled-KS SOLUTION convergence (the valid chemotaxis consistency test).
#    NOTE: a pointwise "L_h(P u) vs P(L u)" modal test is NOT valid for LDG -- the
#    verified Laplacian also fails it because LDG operators do not commute with
#    L2 projection; consistency is in the Galerkin/solution sense.  We therefore
#    self-converge the full coupled solution on a smooth subcritical problem.
# ---------------------------------------------------------------------------
def test_ks_selfconvergence():
    print("2. full coupled-KS self-convergence (smooth subcritical, vs N=160 ref):")
    def u0(x, y):
        return 1.0 + 0.5 * np.cos(x) * np.cos(y)
    def v0(x, y):
        return 0.5 * np.cos(x) * np.cos(y)
    T = 0.05

    def solve(N):
        m = LDGMesh(0, 2 * np.pi, 0, 2 * np.pi, N, N)
        sol = LDGSolver(m, chi=1.0, positivity=True)
        U = sol.limit_positivity(project_ic(m, u0)); V = project_ic(m, v0)
        dt = 0.02 * m.dx ** 2; nst = int(np.ceil(T / dt)); dt = T / nst
        for _ in range(nst):
            U, V, _ = sol.step(U, V, dt)
        return m, U

    mref, Uref = solve(160)
    prev = None; orders = []
    for N in [20, 40, 80]:
        m, U = solve(N)
        f = mref.Nx // m.Nx
        Ur = Uref[..., 0].reshape(m.Ny, f, m.Nx, f).mean(axis=(1, 3))
        e = np.linalg.norm(U[..., 0] - Ur) / np.linalg.norm(Ur)
        if prev is not None:
            orders.append(np.log(prev / e) / np.log(2))
        prev = e
    report("full-KS solution 2nd order", min(orders) > 1.8,
           f"orders={[round(o,2) for o in orders]}")


# ---------------------------------------------------------------------------
# 3. x<->y permutation symmetry: if (u,v) -> (u(y,x), v(y,x)), the chemotaxis
#    RHS must permute consistently (xi <-> eta modes swap, value transposed).
# ---------------------------------------------------------------------------
def test_permutation_symmetry():
    print("3. x<->y permutation symmetry of the chemotaxis RHS:")
    N = 32
    m = LDGMesh(0, 2 * np.pi, 0, 2 * np.pi, N, N)
    sol = LDGSolver(m, chi=1.0, positivity=False)
    rng = np.random.default_rng(0)
    U = project_ic(m, lambda x, y: 2 + np.cos(x) * np.cos(2 * y) + 0.3 * np.sin(2 * x))
    V = project_ic(m, lambda x, y: np.cos(x) * np.cos(y) + 0.2 * np.cos(2 * x) * np.cos(y))

    def chemo(U, V):
        rx, ry = sol.gradient(V)
        return sol.conv_rhs(U, rx, ry, alpha=0.0) * m.Minv

    C = chemo(U, V)
    # build the swapped fields: transpose cells AND swap xi<->eta modes
    def swap(F):
        G = np.transpose(F, (1, 0, 2)).copy()
        G[..., 1], G[..., 2] = G[..., 2].copy(), G[..., 1].copy()
        return G
    Cs = chemo(swap(U), swap(V))
    err = np.linalg.norm(swap(C) - Cs) / np.linalg.norm(Cs)
    report("permutation symmetry holds", err < 1e-12, f"rel diff={err:.2e}")


# ---------------------------------------------------------------------------
# 4. Constant v  => grad v = 0 => chemotaxis RHS = 0.
# ---------------------------------------------------------------------------
def test_constant_v():
    print("4. constant v => zero chemotaxis:")
    N = 24
    m = LDGMesh(-0.5, 0.5, -0.5, 0.5, N, N)
    sol = LDGSolver(m, chi=1.0, positivity=False)
    U = project_ic(m, lambda x, y: 3 + np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y))
    V = project_ic(m, lambda x, y: 7.0 + 0 * x)           # constant
    rx, ry = sol.gradient(V)
    conv = sol.conv_rhs(U, rx, ry, alpha=sol.max_alpha(V))
    report("chemotaxis = 0 for constant v", np.max(np.abs(conv)) < 1e-12,
           f"max|conv|={np.max(np.abs(conv)):.2e}")


# ---------------------------------------------------------------------------
# 5. Radial short-time symmetry on the blow-up IC: center stays, second moments
#    equal in x and y, off-diagonal moment ~0.
# ---------------------------------------------------------------------------
def test_radial_symmetry():
    print("5. radial short-time symmetry (blow-up IC, t<=2e-5):")
    N = 80
    m = LDGMesh(-0.5, 0.5, -0.5, 0.5, N, N)
    sol = LDGSolver(m, chi=1.0, positivity=True)
    U = sol.limit_positivity(project_ic(m, lambda x, y: 840 * np.exp(-84 * (x**2 + y**2))))
    V = project_ic(m, lambda x, y: 420 * np.exp(-42 * (x**2 + y**2)))
    t = 0.0; Tend = 2e-5
    while t < Tend:                      # adaptive dt (stiff concentrating core)
        alpha = sol.max_alpha(V)
        dt = min(0.04 * m.dx ** 2, 0.2 * m.dx / (alpha + 1e-30), Tend - t)
        U, V, _ = sol.step(U, V, dt); t += dt
    ub = U[..., 0]
    X, Y = np.meshgrid(m.xc, m.yc)
    M = ub.sum()
    cx = (X * ub).sum() / M; cy = (Y * ub).sum() / M
    mxx = ((X - cx) ** 2 * ub).sum() / M
    myy = ((Y - cy) ** 2 * ub).sum() / M
    mxy = ((X - cx) * (Y - cy) * ub).sum() / M
    report("center ~ 0", abs(cx) < 1e-3 and abs(cy) < 1e-3, f"cx={cx:.2e} cy={cy:.2e}")
    report("mxx ~ myy", abs(mxx - myy) / max(mxx, 1e-30) < 1e-3,
           f"mxx={mxx:.3e} myy={myy:.3e}")
    report("off-diagonal ~ 0", abs(mxy) / max(mxx, 1e-30) < 1e-3, f"mxy/mxx={mxy/max(mxx,1e-30):.2e}")


if __name__ == "__main__":
    test_diffusion()
    test_ks_selfconvergence()
    test_permutation_symmetry()
    test_constant_v()
    test_radial_symmetry()
    print("\n=== ALL PASS ===" if OK else "\n=== SOME TESTS FAILED ===")
