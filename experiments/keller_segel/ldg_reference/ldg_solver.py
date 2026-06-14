"""
Direct LDG reference for the fully parabolic-parabolic Keller-Segel model,
following Li, Shu & Yang, "Local discontinuous Galerkin method for the
Keller-Segel chemotaxis model" (J. Sci. Comput.).
==========================================================================

Model (their (1.1), chi = 1, homogeneous Neumann on a rectangle Omega):

    u_t - div( grad u - u grad v ) = 0,
    v_t -  Delta v = u - v.

LDG with auxiliary variables p = grad u, r = grad v (their (2.1)-(2.4)):

    u_t = -div(r u) + div p,   p = grad u,
    v_t =  div r + u - v,      r = grad v.

P^1 MODAL DG on a uniform Cartesian mesh, basis {1, xi, eta} per cell with
xi = 2(x-xc)/dx, eta = 2(y-yc)/dy in [-1,1]; mass matrix diag(A, A/3, A/3).
Alternating diffusion fluxes (their (2.5)):
    u_hat = u^+  (trace from the right/top neighbour),  p_hat = p^-  (left/bottom);
    v_hat = v^+,  r_hat = r^-.
Lax-Friedrichs chemotaxis flux (their (2.8)):
    (r u)_hat = 1/2 (r^+ u^+ + r^- u^-) - alpha/2 nu (u^+ - u^-),  alpha = max|grad v|.
Homogeneous Neumann: zero diffusive AND convective flux through dOmega.
Zhang-Shu P^1 positivity scaling limiter on u (their Section 4); SSP-RK3 in time.

Diffusion integrals of the linear traces are evaluated ANALYTICALLY
(int_edge (a+b s) ds = a*len ; int_edge (a+b s) s ds = b*len/3); the chemotaxis
products use 2-point Gauss along each edge (exact for the quadratic integrand).
"""
import numpy as np

S2 = np.array([-1.0, 1.0]) / np.sqrt(3.0)     # 2-pt Gauss nodes
W2 = np.array([1.0, 1.0])                      # weights


class LDGMesh:
    def __init__(self, xa, xb, ya, yb, Nx, Ny):
        self.xa, self.xb, self.ya, self.yb = xa, xb, ya, yb
        self.Nx, self.Ny = Nx, Ny
        self.dx = (xb - xa) / Nx
        self.dy = (yb - ya) / Ny
        self.xc = xa + (np.arange(Nx) + 0.5) * self.dx
        self.yc = ya + (np.arange(Ny) + 0.5) * self.dy
        self.A = self.dx * self.dy
        self.Minv = np.array([1.0 / self.A, 3.0 / self.A, 3.0 / self.A])


def project_ic(mesh, func, nq=6):
    gp, gw = np.polynomial.legendre.leggauss(nq)
    XI, ETA = np.meshgrid(gp, gp, indexing="ij")
    WW = np.outer(gw, gw).ravel()
    phi = np.stack([np.ones_like(XI.ravel()), XI.ravel(), ETA.ravel()], axis=-1)
    U = np.zeros((mesh.Ny, mesh.Nx, 3))
    for j in range(mesh.Ny):
        y = mesh.yc[j] + 0.5 * mesh.dy * ETA.ravel()
        for i in range(mesh.Nx):
            x = mesh.xc[i] + 0.5 * mesh.dx * XI.ravel()
            f = func(x, y)
            rhs = (phi * (f * WW)[:, None]).sum(0) * (0.25 * mesh.A)
            U[j, i] = mesh.Minv * rhs
    return U


def cell_average(U):
    return U[..., 0]


# ---------------------------------------------------------------------------
# Edge traces of a P1 field F (Ny,Nx,3): each is (const a, slope b), trace = a+b s.
#   right  (xi=+1): a=F0+F1, b=F2 (s=eta) ;  left (xi=-1): a=F0-F1, b=F2
#   top   (eta=+1): a=F0+F2, b=F1 (s=xi)  ;  bottom(eta=-1): a=F0-F2, b=F1
# ---------------------------------------------------------------------------
def traces_x(F):
    return (F[..., 0] + F[..., 1], F[..., 2]), (F[..., 0] - F[..., 1], F[..., 2])


def traces_y(F):
    return (F[..., 0] + F[..., 2], F[..., 1]), (F[..., 0] - F[..., 2], F[..., 1])


class LDGSolver:
    def __init__(self, mesh, chi=1.0, positivity=True, eps=1e-13):
        self.m = mesh
        self.chi = chi
        self.positivity = positivity
        self.eps = eps

    # ----- LDG gradient q=grad F, flux F_hat = F^+ (right/top neighbour) -----
    def gradient(self, F):
        m = self.m
        (aR, bR), (aL, bL) = traces_x(F)          # own right/left traces
        # F_hat at the right face of cell i = F^+ = LEFT trace of cell i+1
        fR_a = np.zeros_like(aL); fR_b = np.zeros_like(bL)
        fR_a[:, :-1] = aL[:, 1:]; fR_b[:, :-1] = bL[:, 1:]
        # F_hat at the left face of cell i = F^+ = own LEFT trace
        fL_a, fL_b = aL.copy(), bL.copy()
        # Neumann boundary (their (2.6)-(2.7)): u_hat is the INTERIOR trace of the
        # boundary cell (NOT zero -- only the diffusive flux p_hat.n is zeroed, in
        # div_minus).  Right boundary: u_hat = own RIGHT trace; left: own LEFT trace.
        fR_a[:, -1] = aR[:, -1]; fR_b[:, -1] = bR[:, -1]
        # fL already equals the own LEFT trace, including col 0 -> leave as is
        dy = m.dy
        qx = np.zeros_like(F)
        # DOF0: (aR - aL)*dy ; DOF1: -(2/dx)F0 A + aR*dy + aL*dy ; DOF2: (bR-bL)*dy/3
        qx[..., 0] = (fR_a - fL_a) * dy
        qx[..., 1] = -(2.0 / m.dx) * F[..., 0] * m.A + (fR_a + fL_a) * dy
        qx[..., 2] = (fR_b - fL_b) * dy / 3.0
        qx *= m.Minv

        (aT, bT), (aB, bB) = traces_y(F)
        fT_a = np.zeros_like(aB); fT_b = np.zeros_like(bB)
        fT_a[:-1, :] = aB[1:, :]; fT_b[:-1, :] = bB[1:, :]
        fB_a, fB_b = aB.copy(), bB.copy()
        # Neumann: top boundary u_hat = own TOP trace; bottom = own BOTTOM trace
        fT_a[-1, :] = aT[-1, :]; fT_b[-1, :] = bT[-1, :]
        # fB already equals the own BOTTOM trace, including row 0 -> leave as is
        dx = m.dx
        qy = np.zeros_like(F)
        qy[..., 0] = (fT_a - fB_a) * dx
        qy[..., 2] = -(2.0 / m.dy) * F[..., 0] * m.A + (fT_a + fB_a) * dx
        qy[..., 1] = (fT_b - fB_b) * dx / 3.0
        qy *= m.Minv
        return qx, qy

    # ----- divergence of a diffusive vector G=(Gx,Gy) with G_hat = G^- ------
    # weak form (G, grad w) - <G_hat . n, w> with G_hat from the LEFT/BOTTOM (-).
    def div_minus(self, Gx, Gy):
        """Return the DG rhs (before Minv) of -<G^-.n,w> + (G,grad w) for div G,
        i.e. the contribution of div(G) to (.,w) with the alternating partner
        flux G_hat=G^-.  Used for p in u_t (p_hat=p^-) and r in v_t (r_hat=r^-)."""
        m = self.m
        out = np.zeros_like(Gx)
        # volume (G, grad w): DOF1 += (2/dx) int Gx = (2/dx) Gx0 A ; DOF2 += (2/dy) Gy0 A
        out[..., 1] += (2.0 / m.dx) * Gx[..., 0] * m.A
        out[..., 2] += (2.0 / m.dy) * Gy[..., 0] * m.A
        # x-faces: G_hat = Gx^- = own RIGHT trace at right face; at left face = right trace of i-1
        (aR, bR), (aL, bL) = traces_x(Gx)
        gR_a, gR_b = aR.copy(), bR.copy()           # right face uses own right trace (^-)
        gL_a = np.zeros_like(aR); gL_b = np.zeros_like(bR)
        gL_a[:, 1:] = aR[:, :-1]; gL_b[:, 1:] = bR[:, :-1]   # left face = right trace of i-1
        gR_a[:, -1] = 0.0; gR_b[:, -1] = 0.0        # Neumann
        gL_a[:, 0] = 0.0; gL_b[:, 0] = 0.0
        dy = m.dy
        # -<G_hat n_x, w>: right n=+1, left n=-1
        out[..., 0] -= (gR_a - gL_a) * dy
        out[..., 1] -= (gR_a + gL_a) * dy
        out[..., 2] -= (gR_b - gL_b) * dy / 3.0
        # y-faces
        (aT, bT), (aB, bB) = traces_y(Gy)
        gT_a, gT_b = aT.copy(), bT.copy()
        gB_a = np.zeros_like(aT); gB_b = np.zeros_like(bT)
        gB_a[1:, :] = aT[:-1, :]; gB_b[1:, :] = bT[:-1, :]
        gT_a[-1, :] = 0.0; gT_b[-1, :] = 0.0
        gB_a[0, :] = 0.0; gB_b[0, :] = 0.0
        dx = m.dx
        out[..., 0] -= (gT_a - gB_a) * dx
        out[..., 2] -= (gT_a + gB_a) * dx
        out[..., 1] -= (gT_b - gB_b) * dx / 3.0
        # `out` above accumulates +(G,grad w) - <G^- n, w>; the weak divergence
        # contribution to (div G, w) is -(G,grad w) + <G^- n, w> = -out.
        return -out

    # ----- chemotaxis convection -div(r u) with Lax-Friedrichs flux ---------
    def conv_rhs(self, U, rx, ry, alpha):
        """DG rhs (before Minv) of (r u, grad w) - <(r u)_hat . n, w>,
        LF flux with speed alpha.  This is +div-form contribution to u_t as
        u_t = -div(r u)+...  -> the weak form (2.1) already carries the sign:
        (u_t,w) = (r u - p, grad w) - <(r u)_hat - p_hat, w>.  Here we return the
        (r u) part: (r u, grad w) - <(r u)_hat n, w>."""
        m = self.m
        out = np.zeros_like(U)
        # volume (r u, grad w): need int_K rx*u and int_K ry*u (P2) via 2x2 Gauss
        Ix, Iy = self._cell_int_prod(U, rx, ry)
        out[..., 1] += (2.0 / m.dx) * Ix
        out[..., 2] += (2.0 / m.dy) * Iy
        # x-face LF flux.  _lf_face_x already folds the basis FACE VALUE into each
        # component (integ uses xival for phi1=xi and s for phi2=eta), so the only
        # edge sign left to apply is n_x = +1 (right) / -1 (left): uniformly R - L
        # for EVERY mode.  (Previous code used fxR[1]+fxL[1] for the xi mode -- a
        # double-counted sign; the cell-average (phi0) test did not catch it.)
        fxR, fxL = self._lf_face_x(U, rx, alpha)
        out[..., 0] -= (fxR[0] - fxL[0])
        out[..., 1] -= (fxR[1] - fxL[1])
        out[..., 2] -= (fxR[2] - fxL[2])
        # y-face LF flux.  Component order is [phi0, phi1=xi, phi2=eta]; n_y = +1
        # (top) / -1 (bottom).  Update out[...,m] with component m (no xi<->eta
        # swap), uniformly T - B.
        fyT, fyB = self._lf_face_y(U, ry, alpha)
        out[..., 0] -= (fyT[0] - fyB[0])
        out[..., 1] -= (fyT[1] - fyB[1])
        out[..., 2] -= (fyT[2] - fyB[2])
        return out

    def _cell_int_prod(self, U, rx, ry):
        """int_K rx*u and ry*u via 2x2 Gauss (exact for P1*P1)."""
        m = self.m
        XI, ETA = np.meshgrid(S2, S2, indexing="ij")
        xi = XI.ravel(); eta = ETA.ravel()
        phi = np.stack([np.ones_like(xi), xi, eta], axis=0)      # (3, 4)
        w = np.outer(W2, W2).ravel() * (0.25 * m.A)             # (4,)
        uval = np.einsum('yxk,kq->yxq', U, phi)
        rxv = np.einsum('yxk,kq->yxq', rx, phi)
        ryv = np.einsum('yxk,kq->yxq', ry, phi)
        Ix = np.einsum('yxq,q->yx', rxv * uval, w)
        Iy = np.einsum('yxq,q->yx', ryv * uval, w)
        return Ix, Iy

    def _lf_face_x(self, U, rx, alpha):
        """LF flux integrals on x-faces, returning (right, left) each a 3-list of
        DOF integrals [<.,1>, <.,xi>, <.,eta=s>]."""
        m = self.m
        (uaR, ubR), (uaL, ubL) = traces_x(U)
        (raR, rbR), (raL, rbL) = traces_x(rx)
        dy = m.dy
        Ny, Nx = m.Ny, m.Nx
        s = S2; wq = W2 * (dy / 2.0)
        # at a face: minus side = right trace of left cell; plus side = left trace of right cell
        def face_flux(uminus, rminus, uplus, rplus):
            um = uminus[0][..., None] + uminus[1][..., None] * s
            up = uplus[0][..., None] + uplus[1][..., None] * s
            rm = rminus[0][..., None] + rminus[1][..., None] * s
            rp = rplus[0][..., None] + rplus[1][..., None] * s
            return 0.5 * (rp * up + rm * um) - 0.5 * alpha * (up - um)    # (.,2) nu=+1
        # build per-face plus/minus traces for interior faces; faces are Nx+1
        # face k between cell k-1 (minus) and k (plus). We assemble cell rhs.
        # right face of cell i = face i+1: minus=own right trace, plus=left trace of i+1
        # left  face of cell i = face i  : minus=right trace of i-1, plus=own left trace
        fluxR = np.zeros((Ny, Nx, len(s))); fluxL = np.zeros((Ny, Nx, len(s)))
        # interior right faces (i=0..Nx-2)
        umR = (uaR, ubR)                 # own right trace (minus at right face)
        upR_a = np.zeros_like(uaR); upR_b = np.zeros_like(ubR)
        upR_a[:, :-1] = uaL[:, 1:]; upR_b[:, :-1] = ubL[:, 1:]
        rmR = (raR, rbR)
        rpR_a = np.zeros_like(raR); rpR_b = np.zeros_like(rbR)
        rpR_a[:, :-1] = raL[:, 1:]; rpR_b[:, :-1] = rbL[:, 1:]
        fR = face_flux(umR, rmR, (upR_a, upR_b), (rpR_a, rpR_b))
        fR[:, -1, :] = 0.0               # Neumann: zero convective flux at right boundary
        fluxR = fR
        # left faces: minus = right trace of i-1, plus = own left trace
        umL_a = np.zeros_like(uaR); umL_b = np.zeros_like(ubR)
        umL_a[:, 1:] = uaR[:, :-1]; umL_b[:, 1:] = ubR[:, :-1]
        rmL_a = np.zeros_like(raR); rmL_b = np.zeros_like(rbR)
        rmL_a[:, 1:] = raR[:, :-1]; rmL_b[:, 1:] = rbR[:, :-1]
        fL = face_flux((umL_a, umL_b), (rmL_a, rmL_b), (uaL, ubL), (raL, rbL))
        fL[:, 0, :] = 0.0                # Neumann left boundary
        fluxL = fL
        # integrate against phi(xi=+1: [1,1,s]) on right, phi(xi=-1: [1,-1,s]) on left
        def integ(flux, xival):
            i0 = (flux * wq).sum(-1)
            i1 = (flux * (xival * wq)).sum(-1)
            i2 = (flux * (s * wq)).sum(-1)
            return [i0, i1, i2]
        return integ(fluxR, 1.0), integ(fluxL, -1.0)

    def _lf_face_y(self, U, ry, alpha):
        m = self.m
        (uaT, ubT), (uaB, ubB) = traces_y(U)
        (raT, rbT), (raB, rbB) = traces_y(ry)
        dx = m.dx
        s = S2; wq = W2 * (dx / 2.0)
        Ny, Nx = m.Ny, m.Nx

        def face_flux(um_, rm_, up_, rp_):
            um = um_[0][..., None] + um_[1][..., None] * s
            up = up_[0][..., None] + up_[1][..., None] * s
            rm = rm_[0][..., None] + rm_[1][..., None] * s
            rp = rp_[0][..., None] + rp_[1][..., None] * s
            return 0.5 * (rp * up + rm * um) - 0.5 * alpha * (up - um)
        # top face of cell j = minus own top trace, plus bottom trace of j+1
        upT_a = np.zeros_like(uaB); upT_b = np.zeros_like(ubB)
        upT_a[:-1, :] = uaB[1:, :]; upT_b[:-1, :] = ubB[1:, :]
        rpT_a = np.zeros_like(raB); rpT_b = np.zeros_like(rbB)
        rpT_a[:-1, :] = raB[1:, :]; rpT_b[:-1, :] = rbB[1:, :]
        fT = face_flux((uaT, ubT), (raT, rbT), (upT_a, upT_b), (rpT_a, rpT_b))
        fT[-1, :, :] = 0.0
        umB_a = np.zeros_like(uaT); umB_b = np.zeros_like(ubT)
        umB_a[1:, :] = uaT[:-1, :]; umB_b[1:, :] = ubT[:-1, :]
        rmB_a = np.zeros_like(raT); rmB_b = np.zeros_like(rbT)
        rmB_a[1:, :] = raT[:-1, :]; rmB_b[1:, :] = rbT[:-1, :]
        fB = face_flux((umB_a, umB_b), (rmB_a, rmB_b), (uaB, ubB), (raB, rbB))
        fB[0, :, :] = 0.0

        def integ(flux, etaval):
            return [(flux * wq).sum(-1), (flux * (s * wq)).sum(-1),
                    (flux * (etaval * wq)).sum(-1)]
        return integ(fT, 1.0), integ(fB, -1.0)

    # ---------------- full spatial operator L(U,V) ----------------
    def L(self, U, V):
        m = self.m
        px, py = self.gradient(U)               # p = grad u
        rx, ry = self.gradient(V)               # r = grad v
        # chemotaxis speed alpha = max |grad v| over cell averages of r
        alpha = float(np.max(np.sqrt(rx[..., 0] ** 2 + ry[..., 0] ** 2)) + 1e-30)
        # u_t = (r u - p, grad w) - <(r u)_hat - p_hat, w>
        conv = self.chi * self.conv_rhs(U, rx, ry, alpha)     # contrib of -div(r u)
        diff_u = self.div_minus(px, py)                       # contrib of div p = Lap u
        Ut = (diff_u + conv) * m.Minv
        # v_t = div r + (u - v) = [(r,grad w) - <r^-,w>] + (u-v)
        diff_v = self.div_minus(rx, ry)
        Vt = diff_v * m.Minv + (U - V)
        return Ut, Vt, alpha

    # ---------------- Zhang-Shu P1 positivity limiter on u ----------------
    def limit_positivity(self, U):
        if not self.positivity:
            return U
        ub = U[..., 0]
        # cell min of a P1 field over [-1,1]^2 is at a corner: ub - |U1| - |U2|
        bmin = ub - np.abs(U[..., 1]) - np.abs(U[..., 2])
        eps = self.eps
        Uout = U.copy()
        need = (ub > eps) & (bmin < eps)
        theta = np.where(need, (ub - eps) / np.maximum(ub - bmin, 1e-300), 1.0)
        Uout[..., 1] = U[..., 1] * theta
        Uout[..., 2] = U[..., 2] * theta
        # vacuum cells (ub<=eps): keep as is (treated as ~0)
        return Uout

    # ---------------- SSP-RK3 step ----------------
    def step(self, U, V, dt):
        U0, V0 = U, V
        Ut, Vt, a1 = self.L(U0, V0)
        U1 = self.limit_positivity(U0 + dt * Ut); V1 = V0 + dt * Vt
        Ut, Vt, a2 = self.L(U1, V1)
        U2 = self.limit_positivity(0.75 * U0 + 0.25 * (U1 + dt * Ut))
        V2 = 0.75 * V0 + 0.25 * (V1 + dt * Vt)
        Ut, Vt, a3 = self.L(U2, V2)
        Un = self.limit_positivity((1.0 / 3.0) * U0 + (2.0 / 3.0) * (U2 + dt * Ut))
        Vn = (1.0 / 3.0) * V0 + (2.0 / 3.0) * (V2 + dt * Vt)
        return Un, Vn, max(a1, a2, a3)

    def max_alpha(self, V):
        rx, ry = self.gradient(V)
        return float(np.max(np.sqrt(rx[..., 0] ** 2 + ry[..., 0] ** 2)) + 1e-30)


def field_L2(U, mesh):
    """L2 norm of the P1 field over Omega: sum_K (A U0^2 + (A/3)(U1^2+U2^2))."""
    A = mesh.A
    return float(np.sqrt(np.sum(A * U[..., 0] ** 2 + (A / 3.0) * (U[..., 1] ** 2 + U[..., 2] ** 2))))


def total_mass(U, mesh):
    return float(np.sum(U[..., 0]) * mesh.A)


def eval_on_grid(U, mesh, nsub=1):
    """Evaluate the P1 field at cell centers (nsub=1) -> (Ny,Nx) for plotting/peak."""
    return U[..., 0].copy()


def field_peak(U, mesh):
    """Max over corner values (P1 max is at a corner)."""
    corners = (U[..., 0][..., None]
               + U[..., 1][..., None] * np.array([1, 1, -1, -1])
               + U[..., 2][..., None] * np.array([1, -1, 1, -1]))
    return float(np.max(corners))


def field_min(U, mesh):
    corners = (U[..., 0][..., None]
               + U[..., 1][..., None] * np.array([1, 1, -1, -1])
               + U[..., 2][..., None] * np.array([1, -1, 1, -1]))
    return float(np.min(corners))
