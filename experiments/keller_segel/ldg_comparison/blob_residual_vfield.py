"""
Solver-level TWO-LEVEL BLOB RESIDUAL chemical field v for the parabolic-parabolic
Keller-Segel particle drift  (solver-hybrid plan, "blob residual" / Form-I-smooth).
=====================================================================================

This is the smoother local operator the taper sweep pointed to (REVISION_RESULTS
§3.8): instead of a high-Kl LOCAL SPECTRUM (P_Kl mu, whose gradient amplifies
Monte-Carlo high-mode noise ~Kl), the local high part is an eta_h GAUSSIAN BLOB,
i.e. a kernel density estimate of the v-cloud.  Everything else is kept IDENTICAL
to the Form-I spectral residual (`hybrid_vfield.HybridVField`): same low field
v_lo = P_{Kg} mu_v, same radial taper chi (fractions of the window L), same drift
usage  b_u = chi * grad v_hat.  The ONLY change is the local operator
    P_{Kl} mu_v   ->   eta_h * mu_v ,
so the comparison isolates spectral-vs-blob local reconstruction.

Field (plan §2.1):
    v_hat(x) = v_lo(x) + chi_W(x) [ eta_h*(mu_v|_Wpad)(x)  -  (eta_h * v_lo dx)(x) ].
The SUBTRACTED low-blob term avoids double-counting the low-frequency mass already
in v_lo: the bracket is the blob of the SIGNED residual measure  mu_v - v_lo dx.

Gradient (plan §2.2):
    grad v_hat = grad v_lo + chi grad r_W + r_W grad chi,
    r_W = eta_h*(mu_v|_Wpad) - (eta_h * v_lo dx),
    eta_h(z) = exp(-|z|^2/2h^2)/(2 pi h^2),  grad_x eta_h(x-y) = -((x-y)/h^2) eta_h.
The taper term r_W grad chi must NOT be dropped (FD-verified, test 1).

CONVENTIONS (match field_pp / adaptive_window exactly):
    physical v(x) = mass_v (pi/L)^2 rho_y(y),  y = (x - x_c)(pi/L),  dx = (L/pi)^2 dy.
    The v-cloud carries total physical mass mass_v: weight per particle omega_v =
    mass_v / N_v_in.  So eta_h * mu_v with mu_v = omega_v sum delta_{X_j} is a
    physical density (integrates to mass_v), consistent with v_lo.

BANDWIDTH (plan §2.3 + a cost/scale correction).  The plan's spacing rule
h = c_h R_0.8 / sqrt(N_0.8) gives, with the actual window L ~ gamma R_0.8 (gamma=3)
and N_0.8 ~ 0.8 N ~ 6e4, h/L ~ c_h/(3 sqrt(N_0.8)) ~ 1e-3 -- i.e. ~25x FINER than
the spectral scale L/Kl ~ L/24.  Such a blob is NOISIER, not smoother, than Kl=24
and needs ~sqrt(N) grid points to resolve.  To actually reach the SMOOTHER-than-Kl
regime the relevant knob is h as a FRACTION of L:  h = c_h * L with c_h ~ 1/Kl..3/Kl.
We implement all three rules; the production default is `frac_L`.

PRODUCTION PATH = FFT ON A LOCAL GRID (plan §3.2, chosen from the start because the
naive O(N_u N_v) kernel sum is infeasible for 800 steps).  Deposit mu_v (CIC) and
sample v_lo on an (n,n) grid over the window box; multiply both FFTs by the Gaussian
transfer  G(k)=exp(-h^2|k|^2/2)  (the blob's exact Fourier symbol); the residual
spectrum  R_hat = (FFT(dep) - FFT(v_lo grid)) G  gives r_W and, via i k R_hat, its
gradient on the grid; bilinearly interpolate r_W, grad r_W to the u-particles.  This
is the physical-space convolution done in O(n^2 log n + N), independent of any K.

VERIFICATION PATH = EXACT KERNEL SUM (eval_exact/grad_exact), used by the tests to
FD-check the analytic eta_h, grad eta_h, chi, grad chi, and v_lo math; the grid path
is then validated against it (test 5).
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from adaptive_window import density_coeffs_y
from field_pp import grad_v_from_cloud
from hybrid_vfield import eval_v_from_cloud, radial_taper


# ---------------------------------------------------------------------------
# Gaussian blob kernel (exact-path helpers; plan §2.2).
# ---------------------------------------------------------------------------
def gaussian_kernel(dx, h):
    """eta_h(dx) = exp(-|dx|^2 / 2h^2) / (2 pi h^2),  dx shape (...,2)."""
    r2 = np.sum(dx * dx, axis=-1)
    return np.exp(-0.5 * r2 / (h * h)) / (2.0 * np.pi * h * h)


def gaussian_kernel_grad(dx, h):
    """grad_x eta_h(x-y) = -((x-y)/h^2) eta_h(x-y),  dx=x-y shape (...,2)."""
    k = gaussian_kernel(dx, h)              # (...,)
    return -(dx / (h * h)) * k[..., None]   # (...,2)


def _bilinear(grid, X_eval, x0, dx):
    """Bilinear-interpolate a scalar (n,n) grid (axis0=x, axis1=y) at X_eval.

    grid node (a,b) sits at physical (x0[0]+a*dx, x0[1]+b*dx).  Points outside the
    grid are clamped to the edge (safe: the residual is used only where chi>0, well
    inside the box)."""
    n = grid.shape[0]
    fx = (X_eval[:, 0] - x0[0]) / dx
    fy = (X_eval[:, 1] - x0[1]) / dx
    ix = np.clip(np.floor(fx).astype(np.int64), 0, n - 2)
    iy = np.clip(np.floor(fy).astype(np.int64), 0, n - 2)
    wx = np.clip(fx - ix, 0.0, 1.0)
    wy = np.clip(fy - iy, 0.0, 1.0)
    g00 = grid[ix, iy]; g10 = grid[ix + 1, iy]
    g01 = grid[ix, iy + 1]; g11 = grid[ix + 1, iy + 1]
    return ((1 - wx) * (1 - wy) * g00 + wx * (1 - wy) * g10
            + (1 - wx) * wy * g01 + wx * wy * g11)


def _cic_deposit(X_v, omega_v, x0, dx, n):
    """Cloud-in-cell deposit of a point measure omega_v sum delta_{X_j} onto an
    (n,n) grid over box [x0, x0 + n*dx]^2, returned as a DENSITY (mass/area).
    Sum(density)*dx^2 == total deposited mass (conservative, up to edge clamping)."""
    mass = np.zeros((n, n), dtype=np.float64)
    fx = (X_v[:, 0] - x0[0]) / dx
    fy = (X_v[:, 1] - x0[1]) / dx
    ix = np.clip(np.floor(fx).astype(np.int64), 0, n - 2)
    iy = np.clip(np.floor(fy).astype(np.int64), 0, n - 2)
    wx = np.clip(fx - ix, 0.0, 1.0)
    wy = np.clip(fy - iy, 0.0, 1.0)
    np.add.at(mass, (ix, iy), omega_v * (1 - wx) * (1 - wy))
    np.add.at(mass, (ix + 1, iy), omega_v * wx * (1 - wy))
    np.add.at(mass, (ix, iy + 1), omega_v * (1 - wx) * wy)
    np.add.at(mass, (ix + 1, iy + 1), omega_v * wx * wy)
    return mass / (dx * dx)


class BlobResidualVField:
    """Two-level blob-residual solver field for v on the window (x_c, L).

    Parameters
    ----------
    X_v : (N_v_in,2)  in-window v-cloud (same masking as the spectral path).
    x_c, L : window center and half-extent (physical box [x_c +- L]).
    mass_v : in-window physical v-mass (== M_v_eff); omega_v = mass_v / N_v_in.
    Kg : low global bandwidth for v_lo = P_{Kg} mu_v.
    taper_s : low-pass width for v_lo's spectral gradient (same as the rest of the code).
    frac_in, frac_out : radial taper chi = 1 for r<=frac_in*L, 0 for r>=frac_out*L
        (kept identical to hybrid_vfield.HybridVField so only the local operator changes).
    h_rule : 'frac_L' (h=c_h*L, default), 'core_spacing' (h=c_h*R/sqrt(N)),
             'inner_spacing' (same with the inner-core R,N).
    c_h : bandwidth coefficient.
    R_for_h, N_for_h : the (R, N) used by the spacing rules (ignored for frac_L).
    n_quad : grid points/axis; if None, chosen so the grid spacing ~ h/3 (clamped).
    """

    def __init__(self, X_v, x_c, L, mass_v, Kg, taper_s=0.5,
                 frac_in=0.5, frac_out=0.85, h_rule="frac_L", c_h=0.05,
                 R_for_h=None, N_for_h=None, n_quad=None, n_quad_cap=512):
        self.x_c = np.asarray(x_c, dtype=np.float64)
        self.L = float(L)
        self.mass_v = float(mass_v)
        self.Kg = int(Kg)
        self.taper_s = float(taper_s)
        self.frac_in = float(frac_in)
        self.frac_out = float(frac_out)
        self.h_rule = h_rule
        self.c_h = float(c_h)

        Xv = np.asarray(X_v, dtype=np.float64)
        self.X_v = Xv
        self.N_v_in = int(Xv.shape[0])
        self.omega_v = self.mass_v / max(self.N_v_in, 1)

        # ---- bandwidth h --------------------------------------------------
        if h_rule == "frac_L":
            h = self.c_h * self.L
        elif h_rule in ("core_spacing", "inner_spacing"):
            if R_for_h is None or N_for_h is None:
                raise ValueError(f"h_rule={h_rule} needs R_for_h and N_for_h")
            h = self.c_h * float(R_for_h) / np.sqrt(max(float(N_for_h), 1.0))
        else:
            raise ValueError(f"unknown h_rule: {h_rule}")
        h = max(h, 1e-12)

        # ---- low field coefficients (P_Kg) --------------------------------
        Yv = (Xv - self.x_c) * (np.pi / self.L)
        self.coeff_lo = density_coeffs_y(jnp.asarray(Yv), self.Kg)

        # ---- local grid resolving the blob (spacing ~ h/3) ----------------
        if n_quad is None:
            n_quad = int(np.ceil(6.0 * self.L / h))      # dx = 2L/n <= h/3
        n_quad = int(np.clip(n_quad, 64, n_quad_cap))
        # if the grid had to be capped, the blob is sub-grid: raise h to the grid
        # scale so the deposit does not alias (h_eff = max(h, 3*dx)); this is the
        # noisy-regime clamp -- recorded in diagnostics().
        self.n_quad = n_quad
        self.dx = 2.0 * self.L / n_quad
        self.h_requested = float(h)
        self.h = float(max(h, 3.0 * self.dx))
        self.h_clamped = bool(self.h > h * (1.0 + 1e-9))

        # grid origin (node (0,0)) and node coordinates
        self.x0 = self.x_c - self.L
        ax = self.x0[0] + self.dx * np.arange(n_quad)
        ay = self.x0[1] + self.dx * np.arange(n_quad)
        GX, GY = np.meshgrid(ax, ay, indexing="ij")       # (n,n) axis0=x, axis1=y
        nodes = np.stack([GX.ravel(), GY.ravel()], axis=1)

        # v_lo sampled on the grid (physical low field)
        vlo_grid = np.asarray(eval_v_from_cloud(
            jnp.asarray(nodes), self.coeff_lo, jnp.asarray(self.x_c),
            self.L, self.mass_v, self.taper_s)).reshape(n_quad, n_quad)
        # mu_v deposited on the grid (CIC density)
        dep_grid = _cic_deposit(Xv, self.omega_v, self.x0, self.dx, n_quad)

        # ---- residual spectrum R_hat = (FFT(dep) - FFT(vlo)) * G(k) -------
        kfreq = 2.0 * np.pi * np.fft.fftfreq(n_quad, d=self.dx)   # angular freq
        KX, KY = np.meshgrid(kfreq, kfreq, indexing="ij")
        G = np.exp(-0.5 * self.h * self.h * (KX * KX + KY * KY))
        R_hat = (np.fft.fft2(dep_grid) - np.fft.fft2(vlo_grid)) * G
        self.r_grid = np.real(np.fft.ifft2(R_hat))
        self.gx_grid = np.real(np.fft.ifft2(1j * KX * R_hat))
        self.gy_grid = np.real(np.fft.ifft2(1j * KY * R_hat))

        # ---- diagnostics: residual energy fraction inside the taper -------
        rr = (np.arange(n_quad) - (n_quad - 1) / 2.0)
        # distance of each node from x_c in grid units -> physical
        DXg = (GX - self.x_c[0]); DYg = (GY - self.x_c[1])
        rad = np.sqrt(DXg * DXg + DYg * DYg)
        chi_grid = _raised_cosine_scalar(rad, self.frac_in * self.L,
                                         self.frac_out * self.L)
        area = self.dx * self.dx
        e_res = np.sqrt(np.sum((chi_grid * self.r_grid) ** 2) * area)
        e_lo = np.sqrt(np.sum(vlo_grid ** 2) * area) + 1e-300
        self.residual_energy_fraction = float(e_res / e_lo)
        self._vlo_grid_peak = float(np.max(vlo_grid))
        self._r_grid_absmax = float(np.max(np.abs(self.r_grid)))

    # -- production path: grid interpolation --------------------------------
    def _interp_residual(self, Xe_np):
        r = _bilinear(self.r_grid, Xe_np, self.x0, self.dx)
        gx = _bilinear(self.gx_grid, Xe_np, self.x0, self.dx)
        gy = _bilinear(self.gy_grid, Xe_np, self.x0, self.dx)
        return r, np.stack([gx, gy], axis=1)

    def grad(self, X_eval):
        """+grad v_hat at X_eval (N,2) -- the production drift field."""
        Xe = jnp.asarray(X_eval)
        glo = np.asarray(grad_v_from_cloud(
            Xe, self.coeff_lo, jnp.asarray(self.x_c), self.L,
            self.mass_v, self.taper_s))
        chi, gchi = radial_taper(Xe, jnp.asarray(self.x_c), self.L,
                                 self.frac_in, self.frac_out)
        chi = np.asarray(chi); gchi = np.asarray(gchi)
        Xe_np = np.asarray(X_eval, dtype=np.float64)
        r, gr = self._interp_residual(Xe_np)
        g = glo + chi[:, None] * gr + r[:, None] * gchi
        return jnp.asarray(g)

    def eval(self, X_eval):
        """v_hat at X_eval (production path)."""
        Xe = jnp.asarray(X_eval)
        vlo = np.asarray(eval_v_from_cloud(
            Xe, self.coeff_lo, jnp.asarray(self.x_c), self.L,
            self.mass_v, self.taper_s))
        chi, _ = radial_taper(Xe, jnp.asarray(self.x_c), self.L,
                              self.frac_in, self.frac_out)
        chi = np.asarray(chi)
        Xe_np = np.asarray(X_eval, dtype=np.float64)
        r, _ = self._interp_residual(Xe_np)
        return jnp.asarray(vlo + chi * r)

    # -- verification path: exact kernel sum (tests only; O(N_eval * N)) -----
    def _blob_exact(self, Xe_np):
        """eta_h * mu_v and its gradient at Xe_np by direct kernel sum."""
        dx = Xe_np[:, None, :] - self.X_v[None, :, :]          # (Ne,Nv,2)
        k = gaussian_kernel(dx, self.h)                         # (Ne,Nv)
        B = self.omega_v * np.sum(k, axis=1)                   # (Ne,)
        gB = self.omega_v * np.sum(
            -(dx / (self.h * self.h)) * k[..., None], axis=1)  # (Ne,2)
        return B, gB

    def _lowblob_exact(self, Xe_np, n_quad_q=None):
        """(eta_h * v_lo dx) and its gradient at Xe_np by quadrature on the box."""
        nq = self.n_quad if n_quad_q is None else n_quad_q
        # quadrature grid (cell-centered) spanning the same window box
        step = 2.0 * self.L / nq
        ax = self.x0[0] + step * (np.arange(nq) + 0.5)
        ay = self.x0[1] + step * (np.arange(nq) + 0.5)
        GX, GY = np.meshgrid(ax, ay, indexing="ij")
        nodes = np.stack([GX.ravel(), GY.ravel()], axis=1)
        vlo = np.asarray(eval_v_from_cloud(
            jnp.asarray(nodes), self.coeff_lo, jnp.asarray(self.x_c),
            self.L, self.mass_v, self.taper_s))                # (nq^2,)
        dA = step * step
        dx = Xe_np[:, None, :] - nodes[None, :, :]             # (Ne,Nq2,2)
        k = gaussian_kernel(dx, self.h)                        # (Ne,Nq2)
        Q = np.sum(k * vlo[None, :], axis=1) * dA
        gQ = np.sum(-(dx / (self.h * self.h)) * (k * vlo[None, :])[..., None],
                    axis=1) * dA
        return Q, gQ

    def grad_exact(self, X_eval, n_quad_q=None):
        Xe_np = np.asarray(X_eval, dtype=np.float64)
        glo = np.asarray(grad_v_from_cloud(
            jnp.asarray(Xe_np), self.coeff_lo, jnp.asarray(self.x_c),
            self.L, self.mass_v, self.taper_s))
        B, gB = self._blob_exact(Xe_np)
        Q, gQ = self._lowblob_exact(Xe_np, n_quad_q)
        r = B - Q; gr = gB - gQ
        chi, gchi = radial_taper(jnp.asarray(Xe_np), jnp.asarray(self.x_c),
                                 self.L, self.frac_in, self.frac_out)
        chi = np.asarray(chi); gchi = np.asarray(gchi)
        return glo + chi[:, None] * gr + r[:, None] * gchi

    def eval_exact(self, X_eval, n_quad_q=None):
        Xe_np = np.asarray(X_eval, dtype=np.float64)
        vlo = np.asarray(eval_v_from_cloud(
            jnp.asarray(Xe_np), self.coeff_lo, jnp.asarray(self.x_c),
            self.L, self.mass_v, self.taper_s))
        B, _ = self._blob_exact(Xe_np)
        Q, _ = self._lowblob_exact(Xe_np, n_quad_q)
        r = B - Q
        chi, _ = radial_taper(jnp.asarray(Xe_np), jnp.asarray(self.x_c),
                              self.L, self.frac_in, self.frac_out)
        return vlo + np.asarray(chi) * r

    def diagnostics(self):
        return dict(
            solver_field_mode="two_level_blob_residual",
            h=self.h, h_requested=self.h_requested, h_clamped=self.h_clamped,
            h_rule=self.h_rule, c_h=self.c_h, n_quad=self.n_quad, dx=self.dx,
            N_v_in=self.N_v_in, Kg=self.Kg,
            residual_energy_fraction=self.residual_energy_fraction,
            r_absmax=self._r_grid_absmax, vlo_peak=self._vlo_grid_peak,
        )


def _raised_cosine_scalar(r, r_in, r_out):
    """Scalar raised-cosine cutoff (numpy), matching hybrid_vfield.radial_taper."""
    t = np.clip((r - r_in) / max(r_out - r_in, 1e-30), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * t))


if __name__ == "__main__":
    # quick self-check: build a field on a Gaussian v-cloud and FD-verify grad_exact
    rng = np.random.default_rng(0)
    a = 42.0
    sig = 1.0 / np.sqrt(2 * a)
    Xv = rng.normal(0.0, sig, size=(200000, 2))
    x_c = np.array([0.002, -0.003]); L = 0.08; mass_v = 10 * np.pi
    fld = BlobResidualVField(Xv, x_c, L, mass_v, Kg=8, taper_s=0.5,
                             h_rule="frac_L", c_h=0.06)
    print("[blob] diagnostics:", fld.diagnostics())
    Xe = x_c + rng.uniform(-0.5, 0.5, size=(300, 2)) * L
    g = fld.grad_exact(Xe)
    eps = 1e-6
    gfd = np.zeros_like(g)
    for d in range(2):
        e = np.zeros((1, 2)); e[0, d] = eps
        vp = fld.eval_exact(Xe + e); vm = fld.eval_exact(Xe - e)
        gfd[:, d] = (vp - vm) / (2 * eps)
    rel = np.linalg.norm(g - gfd) / (np.linalg.norm(gfd) + 1e-30)
    print(f"[blob] FD grad_exact check: rel err = {rel:.3e} "
          f"(max|g|={np.max(np.abs(g)):.2e})")
    # grid path vs exact path
    gg = np.asarray(fld.grad(Xe))
    relg = np.linalg.norm(gg - g) / (np.linalg.norm(g) + 1e-30)
    print(f"[blob] grid-vs-exact grad: rel err = {relg:.3e}")
    print("PASS" if rel < 1e-4 else "FAIL -- exact-path math wrong")
