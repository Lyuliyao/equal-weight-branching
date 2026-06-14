"""reconstructors.py -- global spectrum + local residual reconstructions.
===========================================================================

Reconstruction operators that turn an empirical particle measure into a field,
for the resolution-diagnostic demos (CLAUDE.md Sec. 6).  The point is NOT a new
solver: it is to show that the reconstruction of a *concentrating* measure is a
separate approximation problem, and that a uniform global Fourier bandwidth is
inefficient -- the bandwidth sensitivity is localized near the particle core /
growth islands, so resolution should be spent there.

All densities use the SAME cos/sin Fourier convention as the rest of the repo
(experiments/branch_vs_weighted/common_particle.py): on a box [x0,x1]x[y0,y1],

    coeff = norm/(Lx Ly) * (weighted) mean over particles of the cos/sin basis,

so P_K integrates to 1 over the box (a PROBABILITY density); multiply by the
physical mass to get the field.  numpy only (no JAX dependency), self-contained.

Reconstructions provided:
  * global_low / global_high : P_{K} mu on the full box (the baselines).
  * hybrid_spectrum_window   : OPTION B -- low global spectrum + a high-resolution
                               local spectral window, added as a RESIDUAL
                               (local-K_l minus local-K_g) so the low modes are
                               not double counted.
  * hybrid_blob_residual     : OPTION A -- low global spectrum + a local Gaussian
                               blob residual (blob minus its low-pass projection).

Reconstruction-free diagnostics (peak/L2 are bandwidth-SENSITIVE; core radii and
core mass are reconstruction-FREE, computed straight from particles) live in
`detect_windows.py`.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Fourier reconstruction on a rectangular box (cos/sin, repo convention)
# ---------------------------------------------------------------------------
def _norm(K):
    n = np.zeros((K, K))
    n[0, 0] = 1.0; n[0, 1:] = 2.0; n[1:, 0] = 2.0; n[1:, 1:] = 4.0
    return n


def fourier_coeffs(X, w, box, K):
    """cos/sin coefficient tensors of the empirical density on `box` at bandwidth K.

    X    : (N,2) positions.  w : (N,) weights (None -> ones).  box : [[x0,x1],[y0,y1]].
    Returns a dict with the four (K,K) tensors + box metadata; P_K integrates to 1.
    """
    X = np.asarray(X, dtype=np.float64)
    (x0, x1), (y0, y1) = box
    Lx, Ly = x1 - x0, y1 - y0
    w = np.ones(X.shape[0]) if w is None else np.asarray(w, dtype=np.float64)
    fk = np.arange(K)
    tx = 2.0 * np.pi * fk[None, :] * (X[:, 0:1] - x0) / Lx     # (N,K)
    ty = 2.0 * np.pi * fk[None, :] * (X[:, 1:2] - y0) / Ly
    cx, sx = np.cos(tx), np.sin(tx)
    cy, sy = np.cos(ty), np.sin(ty)
    den = w.sum() if w.sum() > 0 else 1.0
    wc = w[:, None]
    nf = _norm(K) / (Lx * Ly)
    return {
        "cos-cos": nf * ((wc * cx).T @ cy) / den,
        "cos-sin": nf * ((wc * cx).T @ sy) / den,
        "sin-cos": nf * ((wc * sx).T @ cy) / den,
        "sin-sin": nf * ((wc * sx).T @ sy) / den,
        "box": box, "K": K,
    }


def eval_grid(coeff, XX, YY):
    """Evaluate the probability density on a meshgrid XX,YY (indexing='xy')."""
    (x0, x1), (y0, y1) = coeff["box"]
    Lx, Ly = x1 - x0, y1 - y0
    K = coeff["K"]
    fk = np.arange(K)
    xv = XX[0, :]; yv = YY[:, 0]
    tx = 2.0 * np.pi * fk[None, :] * (xv[:, None] - x0) / Lx   # (Nx,K)
    ty = 2.0 * np.pi * fk[None, :] * (yv[:, None] - y0) / Ly   # (Ny,K)
    cxg, sxg = np.cos(tx), np.sin(tx)
    cyg, syg = np.cos(ty), np.sin(ty)
    Z = (cyg @ coeff["cos-cos"].T @ cxg.T
         + syg @ coeff["cos-sin"].T @ cxg.T
         + cyg @ coeff["sin-cos"].T @ sxg.T
         + syg @ coeff["sin-sin"].T @ sxg.T)
    return Z                                                   # (Ny,Nx)


def eval_points(coeff, pts):
    """Evaluate the probability density at points (M,2)."""
    (x0, x1), (y0, y1) = coeff["box"]
    Lx, Ly = x1 - x0, y1 - y0
    K = coeff["K"]
    fk = np.arange(K)
    tx = 2.0 * np.pi * fk[None, :] * (pts[:, 0:1] - x0) / Lx
    ty = 2.0 * np.pi * fk[None, :] * (pts[:, 1:2] - y0) / Ly
    cx, sx = np.cos(tx), np.sin(tx)
    cy, sy = np.cos(ty), np.sin(ty)
    Z = (np.einsum('nk,nl,kl->n', cx, cy, coeff["cos-cos"])
         + np.einsum('nk,nl,kl->n', cx, sy, coeff["cos-sin"])
         + np.einsum('nk,nl,kl->n', sx, cy, coeff["sin-cos"])
         + np.einsum('nk,nl,kl->n', sx, sy, coeff["sin-sin"]))
    return Z


# ---------------------------------------------------------------------------
# Baselines: global low-K and high-K reconstruction on the full box.
# ---------------------------------------------------------------------------
def global_recon(X, w, box, K, mass, XX, YY):
    """Physical field = mass * P_K mu  on the grid XX,YY."""
    coeff = fourier_coeffs(X, w, box, K)
    return mass * eval_grid(coeff, XX, YY), coeff


# ---------------------------------------------------------------------------
# Window geometry helpers
# ---------------------------------------------------------------------------
def window_box(center, half):
    """Square window box of half-size `half` centered at `center`."""
    cx, cy = center
    return [[cx - half, cx + half], [cy - half, cy + half]]


def in_window(X, wbox):
    (x0, x1), (y0, y1) = wbox
    Xn = np.asarray(X)
    return ((Xn[:, 0] >= x0) & (Xn[:, 0] <= x1)
            & (Xn[:, 1] >= y0) & (Xn[:, 1] <= y1))


def radial_taper(XX, YY, center, r_in, r_out):
    """Smooth raised-cosine taper chi: 1 for r<=r_in, 0 for r>=r_out."""
    r = np.sqrt((XX - center[0]) ** 2 + (YY - center[1]) ** 2)
    chi = np.ones_like(r)
    band = (r > r_in) & (r < r_out)
    chi[band] = 0.5 * (1.0 + np.cos(np.pi * (r[band] - r_in) / (r_out - r_in)))
    chi[r >= r_out] = 0.0
    return chi


# ---------------------------------------------------------------------------
# OPTION B: global low spectrum + local high-resolution spectral window.
# The local correction is a RESIDUAL (local-K_l minus local-K_g) so the modes
# already in the global low spectrum are not double counted.
# ---------------------------------------------------------------------------
def hybrid_spectrum_window(X, w, box, mass, mass_per_particle, center, half,
                           Kg, Kl, XX, YY, pad=1.5, taper_frac=0.85):
    """u_hyb = mass*P_{Kg}^box  +  chi * massW*(P_{Kl}^W - P_{Kg}^W).

    center, half : the local window (from particle diagnostics, NOT the image).
    pad          : reconstruct the local cloud on a padded window (pad*half) to
                   reduce periodic-window edge bias; the residual is tapered to 0
                   well inside the padded edge by `taper_frac`.
    Returns (u_hyb, u_lo, u_res, info).
    """
    # global low background
    u_lo = mass * eval_grid(fourier_coeffs(X, w, box, Kg), XX, YY)

    # local cloud on the padded window
    wbox = window_box(center, pad * half)
    sel = in_window(X, wbox)
    Xw = np.asarray(X)[sel]
    ww = None if w is None else np.asarray(w)[sel]
    massW = mass_per_particle * (Xw.shape[0] if ww is None else ww.sum())

    coeff_hi = fourier_coeffs(Xw, ww, wbox, Kl)
    u_loc_hi = massW * eval_grid(coeff_hi, XX, YY)         # high-resolution local field

    # Pi_lo^{W} : the low-mode component already represented by the GLOBAL spectrum
    # is exactly u_lo restricted to the window.  Subtracting u_lo (not the window's
    # OWN low reconstruction, which resolves finer on the smaller box and would
    # over-subtract) avoids both double counting AND leaving a low-mode gap.  The
    # result is a smooth blend: local high-res inside the window, global low
    # outside, with the taper chi handling the transition.
    chi = radial_taper(XX, YY, center, taper_frac * half, pad * half)
    u_res = chi * (u_loc_hi - u_lo)                        # residual over the global low
    u_hyb = u_lo + u_res
    info = dict(n_local=int(Xw.shape[0]), mass_window=float(massW),
                global_modes=Kg * Kg, local_modes=Kl * Kl,
                wbox=wbox, center=list(center), half=float(half))
    return u_hyb, u_lo, u_res, info


# ---------------------------------------------------------------------------
# OPTION A: global low spectrum + local Gaussian-blob residual.
# residual = chi * (blob - lowpass_{Kg}(blob)) so the low modes are not doubled.
# ---------------------------------------------------------------------------
def local_blob_density(X, w, mass_per_particle, center, half, h, XX, YY, pad=1.5):
    """Gaussian-blob kernel density of the in-(padded)-window particles on the grid.

    Returns the PHYSICAL field sum_i (mass_pp w_i) eta_h(x - X_i), eta_h a 2D
    Gaussian of width h, computed by deposit-then-smooth (histogram on the grid +
    Gaussian filter) so the cost is O(N + grid) rather than O(N * grid).  The
    deposition conserves the in-window mass exactly; the Gaussian filter spreads
    each deposited mass over width h.
    """
    from scipy.ndimage import gaussian_filter
    wbox = window_box(center, pad * half)
    sel = in_window(X, wbox)
    Xw = np.asarray(X)[sel]
    ww = (np.ones(Xw.shape[0]) if w is None else np.asarray(w)[sel])
    xs = XX[0, :]; ys = YY[:, 0]
    dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
    # bin edges aligned with the grid cell centers
    xe = np.concatenate([[xs[0] - dx / 2], xs + dx / 2])
    ye = np.concatenate([[ys[0] - dy / 2], ys + dy / 2])
    H, _, _ = np.histogram2d(Xw[:, 1], Xw[:, 0], bins=[ye, xe], weights=ww)  # (Ny,Nx)
    # mass per cell = mass_pp * weight; density = mass / cell_area, then smooth
    density = mass_per_particle * H / (dx * dy)
    sigma_pix = (h / dx, h / dy)
    field = gaussian_filter(density, sigma=sigma_pix, mode="constant")
    return field, int(Xw.shape[0])


def lowpass_field(field, XX, YY, center, half, Kg, pad=1.5):
    """Low-pass a window field to its first Kg Fourier modes on the padded window.

    Returns the K_g-bandlimited reconstruction of `field` on the SAME grid by
    projecting the gridded field onto the cos/sin basis (deterministic quadrature
    on the padded window)."""
    wbox = window_box(center, pad * half)
    (x0, x1), (y0, y1) = wbox
    sel = (XX >= x0) & (XX <= x1) & (YY >= y0) & (YY <= y1)
    # build a regular sub-grid mask region; approximate projection by quadrature
    Lx, Ly = x1 - x0, y1 - y0
    fk = np.arange(Kg)
    # use the grid nodes inside the window for a midpoint quadrature
    out = np.zeros_like(field)
    xs = XX[0, :]; ys = YY[:, 0]
    inx = (xs >= x0) & (xs <= x1); iny = (ys >= y0) & (ys <= y1)
    if inx.sum() < 2 or iny.sum() < 2:
        return out
    sub = field[np.ix_(iny, inx)]
    Xc = xs[inx]; Yc = ys[iny]
    dx = (Xc[-1] - Xc[0]) / (Xc.size - 1); dy = (Yc[-1] - Yc[0]) / (Yc.size - 1)
    txi = 2 * np.pi * fk[None, :] * (Xc[:, None] - x0) / Lx
    tyi = 2 * np.pi * fk[None, :] * (Yc[:, None] - y0) / Ly
    cxi, sxi = np.cos(txi), np.sin(txi)
    cyi, syi = np.cos(tyi), np.sin(tyi)
    nf = _norm(Kg) / (Lx * Ly)
    # coefficients via quadrature: integral field * basis / (Lx Ly) * norm
    Ccc = nf * (cyi.T @ sub.T @ cxi) * dx * dy
    Ccs = nf * (syi.T @ sub.T @ cxi) * dx * dy
    Scc = nf * (cyi.T @ sub.T @ sxi) * dx * dy
    Scs = nf * (syi.T @ sub.T @ sxi) * dx * dy
    coeff = {"cos-cos": Ccc.T, "cos-sin": Ccs.T, "sin-cos": Scc.T,
             "sin-sin": Scs.T, "box": wbox, "K": Kg}
    out[np.ix_(iny, inx)] = eval_grid(coeff, XX[np.ix_(iny, inx)], YY[np.ix_(iny, inx)])
    return out


def hybrid_blob_residual(X, w, box, mass, mass_per_particle, center, half, h,
                         Kg, XX, YY, pad=1.5, taper_frac=0.85):
    """u_hyb = u_lo + chi*(blob - u_lo)  with u_lo = mass*P_{Kg}^box.

    Same clean-blend residual as `hybrid_spectrum_window`: the low-mode component
    already represented by the global spectrum is exactly u_lo on the window, so
    subtracting u_lo (rather than a separate low-pass of the blob) both avoids
    double counting and leaves no low-mode gap.  Inside the window the blob
    supplies the high-frequency core; outside, the global low spectrum.  Returns
    (u_hyb, u_lo, u_res, info)."""
    u_lo = mass * eval_grid(fourier_coeffs(X, w, box, Kg), XX, YY)
    blob, n_local = local_blob_density(X, w, mass_per_particle, center, half, h,
                                       XX, YY, pad=pad)
    chi = radial_taper(XX, YY, center, taper_frac * half, pad * half)
    u_res = chi * (blob - u_lo)
    u_hyb = u_lo + u_res
    info = dict(n_local=n_local, blob_h=float(h), global_modes=Kg * Kg)
    return u_hyb, u_lo, u_res, info


# ---------------------------------------------------------------------------
# Residual-particle retention / accept-rate (CLAUDE.md Sec. 6, updated).
#
# IMPORTANT: the "accept rate" here is a RECONSTRUCTION-ENRICHMENT rate -- a
# fraction of particles RETAINED for the local residual reconstruction.  It is
# NOT a Metropolis acceptance probability and NOT a new particle-dynamics
# resampling step; the PDE dynamics are untouched.  This is applied to SAVED
# particle snapshots.
# ---------------------------------------------------------------------------
def _deposit_blob(pts, w, h, XX, YY):
    """Deposit weighted points and Gaussian-smooth to width h (mass-conserving)."""
    from scipy.ndimage import gaussian_filter
    xs = XX[0, :]; ys = YY[:, 0]
    dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
    xe = np.concatenate([[xs[0] - dx / 2], xs + dx / 2])
    ye = np.concatenate([[ys[0] - dy / 2], ys + dy / 2])
    H, _, _ = np.histogram2d(np.asarray(pts)[:, 1], np.asarray(pts)[:, 0],
                             bins=[ye, xe], weights=w)
    density = H / (dx * dy)
    return gaussian_filter(density, sigma=(h / dx, h / dy), mode="constant")


def residual_particle_acceptance(X, weights, u_lo_eval, u_pilot_eval, window,
                                 mode="ht", B_target=None, q_min=0.0, eps=1e-12,
                                 rng=None, stochastic=True):
    """Return particles RETAINED for a local residual reconstruction.

    mode="ht"       : unbiased Horvitz-Thompson sketch of the empirical part of
                      mu - P_Kg mu dx (retained weights divided by q_i).
    mode="positive" : positive-excess thinning (retain particles representing the
                      positive local excess above the global spectrum).

    u_lo_eval(pts)    -> global low field at points.
    u_pilot_eval(pts) -> nonnegative local pilot density at points (or None, in
                         which case the HT score is the taper only, i.e. uniform
                         in-window thinning).
    Returns (res_idx, res_weights, aux).  NOT a particle-dynamics resampling step.
    """
    X = np.asarray(X)
    weights = np.ones(X.shape[0]) if weights is None else np.asarray(weights)
    idx = np.where(window.in_padded(X))[0]
    if idx.size == 0:
        return idx, np.zeros(0), dict(window_indices=idx, accept_prob=np.zeros(0),
                                      mass_accept_rate=0.0, expected_residual_count=0.0,
                                      mean_accept_rate=0.0, n_retained=0,
                                      effective_HT_sample_size=0.0)
    u_lo_i = u_lo_eval(X[idx])
    u_pilot_i = (np.maximum(u_pilot_eval(X[idx]), 0.0)
                 if u_pilot_eval is not None else None)
    chi_i = window.taper(X[idx])

    if mode == "ht":
        if u_pilot_i is not None:
            score = chi_i * np.abs(u_pilot_i - u_lo_i) / (
                u_pilot_i + np.abs(u_lo_i) + eps)
        else:
            score = chi_i                                  # simple default score
        if B_target is None:
            raise ValueError("B_target is required for the HT residual sketch")
        q = np.minimum(1.0, np.maximum(q_min, B_target * score / (score.sum() + eps)))
        if stochastic:
            if rng is None:
                raise ValueError("rng required for stochastic HT thinning")
            keep = rng.random(idx.size) < q
        else:
            keep = q > 0
        res_idx = idx[keep]
        res_weights = weights[res_idx] / np.maximum(q[keep], eps)
        accept_prob = q
    elif mode == "positive":
        if u_pilot_i is None:
            raise ValueError("u_pilot_eval required for the positive-excess mode")
        u_lo_pos = np.maximum(u_lo_i, 0.0)
        alpha = np.clip((u_pilot_i - u_lo_pos) / (u_pilot_i + eps), 0.0, 1.0)
        if stochastic:
            if rng is None:
                raise ValueError("rng required for stochastic positive thinning")
            keep = rng.random(idx.size) < alpha
            res_idx = idx[keep]; res_weights = weights[res_idx]
        else:
            keep = alpha > 0
            res_idx = idx[keep]; res_weights = weights[res_idx] * alpha[keep]
        accept_prob = alpha
    else:
        raise ValueError(f"unknown residual thinning mode: {mode}")

    htw = res_weights  # = omega_i/q_i for HT
    eff = float((htw.sum() ** 2) / np.sum(htw ** 2)) if htw.size and np.sum(htw**2) > 0 else 0.0
    aux = dict(
        window_indices=idx, accept_prob=accept_prob,
        mean_accept_rate=float(np.mean(accept_prob)),
        mass_accept_rate=float(np.sum(weights[idx] * accept_prob)
                               / max(np.sum(weights[idx]), 1e-300)),
        min_accept_rate=float(np.min(accept_prob)),
        max_accept_rate=float(np.max(accept_prob)),
        expected_residual_count=float(np.sum(accept_prob)),
        n_retained=int(res_idx.size),
        n_window=int(idx.size),
        effective_HT_sample_size=eff,
    )
    return res_idx, res_weights, aux


def ht_residual_reconstruction(X, weights, box, mass, mass_per_particle, window,
                               Kg, h, XX, YY, B_target, rng, u_pilot_eval=None):
    """u_hyb = u_lo + chi*( eta_h * mu_retained^HT  -  eta_h*(u_lo 1_W dx) ).

    The empirical residual is a blob of the HT-RETAINED particles (weights
    omega_i/q_i); the low-spectrum part eta_h*(u_lo 1_W) is subtracted
    deterministically so the global spectrum is not double counted.  Returns
    (u_hyb, u_lo, u_res, aux).
    """
    coeff_lo = fourier_coeffs(X, weights, box, Kg)
    u_lo = mass * eval_grid(coeff_lo, XX, YY)
    u_lo_eval = lambda pts: mass * eval_points(coeff_lo, pts)

    res_idx, res_w, aux = residual_particle_acceptance(
        X, weights, u_lo_eval, u_pilot_eval, window, mode="ht",
        B_target=B_target, rng=rng, stochastic=True)

    # empirical blob of retained particles, weighted by mass_pp * (omega_i/q_i)
    if res_idx.size:
        emp = _deposit_blob(np.asarray(X)[res_idx], mass_per_particle * res_w, h, XX, YY)
    else:
        emp = np.zeros_like(XX)
    # deterministic low part: eta_h * (u_lo restricted to padded window)
    from scipy.ndimage import gaussian_filter
    wmask = window.in_padded(np.stack([XX.ravel(), YY.ravel()], axis=1)).reshape(XX.shape)
    xs = XX[0, :]; dx = xs[1] - xs[0]; dy = YY[:, 0][1] - YY[:, 0][0]
    det_low = gaussian_filter(u_lo * wmask, sigma=(h / dy, h / dx), mode="constant")

    chi = radial_taper(XX, YY, window.center, window.r_in, window.r_out)
    u_res = chi * (emp - det_low)
    u_hyb = u_lo + u_res
    aux["residual_energy_fraction"] = (
        field_L2(u_res, XX, YY) / max(field_L2(u_lo, XX, YY), 1e-30))
    aux["B_target"] = int(B_target)
    return u_hyb, u_lo, u_res, aux


def positive_residual_reconstruction(X, weights, box, mass, mass_per_particle,
                                     window, Kg, h, XX, YY, rng):
    """Positive-excess residual reconstruction (positivity-preserving, labelled
    positive-only).  Pilot = local blob; retain positive excess over u_lo^+.
    Returns (u_hyb_plus, u_lo, u_res_plus, aux) and reports the residual mass
    imbalance Delta M = int (r+ - r-)."""
    coeff_lo = fourier_coeffs(X, weights, box, Kg)
    u_lo = mass * eval_grid(coeff_lo, XX, YY)
    u_lo_eval = lambda pts: mass * eval_points(coeff_lo, pts)
    # local pilot density (blob) evaluated at particle points via the grid
    pilot_grid, _ = local_blob_density(X, weights, mass_per_particle,
                                       window.center, window.half, h, XX, YY,
                                       pad=window.pad)
    pilot_eval = _grid_interpolator(pilot_grid, XX, YY)
    res_idx, res_w, aux = residual_particle_acceptance(
        X, weights, u_lo_eval, pilot_eval, window, mode="positive", rng=rng,
        stochastic=True)
    if res_idx.size:
        emp = _deposit_blob(np.asarray(X)[res_idx], mass_per_particle * res_w, h, XX, YY)
    else:
        emp = np.zeros_like(XX)
    chi = radial_taper(XX, YY, window.center, window.r_in, window.r_out)
    u_res = chi * emp                                       # positive-only
    u_hyb = u_lo + u_res
    # residual mass imbalance r+ - r- over the window
    u_lo_pos = np.maximum(u_lo, 0.0)
    rplus = np.maximum(pilot_grid - u_lo_pos, 0.0)
    rminus = np.maximum(u_lo_pos - pilot_grid, 0.0)
    dx = XX[0, 1] - XX[0, 0]; dy = YY[1, 0] - YY[0, 0]
    wmask = window.in_padded(np.stack([XX.ravel(), YY.ravel()], axis=1)).reshape(XX.shape)
    aux["positive_residual_mass"] = float(np.sum(rplus * wmask) * dx * dy)
    aux["negative_residual_mass"] = float(np.sum(rminus * wmask) * dx * dy)
    aux["residual_mass_imbalance"] = aux["positive_residual_mass"] - aux["negative_residual_mass"]
    aux["residual_energy_fraction"] = (
        field_L2(u_res, XX, YY) / max(field_L2(u_lo, XX, YY), 1e-30))
    return u_hyb, u_lo, u_res, aux


def _grid_interpolator(field, XX, YY):
    """Return a callable pts->field value by nearest-grid lookup (cheap pilot eval)."""
    xs = XX[0, :]; ys = YY[:, 0]
    x0, dx = xs[0], xs[1] - xs[0]
    y0, dy = ys[0], ys[1] - ys[0]
    nx, ny = xs.size, ys.size

    def ev(pts):
        pts = np.asarray(pts)
        ix = np.clip(np.round((pts[:, 0] - x0) / dx).astype(int), 0, nx - 1)
        iy = np.clip(np.round((pts[:, 1] - y0) / dy).astype(int), 0, ny - 1)
        return field[iy, ix]
    return ev


# ---------------------------------------------------------------------------
# Field diagnostics (bandwidth-SENSITIVE -- always label as such).
# ---------------------------------------------------------------------------
def field_peak(u):
    return float(np.max(u))


def field_L2(u, XX, YY):
    dx = XX[0, 1] - XX[0, 0]; dy = YY[1, 0] - YY[0, 0]
    return float(np.sqrt(np.sum(u ** 2) * abs(dx * dy)))


def field_min(u):
    return float(np.min(u))
