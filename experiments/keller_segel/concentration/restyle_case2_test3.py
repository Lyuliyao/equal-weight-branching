"""
restyle_case2_test3.py — regenerate the two LEGACY figures of the
Keller--Segel concentration (pre-blow-up) case from archived data, restyled
with the unified paper style.

Figures:
  case2_u.pdf     f=0.8   1x4 imshow row: particle solution u at
                          t = 0, 5e-5, 1e-4, 1.5e-4 (per-panel color scale).
                          Data: ../case2_test1/samples/samples_N_40000.npz
                          (the archive test.ipynb cell 6 of this case loads:
                          dt = 1e-8, frames saved every 100 steps, so
                          t = 1e-8 * frame index -> X1_{0,5000,10000,15000}).
                          u is the normalized 2-D histogram density of X1
                          (cf. test.ipynb cell 6 plt.hist2d, bins=100):
                          u = rho_hist / C_u with C_u = 1/(10*pi); the X1
                          count is constant in this model so no mass factor.
  comparison.pdf  f=0.85  2x3 grid at t = 5e-5: finite difference at
                          200^2/400^2/800^2 (top) vs particle method at
                          N = 4e4/1.6e5/6.4e5 (bottom).  Faithful re-run of
                          test.ipynb cell 5 (the cell that wrote the
                          byte-identical comparison.pdf used by the paper):
                          zero-extended Fourier reconstruction, n_freq=10,
                          1%/99% percentile window, vmin=0, per-panel vmax
                          (the cell defined norm/u_max but never passed it to
                          imshow), cmap RdBu_r, colorbar attached to the
                          bottom-right panel.
                          DATA NOTE: the archived f2_dt_1e_10 files for
                          200^2/400^2 retain only the initial frame, so the
                          top row uses ../case2_test1/f2_dt_1e_8 frame 50
                          (same fields, dt = 1e-8; frame-50 maxima
                          20461.0/20890.1 vs 20462.1/20891.3 printed in the
                          notebook's stored cell-5 output); the 800^2 frame
                          comes from the local f2_dt_1e_10 archive whose
                          frame-50 max 21003.38892824762 matches the stored
                          output exactly.
"""
import os
import sys

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = "/mnt/gs21/scratch/lyuliyao/SDE_PDE/Numerical_experiment"
OUT = os.path.join(ROOT, "restyled_figures")
sys.path.insert(0, ROOT)

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from paper_style import apply_style, TEXTWIDTH_IN

apply_style()
os.makedirs(OUT, exist_ok=True)

C_u = 1 / (10 * jnp.pi)
SAMPLES_DIR = os.path.join(CASE_DIR, "..", "case2_test1", "samples")


# ----------------------------------------------------------------------------
# zero-extended Fourier density estimator — copied verbatim (math unchanged)
# from test.ipynb cell 2 of this case
# ----------------------------------------------------------------------------
def generate_density_estimation(n_freq=10, extend="periodic", period=None):
    K = n_freq
    norm = jnp.zeros((K, K))
    norm = norm.at[0, 0].set(1)
    norm = norm.at[0, 1:].set(2)
    norm = norm.at[1:, 0].set(2)
    norm = norm.at[1:, 1:].set(4)

    def density_estimation(data, mask=None, coeff=None):
        x_data, y_data = data[:, 0], data[:, 1]
        if extend == "periodic":
            x_min, x_max = period[0][0], period[0][1]
            y_min, y_max = period[1][0], period[1][1]
        elif extend == "zero":
            x_min, x_max = coeff["x_min"], coeff["x_max"]
            y_min, y_max = coeff["y_min"], coeff["y_max"]
        Lx = x_max - x_min
        Ly = y_max - y_min
        coeffs = {}
        freq_k = jnp.arange(K)
        freq_l = jnp.arange(K)
        theta_x = 2 * jnp.pi * freq_k[None, :] * (x_data[:, None] - x_min) / Lx
        theta_y = 2 * jnp.pi * freq_l[None, :] * (y_data[:, None] - y_min) / Ly
        basis_cos_cos = jnp.cos(theta_x[..., None]) * jnp.cos(theta_y[:, None, :])
        basis_cos_sin = jnp.cos(theta_x[..., None]) * jnp.sin(theta_y[:, None, :])
        basis_sin_cos = jnp.sin(theta_x[..., None]) * jnp.cos(theta_y[:, None, :])
        basis_sin_sin = jnp.sin(theta_x[..., None]) * jnp.sin(theta_y[:, None, :])
        norm_factor = norm / (Lx * Ly)
        if mask is not None:
            coeffs["cos-cos"] = norm_factor * jnp.sum(basis_cos_cos * mask[:, None, None], axis=0) / jnp.sum(mask)
            coeffs["cos-sin"] = norm_factor * jnp.sum(basis_cos_sin * mask[:, None, None], axis=0) / jnp.sum(mask)
            coeffs["sin-cos"] = norm_factor * jnp.sum(basis_sin_cos * mask[:, None, None], axis=0) / jnp.sum(mask)
            coeffs["sin-sin"] = norm_factor * jnp.sum(basis_sin_sin * mask[:, None, None], axis=0) / jnp.sum(mask)
        else:
            coeffs["cos-cos"] = norm_factor * jnp.mean(basis_cos_cos, axis=0)
            coeffs["cos-sin"] = norm_factor * jnp.mean(basis_cos_sin, axis=0)
            coeffs["sin-cos"] = norm_factor * jnp.mean(basis_sin_cos, axis=0)
            coeffs["sin-sin"] = norm_factor * jnp.mean(basis_sin_sin, axis=0)
        coeffs["Lx"], coeffs["Ly"] = Lx, Ly
        coeffs["x_min"], coeffs["x_max"] = x_min, x_max
        coeffs["y_min"], coeffs["y_max"] = y_min, y_max
        coeffs["K"] = K
        return coeffs

    def density_evaluate(points, coeff):
        x_data, y_data = points[0], points[1]
        x_min, x_max = coeff["x_min"], coeff["x_max"]
        y_min, y_max = coeff["y_min"], coeff["y_max"]
        if extend == "periodic":
            x_data = jnp.mod(x_data - x_min, coeff["Lx"]) + x_min
            y_data = jnp.mod(y_data - y_min, coeff["Ly"]) + y_min
        Lx, Ly, K = coeff["Lx"], coeff["Ly"], coeff["K"]
        freq_k = jnp.arange(K)
        freq_l = jnp.arange(K)
        theta_x = 2 * jnp.pi * freq_k * (x_data - x_min) / Lx
        theta_y = 2 * jnp.pi * freq_l * (y_data - y_min) / Ly
        Z = jnp.zeros((K, K))
        Z += coeff["cos-cos"] * jnp.cos(theta_x[..., None]) * jnp.cos(theta_y[None, :])
        Z += coeff["cos-sin"] * jnp.cos(theta_x[..., None]) * jnp.sin(theta_y[None, :])
        Z += coeff["sin-cos"] * jnp.sin(theta_x[..., None]) * jnp.cos(theta_y[None, :])
        Z += coeff["sin-sin"] * jnp.sin(theta_x[..., None]) * jnp.sin(theta_y[None, :])
        Z = jnp.sum(Z)
        if extend == "zero":
            Z = Z * (x_data < x_max) * (x_data > x_min) * (y_data < y_max) * (y_data > y_min)
        return Z

    return density_estimation, density_evaluate


# ----------------------------------------------------------------------------
# Figure 5: case2_u.pdf  (f = 0.8)
# ----------------------------------------------------------------------------
N0 = 40000
data = np.load(os.path.join(SAMPLES_DIR, f"samples_N_{N0}.npz"))
times = [
    (0, r"$t = 0$"),
    (5000, r"$t = 5\times 10^{-5}$"),
    (10000, r"$t = 1\times 10^{-4}$"),
    (15000, r"$t = 1.5\times 10^{-4}$"),
]
BINS = 100
edges = np.linspace(-0.5, 0.5, BINS + 1)

w = 0.8 * TEXTWIDTH_IN
fig, axs = plt.subplots(1, 4, figsize=(w, 0.36 * w), constrained_layout=True)
for ax, (idx, title) in zip(axs, times):
    X1 = data[f"X1_{idx}"]
    H, _, _ = np.histogram2d(X1[:, 0], X1[:, 1], bins=[edges, edges], density=True)
    u = (X1.shape[0] / N0) * H / float(C_u)
    im = ax.imshow(
        u.T,
        extent=(-0.5, 0.5, -0.5, 0.5),
        origin="lower",
        cmap="RdBu_r",
        vmin=0.0,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cb = fig.colorbar(im, ax=ax, location="bottom", fraction=0.08, pad=0.04)
    cb.formatter = fmt
    cb.ax.tick_params(labelsize=6)
    cb.ax.xaxis.get_offset_text().set_size(6)
    cb.update_ticks()
fig.savefig(os.path.join(OUT, "case2_u.pdf"))
plt.close(fig)
print("wrote case2_u.pdf")

# ----------------------------------------------------------------------------
# Figure 6: comparison.pdf  (f = 0.85) — faithful re-run of test.ipynb cell 5
# ----------------------------------------------------------------------------
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))

x_range = 0.05

w = 0.85 * TEXTWIDTH_IN
fig, axs = plt.subplots(
    2, 3, figsize=(w, 0.667 * w), sharex=True, sharey=True, constrained_layout=True
)


def imshow_ax(ax, X, Y, Z, vmin=0.0, vmax=None):
    extent = (X.min(), X.max(), Y.min(), Y.max())
    im = ax.imshow(
        Z,
        extent=extent,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap="RdBu_r",
    )
    ax.set_xlim(-x_range * 0.99, x_range * 0.99)
    ax.set_ylim(-x_range * 0.99, x_range * 0.99)
    ax.set_xticks([-0.03, 0.0, 0.03], labels=["-0.03", "0", "0.03"])
    ax.set_yticks([-0.03, 0.0, 0.03], labels=["-0.03", "0", "0.03"])
    ax.grid(False)
    return im


# --- top row: finite-difference fields at t = 5e-5 (frame 50) ---
FD_FILES = {
    200: os.path.join(CASE_DIR, "..", "case2_test1", "f2_dt_1e_8", "finite_difference_200_200.npz"),
    400: os.path.join(CASE_DIR, "..", "case2_test1", "f2_dt_1e_8", "finite_difference_400_400.npz"),
    800: os.path.join(CASE_DIR, "f2_dt_1e_10", "finite_difference_800_800.npz"),
}
for index, N in enumerate([200, 400, 800]):
    fd = np.load(FD_FILES[N])
    x = np.linspace(-0.5, 0.5, N, endpoint=False)
    y = np.linspace(-0.5, 0.5, N, endpoint=False)
    X, Y = np.meshgrid(x, y)
    imshow_ax(axs[0, index], X, Y, fd["u"][50])
    print(f"FD {N}: max u = {np.max(fd['u'][50]):.1f}")

# --- bottom row: particle reconstruction at t = 5e-5 (X1_5000, dt = 1e-8) ---
fcn_density_estimation, fcn_density_evaluate = generate_density_estimation(
    n_freq=10, extend="zero"
)
last_im = None
for index, n_samples in enumerate([40000, 160000, 640000]):
    d = np.load(os.path.join(SAMPLES_DIR, f"samples_N_{n_samples}.npz"))
    X1 = d["X1_5000"]
    coeff1 = {
        "x_max": jnp.percentile(X1[:, 0], 99),
        "x_min": jnp.percentile(X1[:, 0], 1),
        "y_max": jnp.percentile(X1[:, 1], 99),
        "y_min": jnp.percentile(X1[:, 1], 1),
    }
    coeff_rho_1 = fcn_density_estimation(X1, np.ones(X1.shape[0]), coeff1)
    x = np.linspace(-x_range, x_range, 1000, endpoint=True)
    y = np.linspace(-x_range, x_range, 1000, endpoint=True)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    density = jax.vmap(fcn_density_evaluate, (0, None))(points, coeff_rho_1)
    density = np.asarray(density).reshape(X.shape)
    u = density / C_u
    print(f"N={n_samples}: max u = {np.max(u):.1f}")
    last_im = imshow_ax(axs[1, index], X, Y, np.asarray(u))

cbar = fig.colorbar(last_im, ax=axs, location="right", fraction=0.035, pad=0.02)
cbar.formatter = formatter
cbar.update_ticks()
cbar.set_label(r"$u$")
# keep the exact f*linewidth page width (a tight bbox would crop ~8 pt of
# horizontal slack left by the equal-aspect panels)
from matplotlib.transforms import Bbox

fig.savefig(
    os.path.join(OUT, "comparison.pdf"),
    bbox_inches=Bbox([[0, 0], [w, 0.667 * w]]),
)
plt.close(fig)
print("wrote comparison.pdf")
