"""
restyle_case1.py — regenerate the four LEGACY figures of the Keller--Segel
mass-balance case (Keller_Segel/case1) from the archived data, restyled with
the unified paper style (paper_style.apply_style).

Figures (include fraction f -> physical width f * 4.773 in):
  total_mass.pdf  f=0.8   mass of v vs t (left) + mass error (right)
                          [results/C_v_list.npz; notebook test.ipynb cells 2-3]
  case1_u.pdf     f=1.0   2x5 imshow: FD (top) vs particle (bottom) for u
                          [finite_difference.npz + samples_320000.npz,
                           mapping of test.ipynb cell 5: u = rho_hat / C_u]
  case1_v.pdf     f=1.0   same for v
                          [mapping of test.ipynb cell 8:
                           v = (|X2_i|/N) * rho_hat / C_v]
  case1_err.pdf   f=0.8   relative L2 error of u and v vs N0 (log-log)
                          [results/u_save.npz, results/v_save.npz
                           (test.ipynb cells 6/10: polyfit in log-log)]

NOTE on the time axis: the archived run (simulation.py, finite_difference.py)
uses T = 0.2, tau = 1e-3 (200 frames). The legacy PDFs in jcp_v3/figure were
produced from an older T = 0.5 run whose raw data is not in the archive, so
the regenerated figures span t in [0, 0.2] with snapshots at
t = 0, 0.05, 0.1, 0.15, 0.2.
"""
import os
import sys

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = "/mnt/gs21/scratch/lyuliyao/SDE_PDE/Numerical_experiment"
OUT = os.path.join(ROOT, "restyled_figures")
sys.path.insert(0, ROOT)
sys.path.insert(0, CASE_DIR)

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from paper_style import apply_style, TEXTWIDTH_IN

apply_style()
os.makedirs(OUT, exist_ok=True)

from density import generate_density_estimation  # the module the notebook used

C_u = 1 / np.pi**2          # test.ipynb cell 1
C_v = 1 / (8 * np.pi**2)

# ----------------------------------------------------------------------------
# Figure 1: total_mass.pdf   (f = 0.8)
# data mapping: test.ipynb cells 2-3 -> results/C_v_list.npz
#   C_v_list[L] = (1/C_v) * |X2_i| / N  (total mass of v), N = 10000 * 2^L
#   C_v_exact   = 1/C_u + (1/C_v - 1/C_u) * exp(-t)
# legacy figure shows N = 20000, 80000, 320000 (L = 1, 3, 5)
# ----------------------------------------------------------------------------
d = np.load(os.path.join(CASE_DIR, "results", "C_v_list.npz"))
t, mass_exact, mass = d["t"], d["C_v_exact"], d["C_v_list"]
sel = [(1, "20000", "C1"), (3, "80000", "C2"), (5, "320000", "C3")]

w = 0.8 * TEXTWIDTH_IN
fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(w, 0.48 * w), sharex=True, constrained_layout=True
)
axL.plot(t, mass_exact, "k", label="Exact")
for L, lab, c in sel:
    axL.plot(t, mass[L], color=c, label=lab)
axL.set_xlabel("Time")
axL.set_ylabel("Total Mass")
axL.legend()

axR.axhline(0.0, color="k", lw=0.8, label="Exact")
for L, lab, c in sel:
    axR.plot(t, mass[L] - mass_exact, color=c, label=lab)
axR.set_xlabel("Time")
axR.set_ylabel("Error of Total Mass")
axR.yaxis.set_label_position("right")
axR.yaxis.tick_right()

fig.savefig(os.path.join(OUT, "total_mass.pdf"))
plt.close(fig)
print("wrote total_mass.pdf")

# ----------------------------------------------------------------------------
# Figures 2-3: case1_u.pdf / case1_v.pdf  (f = 1.0)
# top row   : finite difference (finite_difference.npz, 100x100 grid)
# bottom row: particle reconstruction, N = 320000 (largest count shown in the
#             paper legends), Fourier reconstruction K = 5 periodic on
#             [0,2pi]^2 exactly as test.ipynb cells 5 (u) and 8 (v):
#                 u = rho_hat(X1_i) / C_u
#                 v = (|X2_i|/N) * rho_hat(X2_i) / C_v
# snapshots at frame indices 0, 49, 99, 149, 199  <->  t = 0,...,0.2
# ----------------------------------------------------------------------------
fd = np.load(os.path.join(CASE_DIR, "finite_difference.npz"))
u_fd, v_fd = fd["u"], fd["v"]

N0 = 320000
samples = np.load(os.path.join(CASE_DIR, "samples", f"samples_{N0}.npz"))

fcn_density_estimation, fcn_density_evaluate = generate_density_estimation(
    n_freq=5, extend="periodic", period=jnp.array([[0, 2 * np.pi], [0, 2 * np.pi]])
)
x = np.linspace(0, 2 * np.pi, 100, endpoint=False)
y = np.linspace(0, 2 * np.pi, 100, endpoint=False)
X, Y = np.meshgrid(x, y)
points = np.column_stack([X.ravel(), Y.ravel()])

frames = [0, 49, 99, 149, 199]
tlabels = ["0.0", "0.05", "0.1", "0.15", "0.2"]


def particle_field(key_prefix, idx, norm_const, mass_factor):
    P = samples[f"{key_prefix}_{idx}"]
    coeff = fcn_density_estimation(P)
    dens = jax.vmap(fcn_density_evaluate, (0, None))(points, coeff)
    dens = np.asarray(dens).reshape(X.shape) / norm_const
    if mass_factor:
        dens = (P.shape[0] / N0) * dens
    return dens


def grid_figure(top, bottom, tlabels, vmin, vmax, fname, frac):
    wfig = frac * TEXTWIDTH_IN
    fig, axs = plt.subplots(
        2, 5, figsize=(wfig, 0.42 * wfig), constrained_layout=True
    )
    im = None
    for j in range(5):
        for i, row in enumerate((top, bottom)):
            ax = axs[i, j]
            im = ax.imshow(
                row[j],
                extent=(0, 2 * np.pi, 0, 2 * np.pi),
                origin="lower",
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        axs[0, j].set_title(f"t = {tlabels[j]}")
    axs[0, 0].set_ylabel("reference", fontsize=8)
    axs[1, 0].set_ylabel("particle", fontsize=8)
    fig.colorbar(im, ax=axs, fraction=0.046, pad=0.015)
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print(f"wrote {fname}")


# --- u ---
top_u = [u_fd[k] for k in frames]
bot_u = [particle_field("X1", k, C_u, mass_factor=False) for k in frames]
grid_figure(top_u, bot_u, tlabels, 0.0, 1.0, "case1_u.pdf", frac=1.0)

# --- v ---
top_v = [v_fd[k] for k in frames]
bot_v = [particle_field("X2", k, C_v, mass_factor=True) for k in frames]
grid_figure(top_v, bot_v, tlabels, 0.0, 4.0, "case1_v.pdf", frac=1.0)

# ----------------------------------------------------------------------------
# Figure 4: case1_err.pdf  (f = 0.8)
# relative L2 error vs N0 from results/u_save.npz and results/v_save.npz,
# log-log with fitted slope (test.ipynb cells 6 and 10)
# ----------------------------------------------------------------------------
w = 0.8 * TEXTWIDTH_IN
fig, axes = plt.subplots(
    1, 2, figsize=(w, 0.5 * w), sharey=True, constrained_layout=True
)
for ax, fnpz, key in zip(axes, ["u_save.npz", "v_save.npz"], ["u", "v"]):
    d = np.load(os.path.join(CASE_DIR, "results", fnpz))
    ns, err = d["n_samples"], d["error"]
    slope, intercept = np.polyfit(np.log(ns), np.log(err), 1)
    fit = np.exp(intercept) * ns**slope
    ax.loglog(ns, err, ".-", color="C0", label="Error")
    ax.loglog(ns, fit, "--", color="C1", label=f"Fit: slope={slope:.2f}")
    ax.set_xlabel("Number of Samples")
    ax.legend()
axes[0].set_ylabel("Relative Error")
fig.savefig(os.path.join(OUT, "case1_err.pdf"))
plt.close(fig)
print("wrote case1_err.pdf")
