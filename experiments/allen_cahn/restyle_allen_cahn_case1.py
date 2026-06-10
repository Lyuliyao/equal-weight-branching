"""
restyle_allen_cahn_case1.py — regenerate the LEGACY Allen--Cahn figure
(case_allen_cahn.pdf, include fraction f = 0.8) from archived data, restyled
with the unified paper style.

Data mapping (finite_difference.ipynb cell 2 of this case, which created
kde_density_2pi_domain.npz):
  - "u_<k>"   : finite-difference reference u at frame k (dt = 1e-3, so
                t = k * 1e-3), 100x100 grid on [0,2pi]^2.
  - "rho_<k>" : particle-method field, Fourier KDE of X1_<k> from
                samples_N_400000.npz already normalized by
                (|X1_k|/N)/C_u  ->  directly comparable to u.
Layout: 2x5 grid at t = 0, 0.2, 0.4, 0.6, 0.8 (manuscript caption);
top row FD reference, bottom row particle method; vmin = 0, vmax = 1
(as in the creating notebook), cmap RdBu_r as in the legacy PDF, one shared
colorbar.
"""
import os
import sys

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = "/mnt/gs21/scratch/lyuliyao/SDE_PDE/Numerical_experiment"
OUT = os.path.join(ROOT, "restyled_figures")
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
from paper_style import apply_style, TEXTWIDTH_IN

apply_style()
os.makedirs(OUT, exist_ok=True)

d = np.load(os.path.join(CASE_DIR, "kde_density_2pi_domain.npz"))
keys = [0, 200, 400, 600, 800]
tlabels = ["0.0", "0.2", "0.4", "0.6", "0.8"]
top = [d[f"u_{k}"] for k in keys]          # FD reference
bottom = [d[f"rho_{k}"] for k in keys]     # particle method

w = 0.8 * TEXTWIDTH_IN
fig, axs = plt.subplots(2, 5, figsize=(w, 0.45 * w), constrained_layout=True)
im = None
for j in range(5):
    for i, row in enumerate((top, bottom)):
        ax = axs[i, j]
        im = ax.imshow(
            row[j],
            extent=(0, 2 * np.pi, 0, 2 * np.pi),
            origin="lower",
            cmap="RdBu_r",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    axs[0, j].set_title(f"t = {tlabels[j]}")
axs[0, 0].set_ylabel("reference", fontsize=8)
axs[1, 0].set_ylabel("particle", fontsize=8)
fig.colorbar(im, ax=axs, fraction=0.046, pad=0.015)
fig.savefig(os.path.join(OUT, "case_allen_cahn.pdf"))
plt.close(fig)
print("wrote case_allen_cahn.pdf")
