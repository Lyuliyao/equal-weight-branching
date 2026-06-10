"""
restyle_case3_2.py — regenerate the two LEGACY figures of the non-conservative
logistic Keller--Segel case (Keller_Segel/case3_2) from archived data,
restyled with the unified paper style.

Figures (f = 0.8 -> width 0.8 * 4.773 in):
  case3_u.pdf   2x5 imshow: particle method (top row of the legacy figure is
                the FD reference; we keep FD on top, particle below) for u at
                t = 0.0, 0.1, 0.2, 0.3, 0.4.
  case3_v.pdf   same for v.

Data mapping (test.ipynb of this case):
  - FD: finite_difference.npz, arrays u, v of shape (10000, 100, 100)
        (finite_difference.py: T = 1, dt = 1e-4) -> frame k is t ~ k*1e-4.
        Loaded lazily: only the 5 required 100x100 slices are read from the
        uncompressed zip members (the file is 1.6 GB).
  - particle: samples/samples_N_320000.npz (largest N used in test.ipynb
        cell 3), frames X1_i / X2_i at t ~ i*1e-3 (simulation.py: T = 1,
        dt = 1e-3).  Reconstruction exactly as test.ipynb cells 3 (u) and
        8 (v): Fourier estimation n_freq = 10, periodic on [0,2pi]^2,
            u = (|X1_i|/N) * rho_hat / C_u,  C_u = 1/(3*pi)
            v = (|X2_i|/N) * rho_hat / C_v,  C_v = 1/(4*pi)
        evaluated on the same 100x100 grid as the FD solution.
The legacy figure shows snapshots up to t = 0.4 (the dynamics are nearly
steady afterwards); we keep the same five times.
"""
import os
import struct
import sys
import zipfile

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

from density import generate_density_estimation  # module used by test.ipynb

C_u = 1 / (3 * jnp.pi)   # test.ipynb cell 1
C_v = 1 / (4 * jnp.pi)
N0 = 320000


def read_npz_frames(npz_path, member, frame_indices):
    """Read selected leading-axis slices of a large array stored
    uncompressed inside an .npz, without loading the whole array."""
    out = []
    with zipfile.ZipFile(npz_path) as zf:
        info = zf.getinfo(member)
        assert info.compress_type == zipfile.ZIP_STORED, "member is compressed"
        with zf.open(member) as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            else:
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            assert not fortran
            header_end = f.tell()
            frame_bytes = int(np.prod(shape[1:])) * dtype.itemsize
            for k in frame_indices:
                f.seek(header_end + k * frame_bytes)
                buf = f.read(frame_bytes)
                out.append(np.frombuffer(buf, dtype=dtype).reshape(shape[1:]))
    return out


# ---------------------------------------------------------------------------
# load data
# ---------------------------------------------------------------------------
fd_path = os.path.join(CASE_DIR, "finite_difference.npz")
fd_frames = [0, 999, 1999, 2999, 3999]          # t = 0, 0.1, 0.2, 0.3, 0.4
pt_frames = [0, 99, 199, 299, 399]
tlabels = ["0.0", "0.1", "0.2", "0.3", "0.4"]

u_fd = read_npz_frames(fd_path, "u.npy", fd_frames)
v_fd = read_npz_frames(fd_path, "v.npy", fd_frames)

samples = np.load(os.path.join(CASE_DIR, "samples", f"samples_N_{N0}.npz"))

fcn_density_estimation, fcn_density_evaluate = generate_density_estimation(
    n_freq=10, extend="periodic", period=jnp.array([[0, 2 * np.pi], [0, 2 * np.pi]])
)
x = np.linspace(0, 2 * np.pi, 100, endpoint=False)
y = np.linspace(0, 2 * np.pi, 100, endpoint=False)
X, Y = np.meshgrid(x, y)
points = np.column_stack([X.ravel(), Y.ravel()])


def particle_field(prefix, idx, norm_const):
    P = samples[f"{prefix}_{idx}"]
    coeff = fcn_density_estimation(P)
    dens = jax.vmap(fcn_density_evaluate, (0, None))(points, coeff)
    dens = np.asarray(dens).reshape(X.shape)
    return (P.shape[0] / N0) * dens / float(norm_const)


def grid_figure(top, bottom, fname):
    w = 0.8 * TEXTWIDTH_IN
    fig, axs = plt.subplots(2, 5, figsize=(w, 0.45 * w), constrained_layout=True)
    vmax = max(float(np.max(a)) for a in (list(top) + list(bottom)))
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
    print(f"wrote {fname} (vmax = {vmax:.3f})")


u_pt = [particle_field("X1", k, C_u) for k in pt_frames]
grid_figure(u_fd, u_pt, "case3_u.pdf")

v_pt = [particle_field("X2", k, C_v) for k in pt_frames]
grid_figure(v_fd, v_pt, "case3_v.pdf")
