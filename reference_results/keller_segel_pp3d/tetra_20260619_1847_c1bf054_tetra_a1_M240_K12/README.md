# Tetra Figure-C production — 3D multi-cluster chemotactic aggregation vs diffusion control

Model `u_t = D Δu − χ∇·(u∇v)`, `v_t = D Δv + αu − βv` on the periodic box L=12, **v0=0**.
Four equal Gaussian cell clusters (σ_c=0.25) at the vertices of `a·TETRA` (a=1.0, vertex
radius ≈1.73, nearest-neighbour centroid distance d_min(0)≈2.83), total mass **M=240**
(selected cluster mass; each cluster focuses at this fixed reconstruction bandwidth — a
fixed-bandwidth numerical focusing, NOT a continuum critical mass), `D=α=β=1`, K_dyn=**12**,
τ=1e-3, T=3.0, N=80000, minvar injection, `--fast` JIT buffer. Cluster labels preserved
(u-cloud conservative). R_0.5 is reconstruction-free as a readout, but its dynamics still
depend on K_dyn through the reconstructed drift ∇v.

## Arms (mandatory diffusion control, common randomness)

| arm | χ | seeds | role |
|-----|---|-------|------|
| `active`  | 1 | 0–3 | chemotactic |
| `control` | 0 | 0–3 | diffusion only (same seed ⇒ same initial particles) |

Run dirs: `{active,control}_chi{χ}_N80000_K12_tau1e-03_seed{0-3}/diagnostics.csv` +
`manifest.json` (git, versions, devices, fast, buffer_factor, all params) + `config_used.json`.

## Result — mutual chemotactic attraction + individual collapse

| quantity | active (seed-mean, T) | control (seed-mean, T) |
|----------|----------------------|------------------------|
| d_min (nearest-neighbour centroid) | 2.33 (↓ from 2.83) | 2.80 (flat) |
| mean per-cluster R_0.5 | **0.16** (clusters collapse) | **3.78** (clusters spread) |
| overlap O = min d/(R_i+R_j) | ~7 (tight, separated) | ~0.4 (diffuse, overlapping) |
| m_center (ball r=1 at origin) | ~0.05 (mass at vertices) | ~3.2 (diffusion fills centre) |

`plot_tetra_control.py` verdict: **mutual chemotactic ATTRACTION + individual COLLAPSE**
(d_min < control AND per-cluster R_0.5 collapses vs spreading control). The active clusters
attract one another (d_min decreases vs the flat control) and each focuses into a coherent
aggregate (R_0.5≈0.16 vs control 3.78, a ~23× contrast); they do NOT merge into a single
central mass on T=3 (overlap stays >1, m_center stays low — mass collapses at the vertices,
not the centre).

Caveats (honest):
- The control d_min is mildly noisy at late t because diffuse control clusters (R_0.5~3.8,
  approaching the box) have ill-defined torus centroids; does not affect the separation.
- The deep collapsed-core radius (0.16 < h_K=L/(2K+1)=0.48 at K=12) is reconstruction-limited,
  as in the radial case. The attraction/migration (scales ≳1.5 ≫ h_K) is well-resolved and is
  the robust signal. m_center is confounded by the geometry (clusters collapse off-centre) and
  is NOT the headline.
- Numerically stable: no abort, no population control, max v-occupancy ≤76.3k/128k.

## Regenerate the figure (no solver)

```
python experiments/keller_segel/fully_parabolic_3d/plot_tetra_control.py --run_dir <this dir>
```
Reads `*/diagnostics.csv` only → `figures/figureC_tetra_control.{pdf,png}` and
`plot_data/figureC_tetra_control.npz`.

Pilots that set these params: `tetra_pilot_*` (M-scan at a=1.0) and `tetra_pilot2_*`
(M=240/320 at a=1.0 collapse; a=0.6/M=120 spreads — rejected). a=1.0/M=240 chosen because
clusters stay coherent (collapse) AND attract, giving the cleanest active-vs-control contrast.
