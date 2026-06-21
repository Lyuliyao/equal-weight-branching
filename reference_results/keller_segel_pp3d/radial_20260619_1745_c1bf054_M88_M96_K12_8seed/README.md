# Radial Figure-B production — 3D fully parabolic–parabolic Keller–Segel

Model `u_t = D Δu − χ∇·(u∇v)`, `v_t = D Δv + αu − βv` on the periodic box L=12,
**v0 = 0** (chemical created from scratch by `u→v` injection). Normalized regime
`D_u=D_v=α=β=χ=1`, σ=0.45, τ=1e-3, T=2.0, minvar injection, K_dyn=**12**,
JITTED fixed-capacity grad-v buffer (`--fast`, verified in `test_buffer_equiv.py`).
Reconstruction-free particle radii `R_q` (q=0.2/0.5/0.8). 8 seeds.

The effective coupling is `F=χαM/β=M`, so the initial cell mass M is the single knob.
**The fixed-bandwidth numerical transition mass is bandwidth-dependent** (see
`../PILOT_FINDINGS_msweep.md`): M\*≈(88,92] at K=8 but M\*∈(72,80] at the production
bandwidth K=12. This is a numerical transition at a fixed reconstruction bandwidth, NOT
a continuum critical mass. Configs were therefore re-selected AT K=12. Note: R_0.5 is
reconstruction-free as a readout, but its *dynamics* still depend on K_dyn through the
reconstructed drift ∇v (see the validation-closure resolution audit).

## Run directories (label_chiX_NX_KX_tauX_seedX/)

| label | M | role | N | seeds | R_0.5(T)/R_0.5(0) |
|-------|---|------|---|-------|-------------------|
| `weak`        | 72 | diffusive arm (Fig B panel c) | 100000 | 0–7 | **3.75** (expands) |
| `delayed`     | 96 | delayed-focusing arm (panels a,b,c,d) | 100000 | 0–7 | **0.21** (focuses) |
| `delayed`     | 96 | N-refinement (panel d) | 20000  | 0–3 | 0.22 |
| `delayed`     | 96 | N-refinement (panel d) | 320000 | 0–3 | 0.21 |
| `longdelay88` | 88 | record only: long-delayed focusing at K=12 (NOT the weak arm; M=88 focuses at K=12) | 100000 | 0–7 | 0.27 |

Each run dir has `diagnostics.csv`, `manifest.json` (git, versions, devices, seed,
fast, buffer_factor, all params), `config_used.json`.

## Key results

- **Delayed focusing (M=96)**: core R_0.5 collapses 0.69→0.15 (ratio 0.21), delayed
  turnover t_turn≈0.20–0.24 (cloud first expands ~25% then focuses), t_focus10≈0.44.
  Tight across 8 seeds (0.203–0.219).
- **Reconstruction-free R_0.5 is N-consistent**: seed-mean final ratio 0.219 / 0.211 /
  0.206 across N=20k / 100k / 320k (a 16× range), a slight monotone decrease of ~6%
  (Fig B panel d, curves nearly overlap). Per-seed spread at N=100k is tiny (std 0.005,
  min 0.203, max 0.219).
- **Weak vs delayed contrast (panel c)**: M=72 diffuses (R_0.5 →2.60, ratio 3.75 monotone,
  G_v≈0.33) vs M=96 focuses (R_0.5 →0.15, ratio 0.21, G_v≈21). Clean diffusion-vs-focusing
  across the K=12 fixed-bandwidth transition. Delay is genuine: M=96 max R_0.5/R_0.5(0)≈1.26 (cloud
  expands ~25%) before turning over at t_turn≈0.16–0.24.
- **Caveat — R_0.8 (halo) is seed- and N-scattered, NOT a clean diagnostic**: seed-mean
  R_0.8(T) = 1.47 / 1.96 / 1.87 at N=20k / 100k / 320k (non-monotone), with large low-N
  spread (N=20k seeds span 0.31–2.12). The 80% quantile straddles the bimodal core/halo
  split, so R_0.8 is dominated by which seeds' halos have collapsed by T. **R_0.5 (core) is
  the robust reconstruction-free diagnostic; do NOT headline R_0.8 or quote a single value.**
- Numerically stable throughout: drift-resolution number ≤0.10, max v-occupancy
  ≤277k/512k (N=320k), no abort, no population control. Mass law M_v(t)=M(1−e^{−t})
  unbiased — relative error <1% for t≥0.2 at N≥100k (worst individual run ~4% at N=20k
  seed2, the small-N integer-injection sampling noise; e.g. N=320k seed0 max 0.24%).

## Regenerate the figure (no solver)

```
python experiments/keller_segel/fully_parabolic_3d/plot_radial_response.py \
    --run_dir <this dir> --baseN 100000 --K 12
```
Reads `*/diagnostics.csv` only. Outputs `figures/figureB_radial_response.{pdf,png}` and
`plot_data/figureB_radial_response.npz`.

NOTE: the directory name says `M88_M96` for historical reasons (M=88 was the pilot/K=8
weak pick); the actual weak arm is **M=72** (K=12). See the table above.
