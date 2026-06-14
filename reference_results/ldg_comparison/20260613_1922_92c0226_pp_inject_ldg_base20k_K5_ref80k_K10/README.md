# LDG-style parabolic–parabolic Keller–Segel — reference run

Run id: `20260613_1922_92c0226_pp_inject_ldg_base20k_K5_ref80k_K10`
Git commit: `92c0226`  ·  Python 3.12.7  ·  numpy 2.4.4, jax 0.10.0, matplotlib 3.10.8

This is a paired **base / refined** run of the LDG-style parabolic–parabolic
Keller–Segel particle method (cross-species injection chemical equation), using
the Li–Shu–Yang concentrated initial condition and reporting times. The full
model description, injection kernel, LDG-alignment notes, reporting-language
conventions, and the boundary-condition disclosure are in
`experiments/keller_segel/ldg_comparison/README.md`. This file is the
run-specific summary.

## What was run

| run | N | K | τ | n_steps | t_final | result |
|---|---|---|---|---|---|---|
| `base/`    | 20000 | 5  | 1e-6   | 200 | 2.0e-4 | completed |
| `refined/` | 80000 | 10 | 2.5e-7 | 800 | 2.0e-4 | completed |

The refined run uses a 4× smaller τ than the base run: with the same τ=1e-6 the
sharper K=10 reconstruction trips the chemotactic-drift CFL guard near t≈7.6e-5
(the core gradient grows and the window L(t) shrinks, pushing the drift CFL past
the abort threshold). Reducing τ to 2.5e-7 keeps the drift CFL ≲ 4 and lets the
refined run reach t=2e-4 with all three snapshots. The `(N,K)→(4N,2K)` pairing
required by `tgap.py` is satisfied; `tgap.py` interpolates both S_L2(t) series to
the common time span before forming the ratio, so the differing τ is handled
correctly.

Commands (paths relative to `experiments/keller_segel/ldg_comparison/`):

```bash
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
# BASE  (N=20000, K=5)
$PY simulation.py --N 20000 --K 5  --tau 1e-6   --n_steps 200 --diag_every 4 \
    --seed 0 --report_times 6e-5 1.2e-4 2.0e-4 --cfl_abort 8.0 --verbose \
    --outdir <this_run>/base
# REFINED (N=80000, K=10)
$PY simulation.py --N 80000 --K 10 --tau 2.5e-7 --n_steps 800 --diag_every 16 \
    --seed 0 --report_times 6e-5 1.2e-4 2.0e-4 --cfl_abort 8.0 --verbose \
    --outdir <this_run>/refined
# resolution-gap indicator
$PY tgap.py --pairs <this_run>/base/diag_*.csv:<this_run>/refined/diag_*.csv \
    --threshes 1.05 1.10 --out <this_run>/tgap_table
# figures (refined has all 3 snapshots + full diag)
$PY plot_ldg_style.py --results_dir <this_run>/refined --out_dir <this_run>/figures
```

## Key numbers

Reconstructed L² norm `S_{K,N}(t)` (**bandwidth-sensitive**) and reconstructed
peak at the report times:

| t | base S_L2 (K=5) | refined S_L2 (K=10) | refined peak (BW-sensitive) |
|---|---|---|---|
| 6e-5   | 458.7  | 1096.5 | 8.19e4 |
| 1.2e-4 | 894.2  | 2176.8 | 2.46e5 |
| 2.0e-4 | 1274.5 | 2685.6 | 3.31e5 |

Resolution-gap indicator (`tgap_table.csv`, ratio `S_{2K,4N}/S_{K,N}`):

```
t_gap(θ=1.05) = 8e-6
t_gap(θ=1.10) = 1.2e-5
ratio_max     = 2.71
ratio_final   = 2.11
```

The resolution gap opens **well before** the first report time 6e-5: this is a
**resolution-gap indicator, NOT a blow-up time**.

Core radii (**reconstruction-free**, refined run): `R_0.5` collapses from ≈0.09 to
≈0.01 by t≈6e-5 then stabilizes; `R_0.8` from ≈0.13 to ≈0.025. The core radii are
stable (reconstruction-free), while the reconstructed peak and S_L2 keep rising
(bandwidth-sensitive).

Mass: `M_u` conserved exactly at `10π = 31.4159` in both runs. `M_v` follows the
exact chemical law `M_v(t)=M_u+(M_v0−M_u)e^{−t}` (here ≈ constant since
`M_u0=M_v0=10π`); max relative error vs the law is **8.75e-5** (refined),
**1.0e-4** (base) — the small residual is from the discrete integer
injection/decay events.

## Files

```
base/, refined/        config.json, manifest.json, diag_*.csv, snapshots/*.npz,
                       plot_data/*.npz, figures/*.{pdf,png}, run.log, README_run.md
figures/               main figures (from the REFINED run)
plot_data/             figure data arrays (from the REFINED run)
tgap_table.{csv,json}  resolution-gap indicator from the (base, refined) pair
README.md              this file
```

## Regenerate figures (no solver rerun)

```bash
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
$PY <repo>/experiments/keller_segel/ldg_comparison/plot_ldg_style.py \
    --results_dir refined --out_dir figures
```

Figures depend only on `refined/snapshots/*.npz` and `refined/diag_*.csv`.
