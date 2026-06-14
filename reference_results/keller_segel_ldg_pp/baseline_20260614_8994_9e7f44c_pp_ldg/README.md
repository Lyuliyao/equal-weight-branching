# Grid FVM baseline (§5.4)

Deterministic fully parabolic-parabolic Keller-Segel grid reference. Regenerate:
`cd experiments/keller_segel/ldg_pp_baseline && for n in 128 256 512; do python fvm_baseline.py --n $n --T 2e-4 --out_dir results/n$n; done`. See `manifest.json` and the experiment README. Figures regenerate from `nNNN/S_curves.csv` + `nNNN/snapshots.npz` via `plot_baseline.py`.
