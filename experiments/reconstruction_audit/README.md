# Reconstruction-bandwidth & KDE audit (CLAUDE.md §7)

Appendix robustness check for the branching-vs-weighted comparisons. The main
tables report the relative `L²` error of the Fourier-reconstructed solver output
`u_h^{N,K} = P_K μ_t^N`. This audit verifies that the reported method **ordering**
is not an artifact of a single Fourier bandwidth `K` (too large → dominated by
Monte-Carlo coefficient noise; too small → dominated by truncation bias), and that
it survives a common periodic Gaussian smoothing of both the particle measure and
the deterministic reference.

This is **not** a new dynamics experiment. The script reruns the *exact* production
dynamics of §5.2 (`experiment.py`) and §5.3 (`experiment_switch.py`) for a few
seeds — same configs, same common-random-number key sequence — captures the final
particle clouds, and only changes the *reconstruction* applied to them.

## What it computes

For the final-time cloud of each method (§5.2: `weighted`, `minvar`; §5.3:
`weighted`, `weighted_ess`, `minvar`):

**Fourier `K`-sweep** (`fourier_k_sweep.csv`), `K ∈ {8,12,16,24}` (§5.2),
`{32,48,64}` (§5.3):

```
E_total(K)    = ||P_K μ^N − u_ref||    / ||u_ref||        (= the reported solver error)
E_particle(K) = ||P_K μ^N − P_K u_ref|| / ||P_K u_ref||    (particle error at fixed scale)
E_proj(K)     = ||P_K u_ref − u_ref||  / ||u_ref||         (Fourier truncation bias of ref.)
```

`P_K u_ref` is the band-limited `L²` projection of the deterministic reference onto
the **same** folded cos/sin basis used by the particle reconstruction, so
`E_particle` isolates representation error at a fixed reconstruction scale.

**KDE `h`-sweep** (`kde_h_sweep.csv`), common periodic Gaussian scale
`h ∈ {0.10,0.15,0.20,0.25}` (§5.2), `{0.06,0.10,0.15}` (§5.3), via FFT
(deposit → FFT → `×exp(−½h²|k|²)` → iFFT):

```
E_KDE_rep(h) = ||u_h^N − u_{ref,h}|| / ||u_{ref,h}||      (representation error, same h for all methods)
E_bias(h)    = ||u_{ref,h} − u_ref|| / ||u_ref||          (smoothing bias of the reference)
```

The same `h` is used for every method — no per-method or data-driven bandwidth.

For §5.3 every error is reported **global** and **local** over `B_A` (old growth)
and `B_B` (new growth), to check that the Table-6 mechanism (ESS competitive in
`B_A`, branching better in `B_B`) is stable across `K` and `h`.

## Validation (byte-faithful rerun)

The replicated dynamics are checked against the archived production numbers: the
per-seed `E_total` at the production bandwidth (global) must reproduce the final
`L2_rel_err` rows in `reference_results/{branch_vs_weighted,switch}/metrics.csv`,
and for §5.3 the seed-0 reconstruction additionally matches
`reference_results/switch/fields_seed0.npz` to floating-point tolerance. The match
tolerances are recorded in `manifest.json` (`validation_seed0`).

## Run

```bash
# heat conda env (jax, x64). From this directory:
python audit_fourier_kde.py --seeds 0 1 2      # both experiments, production configs
python audit_fourier_kde.py --smoke            # tiny configs, code check only
python plot_audit.py                           # figures from saved sweep data
```

Outputs land in `reference_results/reconstruction_audit/{localized_growth,switching_growth}/`:

```
fourier_k_sweep.csv   kde_h_sweep.csv          # full schema, all seeds
config_used.json      manifest.json            # repro record + validation tolerances
snapshots/<exp>_seed0.npz                      # seed-0 final clouds (recompute without rerun)
plot_data/<exp>_sweeps.npz                     # aggregated mean±std curves
figures/<exp>_audit.{pdf,png}                  # appendix figures
```

CSV schema (blank where not applicable):

```
experiment, method, seed, K, h, region, E_total, E_particle, E_proj,
E_KDE_rep, E_bias, global_nESS, local_nESS_or_count, N_active, particle_steps
```

## Reading the result

The audit is appendix-worthy (CLAUDE.md §7.6) if the §5.2 method ordering and the
§5.3 old/new-region mechanism are stable over moderate `K` and `h`; very small `K`
or very large `h` smooths away differences (expected), and very large `K` increases
particle-noise sensitivity (expected). No conclusion should depend on one isolated
`K` or `h`. See `manifest.json` and the parent `CLAUDE.md` §7 for the policy.
