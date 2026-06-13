# kinetic_ks — field-coupled 6D kinetic Keller-Segel particle method

Genuinely field-coupled 6-dimensional kinetic Keller-Segel in phase space
`z = (x, v)`, `x ∈ T³ = [-π,π]³` (period `L = 2π`), `v ∈ ℝ³`. `d = 6`,
`d_x = d_v = 3`. Reuses the validated kinetic-particle infrastructure in
`../highdim/common_highdim.py` (three reaction kernels, branching compaction,
nESS, torus wrap) and adds a spectral screened-Poisson field solve.

> **No code in this directory is run automatically.** Every execution is gated
> behind a separate Codex cold-verification step. See `run_me` for the SMOKE and
> production invocations.

## Model (authoritative, corrected plan §5.3)

Kinetic PDE:

    ∂_t f + v·∇_x f = γ_v ∇_v·((v − χ∇_x c) f) + D_v Δ_v f + r[ρ,c](x) f

Spatial marginal `ρ(t,x) = ∫ f dv`. Chemical field on `T³`:

    −Δ_x c + κ² c = ρ − ρ̄ ,   ρ̄ = mean of ρ over T³.

Particle SDE:

    dX = V dt ,   dV = −γ_v (V − χ∇_x c(t,X)) dt + √(2 D_v) dW.

Reaction rate:

    r[ρ,c](x) = λ_g S_c(c(x)) − α_ρ S_ρ(ρ(x)) − β ,
    S_c(c)   = ½(1 + tanh((c − c0)/δ_c)) ,
    S_ρ(ρ)   = ρ/(ρ + ρ0).

Parameters: `L=2π, γ_v=2, D_v=1, χ=1.5, κ=0.5, λ_g=4, α_ρ=1, β=0.2, c0=0.1,
δ_c=0.05, ρ0=0.2, T=2 (or 1.5), τ=2e-3 (fallback 1e-3), N0∈{2e4,8e4,3.2e5},
seeds {0,1,2,3} (pilot {0,1}), K_x=8`. See `parameter_log.md`.

## File map

| file                    | role                                                        |
|-------------------------|-------------------------------------------------------------|
| `field_kinetic.py`      | spectral screened-Poisson solve + c/∇c/ρ evaluation + `selftest_field()` |
| `common_kinetic.py`     | OU velocity+transport step, reaction rate, diagnostics; imports the 3 branching kernels from `../highdim/common_highdim.py` |
| `experiment_kinetic.py` | main loop: 4 methods under shared transport CRN, buffer/mask, CSV, npz clouds, reaction histograms, config copy, README |
| `config_pilot.json`     | N0=2e4, seeds {0,1}, T=2, τ=2e-3, K_x=8, buffer_mult=8       |
| `config_prod.json`      | N0=8e4, seeds {0,1,2,3} (note inside for the 3.2e5 case)     |
| `run_me`                | SLURM script; SMOKE block clearly labeled                   |
| `parameter_log.md`      | record of any parameter change                              |

## Field solve (spectral, grid-free in the x-marginal)

K_x Fourier modes per spatial dim (default 8, wavenumbers `n_j ∈ {−K_x..K_x}`).

- Empirical density coefficients (mass-normalized so `∫_{T³} ρ dx = M_f`):
  `ρ̂_k = (1/((2π)³ N0)) Σ_i w_i e^{−i k·X_i}` (active particles; `w_i=1` for branching).
- Screened-Poisson divide: `ĉ_k = ρ̂_k/(|k|²+κ²)` for `k≠0`; **`ĉ_0 = 0` gauge fix**.
- Evaluate at particles: `c(X_i)=Re Σ_k ĉ_k e^{ik·X_i}`,
  `∇c(X_i)=Re Σ_k (ik) ĉ_k e^{ik·X_i}` via a real half-spectrum cos/sin assembly.

`selftest_field()` checks: analytic single mode `ρ−ρ̄=cos(x1) ⇒ c=cos(x1)/(1+κ²)`;
chemotactic gradient sign; finite-difference gradient.

## Velocity update (exact OU for the linear drift)

    V_{n+1} = χ∇c(X_n) + e^{−γ_v τ}(V_n − χ∇c(X_n))
              + √( (D_v/γ_v)(1 − e^{−2 γ_v τ}) ) ξ_n ,
    X_{n+1} = X_n + V_n τ   (mod 2π).

Velocity lives in `ℝ³` (NOT wrapped); only `X` is wrapped to the torus.
An Euler-Maruyama option is available via `velocity_scheme="euler"`.

## Methods compared (shared transport CRN)

`weighted`, `weighted_resample` (resample when global nESS < `ess_thresh`,
mass-preserving common weight), `poisson` branching, `minvar` branching. Transport
randomness is shared across methods via the full-buffer `xi_buf` pattern (weighted
uses `xi_buf[:N0]`, branching uses the full buffer).

## Run

SMOKE (tiny, separate scratch dir; **gated behind Codex cold verification**):

    python3 experiment_kinetic.py --smoke
    # or via SLURM, after editing run_me to enable the SMOKE line.

Production / pilot:

    python3 experiment_kinetic.py --config config_pilot.json
    python3 experiment_kinetic.py --config config_prod.json
    sbatch run_me "--config config_pilot.json"
    sbatch run_me                                  # defaults to config_prod.json

## Diagnostics (per snapshot, ~21 over [0,T]) → metrics.csv

Total mass `M(t)=M_f`; `||c||_∞`, `||ρ||_∞` on a coarse 32³ eval grid AND at
particle positions; quantile core radii `R_0.5,R_0.9` of the x-marginal about the
torus mass centroid (reconstruction-free); local phase-space ball
`B={|x−x_c|≤r_x, |v−v_c|≤r_v}` local ESS (weighted) and equal-weight count
(branching); global nESS, max:mean weight, `N_act`; reaction-coupling evidence
`corr(r,c)`, `corr(r,ρ)`, reaction-rate histograms (`reaction_histograms.npz`),
mass-fraction with `r>0` vs `r<0`, inside-vs-outside-core mean `S_c`, `S_ρ`, `r`;
CFL proxy `max|V|τ/dx`. Final-time particle clouds → `cloud_d6_seed*.npz`.
