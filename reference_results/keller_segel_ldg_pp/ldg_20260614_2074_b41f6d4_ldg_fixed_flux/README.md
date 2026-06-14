# Direct LDG reference (Li–Shu–Yang) for the Keller–Segel blow-up benchmark

A from-scratch implementation of the **local discontinuous Galerkin** method of

> X. H. Li, C.-W. Shu, Y. Yang, *Local discontinuous Galerkin method for the
> Keller–Segel chemotaxis model*, J. Sci. Comput.

This is the **direct LDG reference** required for §5.4 (plan §6) — it is *not* the
finite-volume baseline. It solves the same fully parabolic–parabolic system,
boundary condition, initial data, and reporting times as the paper.

## Method (`ldg_solver.py`), following the paper

- System (their (1.1), χ=1, homogeneous **Neumann** on a rectangle):
  `u_t − div(∇u − u∇v) = 0`, `v_t = Δv + u − v`.
- Auxiliary variables `p=∇u`, `r=∇v` (their (2.1)–(2.4)); **P¹ modal DG** on a
  uniform Cartesian mesh.
- **Alternating diffusion fluxes** `(û,p̂)=(u⁺,p⁻)`, `(v̂,r̂)=(v⁺,r⁻)` (their (2.5));
  Neumann boundary: diffusive flux `p̂·n=0`, `û` = interior trace (their (2.6)–(2.7)).
- **Lax–Friedrichs chemotaxis flux** `½(r⁺u⁺+r⁻u⁻) − ½α ν(u⁺−u⁻)`, `α=max|∇v|` (their (2.8)).
- **Zhang–Shu P¹ positivity-preserving scaling limiter** on `u` (their §4):
  `θ=(ū−ε)/(ū−b)`, `b=ū−|u_ξ|−|u_η|`.
- **SSP-RK3** in time (their §4.1), adaptive `dt=min(c_diff dx², c_conv dx/α)`
  (the concentrating core makes α grow; the paper likewise notes adaptive stepping).

## Verification (the correctness gate)

```
pure heat u_t=Δu, Neumann, e^{-2t}cos x cos y:  relL2 order 2.00, 2.03, 2.02  (2nd order ✓)
LDG Laplacian matrix L: ||L−Lᵀ||/||L|| = 5.5e-17  (symmetric negative-definite ✓)
chemotaxis −div(u∇v) consistency:               order 3.0  (✓)
u-mass on the blow-up IC:                        = 10π exactly, drift ~1e-14  (✓)
positivity (limiter on):                         u_min ≳ −1e-11 (machine ε)  (✓)
```

`python -c "..."` verification snippets are in the commit message / session log;
the accuracy structure matches the paper's Table 5.1 (second order with the limiter).

## Example 5.2 (blow-up benchmark) and the numerical-blow-up time

IC `u0=840 exp(−84 r²)`, `v0=420 exp(−42 r²)` on `[−½,½]²`, `M_u=10π`; report at
`t=6e-5, 1.2e-4, 2e-4`. The paper cites a reference numerical blow-up time
`≈1.21×10⁻⁴` and shows P¹ snapshots at N=160. We additionally compute their
**numerical-blow-up-time indicator** (their (5.2)) for KS — which the paper itself
applied only to a 1-D manufactured test:

```
tb(N; θ) = inf{ t : S(2N,t) ≥ θ S(N,t) },   S(N,t) = ||u_N(t)||_L2,   θ=1.05.
```

`tb(N)` is a numerical resolution-gap indicator that increases toward the blow-up
time as N refines (the paper proves first-order convergence on the manufactured
test); it is **not** a continuum blow-up time at finite N.

## Run

```bash
for N in 80 160 320; do python run_ldg.py --N $N --T 2e-4 --out_dir results/N$N; done
# tb(N) from a refinement pair:
python - <<'PY'  # or reuse ldg_pp_baseline/tb_from_pair.py after renaming the N column
PY
```

Outputs (`reference_results/keller_segel_ldg_pp/ldg_<run_id>/N{N}/`): `S_curves.csv`
(t, S_L2, peak, umin, mass_u, mass_v), `snapshots.npz` (u,v at report times),
`config_used.json`. `tb_ldg.json` holds the resolution-gap times.
