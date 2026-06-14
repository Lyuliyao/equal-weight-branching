# Residual-hybrid reconstruction plan for the LDG Keller--Segel benchmark

This note fixes the intended reconstruction refinement.  The point is not a generic adaptive KDE.  The intended object is

```text
low global spectrum + local residual correction in particle-detected windows.
```

The local correction must not double-count the low global spectrum.

## 1. Low global spectrum

Use a deliberately low global reconstruction

```math
u_lo(x) = P_{K_g} mu(x),    K_g << K_full.
```

This represents the smooth background.

## 2. Particle-detected windows and pilot field

Detect windows W_j from the particle cloud.  Each window has a padded support W_j^pad and a smooth taper chi_j.  Inside W_j^pad, build a cheap pilot field, for example a local blob/KDE or local spectrum,

```math
u_pilot,j(x) approx eta_{h_j} * mu(x).
```

The pilot only identifies where the global spectrum fails.  It must not be added directly to u_lo.

## 3. Residual score and retained residual particles

For particles in W_j^pad define

```math
s_i^(j) = chi_j(X_i) |nu_pilot,j(X_i)-nu_lo(X_i)| /(nu_pilot,j(X_i)+|nu_lo(X_i)|+eps).
```

Given target budget B_j,

```math
q_i^(j) = min(1, max(q_min, B_j s_i^(j)/(sum_{k in W_j^pad} s_k^(j)+eps))),
A_i^(j) ~ Bernoulli(q_i^(j)).
```

The average rate

```math
qbar_j = N(W_j^pad)^(-1) sum_{i in W_j^pad} q_i^(j)
```

is a reconstruction-enrichment diagnostic:

```text
low qbar_j      : low global spectrum already explains the window
high qbar_j     : core/island has unresolved residual structure
growing qbar_j  : reconstruction resolution should be concentrated here
```

Do not call q_i a Metropolis acceptance probability.  This is not a dynamics resampling step.

## 4. Signed residual and Horvitz--Thompson correction

The residual mu - P_{K_g}mu dx is signed.  Retained particles alone are not the full residual.  For quantitative reconstruction use

```math
u_res,j^HT(x)
= chi_j(x)[ sum_{i in W_j^pad} A_i^(j)/q_i^(j) * omega_i * eta_{h_j}(x-X_i)
           - int_{W_j^pad} eta_{h_j}(x-y) nu_lo(y) dy ].
```

Conditioned on the current particle cloud,

```math
E_A nu_res,j^HT(x)
= chi_j(x) [ eta_{h_j} * (mu|_{W_j^pad} - nu_lo 1_{W_j^pad} dx) ](x).
```

Thus accepted particles estimate the empirical local blob part, while the low-spectrum part is subtracted deterministically.

## 5. Positive-excess visualization only

For plots, one may use

```math
nu_lo^+(x)=max(P_{K_g}mu(x),0),
r_j^+(x)=(nu_pilot,j(x)-nu_lo^+(x))_+,
alpha_j(x)=min(1, r_j^+(x)/(nu_pilot,j(x)+eps_dens)).
```

This highlights particles in the positive local excess, but it is not an unbiased signed residual estimator.  Label it as positive-residual-only diagnostic.

## 6. Two hybrid reconstruction options

### Option A: global spectrum + residual blob

Use

```math
u_hyb(x) = P_{K_g}mu(x) + sum_j chi_j(x)[ eta_{h_j}*(mu - P_{K_g}mu dx) ](x),
```

or replace the residual term by the HT estimator above.

### Option B: global spectrum + local spectral residual window

Compute local high-order spectra in particle windows,

```math
nu_loc,j = P_{K_l}^{W_j}(mu|_{W_j^pad}),    K_l > K_g.
```

For quantitative comparison, prefer the global-residual spectral form

```math
nu_hyb = nu_lo + sum_j chi_j[ P_{K_l}^{W_j}(mu|_{W_j^pad})
                            - P_{K_l}^{W_j}(nu_lo 1_{W_j^pad} dx) ].
```

The alternative local high-pass form

```math
nu_hyb = nu_lo + sum_j chi_j[nu_loc,j - Pi_lo^{W_j} nu_loc,j]
```

is useful as a local high-frequency diagnostic, but it is not exactly the signed residual mu - P_{K_g}mu dx unless Pi_lo^{W_j} is defined to match the restriction/projection of the global low field.

## 7. Required change for the Keller--Segel LDG benchmark

The current field-only snapshots are insufficient.  At each LDG reporting time, the particle solver must save raw particle clouds:

```text
X_u, X_v
mass_per_u_particle, mass_per_v_particle
M_u, M_v
x_c, L or physical box metadata
seed, N0, K, tau, t
```

Then postprocess the same particle snapshot with:

```text
1. low global spectrum P_{K_g}mu
2. current global Fourier P_K mu
3. global spectrum + HT residual blob
4. global spectrum + local spectral residual window
5. positive-excess retained-particle plot, visualization only
```

Compare these reconstructions with the direct LDG reference, not the FVM surrogate, using S_L2(t), peak(t), R_0.5(t), R_0.8(t), and qbar_j(t).  The key question is whether the residual-hybrid reconstruction moves the bandwidth-sensitive diagnostics toward the LDG refinement trend while preserving the reconstruction-free radius diagnostics.
