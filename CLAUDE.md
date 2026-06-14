# CLAUDE.md — rg_com conservative-force repair plan

Repository: `Lyuliyao/rg_com`  
Current focus: fix the conservative free energy `A_theta(C,s)` before further memory/friction work.

This file is the next-run instruction for Claude Code. It supersedes older branching-particle / unrelated instructions if present in this repository.

---

## 0. Current project status

We are building a CGMD / GLE model for a neutral star-polymer melt. The molecule-level CG variables are

\[
q_i=(C_i,s_i),\qquad s_i=\log(R_{g,i}/R_0),
\]

where \(C_i\in\mathbb T^3\) is the molecular center of mass and \(s_i\) is the log-radius label.

The current learned conservative free energy is a DeePCG-style many-body scalar energy

\[
A_\theta(C,s),
\]

with forces obtained only by autograd:

\[
F_{C_i}^\theta=-\nabla_{C_i}A_\theta(C,s),
\qquad
F_{s_i}^\theta=-\partial_{s_i}A_\theta(C,s).
\]

Stage M4 structure validation showed:

- \(p(s)\), \(p(R_g)\), and \(g_{CC}(r)\) are roughly reasonable;
- the many-body / three-body packing observable is wrong:
  \[
  O_3^{CG}=0.095,\qquad O_3^{ref}=0.067,
  \]
  i.e. about \(+42\%\) over-packing;
- the runtime soft COM core and \(s\)-wall are almost inactive, so the error is genuinely in \(A_\theta\), not in the safety wall/core.

Therefore the next priority is **not** more memory structure. The next priority is to repair the conservative free energy and validate its equilibrium structure.

---

## 1. Key architectural problem to fix

The old / current local coordinate row has the form

\[
\widetilde C_i[j,:]=f_c(r_{ij})\Delta C_{ij}^T.
\]

This is problematic because when two CG particles overlap,

\[
r_{ij}\to0,
\qquad
\Delta C_{ij}\to0,
\]

so the descriptor can vanish instead of becoming singular. Thus the network may not see overlap as an extreme forbidden event. This can allow nonphysical COM overlap or distorted packing.

The other issue is cutoff behavior: if the descriptor or energy does not smoothly vanish at the cutoff, the equilibrium distribution can be biased.

The fix is to use **DeePCG-style inverse-distance descriptors** together with a smooth switching function and an explicit analytic hard-core prior.

The 2018 DeePCG paper uses descriptors of the form

\[
\left\{\frac1{R_{ij}},\frac{x_{ij}}{R_{ij}^2},\frac{y_{ij}}{R_{ij}^2},\frac{z_{ij}}{R_{ij}^2}\right\}
\]

for near neighbors and radial \(1/R_{ij}\) information for farther neighbors. This is the right short-distance behavior: as \(r\to0\), the descriptor becomes singular rather than disappearing. The paper also validates DeePCG by sampling the CG model and comparing RDF, ADF, and higher-order local order statistics, not merely force MSE.

---

## 2. New conservative free-energy form

Implement the conservative energy as

\[
\boxed{
A_\theta(C,s)
=
A_{\rm core}(C,s)
+
A_{\rm one}(s)
+
A_{\rm MB,\theta}(C,s).
}
\]

The many-body neural part is

\[
\boxed{
A_{\rm MB,\theta}(C,s)
=
\sum_i \widetilde U_\theta(D_i,s_i).
}
\]

The analytic core enforces no-overlap physics. The neural many-body residual learns the remaining PMF: RDF, ADF, \(S(k)\), \(R_g\)-density coupling, and higher-order packing.

Forces must always come from the total scalar energy:

\[
F_C=-\partial_C A_\theta,
\qquad
F_s=-\partial_s A_\theta.
\]

Do not add a direct force head.

Do not add a runtime force patch that is not part of the scalar energy.

---

## 3. Smooth switching function

Use a smooth switch

\[
S(r)=
\begin{cases}
1, & r<r_{\rm in},\\[3pt]
\frac12+\frac12\cos\!\left(\pi\frac{r-r_{\rm in}}{r_{\rm out}-r_{\rm in}}\right),
& r_{\rm in}\le r<r_{\rm out},\\[3pt]
0, & r\ge r_{\rm out}.
\end{cases}
\]

This gives

\[
S(r_{\rm out})=0,
\qquad
S'(r_{\rm out})=0.
\]

Use separate cutoffs:

- \(r_{\rm core}\): analytic hard-core cutoff;
- \(r_{\rm out}\): neural descriptor cutoff.

Typically

\[
r_{\rm core}<r_{\rm out}.
\]

The neural cutoff must not be used as a substitute for the hard core.

---

## 4. DeePCG-style inverse-distance descriptor

For molecule \(i\), define the periodic minimum-image displacement

\[
\Delta C_{ij}=C_j-C_i,
\qquad
r_{ij}=|\Delta C_{ij}|_{\rm per}.
\]

For each neighbor \(j\in\mathcal N_i\), define the row

\[
\boxed{
X_i[j,:]
=
\left(
\frac{S(r_{ij})}{r_{ij}},
\frac{S(r_{ij})\Delta C_{ij,x}}{r_{ij}^2},
\frac{S(r_{ij})\Delta C_{ij,y}}{r_{ij}^2},
\frac{S(r_{ij})\Delta C_{ij,z}}{r_{ij}^2}
\right).
}
\]

Use a small numerical guard in code, e.g.

\[
r_{ij}\leftarrow \max(r_{ij},r_{\epsilon})
\]

only to avoid NaNs. The analytic core should prevent actual overlap in sampled configurations.

As \(r_{ij}\to0\), \(X_i[j,:]\) becomes large. As \(r_{ij}\to r_{\rm out}\), \(X_i[j,:]\to0\) smoothly.

---

## 5. Eq. (13)-style invariant construction

Keep the Eq. (13)-style invariant structure from the 2409.11519 model, but replace the old coordinate row with \(X_i\).

Let

\[
G_{1,i}\in\mathbb R^{m_i\times h},
\qquad
G_{2,i}\in\mathbb R^{m_i\times h}.
\]

Rows are produced by embedding networks:

\[
G_{1,i}[j,:]
=
g_{1,\theta}(r_{ij},S(r_{ij}),s_i,s_j),
\]

\[
G_{2,i}[j,:]
=
g_{2,\theta}(r_{ij},S(r_{ij}),s_i,s_j).
\]

No \(b_{ij}\) label is needed in this one-site-per-molecule model.

No explicit \(s_j-s_i\) feature is required, because \(g_1,g_2\) already receive \(s_i,s_j\).

Define

\[
\boxed{
D_i=G_{1,i}^T X_iX_i^T G_{2,i}.
}
\]

Then

\[
a_i=\widetilde U_\theta(D_i,s_i),
\]

and

\[
A_{\rm MB,\theta}(C,s)=\sum_i a_i.
\]

This remains translation-, rotation-, and permutation-invariant. It is many-body because \(D_i\) depends on all neighbors of \(i\), and \(\widetilde U_\theta\) is nonlinear.

---

## 6. Analytic hard-core prior

Add an explicit hard-core energy:

\[
A_{\rm core}(C,s)=\sum_{i<j}u_{\rm core}(r_{ij};s_i,s_j).
\]

First implementation: use a fixed core radius, independent of \(s_i,s_j\):

\[
\boxed{
u_{\rm core}(r)
=
\begin{cases}
\epsilon_{\rm core}\left[
\left(\frac{r_{\rm core}}{r}\right)^{12}
-2\left(\frac{r_{\rm core}}{r}\right)^6
+1
\right],
& 0<r<r_{\rm core},\\[6pt]
0,& r\ge r_{\rm core}.
\end{cases}
}
\]

Use the notation `u_core` in code; the displayed symbol above is the core pair potential.

This satisfies

\[
\lim_{r\to0^+}u_{\rm core}(r)=+\infty,
\]

\[
u_{\rm core}(r_{\rm core})=0,
\]

\[
u_{\rm core}'(r_{\rm core})=0.
\]

This fixes the old problem: finite energy at \(r=0\) and nonzero force/energy at cutoff.

### Optional later extension

If fixed-core is insufficient, use a size-dependent core radius:

\[
r_{{\rm core},ij}=r_{{\rm core},0}\exp\!\left(\eta\frac{s_i+s_j}{2}\right),
\qquad 0\le \eta\le1.
\]

Do not implement this in the first repair unless validation shows it is needed.

---

## 7. One-body \(s\) prior

Include an optional one-body prior

\[
A_{\rm one}(s)=\sum_i u_s(s_i).
\]

Purpose: stabilize \(s\)-range and encode the main \(p(s)\) marginal if helpful.

Acceptable forms:

1. weak quadratic / quartic around \(s=0\);
2. spline fitted to reference \(p(s)\):
   \[
   u_s(s)=-\beta^{-1}\log p_{\rm ref}(s)+\text{smooth regularization}.
   \]

If used, the neural network should learn the residual coupling beyond the one-body \(p(s)\).

Do not let \(A_{\rm one}\) mask a bad \(A_{\rm MB,\theta}\). Always validate \(\mathbb E[s\mid\rho]\) and \(O_3\).

---

## 8. How to choose \(r_{\rm core}\)

Do not choose \(r_{\rm core}\) by copying the old runtime soft-core radius.

Determine \(r_{\rm core}\) from the atomistic mapped COM RDF:

\[
r_{\rm core}=\min\{r:g_{CC}^{\rm ref}(r)>\delta_g\},
\]

with

\[
\delta_g=10^{-3}\text{ to }10^{-2}.
\]

Alternative robust rule:

\[
r_{\rm core}=Q_{0.001}(r_{ij}^{\rm ref})-\Delta.
\]

The core should only exclude configurations that are essentially absent in the reference. It should not distort the normal first-shell structure.

Report the chosen \(r_{\rm core}\), \(\epsilon_{\rm core}\), and the reference RDF criterion used.

---

## 9. Training strategy

Use the current checkpoint as pretraining if compatible.

If the descriptor changes shape incompatibly, initialize a new model but keep the same training/validation protocol.

### 9.1 Force-matching pretraining

Train the total scalar energy:

\[
A_\theta=A_{\rm core}+A_{\rm one}+A_{\rm MB,\theta}.
\]

Use whole-snapshot training. A data item is a full window/snapshot:

```text
C:   (Nmol, 3)
s:   (Nmol,)
F_C: (Nmol, 3)
F_s: (Nmol,)
```

A batch is:

```text
C:   (B, Nmol, 3)
s:   (B, Nmol)
F_C: (B, Nmol, 3)
F_s: (B, Nmol)
```

For each snapshot, compute total scalar energy and all forces by autograd:

\[
F_C^\theta=-\partial_C A_\theta,
\qquad
F_s^\theta=-\partial_s A_\theta.
\]

Force-matching loss:

\[
\mathcal L_{\rm FM}
=
\sum_\ell
\left[
\frac{\|F_C^\theta(C^\ell,s^\ell)-F_C^\ell\|^2}{\sigma_C^2}
+
\lambda_s
\frac{\|F_s^\theta(C^\ell,s^\ell)-F_s^\ell\|^2}{\sigma_s^2}
\right].
\]

Project out global COM force noise in labels before training:

\[
F_{C_i}^\ell\leftarrow F_{C_i}^\ell-\frac1N\sum_jF_{C_j}^\ell.
\]

### 9.2 Structure fine-tuning

After force pretraining, do cheap CG Langevin sampling and compare structure distributions.

If \(O_3\), ADF, \(S(k)\), or \(g(r)\) remain wrong, fine-tune by relative entropy:

\[
D_{\rm KL}(\mu_{\rm ref}\Vert \mu_\theta)
=
\mathbb E_{\rm ref}[\beta A_\theta(q)]
+
\log Z_\theta
+\text{const}.
\]

Gradient:

\[
\boxed{
\nabla_\theta D_{\rm KL}
=
\beta
\left(
\mathbb E_{\rm ref}[\nabla_\theta A_\theta]
-
\mathbb E_{\theta}[\nabla_\theta A_\theta]
\right).
}
\]

Here \(\mathbb E_\theta\) is estimated from the cheap CG Langevin sampler.

Do not use differentiable histogram losses as the first choice. Use structure metrics for validation and early stopping. Relative entropy is the cleaner thermodynamic objective.

---

## 10. Validation: follow DeePCG spirit

DeePCG validates the learned CG potential by running CG NVT and comparing structural distributions, including RDF, ADF, and higher-order local order parameters. Use the same philosophy here.

Validation must include:

\[
g_{CC}(r),
\]

\[
S(k),
\]

\[
p(s),
\]

\[
p(R_g),
\]

\[
\mathbb E[s_i\mid\rho_i],
\]

\[
P(\theta_{ijk})\quad\text{COM ADF},
\]

\[
O_3=\frac1N\sum_i\sum_{j<k}w(r_{ij})w(r_{ik}).
\]

Also report:

```text
core_active_fraction
wall_active_fraction
min_pair_distance_distribution
small-r RDF hole
```

The current failure is \(O_3^{CG}/O_3^{ref}\approx1.42\). The repaired model should reduce this significantly.

---

## 11. Success criteria

The repaired conservative model passes if:

1. no COM overlap occurs in cheap Langevin sampling;
2. the small-\(r\) RDF hole matches reference;
3. \(p(s)\) and \(p(R_g)\) do not degrade;
4. \(g_{CC}(r)\) stays at least as good as current;
5. \(O_3^{CG}/O_3^{ref}\) moves from \(1.42\) toward \(1\), preferably within \(0.9\)--\(1.1\), or at least improves substantially;
6. ADF/triplet statistics improve relative to the current checkpoint;
7. the runtime soft core is removed or becomes identical to the analytic \(A_{\rm core}\) already included in scalar energy;
8. all forces come from autograd of total scalar energy.

Only after these criteria are met should we revisit memory / GLE hydrodynamics.

---

## 12. Implementation tasks

### C5A: implement descriptor and core

1. Add a new conservative model type, e.g. `mb_deepcg_inv`.
2. Implement smooth switch `S(r)`.
3. Implement inverse-distance rows:
   ```text
   [S/r, S*dx/r^2, S*dy/r^2, S*dz/r^2]
   ```
4. Implement Eq. (13)-style invariant:
   ```text
   D_i = G1_i^T X_i X_i^T G2_i
   ```
5. Add analytic `A_core` to total scalar energy.
6. Add optional `A_one(s)`.
7. Ensure `conservative_forces` differentiates the total scalar energy.

### C5B: train / fine-tune

1. Train with force matching on existing dense N512 instantaneous dataset.
2. If feasible, initialize from current checkpoint where compatible; otherwise train new.
3. Run cheap Langevin structure sampler.
4. Evaluate RDF / ADF / \(S(k)\) / \(p(s)\) / \(p(R_g)\) / \(O_3\).
5. If needed, run relative-entropy fine-tuning using CG samples.

### C5C: compare to current model

Compare the new model against `inst_N512_mb_rc10`:

```text
current A_theta
new inverse-descriptor + hard-core A_theta
```

Use the same sampler length, temperature, box, and analysis scripts.

---

## 13. Files to produce

Required outputs:

```text
outputs/stageC5_conservative_fix/config.yaml
outputs/stageC5_conservative_fix/checkpoint_best.pt
outputs/stageC5_conservative_fix/training_metrics.json
outputs/stageC5_conservative_fix/structure_metrics.json
outputs/stageC5_conservative_fix/g_CC.svg
outputs/stageC5_conservative_fix/S_k.svg
outputs/stageC5_conservative_fix/p_s_Rg.svg
outputs/stageC5_conservative_fix/ADF_triplet.svg
outputs/stageC5_conservative_fix/O3_comparison.svg
outputs/stageC5_conservative_fix/stageC5_summary.md
```

Also update:

```text
results/STAGE_C5_SUMMARY.md
```

---

## 14. Codex cold-review requirements

Before expensive runs, ask Codex to cold-review:

1. Does \(u_{\rm core}(r)\to\infty\) as \(r\to0\)?
2. Does \(u_{\rm core}(r_{\rm core})=0\) and \(u_{\rm core}'(r_{\rm core})=0\)?
3. Does \(S(r_{\rm out})=S'(r_{\rm out})=0\)?
4. Does the inverse descriptor avoid vanishing at overlap?
5. Are all forces from scalar-energy autograd?
6. Is training whole-snapshot, not per-molecule independent?
7. Are validation metrics structure-based, not only force-MSE?

After results, ask Codex to cold-review the interpretation before updating any headline.

---

## 15. Do not do in this stage

Do not change memory / friction in this stage.

Do not add non-Markovian \(\tau\)-learning in this stage.

Do not run larger-box hydro validation until the conservative structure issue is addressed.

Do not claim the hydro deficit is memory-related or cage-related until the repaired \(A_\theta\) has been tested.

---

## 16. Final desired answer from this stage

At the end of Stage C5, answer:

1. Does the inverse-distance descriptor + hard-core prior remove overlap and fix the small-\(r\) RDF hole?
2. Does it reduce \(O_3\) over-packing from the current \(+42\%\)?
3. Does it preserve \(p(s)\), \(p(R_g)\), and \(g_{CC}(r)\)?
4. Does relative-entropy fine-tuning help beyond force matching?
5. Is the repaired conservative model good enough to re-run COM hydro tests?
