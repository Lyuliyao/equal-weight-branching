# Stage C5 Correction: Conservative Core Must Be Singular at Overlap and Smooth at Cutoff

Date: 2026-06-14  
Repository: `Lyuliyao/rg_com`

This note corrects the conservative-force form discussed in the previous planning notes. The conservative prior cannot be an arbitrary soft patch. If the pair potential is finite at \(r=0\), the equilibrium distribution gives nonzero probability to overlapping COM sites. If the pair potential or force does not vanish smoothly at the cutoff \(r=r_c\), the Langevin equilibrium distribution and force field acquire cutoff artifacts.

The conservative model should therefore use a proper short-range repulsive prior with the following properties:

\[
U_{\rm core}(r)\to +\infty \quad \text{as } r\to0,
\]

\[
U_{\rm core}(r_c)=0,
\qquad
U'_{\rm core}(r_c)=0,
\]

and preferably also

\[
U''_{\rm core}(r_c)=0
\]

if a \(C^2\) force is desired.

---

## 1. Correct conservative energy form

Use

\[
A_\theta(C,s)
=
U_{\rm core}(C)
+
U_{\rm one}(s)
+
A_{\rm MB,\theta}(C,s),
\]

where

\[
A_{\rm MB,\theta}(C,s)
=
\sum_i \widetilde U_\theta(D_i)
\]

is the Eq. (13)-style DeePCG many-body energy, and

\[
U_{\rm core}(C)=\sum_{i<j}u_{\rm core}(r_{ij};s_i,s_j).
\]

The core term should be a fixed physical prior, not a runtime dynamics patch. The learned network should model the smooth many-body PMF residual on top of the non-overlap prior.

---

## 2. Recommended core: shifted-force inverse-power repulsion

Use an inverse-power core with shifted force:

\[
u(r)=\varepsilon\left(\frac{\sigma_{ij}}{r}\right)^p,
\qquad p\ge 12,
\]

for \(0<r<r_c\), and define

\[
\boxed{
u_{\rm core}(r)
=
\varepsilon\left[\left(\frac{\sigma_{ij}}{r}\right)^p
-\left(\frac{\sigma_{ij}}{r_c}\right)^p
+p\left(\frac{\sigma_{ij}}{r_c}\right)^p\frac{r-r_c}{r_c}
\right],\qquad 0<r<r_c.
}
\]

Set

\[
\boxed{
u_{\rm core}(r)=0,\qquad r\ge r_c.}
\]

This gives

\[
u_{\rm core}(r)\to +\infty \quad \text{as }r\to0,
\]

\[
u_{\rm core}(r_c)=0,
\qquad
\nu'_{\rm core}(r_c)=0.
\]

Thus both the energy and force are continuous at the cutoff, while overlaps are forbidden in equilibrium.

The pair force magnitude is

\[
F_{ij}^{\rm core}(r)
=
-\frac{d\nu_{\rm core}}{dr}
=
\varepsilon p\sigma_{ij}^p
\left(
\frac{1}{r^{p+1}}-rac{1}{r_c^{p+1}}
\right),
\qquad 0<r<r_c.
\]

The vector force on molecule \(i\) from \(j\) is

\[
\boxed{
F_i^{\rm core}
=
F_{ij}^{\rm core}(r_{ij})\,\hat r_{ij},
\qquad
\hat r_{ij}=\frac{C_i-C_j}{r_{ij}}.
}
\]

with the opposite force on \(j\).

---

## 3. Optional \(s\)-dependent effective core size

Because molecule size is part of the CG state, the effective exclusion length may depend on \(s_i,s_j\):

\[
\sigma_{ij}
=\sigma_0\exp\bigl[a_\sigma(s_i+s_j)/2\bigr],
\]

or, more conservatively,

\[
\sigma_{ij}=\sigma_0.
\]

Start with constant \(\sigma_{ij}=\sigma_0\). Only add \(s\)-dependence if the static structure diagnostics show a clear size-dependent exclusion effect that the many-body residual cannot learn.

---

## 4. Smooth \(C^2\) alternative

If \(C^2\) smoothness at the cutoff is needed, multiply the singular repulsion by a switching function that equals one near the origin and has zero value, first derivative, and second derivative at the cutoff.

Let

\[
x=r/r_c.
\]

For \(0<x<1\), define

\[
S(x)=1-10x^3+15x^4-6x^5.
\]

Then

\[
u_{\rm core}^{C^2}(r)
=\varepsilon\left(\frac{\sigma}{r}\right)^p S(r/r_c),
\qquad 0<r<r_c,
\]

and zero for \(r\ge r_c\). This is singular at \(r=0\) and vanishes smoothly at \(r_c\). However, the shifted-force inverse-power form above is simpler and usually sufficient.

---

## 5. What to remove

Do not use a core that is finite at \(r=0\). For example, a polynomial or harmonic overlap penalty of the form

\[
\frac{k}{2}(r_c-r)^2
\]

is not acceptable as the only exclusion prior, because it gives finite energy at \(r=0\).

Do not use a core that is nonzero or has nonzero force at \(r=r_c\). This creates cutoff artifacts in equilibrium structure.

Do not leave the core as a runtime-only patch outside the scalar energy. The conservative force must come from the total scalar energy:

\[
F_C=-\partial_C A_\theta,
\qquad
F_s=-\partial_s A_\theta.
\]

---

## 6. Training implication

The many-body neural network should learn the smooth PMF residual:

\[
A_{\rm res,\theta}(C,s)
=
A_{\rm true}(C,s)-U_{\rm core}(C)-U_{\rm one}(s).
\]

The force-matching label for the residual is therefore

\[
F^{\rm res}_{C_i}
=F^{\rm label}_{C_i}-F^{\rm core}_{C_i}-F^{\rm one}_{C_i},
\]

where \(F^{\rm one}_{C_i}=0\) if the one-body term depends only on \(s_i\). The total predicted force is

\[
F^{\rm pred}_{C_i}
=F^{\rm core}_{C_i}-\partial_{C_i}A_{\rm MB,\theta}.
\]

For \(s\):

\[
F^{\rm pred}_{s_i}
=-\partial_{s_i}U_{\rm one}(s_i)-\partial_{s_i}U_{\rm core}(C,s)-\partial_{s_i}A_{\rm MB,\theta}.
\]

If \(\sigma_{ij}\) is constant, \(U_{\rm core}\) contributes no \(s\)-force. If \(\sigma_{ij}\) depends on \(s\), it contributes to \(F_s\) and must be included consistently.

---

## 7. Conservative validation after correction

After implementing the corrected core, rerun cheap CG Langevin sampling and compare:

\[
g_{CC}(r),
\]

\[
S(k),
\]

\[
p(s),\quad p(R_g),
\]

\[
\mathbb E[s\mid \rho],
\]

and three-body packing / ADF:

\[
O_3=\frac1N\sum_i\sum_{j<k}w(r_{ij})w(r_{ik}).
\]

The immediate success criterion is:

\[
O_3^{\rm CG}/O_3^{\rm ref}
\]

moves substantially closer to one without degrading \(p(s),p(R_g)\), or \(g_{CC}(r)\).

The core-active fraction should no longer be interpreted as a runtime-patch diagnostic; it is now part of the equilibrium PMF. Instead report overlap statistics:

\[
\min_{i<j}r_{ij},
\qquad
\Pr(r_{ij}<\sigma_0),
\qquad
\Pr(r_{ij}<r_c).
\]

---

## 8. Codex tasks

1. Replace runtime-only soft COM core by scalar-energy \(U_{\rm core}(C)\).
2. Use shifted-force inverse-power repulsion as the default core.
3. Ensure \(U_{\rm core}(r)\to\infty\) as \(r\to0\).
4. Ensure \(U_{\rm core}(r_c)=0\) and \(U'_{\rm core}(r_c)=0\).
5. Verify force equals negative gradient of scalar energy by finite differences.
6. Retrain or fine-tune the residual many-body network on labels with the core contribution subtracted or included consistently.
7. Rerun cheap Langevin structure validation.
8. Do not proceed to new memory conclusions until the conservative structure distribution has been rechecked.
