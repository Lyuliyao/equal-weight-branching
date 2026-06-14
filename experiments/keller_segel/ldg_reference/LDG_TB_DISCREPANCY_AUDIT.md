# LDG `t_b` discrepancy audit and implementation verification plan

Status: **blocker before using LDG `t_b` in the manuscript**.

This note records why the currently archived value

```text
t_b(80 -> 160; theta=1.05)  = 7.36e-6
t_b(160 -> 320; theta=1.05) = 7.36e-6
```

should **not** yet be treated as a verified reproduction of the Li--Shu--Yang LDG
benchmark.  The direct LDG reference has useful components and passes several checks,
but the `t_b` discrepancy requires a stricter audit of both the mathematical definition
being compared and the implementation of the chemotaxis flux.

---

## 1. Immediate conclusion

Do **not** write any of the following in the manuscript yet:

```text
we reproduce the LDG blow-up time;
t_b = 7.36e-6 is the LDG numerical blow-up time;
our LDG reproduction confirms the paper's 1.21e-4 blow-up time;
particle t_gap can be compared directly to this LDG t_b.
```

Safe current statement:

```text
The new LDG implementation reproduces strong reporting-time concentration and provides
a direct LDG comparator for snapshots, S_L2, peak, mass, and positivity.  However, the
uniform-grid resolution-gap indicator currently computed from S_{2N}/S_N crosses much
earlier than the reference time quoted in the paper.  Until the LDG flux implementation
and the exact definition of the paper's reported time are rechecked, we do not quote a
continuum or LDG blow-up time.
```

---

## 2. What was computed

Current code defines

```math
t_b(N;\theta)=\inf\{t:S_{2N}(t)\ge \theta S_N(t)\},
\qquad S_N(t)=\|u_N(t)\|_{L^2}.
```

The archived direct-LDG run gives

```text
80 -> 160:  theta=1.05 crossing at 7.3578595317725755e-6
160 -> 320: theta=1.05 crossing at 7.3578595317725755e-6
```

This value is far earlier than the reference numerical time often quoted for the LDG
paper's Example 5.2, approximately `1.21e-4`.  Therefore there are two separate issues:

1. **Definition mismatch:** the paper's quoted numerical blow-up time may not be the same
   object as the uniform-grid `S_{2N}/S_N` crossing used here.
2. **Implementation risk:** the current LDG code may have a bug in the chemotaxis flux
   assembly, so the archived `t_b` should not be used until the code passes stricter
   tests.

---

## 3. Why definition mismatch alone is not enough

Even if the implementation were perfect, `t_b(N;theta)` above might not equal the
paper's quoted time.  The paper uses the LDG benchmark to show concentration snapshots
and discusses numerical blow-up behavior, while the `S_{2N}/S_N` crossing is a
resolution-gap proxy.  We must check the paper directly and record:

```text
- the exact equation and boundary condition used in Example 5.2;
- the exact domain;
- whether 1.21e-4 is obtained from a threshold, a visual blow-up criterion, a maximum
  norm criterion, adaptive mesh behavior, or a separate reference computation;
- whether formula (5.2) was actually applied to the 2D Keller--Segel blow-up example or
  only to a manufactured / lower-dimensional convergence test;
- the polynomial degree k, grid sizes, time-step rule, positivity limiter, and plotted
  reporting times.
```

Until this is settled, do not compare our `t_b` number directly with `1.21e-4`.

---

## 4. Code-level concern: chemotaxis modal face assembly

The highest-priority code audit is `experiments/keller_segel/ldg_reference/ldg_solver.py`,
function `conv_rhs`.  This function assembles the DG weak form for the chemotaxis term

```math
-\nabla\cdot(u\nabla v).
```

The current implementation uses face moments returned by `_lf_face_x` and `_lf_face_y`.
The returned moment ordering is

```text
index 0: moment against constant basis 1
index 1: moment against xi
index 2: moment against eta
```

### 4.1 Suspected y-face component swap

In the current code, the y-face terms are added as

```python
out[..., 0] -= (fyT[0] - fyB[0])
out[..., 2] -= (fyT[1] + fyB[1])
out[..., 1] -= (fyT[2] - fyB[2])
```

This appears to swap the `xi` and `eta` modal components.  Since `_lf_face_y` returns
`[constant, xi, eta]`, the y-face contribution should enter the same modal slots:

```python
out[..., 0]  # constant mode
out[..., 1]  # xi mode
out[..., 2]  # eta mode
```

A first candidate correction is therefore

```python
out[..., 0] -= (fyT[0] - fyB[0])
out[..., 1] -= (fyT[1] - fyB[1])
out[..., 2] -= (fyT[2] - fyB[2])
```

This must be verified against the exact DG weak form, not patched blindly.

### 4.2 Suspected x-face odd-mode sign issue

For x-faces, the current code uses

```python
out[..., 0] -= (fxR[0] - fxL[0])
out[..., 1] -= (fxR[1] + fxL[1])
out[..., 2] -= (fxR[2] - fxL[2])
```

However, `_lf_face_x` already integrates the left face against the basis value at
`xi=-1`.  If `fxL[1]` already contains the negative trace value, then adding with `+`
can flip the intended odd-mode contribution.  A candidate correction is

```python
out[..., 0] -= (fxR[0] - fxL[0])
out[..., 1] -= (fxR[1] - fxL[1])
out[..., 2] -= (fxR[2] - fxL[2])
```

Again, this must be checked by a modal manufactured test.

### 4.3 Why this matters

The diffusion checks can pass even if the chemotaxis face modal assembly is wrong.
Mass conservation can also pass because it mostly checks the constant mode.  The blow-up
benchmark is dominated by the chemotactic drift, so an error in the `xi/eta` modal face
assembly can move the resolution-gap crossing by orders of magnitude.

---

## 5. Required verification before rerunning production

The current checks are useful but insufficient.  Before using LDG `t_b`, add the following
verification tests.

### Test A: modal manufactured chemotaxis RHS

Pick smooth non-radial functions, for example

```math
u(x,y)=1+0.2\sin(2\pi x)\cos(3\pi y),
\qquad
v(x,y)=0.3\cos(\pi x)+0.2\sin(2\pi y)+0.1\sin(\pi x)\sin(\pi y).
```

Compute the exact field

```math
R(x,y)=-\nabla\cdot(u\nabla v),
```

project it onto the same P1 modal basis, and compare against `conv_rhs` cellwise.
The error must be checked separately for

```text
constant coefficient,
xi coefficient,
eta coefficient.
```

This test should converge at the expected order under grid refinement.  A cell-average-only
test is not enough.

### Test B: x-y permutation symmetry

For a non-symmetric manufactured pair `(u(x,y), v(x,y))`, define the swapped pair

```math
u^\sharp(x,y)=u(y,x),
\qquad
v^\sharp(x,y)=v(y,x).
```

The computed chemotaxis RHS should satisfy the corresponding x-y swapped relation.
This test is designed to catch exactly the suspected `xi/eta` component swap.

### Test C: constant-gradient transport sanity check

Choose `v(x,y)=a x + b y` locally, with boundary effects avoided or tested on periodic
manufactured data, and choose a smooth `u`.  Then

```math
-\nabla\cdot(u\nabla v)=-(a\partial_x u+b\partial_y u),
```

so the chemotaxis operator reduces to linear advection.  Verify that the sign and modal
components agree with a known DG advection operator.

### Test D: radial symmetry preservation

For the blow-up initial data

```math
u_0=840e^{-84r^2},
\qquad
v_0=420e^{-42r^2},
```

run to a short time such as `1e-5` and record

```text
center of mass,
second moments Mxx, Myy,
off-diagonal moment Mxy,
axis-aligned peak location,
radial profile error.
```

For a correct implementation, the radial symmetry error should be small and should
decrease under refinement.  This is not a replacement for Test A, but it is a useful
blow-up-specific sanity check.

### Test E: snapshot-level comparison to the LDG paper

Before discussing `t_b`, compare the actual LDG snapshots / diagnostics at the reporting
times:

```text
t = 6e-5, 1.2e-4, 2e-4
S_L2(t)
peak(t)
mass_u(t), mass_v(t)
u_min(t)
profile cuts through the core
```

If the paper only gives images, digitize or compare profile/peak ranges with stated
uncertainty.  Do not use `t_b` as the first validation target.

---

## 6. Required code changes / tasks

### Task 1: freeze current LDG result as provisional

Update the current result log to mark the archived `t_b=7.36e-6` as **provisional / not
manuscript-ready** until the chemotaxis flux audit is complete.

### Task 2: add isolated chemotaxis tests

Add a file:

```text
experiments/keller_segel/ldg_reference/test_ldg_chemotaxis.py
```

It should run Tests A--C and print a compact table:

```text
N | constant-mode error | xi-mode error | eta-mode error | symmetry error
```

### Task 3: fix `conv_rhs` only after tests expose the failure

Do not patch by intuition alone.  The test should first fail on the current code and then
pass after the correction.

### Task 4: rerun LDG after correction

After the tests pass, rerun

```bash
for N in 80 160 320; do
  python experiments/keller_segel/ldg_reference/run_ldg.py \
      --N $N --T 2e-4 \
      --report_times 6e-5 1.2e-4 2e-4 \
      --out_dir reference_results/keller_segel_ldg_pp/ldg_fixed_<run_id>/N$N
done
```

Then recompute

```text
S_L2 reporting-time table,
peak reporting-time table,
tb_1_05, tb_1_10,
snapshot/profile comparison.
```

### Task 5: only then rewrite Section 5.4

The Section 5.4 rewrite should wait until the LDG comparator is verified.  If the fixed
LDG `t_b` still does not match `1.21e-4`, then the manuscript should state that the
reporting-time concentration is reproduced, while the blow-up-time proxy is not quoted.

---

## 7. Interpretation after the audit

There are three possible outcomes.

### Outcome 1: flux bug explains the discrepancy

After fixing `conv_rhs`, the LDG curves move substantially and `t_b` shifts toward the
paper's reference behavior.  Then the archived old LDG result should be discarded and
all Section 5.4 numbers regenerated.

### Outcome 2: flux bug exists but `t_b` still differs

Then the implementation was partly wrong, but the remaining discrepancy is likely a
definition / adaptivity / threshold issue.  In this case, use the LDG run for snapshots,
`S_L2`, peak, mass, and qualitative concentration, but do not compare `t_b` to the
paper's quoted time.

### Outcome 3: no flux bug after testing

Then the candidate modal concerns were false, and the discrepancy is primarily about the
meaning of `t_b`.  Still, Section 5.4 must distinguish clearly between reporting-time
concentration and blow-up-time estimation.

---

## 8. Safe manuscript language after successful LDG audit

If LDG snapshots and reporting-time diagnostics match but `t_b` remains sensitive:

```latex
We use the LDG calculation as a direct deterministic comparator for the fully
parabolic--parabolic Keller--Segel benchmark.  The LDG implementation follows the
modal P1 local discontinuous Galerkin discretization of Li--Shu--Yang and is verified
on diffusion, chemotaxis-consistency, mass conservation, and positivity tests.  On the
super-critical Gaussian datum, the LDG solution exhibits the expected strong
resolution-dependent concentration at the reported times.  We do not, however, quote a
continuum blow-up time: uniform-grid resolution-gap indicators and reconstructed peaks
remain sensitive to resolution and reconstruction in the pre-singular window.
```

If the LDG flux audit is not finished before submission:

```latex
The direct LDG reproduction is still under verification and is therefore not used as a
quantitative blow-up-time reference.  We report only reconstruction-free particle radii
and conservative mass diagnostics, and keep the grid comparison as a sanity check rather
than a claimed LDG blow-up-time match.
```

---

## 9. Short instruction for Claude / Codex

```text
Do not trust the archived LDG tb=7.36e-6 yet.
First verify the chemotaxis flux assembly in ldg_solver.py::conv_rhs.
Construct modal manufactured tests that check constant, xi, and eta components, plus
x-y permutation symmetry.  Only after those tests pass should the LDG Example 5.2
production sweep be rerun and Section 5.4 rewritten.
```
