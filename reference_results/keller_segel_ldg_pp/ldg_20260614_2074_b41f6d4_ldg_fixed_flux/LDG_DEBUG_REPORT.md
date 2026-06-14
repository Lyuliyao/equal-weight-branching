# LDG chemotaxis-flux bug: debug report

**Trigger:** the debug note flagging that our direct LDG `tb(1.05) ≈ 7.36×10⁻⁶`
looked wrong against the paper's reference numerical blow-up time `≈1.21×10⁻⁴`,
and suspecting a chemotaxis flux-assembly bug in `ldg_solver.py::conv_rhs`.

## 1. Bug confirmed

`conv_rhs` assembles the boundary part of the chemotaxis term
`−div(r u)` (`r=∇v`). The face-moment helpers `_lf_face_x` / `_lf_face_y`
already fold the **basis face value** into each returned component
(`integ` multiplies by `xival`/`s`/`etaval`), so the only edge sign left to apply
is the outward normal `n` (+1 right/top, −1 left/bottom): the assembly must be a
uniform `R − L` (resp. `T − B`) **for every modal component**.

The previous code instead had:

```python
# x-face:  xi-mode used + instead of -
out[..., 1] -= (fxR[1] + fxL[1])          # BUG: double-counted the xi face sign
# y-face:  xi and eta modal indices were SWAPPED (and the eta line used +)
out[..., 2] -= (fyT[1] + fyB[1])          # BUG: out[eta] fed the xi moment, with +
out[..., 1] -= (fyT[2] - fyB[2])          # BUG: out[xi]  fed the eta moment
```

Fixed to:

```python
out[..., 0] -= (fxR[0] - fxL[0]);  out[..., 1] -= (fxR[1] - fxL[1]);  out[..., 2] -= (fxR[2] - fxL[2])
out[..., 0] -= (fyT[0] - fyB[0]);  out[..., 1] -= (fyT[1] - fyB[1]);  out[..., 2] -= (fyT[2] - fyB[2])
```

The cell-average (`phi0`) component was correct, which is why the previous
cell-average-only consistency test and exact mass conservation **did not catch
it** — the error was entirely in the slope (`xi`,`eta`) modes, i.e. in the
subcell structure that drives the focusing dynamics.

## 2. Verification (now at the modal / symmetry / solution level)

A note on the suggested "modal `L_h(Pu)` vs `P(Lu)`" test: it is **not a valid LDG
consistency test** — the *already-verified* Laplacian also fails it, because LDG
operators do not commute with L² projection (consistency is in the Galerkin /
solution sense, controlled by the energy estimate, not pointwise per mode). We
therefore verify with symmetry invariants and solution convergence
(`test_ldg.py`):

```
[PASS] diffusion 2nd order .............. orders 2.00, 2.03
[PASS] full coupled-KS self-convergence . orders 2.11, 2.46   (the valid chemotaxis test)
[PASS] x<->y permutation symmetry ....... rel diff 1.9e-15    (was FAIL before the fix)
[PASS] constant v => zero chemotaxis .... max|conv| 1.3e-35
[PASS] radial symmetry (blow-up IC) ..... center ~3e-6, m_xx = m_yy exactly, m_xy/m_xx ~1.5e-6
```

The **permutation-symmetry** and **radial-symmetry** tests are the decisive ones:
the buggy asymmetric flux broke x↔y symmetry (and would skew a radial blow-up);
both are now exact. Diffusion, operator symmetry (`||L−Lᵀ||/||L|| = 5.5e-17`),
exact 10π mass, and positivity are unchanged.

## 3. Effect on the result (old vs fixed)

Old and fixed runs are both kept; the old run is **record-only / provisional**.

| quantity | OLD (buggy) | FIXED |
|---|---|---|
| `tb(1.05)`  (80→160) | `7.36×10⁻⁶` | **`5.95×10⁻⁵`** |
| `tb(1.05)`  (160→320) | `7.36×10⁻⁶` | **`8.43×10⁻⁵`** |
| `tb(1.10)`  (80→160) | `1.00×10⁻⁵` | `6.56×10⁻⁵` |
| `tb(1.10)`  (160→320) | `1.07×10⁻⁵` | `9.30×10⁻⁵` |
| `S_L2(2e-4)` N=80 | `893` | `874` |
| `S_L2(2e-4)` N=160 | `1604` | `1604` |
| `S_L2(2e-4)` N=320 | `3274` | `3159` |
| `S_L2(6e-5)` N=160 | `1375` | `655` |
| `S_L2(6e-5)` N=320 | `2424` | `660` |

The bug made the **early-time** concentration spuriously fast (wrong slope modes),
opening the resolution gap ~10× too early and **freezing `tb(N)` at `7.36×10⁻⁶`
for both refinement pairs** (no trend). With the fix:

- `tb(1.05)` moves to `5.95×10⁻⁵` (80→160) and `8.43×10⁻⁵` (160→320) — it now
  **increases monotonically toward** the reference `1.21×10⁻⁴` as the mesh
  refines, which is exactly the convergence behaviour the paper reports for their
  1-D manufactured test (`tb(N) → T*`, first order). The buggy version showed no
  such trend.
- The corrected `tb` is the **same scale as the FVM baseline** (`3.5–5.0×10⁻⁵`).
- The **late-time** concentration `S_L2(2e-4)` is essentially unchanged (the
  cell-average dynamics were always correct); only the early-time / subcell
  structure moved (`S_L2(6e-5)` for N=320 fell from `2424` to `660`).

## 4. Comparison with the paper (after the fix)

The paper's `1.21×10⁻⁴` is a reference numerical blow-up time (from their prior
adaptive-method work [15]); their own `tb(N)=inf{t: S(2N,t)≥1.05 S(N,t)}`
indicator (their (5.2)) was demonstrated only on a **1-D manufactured** test, and
they explicitly defer the KS `tb(N)` to future adaptive-mesh work. Our fixed
uniform-mesh KS `tb(1.05) ≈ 6×10⁻⁵` is therefore the **same order** as their
reference but is a resolution-gap indicator on a uniform mesh, not a converged
continuum blow-up time. We do **not** claim to reproduce the blow-up time.

Safe manuscript language:

> The direct LDG reproduction shows the same pre-singular concentration trend and
> strong resolution dependence at the reported times; the uniform-mesh
> resolution-gap time is of the same order as the reference numerical blow-up
> time but is not a converged continuum value.

## 5. Recommendation for §5.4

- Use the **fixed-flux** run as the LDG comparator; keep the old run as a
  record-only provisional result.
- Report the LDG reporting-time concentration (`S_L2`, peak, snapshots) and the
  resolution trend N=80,160,320, **not** a single `tb` scalar as a blow-up time.
- Do not write "we reproduce the LDG blow-up time." Do write that the LDG and the
  particle method show the same pre-singular concentration with strong resolution
  dependence.
