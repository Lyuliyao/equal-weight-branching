# Manuscript

LaTeX source for the manuscript

> *Adaptive equal-weight branching particle methods for non-conservative
> transport–reaction–diffusion equations*, Liyao Lyu and Huan Lei.

`cmame-main.tex` is the main file (Elsevier `cas-sc` class). The figures in
`figure/` are produced by the experiment plot scripts under `../experiments/`;
the production data behind the new §5.2 / §5.4 / §5.5 / §5.x figures live under
`../reference_results/`.

## Build

```bash
pdflatex cmame-main
bibtex   cmame-main
pdflatex cmame-main
pdflatex cmame-main
```

Requires a TeX Live (or MiKTeX) installation. The local class/style/bib-style
files (`cas-sc.cls`, `cas-common.sty`, `lyu.sty`, `elsarticle-num.bst`) and the
author `thumbnails/` are vendored here so the build is self-contained.

## Revision contents (this branch)

The numerical section was revised to add: the separated growth-island benchmark
(§5.2, "global ESS is not a local degeneracy diagnostic"), the cross-species
injection kernel for the parabolic–parabolic Keller–Segel chemical equation
(coupled-system section + §5.4), the LDG-aligned parabolic–parabolic concentration
comparison (§5.5), and the local-reconstruction diagnostics (§5.x). The figures
and tables are backed by the reproducibility records in `../reference_results/`.
