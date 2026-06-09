# Third-party code

`experiments/highdim/` vendors the following files from
[Xun-Tang123/FHT_for_deans_equation](https://github.com/Xun-Tang123/FHT_for_deans_equation)
(reference implementation of: Xun Tang and Lexing Ying, "Solving the Fokker-Planck equation of
discretized Dean-Kawasaki models with functional hierarchical tensor", arXiv:2503.22816; MIT license):

- `fht_utils.py` (verbatim)
- `functional_hierarchical_tensor_sketch.py` (verbatim)
- `functional_hierarchical_tensor_fourier.py` (one documented patch: ghost (padding) dimensions of non-power-of-two
  problem sizes are marginalized through their constant-mode projection, enabling d=6 via padding to 8 leaves)

These files are used only for the low-rank density *diagnostics* of the high-dimensional experiment; the particle
dynamics never depend on them. They remain under the upstream MIT license and copyright.
