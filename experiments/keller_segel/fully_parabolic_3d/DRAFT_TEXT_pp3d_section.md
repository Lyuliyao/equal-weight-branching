# Draft manuscript text — §5.6 three-dimensional fully parabolic–parabolic Keller–Segel

> NOT for direct commit into `paper/` (Overleaf-managed). Paste/adapt into the manuscript
> by hand. Numbers traceable to `reference_results/keller_segel_pp3d/`. Figures:
> Figure B = `radial_*_M88_M96_K12_8seed/figures/figureB_radial_response.pdf`;
> Figure C = `tetra_*/figures/figureC_tetra_control.pdf` (pending).

## Setup paragraph

We test the particle–field method on the three-dimensional fully parabolic–parabolic
Keller–Segel system on the periodic box \(\mathbb T_L^3=[-L/2,L/2]^3\), \(L=12\),
\[
\partial_t u = D_u\Delta u-\chi\nabla\!\cdot(u\nabla v),\qquad
\partial_t v = D_v\Delta v+\alpha u-\beta v,
\]
with the chemical initialized at \(v_0\equiv 0\): the entire chemical field is created from
the cell distribution by the cross-species injection step, so this is a stringent test of the
multi-species mechanism. Cell particles are conservative (no birth/death, since the \(u\)
equation is conservative); the chemical measure evolves by the exact decay–injection substep
\(\mu_v^{n+1}=e^{-\beta\tau}\mu_v^{*}+(\alpha/\beta)(1-e^{-\beta\tau})\mu_u^{*}\) — existing
chemical particles survive with probability \(e^{-\beta\tau}\) and transported cell particles
inject new chemical particles with the minimum-variance integer kernel of mean
\((\alpha/\beta)(1-e^{-\beta\tau})\omega_u/\omega_v\). We use equal particle masses
\(\omega_u=\omega_v\), a first-order Lie split, and reconstruct \(\nabla v\) at the cell
particles from a periodic Fourier representation of the chemical cloud with bandwidth
\(K_{\rm dyn}\). All concentration diagnostics are the **reconstruction-free** particle
quantile radii \(R_q(t)\) (the radius of the smallest torus ball about the cloud centroid
containing a fraction \(q\) of the cell mass), which require no field reconstruction.

We work in the normalized regime \(D_u=D_v=\alpha=\beta=\chi=1\); the dynamics then depend on
a single effective coupling \(F=\chi\alpha M/\beta=M\), the initial cell mass. A first
experiment confirms the coupled algorithm against the exact linear modes (chemical mass law
\(M_v(t)=M_u(0)(1-e^{-t})\) and Fourier-mode errors decreasing at the Monte-Carlo rate
\(N^{-1/2}\); Appendix/Experiment A).

## Radial delayed-response paragraph (Figure B)

For a single radial Gaussian cell cloud (\(\sigma=0.45\)) we sweep the mass \(M\). The cloud
exhibits a sharp transition between a diffusion-dominated regime, in which the core radius
\(R_{0.5}(t)\) grows monotonically, and a **delayed-focusing** regime, in which the cloud
first spreads while the chemical builds up from zero and then turns over and concentrates.
Figure B reports the delayed config \(M=96\) (8 seeds, \(N=10^5\), \(K_{\rm dyn}=12\)). The
chemical mass tracks the exact law \(M_v(t)=M(1-e^{-t})\) [panel (a)]. The core radius first
rises to \(\approx 1.26\,R_{0.5}(0)\) (the cloud expands while the chemical accumulates),
turns over at \(t_{\rm turn}\approx 0.20\), and then collapses to
\(R_{0.5}(T)\approx 0.21\,R_{0.5}(0)\) [panel (b)]; the delay between chemical buildup and
cell response is the signature of the parabolic–parabolic (non-instantaneous) chemical.
Against a sub-critical control \(M=72\), whose core radius grows by a factor \(3.75\) over
the same interval, the contrast between diffusion and delayed focusing is unambiguous
[panel (c)]. Crucially, the reconstruction-free core radius is **stable under \(N\)-refinement**:
the final ratio is \(0.219,\,0.211,\,0.206\) at \(N=2\times10^4,\,10^5,\,3.2\times10^5\)
(a \(16\times\) range), with a per-seed standard deviation of \(0.005\) at \(N=10^5\)
[panel (d)]. The run is numerically stable throughout (drift-resolution number \(\le 0.1\),
no population control, chemical-buffer occupancy below \(60\%\) of capacity).

## Resolution / bandwidth caveat paragraph

Consistent with the reconstruction analysis elsewhere in the paper, the core radius is a
reliable, reconstruction-free diagnostic, whereas reconstruction-sensitive quantities are
not. Two observations make this concrete. First, the **numerical transition mass is
bandwidth-dependent**: the threshold between diffusion and focusing shifts as \(K_{\rm dyn}\)
is increased (a higher bandwidth resolves sharper chemical gradients and therefore lowers the
apparent critical mass). We therefore do not quote a continuum critical mass; the transition
is a property of the reconstructed drift at a given resolution. Second, the deeper the
collapse, the more the late-time core diagnostics depend on resolution: at fixed \(M\) the
core depth \(R_{0.5}(T)\) decreases with \(K_{\rm dyn}\), while the **delayed turnover time
\(t_{\rm turn}\) is resolution-robust**. We also find that the \(80\%\) quantile radius
\(R_{0.8}\) — which straddles the bimodal split between a focusing core and a diffusing halo
— is strongly seed- and \(N\)-scattered and is not a reliable diagnostic here. We therefore
report \(R_{0.5}\) (and the resolution-robust \(t_{\rm turn}\)) as the headline quantities and
treat reconstructed peaks and \(R_{0.8}\) as bandwidth/sampling-sensitive.

## Tetrahedral multi-cluster paragraph (Figure C)

To probe a fully three-dimensional, nonradial, multi-body configuration we place four equal
Gaussian cell clusters (\(\sigma_c=0.25\)) at the vertices of a regular tetrahedron (vertex
radius \(\approx 1.73\), nearest-neighbour centroid distance \(d_{\min}(0)\approx 2.83\)),
with total mass \(M=240\) (super-critical per cluster), and run an active arm (\(\chi=1\))
against a mandatory diffusion control (\(\chi=0\)). The two arms share the same seed, hence
the same initial particles. Figure C reports four seeds at \(N=8\times10^4\), \(K_{\rm dyn}=12\),
to \(T=3\). In the active arm, after a brief early transient (the mean per-cluster core
radius rises to \(\approx 0.48\) before \(t\approx 0.1\)) each cluster collapses and stays
tight, settling at \(\overline{R_{0.5}}(T)\approx 0.16\), while the cluster centroids migrate
inward and the nearest-neighbour distance falls to \(d_{\min}(T)\approx 2.33\). In the
diffusion control the same clusters simply spread (\(\overline{R_{0.5}}\to 3.8\)) and their
centroids stay at the tetrahedron vertices (\(d_{\min}(T)\approx 2.80\)). The contrast is a
factor of \(\approx 23\) in per-cluster core radius and a clear, control-free separation in
\(d_{\min}\). We summarize this as **mutual chemotactic attraction with individual collapse**:
the clusters attract one another (\(d_{\min}\) decreases relative to the control) and each
focuses into a coherent aggregate, rather than merging into a single central mass on the
accessible time horizon. As in the radial case, the inward migration and the attraction occur
on scales well above the reconstruction length and are robust, whereas the depth of each
collapsed core is reconstruction-limited; we therefore report \(d_{\min}\) and the per-cluster
core radius rather than reconstructed peaks.

(Trace: `reference_results/keller_segel_pp3d/tetra_*_a1_M240_K12/`; verdict printed by
`plot_tetra_control.py`. Note: the control \(d_{\min}\) becomes mildly noisy at late times
because diffuse control clusters (\(R_{0.5}\sim 3.8\)) have ill-defined centroids; this does
not affect the active-vs-control separation.)

## Language guardrails honored

- No continuum blow-up time and no universal critical-mass claim (the transition is described
  as numerical/bandwidth-dependent).
- Cross-species source described as injection (\(e^{-\beta\tau}\mu_v^*+(1-e^{-\beta\tau})\mu_u^*\)),
  never as an \((u-v)/v\) multiplicative rate.
- Reconstruction-free core radii separated from bandwidth-sensitive peaks/\(R_{0.8}\).
