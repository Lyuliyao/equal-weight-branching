# parameter_log.md — kinetic_ks (field-coupled 6D kinetic Keller-Segel)

Record EVERY parameter change here, with date, reason, and the resulting effect
(per the CLAUDE.md "do not cherry-pick silently" rule). The authoritative
baseline parameters below come from corrected plan section 5.3.

## Authoritative baseline (do not edit; copy a row to override)

| symbol     | code key         | value      | meaning                                   |
|------------|------------------|------------|-------------------------------------------|
| L          | L                | 2*pi       | spatial period (torus [-pi,pi]^3)         |
| gamma_v    | gamma_v          | 2          | velocity relaxation rate                  |
| D_v        | D_v              | 1          | velocity diffusion                        |
| chi        | chi              | 1.5        | chemotactic sensitivity                   |
| kappa      | kappa            | 0.5        | chemical screening                        |
| lambda_g   | lambda_g         | 4          | growth amplitude                          |
| alpha_rho  | alpha_rho        | 1          | crowding-death amplitude                  |
| beta       | beta             | 0.2        | baseline death rate                       |
| c0         | c0               | 0.1        | chemo-activation threshold                |
| delta_c    | delta_c          | 0.05       | chemo-activation width                    |
| rho0       | rho0             | 0.2        | crowding half-saturation                  |
| T          | T                | 2 (or 1.5) | horizon                                   |
| tau        | tau              | 2e-3 (1e-3)| time step (fallback 1e-3)                 |
| N0         | N0               | 2e4/8e4/3.2e5 | initial particle count                 |
| seeds      | seeds            | {0,1,2,3}  | pilot {0,1}                               |
| K_x        | K_x              | 8          | Fourier modes/dim (wavenumbers -8..8)     |
| init       | init_kind        | single     | single blob (or "four")                   |
| sigma_x    | sigma_x          | 0.7        | initial x-blob std (0.55 for four)        |
| T_v        | Tv               | 0.5        | velocity temperature = D_v/gamma_v        |

## Change log

| date | who | parameter | old -> new | reason | effect observed |
|------|-----|-----------|------------|--------|-----------------|
| 2026-06-12 | init | (baseline created) | - | corrected plan §5.3 | not yet run |
| 2026-06-12 | tune | K_x | 8 -> 6 | runtime: O(N*(2K_x+1)^3) field solve; kappa=0.5 damps modes >~6 (c spectrum ~1/(k^2+0.25)) | smoke OK |
| 2026-06-12 | tune | smoke (T=0.1) | - | revealed ||c||~0.05 < c0=0.1 => S_c never activates, mass decays, R_0.5 EXPANDS (velocity diffusion dominates) | needs tighter blob + activation |
| 2026-06-12 | tune | mean_field_scan round 1 (alpha_rho=1) | - | scanned sigma_x/c0/chi/lambda_g/beta | activation either OFF (decay+spread) or EXPLOSIVE (mass 14-230x; branching buffer would overflow). alpha_rho=1 cannot self-limit growth |
| 2026-06-12 | tune | alpha_rho | 1 -> ~3-4 (round 2) | self-limiting crowding: saturated-core r=lambda_g-alpha_rho-beta in [0.3,0.8] => bounded LOCALIZED growth (logistic regime the model intends) | round 2: S_c ON, mass bounded 1.2-12x, but ALL SPREAD (R_0.5 grows) -- velocity diffusion D_x=D_v/gamma_v^2=0.25 outpaces chemotaxis |
| 2026-06-12 | tune | D_v, gamma_v (round 3) | D_v 1->0.6, gamma_v 2->3 | reduce spatial diffusion D_x=D_v/gamma_v^2 0.25->0.067 so chemotaxis FOCUSES; keep gamma_v=3 (still kinetic, not overdamped) | round 3: U3 FOCUSES R_0.5 0.758->0.318, S_c ON (mean_Sc->0.99), mass 10.65x, weighted max:mean->6.3 |
| 2026-06-12 | PILOT | (final choice) | U3 regime + beta 0.3->0.4, N0=15000, K_x 8->5, buffer_mult 8->20, sigma_x 0.7->0.5, c0 0.1->0.05, chi 1.5->3, alpha_rho 1->3, gamma_v 2->3, D_v 1->0.6 | beta up caps mass ~4-6x (vs 10.65x); N0/K_x/buffer for runtime+overflow safety | PILOT config_pilot.json; smoke to confirm no overflow |

## CHOSEN PILOT PARAMETERS (config_pilot.json)
gamma_v=3, D_v=0.6 (D_x=0.067), chi=3, kappa=0.5, lambda_g=4, alpha_rho=3, beta=0.4,
c0=0.05, delta_c=0.05, rho0=0.2, sigma_x=0.5, Tv=0.2, K_x=5, T=2, tau=2e-3,
N0=15000, buffer_mult=20, seeds {0,1}. Expected: S_c activates, blob FOCUSES,
mass ~4-6x (self-limited), weighted degenerates while branching keeps equal weights.
DEVIATION from corrected-plan §5.3 defaults recorded above; justified by the smoke
(default c0=0.1 left S_c inactive) + 3-round mean-field tuning. To be stated in MS.

## Open knobs that may need tuning (flag if changed)
- `buffer_mult` (default 8): branching population can grow; if buffer overflow is
  raised, increase buffer_mult OR strengthen alpha_rho/beta (crowding death) OR
  reduce lambda_g / T. Record any change here.
- `K_x` (default 8): controls field resolution AND the eval-grid CFL dx proxy.
  Larger K_x sharpens c but costs O(N*(2K_x+1)^3). Record any change.
- `tau` fallback 1e-3, `T` alternate 1.5: supported; record if used.
- `ess_thresh` (default 0.5): resample trigger for weighted_resample.
