# Model card: 523_infilled_frames_aac_lsf

Frozen 2026-10-19. Code: `call.py` (`run(params) -> dict(capacity=..., converged=...)`, `RV`, `CAP_UNITS`). The calibrated model is `h_paper_database/calib/523_infilled_frames_aac_lsf/model.py`, imported by path (unmodified).

## Structural RVs (independent, in `call.py` RV; no demand RV in the dict)

| name | dist | mean | CoV | CoV basis | note |
|---|---|---|---|---|---|
| fc | lognormal | 30.15 | 0.12 | typical (JCSS/literature order of magnitude, not re-checked) | paper mean; CoV JCSS-type concrete (fcc scaled with fc) |
| fy | lognormal | 430 | 0.06 | typical (JCSS/literature order of magnitude, not re-checked) | paper mean; CoV JCSS-type rebar |
| fu | lognormal | 545 | 0.05 | ASSUMED | paper mean; CoV assumed |
| N_ax | normal | 188438 | 0.1 | ASSUMED | 0.1 fc Ag gravity; CoV assumed |
| Em | lognormal | 2195 | 0.2 | ASSUMED | AAC masonry modulus (derived Table 3); CoV assumed |
| Vs | lognormal | 95000 | 0.25 | ASSUMED | CAL strut peak horizontal force (N) = AAC infill strength; CoV assumed |
| r | normal | 0.2 | 0.3 | ASSUMED | CAL residual strut ratio; assumed |

Means are the calibrated / measured values. Mean CAL parameters (kappa, lam_s, Vs, r, d_f, W_ring, LAM, pit_scale etc.) are used as model-error / scatter RVs; their CoVs are guesses. All RVs independent (physically correlated groups such as the four fy in 556 are NOT correlated here).

## Capacity R (kN)

R = peak lateral force (kN) of a monotonic push of the AAC-infilled frame to 25 mm (right-beam LPDT); peak near 5 mm then strut softening.

## Demand model (applied later, NOT in `call.py`)

S ~ lognormal, CoV 0.30, independent of the structural RVs; median set later so that Pf = P(R < S) = 1e-3 on the N = 1e5 population (g = R - S). Separable: one FE call per structural sample; Rao-Blackwellised Pf = E[F_S(R)] is available.

Numbers from the 30-draw MC (PROFILE.md): see `timing.json` (median s/call, capacity mean / CoV).
