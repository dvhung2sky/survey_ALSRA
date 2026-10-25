# Model card: 578_precast_postcast_joints

Frozen 2026-10-19. Code: `call.py` (`run(params) -> dict(capacity=..., converged=...)`, `RV`, `CAP_UNITS`). The calibrated model is `h_paper_database/calib/578_precast_postcast_joints/model.py`, imported by path (unmodified).

## Structural RVs (independent, in `call.py` RV; no demand RV in the dict)

| name | dist | mean | CoV | CoV basis | note |
|---|---|---|---|---|---|
| fc_cast | lognormal | 31.22 | 0.12 | typical (JCSS/literature order of magnitude, not re-checked) | Table 7 mean; CoV JCSS-type concrete |
| fc_pre | lognormal | 36.56 | 0.1 | ASSUMED | Table 7 mean; CoV assumed (precast plant) |
| fy18 | lognormal | 415.7 | 0.06 | typical (JCSS/literature order of magnitude, not re-checked) | READ D18; CoV JCSS-type rebar |
| fy10 | lognormal | 407.8 | 0.06 | typical (JCSS/literature order of magnitude, not re-checked) | READ D10 waist bars; CoV JCSS-type rebar |
| n_ratio | normal | 0.234 | 0.1 | ASSUMED | READ axial ratio; gravity load CoV assumed |
| kappa | lognormal | 0.58 | 0.15 | ASSUMED | CAL panel strength = model-error RV; CoV assumed |
| lam_s | lognormal | 4 | 0.2 | ASSUMED | CAL slip multiplier = model-error RV; CoV assumed |

Means are the calibrated / measured values. Mean CAL parameters (kappa, lam_s, Vs, r, d_f, W_ring, LAM, pit_scale etc.) are used as model-error / scatter RVs; their CoVs are guesses. All RVs independent (physically correlated groups such as the four fy in 556 are NOT correlated here).

## Capacity R (kN)

R = peak lateral force (kN) of a monotonic push of the PRCJ1 column top to 100 mm under constant axial load (joint-panel governed).

## Demand model (applied later, NOT in `call.py`)

S ~ lognormal, CoV 0.30, independent of the structural RVs; median set later so that Pf = P(R < S) = 1e-3 on the N = 1e5 population (g = R - S). Separable: one FE call per structural sample; Rao-Blackwellised Pf = E[F_S(R)] is available.

Numbers from the 30-draw MC (PROFILE.md): see `timing.json` (median s/call, capacity mean / CoV).
