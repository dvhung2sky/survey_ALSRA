# Model card: 556_xbraced_steel_frame

Frozen 2026-10-19. Code: `call.py` (`run(params) -> dict(capacity=..., converged=...)`, `RV`, `CAP_UNITS`). The calibrated model is `h_paper_database/calib/556_xbraced_steel_frame/model.py`, imported by path (unmodified).

## Structural RVs (independent, in `call.py` RV; no demand RV in the dict)

| name | dist | mean | CoV | CoV basis | note |
|---|---|---|---|---|---|
| fy_cf | lognormal | 356.2 | 0.07 | typical (JCSS/literature order of magnitude, not re-checked) | Table 1 mean; CoV JCSS-type structural steel fy ~0.07 |
| fy_cw | lognormal | 425.6 | 0.07 | typical (JCSS/literature order of magnitude, not re-checked) | Table 1 mean; CoV JCSS-type structural steel fy ~0.07 |
| fy_bf | lognormal | 425.6 | 0.07 | typical (JCSS/literature order of magnitude, not re-checked) | Table 1 mean; CoV JCSS-type structural steel fy ~0.07 |
| fy_bw | lognormal | 376.6 | 0.07 | typical (JCSS/literature order of magnitude, not re-checked) | Table 1 mean; CoV JCSS-type structural steel fy ~0.07 |
| fy_ly | lognormal | 212.6 | 0.08 | ASSUMED | LY225 mean Table 1; CoV assumed |
| E_ly | lognormal | 194000 | 0.03 | ASSUMED | assumed |
| b_ly | normal | 0.004 | 0.3 | ASSUMED | assumed (CAL-ish hardening ratio) |
| W_ring | normal | 105 | 0.03 | ASSUMED | assumed (CAL geometry, not printed) |
| t_ring | normal | 16 | 0.02 | ASSUMED | assumed plate tolerance |
| N_col | normal | 360000 | 0.1 | ASSUMED | assumed gravity load, normal |

Means are the calibrated / measured values. Mean CAL parameters (kappa, lam_s, Vs, r, d_f, W_ring, LAM, pit_scale etc.) are used as model-error / scatter RVs; their CoVs are guesses. All RVs independent (physically correlated groups such as the four fy in 556 are NOT correlated here).

## Capacity R (kN)

R = max lateral force (kN) of a monotonic pushover of the T1 frame (virgin frame + ring damper) to 2.0 % drift (45 mm); no fracture in the model, so this is the force at the drift limit (hardening-controlled).

## Demand model (applied later, NOT in `call.py`)

S ~ lognormal, CoV 0.30, independent of the structural RVs; median set later so that Pf = P(R < S) = 1e-3 on the N = 1e5 population (g = R - S). Separable: one FE call per structural sample; Rao-Blackwellised Pf = E[F_S(R)] is available.

Numbers from the 30-draw MC (PROFILE.md): see `timing.json` (median s/call, capacity mean / CoV).
