# Model card: 635_corrugated_web_steel_frames

Frozen 2026-10-19. Code: `call.py` (`run(params) -> dict(capacity=..., converged=...)`, `RV`, `CAP_UNITS`). The calibrated model is `h_paper_database/calib/635_corrugated_web_steel_frames/model.py`, imported by path (unmodified).

## Structural RVs (independent, in `call.py` RV; no demand RV in the dict)

| name | dist | mean | CoV | CoV basis | note |
|---|---|---|---|---|---|
| fy | lognormal | 333 | 0.07 | typical (JCSS/literature order of magnitude, not re-checked) | Table 1 flange coupon; CoV JCSS-type steel |
| fu | lognormal | 483 | 0.05 | ASSUMED | Table 1; CoV assumed |
| E | lognormal | 200000 | 0.03 | ASSUMED | assumed |
| bf | normal | 125 | 0.02 | ASSUMED | assumed rolled/welded tolerance |
| tf | normal | 7.3 | 0.03 | ASSUMED | READ tf; thickness tolerance assumed |
| h | normal | 266 | 0.015 | ASSUMED | assumed |
| tw | normal | 1.7 | 0.04 | ASSUMED | READ web t; tolerance assumed |
| d_f | lognormal | 25.68 | 0.2 | ASSUMED | CAL fracture stretch of hinge flange (mm); fracture scatter assumed -- dominant RV for catenary/fracture |

Means are the calibrated / measured values. Mean CAL parameters (kappa, lam_s, Vs, r, d_f, W_ring, LAM, pit_scale etc.) are used as model-error / scatter RVs; their CoVs are guesses. All RVs independent (physically correlated groups such as the four fy in 556 are NOT correlated here).

## Capacity R (kN)

R = max vertical load (kN) of the CWES double-span column-removal pushdown, up to the FIRST of (a) bottom-flange hinge fracture (end='fracture') or (b) chord rotation 0.2 rad (UFC 4-023-03), v_lim = 2700*tan(0.2) = 547 mm (end='rotation'). 11/30 MC draws end by rotation; for those, R is a rotation-limited capacity, not a fracture capacity.

## Demand model (applied later, NOT in `call.py`)

S ~ lognormal, CoV 0.30, independent of the structural RVs; median set later so that Pf = P(R < S) = 1e-3 on the N = 1e5 population (g = R - S). Separable: one FE call per structural sample; Rao-Blackwellised Pf = E[F_S(R)] is available.

Numbers from the 30-draw MC (PROFILE.md): see `timing.json` (median s/call, capacity mean / CoV).
