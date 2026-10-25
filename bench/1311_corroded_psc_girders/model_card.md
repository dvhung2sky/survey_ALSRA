# Model card: 1311_corroded_psc_girders

Frozen 2026-10-19. Code: `call.py` (`run(params) -> dict(capacity=..., converged=...)`, `RV`, `CAP_UNITS`). The calibrated model is `h_paper_database/calib/1311_corroded_psc_girders/model.py`, imported by path (unmodified).

## Structural RVs (independent, in `call.py` RV; no demand RV in the dict)

| name | dist | mean | CoV | CoV basis | note |
|---|---|---|---|---|---|
| fcm | lognormal | 44.07 | 0.12 | typical (JCSS/literature order of magnitude, not re-checked) | paper fcm; CoV JCSS-type concrete |
| Pe | normal | 400000 | 0.05 | ASSUMED | effective prestress force; CoV assumed (loss uncertainty) |
| fpu | lognormal | 1883 | 0.025 | typical (JCSS/literature order of magnitude, not re-checked) | strand fpu paper; CoV ~0.025 literature (prestressing strand), fpy scaled with it |
| Ep | lognormal | 195000 | 0.02 | ASSUMED | assumed |
| b | normal | 280 | 0.02 | ASSUMED | assumed geometric tolerance |
| h | normal | 380 | 0.015 | ASSUMED | assumed geometric tolerance |
| c_strand | normal | 1 | 0.05 | ASSUMED | multiplier on strand eccentricity profile (sd ~3-4 mm); assumed |
| pit_scale | lognormal | 1 | 0.3 | ASSUMED | scale of the MEASURED wire-area loss (Table 2); CoV assumed -- corrosion RV |
| LAM | normal | 0.82 | 0.1 | ASSUMED | true/idealised pit-loss factor (Table 9 calibrated); CoV assumed |
| EPU | normal | 0.075 | 0.1 | ASSUMED | strand rupture strain (Table 8); CoV assumed |

Means are the calibrated / measured values. Mean CAL parameters (kappa, lam_s, Vs, r, d_f, W_ring, LAM, pit_scale etc.) are used as model-error / scatter RVs; their CoVs are guesses. All RVs independent (physically correlated groups such as the four fy in 556 are NOT correlated here).

## Capacity R (kN)

R = ultimate load (kN) = peak load over the monotonic push of girder MSG5 (0.1 mm steps, to 40 mm; ruptured wires drop out and the push continues).
Revised 2026-10-20 (v2b). v1 used the test report's "peak load before the first failure event": with pit_scale >~ 2 a pitted wire ruptures at
the first step (~12 kN) while the girder still carries ~210 kN, giving a spurious 1.15% low mode and a demand median of 8.6 kN at Pf = 1e-3.
Both definitions give 284.5 kN at the mean parameters (calibration unchanged). The first-event load is still returned as `Pu_first_event`.
Converged = full push, or descending branch (P_end < 0.8 P_max), or a failure event within 0.15 mm (1.5 steps) before or after the peak;
in the 5e4 population 49,868 rows converge (132 dropped, median capacity 306 kN vs 287 kN: strong girders, negligible effect on Pf).

## Demand model (applied later, NOT in `call.py`)

S ~ lognormal, CoV 0.30, independent of the structural RVs; median set later so that Pf = P(R < S) = 1e-3 on the N = 1e5 population (g = R - S). Separable: one FE call per structural sample; Rao-Blackwellised Pf = E[F_S(R)] is available.

Numbers from the 30-draw MC (PROFILE.md): see `timing.json` (median s/call, capacity mean / CoV).
