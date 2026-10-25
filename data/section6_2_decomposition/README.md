# Section 6.2 per-seed results (precast beam-column joint, specimen P-I-H; 15 random variables; N = 199,857)

- accuracy_16arms_pf3e-03.csv: one row per arm, signed relative error (%) per seed against Pf_RB = 0.002914.
  Error against the crude population count (0.003002): |Pf_RB*(1+e/100)/0.003002 - 1|.
  Arm labels are internal; mapping to Table 10: EFF+CQR, EFF, A12-rand = random design, A8-RB = U-function (beta target),
  A12 = probability-band width (delta-method rule, devised after inspecting these seeds), A11* = T(g) readout on the CQR-UCB design,
  A8 = U-function (margin), A9 = CQR-UCB, A7 = native-band UCB, A10 = CQR-U, A5 = K-fold conformal UCB.
  A1-A4, A6 (Gaussian-process and gradient-boosting arms) are not reported in the paper: fixed-budget implementation with an isotropic
  kernel, not the published AK-MCS.
- eff_across_target_pf.csv: EFF arm at five target probabilities (same population, re-bisected demand level), five seeds each.
