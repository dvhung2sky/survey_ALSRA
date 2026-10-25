# Corpus statistics: 81 AL / AL+UQ papers coded on the 11 design-space axes

Source: `corpus_coding.csv`; regenerate with `python3 corpus_stats.py > corpus_stats.md` (do NOT run corpus_merge.py: it rebuilds the CSV from raw coder output and would undo the 2026-10-19 recode of #223 batch adapt -> kmeans).

## Corpus limitations (quote wherever these numbers are used)
- Convenience collection: 2,887 PDFs from 14 journal folders, 1,590 unique by title, 382 keyword candidates, abstract-level LLM classification, 335 relevant (5 AL, 76 AL+UQ, 254 UQ). "AL papers" = the 81 in {AL, AL+UQ}.
- Unbalanced coverage: Structural Safety 2021-2027; RESS 2027 vols 277-278 only; IEEE Trans. Reliab. vol 75 (2026) + early access; CMAME 2026-27.
- Coding by four parallel language-model coders (abstract + method/example sections); one primary option per axis; confidence mostly medium; no inter-coder agreement measured; only reports_repeats cross-checked against full text. 10 papers adjacent/off-topic (5, 17, 18, 53, 65, 84, 91, 132, 186, 251) remain in the counts.

## 1. Axis tables (n = 81 papers; primary option per paper; `other` = key not in the design-space list; `na` = not stated or not applicable)

### Axis `tgt`: Regression target

| option | n | share |
|---|---|---|
| Margin g(x) (`margin`) | 55 | 68% |
| Conditional β(x_s) (`beta`) | 2 | 2% |
| Other transforms / sign (`othr`) | 18 | 22% |
| na | 6 | 7% |

### Axis `surr`: Surrogate

| option | n | share |
|---|---|---|
| GP / Kriging (`gp`) | 52 | 64% |
| XGBoost (`xgb`) | 1 | 1% |
| PC-Kriging (`pck`) | 1 | 1% |
| PCE (`pce`) | 3 | 4% |
| SVM classifier (`svmc`) | 1 | 1% |
| Deep NN / PINN (`nn`) | 9 | 11% |
| None — direct sampling (`none`) | 6 | 7% |
| Surrogate-agnostic (`agn`) | 5 | 6% |
| other (total) | 3 | 4% |
| &nbsp;&nbsp;other:weibull-mixture-MLE | 1 | |
| &nbsp;&nbsp;other:gaussian-random-field-MLE | 1 | |
| &nbsp;&nbsp;other:rbf | 1 | |

### Axis `read`: Readout (estimator)

| option | n | share |
|---|---|---|
| Hard indicator (`hard`) | 32 | 40% |
| Rao–Blackwellised (`rb`) | 3 | 4% |
| IS-weighted (`isw`) | 10 | 12% |
| Bayesian posterior (`bayes`) | 17 | 21% |
| other (total) | 5 | 6% |
| &nbsp;&nbsp;other:upcrossing-rate | 1 | |
| &nbsp;&nbsp;other:simplex-cubature | 1 | |
| &nbsp;&nbsp;other:tail-fit-CDF | 1 | |
| &nbsp;&nbsp;other:Vorobev-random-set | 1 | |
| &nbsp;&nbsp;other:hypersphere-line-search | 1 | |
| na | 14 | 17% |

### Axis `uq`: UQ source

| option | n | share |
|---|---|---|
| GP posterior σ (`gpsig`) | 48 | 59% |
| CQR additive (`cqra`) | 1 | 1% |
| Bootstrap / ens. (`boot`) | 6 | 7% |
| BQ posterior (`bq`) | 6 | 7% |
| n/a — no surrogate (`na`) | 16 | 20% |
| other (total) | 4 | 5% |
| &nbsp;&nbsp;other:sngp-distance-aware | 1 | |
| &nbsp;&nbsp;other:variance-free | 1 | |
| &nbsp;&nbsp;other:softmax-prob | 1 | |
| &nbsp;&nbsp;other:platt-prob | 1 | |
| na | 16 | 20% |

### Axis `acq`: Acquisition / learning function

| option | n | share |
|---|---|---|
| Sign-ambiguity (U) (`amb`) | 10 | 12% |
| EFF (`eff`) | 1 | 1% |
| Contour improvement (`cont`) | 1 | 1% |
| Max-min boundary distance (`maxmin`) | 1 | 1% |
| H / LIF / SUR (`hls`) | 4 | 5% |
| Pf-variance reduction (`bqv`) | 12 | 15% |
| n/a — no learning loop (`naq`) | 3 | 4% |
| Bandit / RL portfolio (`rl`) | 1 | 1% |
| other (total) | 47 | 58% |
| &nbsp;&nbsp;other:farthest-point-in-critical-set | 2 | |
| &nbsp;&nbsp;other:ActiveCSV-node-reallocation | 1 | |
| &nbsp;&nbsp;other:risk-aware-error-decision | 1 | |
| &nbsp;&nbsp;other:dynamic-collocation-point-selection | 1 | |
| &nbsp;&nbsp;other:sngp-uncertainty | 1 | |
| &nbsp;&nbsp;other:EI-global-local-switch | 1 | |
| &nbsp;&nbsp;other:mdp-selected-LF | 1 | |
| &nbsp;&nbsp;other:max-model-improvement-unit-selection | 1 | |
| &nbsp;&nbsp;other:MCTS-lookahead | 1 | |
| &nbsp;&nbsp;other:MCLF-uncertainty+diversity+geometry | 1 | |
| &nbsp;&nbsp;other:BCE-uncertainty | 1 | |
| &nbsp;&nbsp;other:MOO-Pareto-explore-exploit | 1 | |
| &nbsp;&nbsp;other:credibility-density | 1 | |
| &nbsp;&nbsp;other:global-misclass-prob | 1 | |
| &nbsp;&nbsp;other:expected-info-gain-composite | 1 | |
| &nbsp;&nbsp;other:BO-acquisition | 1 | |
| &nbsp;&nbsp;other:EI-extremum-ranking | 1 | |
| &nbsp;&nbsp;other:gp-sampling-strategy | 1 | |
| &nbsp;&nbsp;other:measurement-count-refinement | 1 | |
| &nbsp;&nbsp;other:collective-learning-fn-over-EFF | 1 | |
| &nbsp;&nbsp;other:U-weighted-by-ensemble-error | 1 | |
| &nbsp;&nbsp;other:bootstrap-misclass-prob | 1 | |
| &nbsp;&nbsp;other:expected-misclassification-prob | 1 | |
| &nbsp;&nbsp;other:conditional-pf-weighted | 1 | |
| &nbsp;&nbsp;other:U-posterior-with-reduction-factor | 1 | |
| &nbsp;&nbsp;other:AEI+EVCUF | 1 | |
| &nbsp;&nbsp;other:info-gain-simplex-classification | 1 | |
| &nbsp;&nbsp;other:expected-info-gain | 1 | |
| &nbsp;&nbsp;other:modified-UP-learning-fn | 1 | |
| &nbsp;&nbsp;other:SEUR-expected-uncertainty-reduction | 1 | |
| &nbsp;&nbsp;other:FPUP-IS-weighted-Pf-uncertainty | 1 | |
| &nbsp;&nbsp;other:expanding-ring-adaptive-sampling | 1 | |
| &nbsp;&nbsp;other:SOM-region-adaptive | 1 | |
| &nbsp;&nbsp;other:bootstrap-misclass-x-joint-pdf | 1 | |
| &nbsp;&nbsp;other:GELF-global-error-lookahead | 1 | |
| &nbsp;&nbsp;other:explore-exploit-extreme-region | 1 | |
| &nbsp;&nbsp;other:U_N-noise-aware-U | 1 | |
| &nbsp;&nbsp;other:1D-Kriging-root-finding | 1 | |
| &nbsp;&nbsp;other:ubmadr-lookahead-multipoint | 1 | |
| &nbsp;&nbsp;other:RBF-merit-distance-weighted | 1 | |
| &nbsp;&nbsp;other:PSO-weighted-learning | 1 | |
| &nbsp;&nbsp;other:ls-intersection-point-error | 1 | |
| &nbsp;&nbsp;other:IS-proposal-adaptation | 1 | |
| &nbsp;&nbsp;other:var-weighted-conditional-Pf | 1 | |
| &nbsp;&nbsp;other:SFCA-scale-free-coverage | 1 | |
| &nbsp;&nbsp;other:classification-uncertainty | 1 | |
| na | 1 | 1% |

### Axis `batch`: Batch & candidate pool

| option | n | share |
|---|---|---|
| Single pt., full pop. (`single`) | 56 | 69% |
| Fixed-size batch (greedy or joint) (`b8`) | 7 | 9% |
| Truncated candidate region (`trunc`) | 1 | 1% |
| K-means batch (`kmeans`) | 3 | 4% |
| Adaptive batch size (`adapt`) | 6 | 7% |
| other (total) | 2 | 2% |
| &nbsp;&nbsp;other:b2 | 1 | |
| &nbsp;&nbsp;other:b40-refinement | 1 | |
| na | 6 | 7% |

### Axis `pop`: Population & sampling

| option | n | share |
|---|---|---|
| Crude Monte Carlo (`mc`) | 34 | 42% |
| Subset simulation (`ss`) | 4 | 5% |
| Classical IS (`isc`) | 3 | 4% |
| Radial / annulus IS (`rad`) | 4 | 5% |
| meta-IS (`meta`) | 1 | 1% |
| CE-IS (`ce`) | 1 | 1% |
| Sequential IS (`sis`) | 5 | 6% |
| Line / directional (`line`) | 3 | 4% |
| Stratified / LHS (`strat`) | 9 | 11% |
| other (total) | 9 | 11% |
| &nbsp;&nbsp;other:PDEM-rep-points | 1 | |
| &nbsp;&nbsp;other:focal-element-enum | 1 | |
| &nbsp;&nbsp;other:global+local-MC | 1 | |
| &nbsp;&nbsp;other:adaptive-GMM-IS | 1 | |
| &nbsp;&nbsp;other:particle-filter-IS | 1 | |
| &nbsp;&nbsp;other:triangulation-cubature | 1 | |
| &nbsp;&nbsp;other:SOM-region-sampling | 1 | |
| &nbsp;&nbsp;other:hyper-shell-sampling | 1 | |
| &nbsp;&nbsp;other:hollow-hypersphere | 1 | |
| na | 8 | 10% |

### Axis `stop`: Stopping criterion

| option | n | share |
|---|---|---|
| Fixed FE budget (`budget`) | 23 | 28% |
| Learning-fn. threshold (`lft`) | 15 | 19% |
| Pf CoV / stability (`pfcov`) | 29 | 36% |
| Error-rate bound (`err`) | 8 | 10% |
| other (total) | 5 | 6% |
| &nbsp;&nbsp;other:N-ESS | 1 | |
| &nbsp;&nbsp;other:parameter-stabilisation | 1 | |
| &nbsp;&nbsp;other:scenario-set-convergence | 1 | |
| &nbsp;&nbsp;other:hypersphere-level-count | 1 | |
| &nbsp;&nbsp;other:levels-until-failure-threshold | 1 | |
| na | 1 | 1% |

### Axis `fid`: Fidelity

| option | n | share |
|---|---|---|
| Single fidelity (`sf`) | 76 | 94% |
| Multi-fidelity co-Kriging (`cok`) | 3 | 4% |
| Adaptive fidelity selection (`adapf`) | 1 | 1% |
| na | 1 | 1% |

### Axis `dim`: Dimension handling

| option | n | share |
|---|---|---|
| Full input space (`full`) | 73 | 90% |
| Active subspace (`assub`) | 3 | 4% |
| PLS / sliced inverse (`pls`) | 1 | 1% |
| Learned latent space (`ae`) | 1 | 1% |
| na | 3 | 4% |

### Axis `ls`: Limit-state composition

| option | n | share |
|---|---|---|
| Composed g = min(·) (`composed`) | 65 | 80% |
| Per-component surrogates (`percomp`) | 4 | 5% |
| Multiple disjoint regions (`disjoint`) | 5 | 6% |
| Adaptive mode identification (`adaptmode`) | 3 | 4% |
| na | 4 | 5% |

## 2. Headline shares

- Kriging/GP family (`gp`+`pck`): 53/81 (65%); `gp` alone 52 (64%); neural (`nn`) 9; all others 19.
- Learning function: sign-ambiguity U/AK family 10 (12%); EFF 1; H/LIF/SUR 4; Pf-variance reduction (Bayesian) 12; contour 1; other:* 47; na 1.
- Population: crude MC 34 (42%); any non-MC sampler (SS, IS, line, strat, other) 39 (48%); na 8.
- Stopping: learning-function threshold 15 (19%); fixed budget 23; Pf CoV/stability 29; error-rate bound 8; conformal 0; other/na 6.
- Batch / multi-point (batch not single/na): 19 (23%); explicitly single-point 56.
- System reliability: task=system 6; limit-state composition other than composed/single (percomp, disjoint, adaptmode) 12; union 13 (16%).
- Dimension reduction (dim not full/na): 5 (6%).
- Multi-fidelity (fid not sf/na): 4 (5%).
- Regression target other than margin (tgt not margin/na): 20.
- Task: pf 32, updating 9, time-dependent 7, system 6, rbdo 4, stochastic-simulator 2, other:surrogate-FE-fullfield 1, other:excursion-set-confidence-region 1, other:deterministic-optimization-FSI 1, other:PINN-porous-flow-forward-inverse 1, other:parametric-variation 1, other:test-planning 1, other:network-reliability 1, other:cyber-threat-detection 1, other:response-distribution-CDF 1, other:regression-risk-prediction 1, other:design-optimization 1, other:NDT-spatial-variability 1, other:instrument-calibration-design 1, other:robust-design-optimization 1, other:fatigue-damage-expectation 1, other:long-term-extreme-response 1, other:response-distribution 1, other:hidden-constraint-feasible-set 1, other:fatigue-iso-pf-contour 1, other:decision-value-optimization 1, other:design-reliability-application 1.
- At least one finite-element example: 56/81 (69%).
- **Reports results over repeated runs/seeds: 61/81 (75%)**; no/unclear 20.
  - among task=pf papers (n=32): 30 (94%).
- Confidence of coding: med 65, low 13, high 3.
- Numerical examples per paper: median 4, mean 4.0, n=81.

### Learning-function genres (regex grouping of `acq`, order of rules matters; see snippet)

| genre | n | share |
|---|---|---|
| Information gain / lookahead / uncertainty reduction (SUR, Pf-variance) | 26 | 32% |
| Sign ambiguity U family (AK-type) | 13 | 16% |
| Other / task-specific | 10 | 12% |
| Improvement-type (EFF, EI, BO) | 9 | 11% |
| Geometry / distance / coverage | 8 | 10% |
| Misclassification-probability / error-based | 8 | 10% |
| No loop / not applicable | 4 | 5% |
| Pf-weighted / IS- or conditional-Pf aware | 3 | 4% |

- Papers whose coder note says the work is not structural-reliability AL (off-topic or adjacent, kept because classified AL by the abstract screen): 10 (5, 17, 18, 53, 65, 84, 91, 132, 186, 251).

## 3. Distributions

### max_dim (largest number of random variables in any example; stated for 50/81)

| bin | n | share of stated |
|---|---|---|
| <= 5 | 7 | 14% |
| 6-10 | 15 | 30% |
| 11-20 | 12 | 24% |
| 21-50 | 4 | 8% |
| > 50 | 12 | 24% |

median 11, max 1533. Not stated: 31.

### min_pf (smallest target Pf in any example; stated for 33/81)

| bin | n | share of stated |
|---|---|---|
| >= 1e-2 | 1 | 3% |
| [1e-3,1e-2) | 4 | 12% |
| [1e-4,1e-3) | 6 | 18% |
| [1e-5,1e-4) | 6 | 18% |
| [1e-6,1e-5) | 5 | 15% |
| < 1e-6 | 11 | 33% |

median 1.0e-05, smallest 6.0e-16. Not stated: 48.

## 4. Papers per year per journal

| year | CMAME | IEEE_TR | RESS | StrucSafety | total |
|---|---|---|---|---|---|
| 2021 | 0 | 0 | 0 | 5 | 5 |
| 2022 | 0 | 0 | 0 | 8 | 8 |
| 2023 | 0 | 0 | 0 | 11 | 11 |
| 2024 | 0 | 0 | 0 | 11 | 11 |
| 2025 | 0 | 0 | 0 | 9 | 9 |
| 2026 | 2 | 7 | 0 | 10 | 19 |
| 2027 | 3 | 0 | 14 | 1 | 18 |
| all | 5 | 7 | 14 | 55 | 81 |

## 5. Papers with an FE example and with repeated runs, by journal (counts)

| journal | n | FE example | repeats |
|---|---|---|---|
| CMAME | 5 | 5 | 2 |
| IEEE_TR | 7 | 2 | 5 |
| RESS | 14 | 10 | 11 |
| StrucSafety | 55 | 39 | 43 |
