# Coding guide: 11 axes. Use ONLY the option keys below (or other:<short>, or na).


## tgt  (Regression target: What the surrogate is fitted on)
- `margin` = Margin g(x)
- `tg` = Compressed T(g)
- `beta` = Conditional β(x_s)
- `othr` = Other transforms

## surr  (Surrogate: Model class)
- `gp` = GP / Kriging
- `xgb` = XGBoost
- `tabicl` = TabICL v2
- `pck` = PC-Kriging
- `pce` = PCE
- `svmc` = SVM classifier
- `svr` = SVR regressor
- `nn` = Deep NN / PINN
- `rsm` = Quadratic response surface
- `none` = None — direct sampling
- `agn` = Surrogate-agnostic

## read  (Readout (estimator): After the loop ends, how the fitted surrogate is turned into one number P̂f. Independent of which points were chosen — same design, different readout, different answer.)
- `hard` = Hard indicator
- `med` = Median quantile
- `bag` = Bag + smooth
- `rb` = Rao–Blackwellised
- `isw` = IS-weighted
- `bayes` = Bayesian posterior

## uq  (UQ source: Where the width w comes from)
- `gpsig` = GP posterior σ
- `kfold` = K-fold conformal
- `native` = TabICL native
- `cqra` = CQR additive
- `cqrm` = CQR multiplicative
- `boot` = Bootstrap / ens.
- `bq` = BQ posterior
- `na` = n/a — no surrogate

## acq  (Acquisition / learning function: Which point to evaluate next)
- `amb` = Sign-ambiguity (U)
- `ucb` = UCB  w × amb
- `eff` = EFF
- `cont` = Contour improvement
- `maxmin` = Max-min boundary distance
- `effc` = EFF + CQR
- `hls` = H / LIF / SUR
- `pbw` = Probability-band width
- `rand` = Random
- `erf` = REIF
- `bqv` = Pf-variance reduction
- `naq` = n/a — no learning loop
- `rl` = Bandit / RL portfolio

## batch  (Batch & candidate pool: How many points, chosen from where)
- `single` = Single pt., full pop.
- `b8` = BATCH = 8
- `cand` = CAND = 6000 subsample
- `trunc` = Truncated candidate region
- `kmeans` = K-means batch
- `parreg` = Parallel adapt. regions
- `adapt` = Adaptive batch size

## pop  (Population & sampling: How the candidate population is drawn)
- `mc` = Crude Monte Carlo
- `ss` = Subset simulation
- `isc` = Classical IS
- `rad` = Radial / annulus IS
- `meta` = meta-IS
- `ce` = CE-IS
- `sis` = Sequential IS
- `line` = Line / directional
- `strat` = Stratified / LHS
- `nis` = Neural IS
- `ttis` = Transport-map IS

## stop  (Stopping criterion: When the loop ends)
- `budget` = Fixed FE budget
- `lft` = Learning-fn. threshold
- `pfcov` = Pf CoV / stability
- `err` = Error-rate bound
- `conf` = Conformal coverage

## fid  (Fidelity: How many model levels the loop uses)
- `sf` = Single fidelity
- `cok` = Multi-fidelity co-Kriging
- `rom` = Reduced-order / coarse mesh
- `adapf` = Adaptive fidelity selection

## dim  (Dimension handling: What the surrogate sees)
- `full` = Full input space
- `fidr` = Failure-informed DR
- `assub` = Active subspace
- `pls` = PLS / sliced inverse
- `ae` = Learned latent space

## ls  (Limit-state composition: What the surrogate is asked to represent)
- `composed` = Composed g = min(·)
- `percomp` = Per-component surrogates
- `disjoint` = Multiple disjoint regions
- `adaptmode` = Adaptive mode identification

## Conventions
- tgt: `margin` = surrogate fitted to the limit-state value g(x) or its sign/class (standard AK-MCS, classifiers included); `tg` compressed/transformed margin; `beta` conditional reliability index/probability regression; `na` = no surrogate.
- surr: primary surrogate (`gp` for Kriging/GP/PC-Kriging-as-GP-variant? no: PC-Kriging = `pck`). Neural nets, BNN, PINN, deep ensembles = `nn`. Several surrogates compared/combined = `agn` or the dominant one.
- read: how Pf is read out. `hard` = indicator sign of surrogate on MC population/IS samples (default for AK-MCS style); `isw` = importance-sampling weighted estimator with surrogate; `bayes` = Pf as posterior random variable / Bayesian quadrature; `rb` Rao-Blackwellised; others per label. `na` if not Pf estimation.
- uq: `gpsig` = GP/Kriging predictive variance; `boot` = bootstrap/ensemble/dropout; `bq` = Bayesian quadrature posterior of the integral; `na` no surrogate.
- acq: the learning function. U/AK/ambiguity-of-sign = `amb`; EFF = `eff`; H/LIF/SUR-type = `hls`; contour/ level-set improvement = `cont`; Pf-variance reduction (Bayesian AL) = `bqv`; REIF = `erf`; otherwise write other:<short> (e.g. other:misclass-prob, other:lookahead, other:knowledge-gradient, other:mutual-info). `na` if none.
- batch: `single` = one point per iteration; `b8` = generic fixed-size multi-point batch; `kmeans` = clustering batch; `parreg` = parallel disjoint regions; `adapt` = adaptive batch size; `cand`/`trunc` = the paper's distinctive feature is a candidate-pool reduction / truncated region. If unclear -> `single` only if text implies sequential; else `na`.
- pop: sampling for the Pf estimate / candidate pool: `mc`, `ss` subset simulation, `isc` classical IS (FORM/design-point), `rad`, `meta` meta-model IS, `ce` cross-entropy IS, `sis` sequential IS, `line` line sampling, `strat` stratified/LHS/Sobol/QMC, `nis` neural IS, `ttis` transport-map IS; other:<short> (e.g. other:directional, other:hollow-hypersphere, other:Gibbs).
- stop: `budget` fixed number of FE/DoE; `lft` learning-function threshold (e.g. min U>=2, EFF<0.001); `pfcov` Pf-stability / CoV of Pf / confidence interval of Pf; `err` bound on estimated misclassification/error rate; `conf` conformal. other:<short> for others.
- fid: `sf` single fidelity; `cok` multi-fidelity (co-Kriging); `rom`; `adapf`.
- dim: `full` full input space; `fidr`/`assub`/`pls`/`ae` reduction methods; `na` if unclear.
- ls: limit-state composition. `composed` = single/composed g (system handled via min/max of g, or single limit state); `percomp` = per-component surrogates for system; `disjoint` = multiple failure regions handled specifically; `adaptmode` = adaptive failure-mode identification.
- task: one of pf | time-dependent | rbdo | updating | sensitivity | system | stochastic-simulator | other:<short>.
- max_dim: largest number of random variables in any numerical example (integer, or `na`). min_pf: smallest reference/target Pf in the examples (number like 1e-6, or `na`).
- n_examples: count of numerical examples/case studies (integer). has_fe_example: yes/no (at least one finite-element / PDE / structural-FE model). reports_repeats: yes/no (results reported over repeated independent runs / seeds / replications with mean+std or CoV or boxplots). compares_to: baseline methods named, short, semicolon separated (e.g. AK-MCS;EFF;MCS;subset). confidence: high/med/low.
- notes: <=20 words free text.
- Code what the paper DOES (method + examples), not what its title suggests. If it does not say -> na. Papers might not be reliability-AL proper (e.g. updating, RBDO, test planning, excursion sets); still code best-fit and set task accordingly; use na for axes that do not apply.
- Method: read the abstract, then grep the Markdown full text (grep -n -i -E "learning function|stopping|Kriging|subset|importance sampling|batch|parallel|system|dimension|fidelity|coefficient of variation|repeat|runs|independent|finite element|ABAQUS|ANSYS|OpenSees") and read method/examples sections. Do not read entire papers. ~3-4 min per paper.
