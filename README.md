# survey_ALSRA

Code and data accompanying the manuscript *Active Learning for Structural Reliability Analysis: a Design-Space Review of Methods, Evidence and Benchmarking* (submitted to *Reliability Engineering & System Safety*).

The repository holds three things:

1. **AL Path Bench** (`alpath/`): an engine and a local web interface that compose an active-learning reliability method as one path through the 11-axis design space of the review (one option per axis), run it on the five benchmarks and compare it with the 12 methods of the benchmark section.
2. **The benchmark harness** (`bench/`): the five test-calibrated finite-element benchmarks (Monte Carlo populations with their finite-element capacities, wrappers and model cards), the harness that runs the 12 methods, and the per-seed results reported in the manuscript.
3. **A memory guard** (`guard/`) that runs each job under a resident-memory cap.

## Quick start

```bash
pip install -r requirements.txt
python3 alpath/server.py            # then open http://127.0.0.1:8765
```

Choose a database and one option on each axis (or start from a method in the list), set the budget, the target failure probability and the seeds, and press **Run path**. Each seed runs `alpath/engine.py` under `guard/run_guarded.py`, one job at a time; results are written to `alpath/runs/`.

Command line, one seed:

```bash
python3 alpath/engine.py --db 578 --path '{"acq":"eff","stop":"lft"}' --seed 0 --budget 300 --out result.json
python3 bench/al_run.py 578 egra 0 --pop bench/578_precast_postcast_joints/pop_merged.npz     # a method of the harness
```

`python3 alpath/test_engine.py 578 25 tgt,read,batch` checks the closed-form learning functions against quadrature and runs every option of the listed axes once.

## What runs where

| Needs | Options |
| --- | --- |
| Only this repository | Every path on the crude Monte Carlo population of a database: a label at a population sample is read from the precomputed capacities (one model call, no analysis time). |
| A TabICL-v2 checkpoint | The TabICL surrogate and the paths built on it. Set `TABICL_CKPT=/path/to/tabicl-regressor-v2-20260212.ckpt`, or place the file in the repository root. |
| The calibrated OpenSees models | Every option that evaluates a point outside the population: subset simulation, the importance-sampling family, line sampling, Latin hypercube sampling, the surrogate-agnostic (MFIS) estimator and the low-fidelity level. The calibrated `model.py` files are **not distributed**; they are available from the corresponding author. Set `CALIB_DIR=/path/to/calib` (one folder per benchmark, e.g. `calib/578_precast_postcast_joints/model.py`). Without them, the interface disables these options and says why. |

## Data of the review

- `data/corpus/`: the coded corpus (`corpus_coding.csv`, one row per reviewed active-learning paper with its placement on the eleven axes), the placement of the 61 named methods on the design space (`designspace_data.json`), corpus statistics, the blind second coding of 20 papers with its agreement statistics (`recode_*`), the coding instructions given to the language-model coders (`coding_prompt_GUIDE.md`) and the publication-trend scan (`trend_*`).
- `data/section6_2_decomposition/`: the per-seed results behind Table 10, the error-versus-target-probability series and the demand-redraw runs of Section 6.2 (see its README).
- `bench/`: the five benchmark populations, model cards, method harness and per-seed results of Section 6 and Section 6.1.

## Repository layout

```
alpath/
  engine.py         one function or class per option of the 11 axes; check() flags invalid combinations
  server.py         local web server (standard library only); one job at a time, each seed under the memory guard
  index.html        the interface
  test_engine.py    formula checks and a smoke run of every option
bench/
  al_run.py         harness of the 12 methods (HARNESS.md documents each)
  <id>_<name>/      pop_merged.npz (population: X, cap, ok, t_call), call.py (wrapper), model_card.md, rv.json (random variables)
  out/              per-seed results: out/<id>/<method>_s<seed>[_b<budget>].json (no suffix = 300 calls)
  RESULTS_S7.md     the result tables of the benchmark section; tab_s7.py rebuilds them from out/
figures/fig_bench_data.py   data of the benchmark figure, from bench/out
guard/              run_guarded.py (one job), run_queue.py (a queue with a runtime projection)
```

The five benchmarks are 556 (X-braced steel frame with a replaceable ring damper), 635 (corrugated-web steel frame under column removal), 1311 (corroded post-tensioned concrete girder), 523 (AAC-infilled light-steel frame) and 578 (precast beam–column joint with a post-cast zone). Every benchmark uses the limit state g = R(x_s) − S, with S lognormal (CoV 0.30) and its median set so that the Rao–Blackwellised reference P_f,RB equals 10⁻³ (`bench/al_run.py`, class `World`).

## Known limitations

- Five options fill open cells of the design space (other target transforms, the bandit portfolio, adaptive batch size, adaptive fidelity and the autoencoder latent space). They are first-cut heuristics, labelled experimental in the interface, not published methods.
- Several options were implemented from their published descriptions rather than the authors' code and may differ in detail: REIF uses a re-derived folded-normal form, and PC-Kriging is fitted in two stages (sparse PCE trend, then a Gaussian process on the residuals).
- The low-fidelity level is the same finite-element model run with a four times coarser load or displacement step, charged a nominal quarter of a full call. On benchmark 578 it agrees with the full model to within 0.5%, so it is a weak test of the fidelity axis there; benchmark 635 has no such level.
- Transport-map importance sampling, per-component surrogates and adaptive mode identification cannot run on these benchmarks and are disabled.
- The TabICL readout is recomputed every fifth refit on large populations, so its convergence curve is coarser than that of a Gaussian-process path.

## Licence

No licence has been chosen yet; until one is added, all rights are reserved by the authors.

## Contact

Viet-Hung Dang (corresponding author), Hanoi University of Civil Engineering, hungdv@huce.edu.vn
