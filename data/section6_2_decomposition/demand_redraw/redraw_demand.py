"""Re-draw the 4 demand variables of the Ex14-PC-D3 population and re-run the margin+indicator U-function arm (A8).

Question (FACTS F8, review R6): the A8-vs-beta gap at fixed U-function was measured on ONE population draw, whose crude failure
count lies +3.0% above Pf_RB. How large is the margin arm's error against Pf_RB over fresh demand draws? (The beta+RB arm does not
depend on the demand draw: it integrates the demand exactly, so its error stays 2.50%.)

usage: python3 redraw_demand.py <redraw> <seed>     redraw 0 = original population (validation: seed 0 must give +7.41 %)
Writes exp/out/redraw<r>_seed<s>.json. Zero FE calls: capacity curves are cached; margins are closed-form in the demand.
"""
import os, sys, json, time
for k, v in dict(OMP_NUM_THREADS="1", DEVICE="cpu", MOD="fe_pcjoint_d2", TAG="pcjointD3", NPOP="199857",
                 PFTAG="3e-03", SIG="0", ARMS="A8").items():
    os.environ.setdefault(k, v)
ROOT = os.environ.get("ALSRA_ROOT", ".")   # folder holding the original experiment scripts
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np
import run_pcjoint_a12 as RP                      # loads the cached population exactly as the paper's runs did
from a8_eff import run_a8u
from a9_tabicl import run_a9
ARM = os.environ.get('RARM', 'A8')

r, sd = int(sys.argv[1]), int(sys.argv[2])
U2 = RP.U.copy()
if r > 0:                                         # columns NS..NS+3 = theta_R, D, L, theta_allow (standard normal)
    U2[:, RP.NS:RP.NS + 4] = np.random.default_rng(10_000 + r).standard_normal((RP.N, 4))
g = np.full(RP.N, -1.0); ok = ~RP.bad
g[ok], _ = RP.Mod.margins(U2[ok], RP.Pd[ok], RP.Pdmax[ok], RP.ufr[ok], RP.ulast[ok], RP.lam)
if r == 0:
    assert np.allclose(g, RP.g_clean), "redraw 0 must reproduce the cached margins"
y = g / RP.S0
count = float((g <= 0).mean())
RP.v1.N_POP, RP.v1.BUDGET = RP.N, RP.BUD; nfe = RP.v1.N0 + RP.BUD
t0 = time.time()
extra = {}
if ARM == 'A11':                                  # CQR-UCB design (A9) read out twice: margin indicator and T(g) refit (paired)
    idx = run_a9(U2, y, sd, 0); pf9 = RP.ind_readout(U2, y, idx); pf = RP.ind_readout(U2, y, idx, RP.T_COMP)
    extra = dict(pf_a9=pf9, a9_err_vs_count_pct=(pf9 - count) / count * 100, a9_err_vs_rb_pct=(pf9 - RP.PF_RB) / RP.PF_RB * 100)
else:
    idx = run_a8u(U2, y, sd, 0, nfe); pf = RP.ind_readout(U2, y, idx)
res = dict(arm=ARM, redraw=r, seed=sd, pf=pf, count=count, pf_rb=RP.PF_RB, **extra,
           err_vs_rb_pct=(pf - RP.PF_RB) / RP.PF_RB * 100, err_vs_count_pct=(pf - count) / count * 100,
           count_vs_rb_pct=(count - RP.PF_RB) / RP.PF_RB * 100, n_fe=len(idx), seconds=time.time() - t0)
os.makedirs(f"{ROOT}/survey_ALSRA/exp/out", exist_ok=True)
json.dump(res, open(f"{ROOT}/survey_ALSRA/exp/out/redraw{r}_seed{sd}{'' if ARM == 'A8' else '_' + ARM}.json", "w"), indent=1)
print(json.dumps(res), flush=True)
