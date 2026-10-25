"""1311: ONE girder (default MSG5: 3-strand girder, pitting corrosion at midspan, calibration Pu ratio 0.98; SSG4 via env GIRDER=SSG4 is a weaker fit, Pu 0.86), ONE analysis = monotonic bending (MSG: three-point, one midspan load; SSG: four-point) (displacement control, 0.1 mm) past the peak (ruptured wires drop out, the push goes on).
capacity = ULTIMATE peak load (kN) over the push.  (v1 used the paper's "peak before the first failure event": with pit_scale >~ 2 a pitted
wire ruptures at the first step, ~12 kN, while the girder still carries ~210 kN -> a spurious 1.15% low mode and s_med = 8.6 kN.)
At the mean parameters both definitions give 284.5 kN, so the calibration is unchanged.
chdir to results/ (model.py writes results/ops.log relative to cwd).
Failure mode: flexural failure of the corroded girder (wire rupture and/or top-concrete crushing), capacity = ultimate peak load."""
import os, sys, copy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from _load import load
import numpy as np
CAP_UNITS = 'kN'
M = load('1311_corroded_psc_girders')
os.makedirs(f'{HERE}/results', exist_ok=True); os.chdir(f'{HERE}/results')
NAME = os.environ.get('GIRDER', 'MSG5'); FAM = NAME[:3]
S0 = copy.deepcopy(M.SPEC[FAM]); W0 = copy.deepcopy(M.WIRES[NAME])
RV = {
    'fcm': ('LN', 44.07, 0.12, 'paper fcm; CoV JCSS-type concrete'),
    'Pe': ('N', S0['Pe'], 0.05, 'effective prestress force; CoV assumed (loss uncertainty)'),
    'fpu': ('LN', S0['fpu'], 0.025, 'strand fpu paper; CoV ~0.025 literature (prestressing strand), fpy scaled with it'),
    'Ep': ('LN', S0['Ep'], 0.02, 'assumed'),
    'b': ('N', S0['b'], 0.02, 'assumed geometric tolerance'),
    'h': ('N', S0['h'], 0.015, 'assumed geometric tolerance'),
    'c_strand': ('N', 1.0, 0.05, 'multiplier on strand eccentricity profile (sd ~3-4 mm); assumed'),
    'pit_scale': ('LN', 1.0, 0.30, 'scale of the MEASURED wire-area loss (Table 2); CoV assumed -- corrosion RV'),
    'LAM': ('N', 0.82, 0.10, 'true/idealised pit-loss factor (Table 9 calibrated); CoV assumed'),
    'EPU': ('N', 0.075, 0.10, 'strand rupture strain (Table 8); CoV assumed'),
}
def run(params=None, dstep=0.1):
    q = {k: v[1] for k, v in RV.items()}; q.update(params or {})
    fcm = q['fcm']; M.FCM = fcm; M.FCK = fcm - 4.07; M.FT = 0.3016 * M.FCK ** (2 / 3)
    M.EC0 = 10000 * fcm ** (1 / 3) * (0.8 + 0.2 * fcm / 88); M.EC1 = 0.7 * fcm ** 0.31 / 1000
    M.LAM = q['LAM']; M.EPU = q['EPU']
    S = copy.deepcopy(S0); S['Pe'] = q['Pe']; S['fpu'] = q['fpu']; S['fpy'] = S0['fpy'] * q['fpu'] / S0['fpu']
    S['Ep'] = q['Ep']; S['b'] = q['b']; S['h'] = q['h']; f0 = S0['strand_y']; S['strand_y'] = lambda x, c=q['c_strand']: f0(x) * c
    S['bot'] = (S0['bot'][0], S0['bot'][1], q['h'] - (S0['h'] - S0['bot'][2])); S['top'] = S0['top']
    M.SPEC[FAM] = S
    M.WIRES[NAME] = [[float(np.clip(a0 - q['pit_scale'] * (a0 - a), 0.25 * a0, a0)) for a0, a in zip(M.A0, row)] for row in W0]
    D, P, ev = M.run(NAME, dstep=dstep)
    evs = [e for e in (ev['rupture'], ev['crush']) if e]
    uf = min(e[0] for e in evs) if evs else None
    Pu = float(P.max()); Pu_first = float(P[D <= uf].max()) if uf is not None else Pu
    mode = 'none' if not evs else ('rupture' if ev['rupture'] and ev['rupture'][0] == uf else 'crush')
    upk = float(D[np.argmax(P)])                    # peak reached: full push, descending branch, or a failure event at/after the peak
    conv = bool(D[-1] >= (29.9 if FAM == 'SSG' else 39.9) or P[-1] < 0.8 * P.max() or any(e[0] >= upk - 0.15 for e in evs))   # 0.15 mm = 1.5 load steps
    return dict(capacity=Pu, Pu_first_event=Pu_first, u_fail=float(uf) if uf is not None else float('nan'), mode_rupture=float(mode == 'rupture'), converged=conv, u_end=float(D[-1]))
if __name__ == '__main__':
    print(run())
