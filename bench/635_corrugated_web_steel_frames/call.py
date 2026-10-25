"""635: ONE specimen (CWES, equal spans), ONE analysis = central-column-removal PUSHDOWN, 1 mm adaptive steps.
Failure mode: bottom-flange fracture in a column-face hinge (catenary action). The run ends at the FIRST of
(a) fracture (end='fracture') or (b) chord rotation 0.2 rad (UFC 4-023-03 catenary limit) of the shorter beam,
v_lim = L_span*tan(0.2) = 547 mm for CWES (L = 2700 mm column centre to pin) (end='rotation').
capacity = max load reached up to the end (kN); also d_end, flexural-stage peak (max load d<=60 mm)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _load import load
import numpy as np
CAP_UNITS = 'kN'
M = load('635_corrugated_web_steel_frames')
RV = {
    'fy': ('LN', 333., 0.07, 'Table 1 flange coupon; CoV JCSS-type steel'),
    'fu': ('LN', 483., 0.05, 'Table 1; CoV assumed'),
    'E': ('LN', 200e3, 0.03, 'assumed'),
    'bf': ('N', 125., 0.02, 'assumed rolled/welded tolerance'),
    'tf': ('N', 7.3, 0.03, 'READ tf; thickness tolerance assumed'),
    'h': ('N', 266., 0.015, 'assumed'),
    'tw': ('N', 1.7, 0.04, 'READ web t; tolerance assumed'),
    'd_f': ('LN', 25.68, 0.20, 'CAL fracture stretch of hinge flange (mm); fracture scatter assumed -- dominant RV for catenary/fracture'),
}
def run(params=None, spec='CWES'):
    q = {k: v[1] for k, v in RV.items()}; q.update(params or {})
    p = dict(M.P); p.update(q); p['b_k'] = (q['fu'] - q['fy']) / (0.1665 - 0.01665) / q['E']
    vlim = min(M.SPEC[spec]['L']) * np.tan(0.2); p['dmax'] = vlim
    try: res, frac = M.run(spec, p)
    except AssertionError as e: return dict(capacity=float('nan'), converged=False, end='error', err=str(e))
    d, P = res[:, 0], res[:, 1]
    end = 'fracture' if frac is not None else ('rotation' if d[-1] >= vlim else 'nonconv'); conv = end != 'nonconv'
    return dict(capacity=float(P.max()), d_fracture=float(frac) if frac is not None else float('nan'), P_flex60=float(P[d <= 60].max()),
                axial_max=float(np.abs(res[:, 2:4]).max()), d_end=float(d[-1]), end=end, converged=bool(conv))
if __name__ == '__main__':
    print(run())
