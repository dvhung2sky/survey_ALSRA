"""556: ONE specimen (T1 virgin frame + REDC), ONE analysis = MONOTONIC pushover to 2.0 % drift (45 mm), no cycles.
capacity = max lateral force (kN) over the push (no fracture in the model: force-at-drift-limit, see PROFILE.md).
Failure mode: none modelled (no fracture); capacity is the lateral force at 2.0 % drift, i.e. a hardening-controlled strength (ring yielding / frame hinging)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _load import load
import openseespy.opensees as ops
CAP_UNITS = 'kN'
M = load('556_xbraced_steel_frame')
RV = {  # name: (dist, mean, cov, source)
    'fy_cf': ('LN', 356.2, 0.07, 'Table 1 mean; CoV JCSS-type structural steel fy ~0.07'),
    'fy_cw': ('LN', 425.6, 0.07, 'Table 1 mean; CoV JCSS-type structural steel fy ~0.07'), 'fy_bf': ('LN', 425.6, 0.07, 'Table 1 mean; CoV JCSS-type structural steel fy ~0.07'), 'fy_bw': ('LN', 376.6, 0.07, 'Table 1 mean; CoV JCSS-type structural steel fy ~0.07'),
    'fy_ly': ('LN', 212.6, 0.08, 'LY225 mean Table 1; CoV assumed'),
    'E_ly': ('LN', 1.94e5, 0.03, 'assumed'), 'b_ly': ('N', 0.004, 0.30, 'assumed (CAL-ish hardening ratio)'),
    'W_ring': ('N', 105.0, 0.03, 'assumed (CAL geometry, not printed)'), 't_ring': ('N', 16.0, 0.02, 'assumed plate tolerance'),
    'N_col': ('N', 360e3, 0.10, 'assumed gravity load, normal'),
}
def run(params=None, drift=2.0, dstep=0.5):
    q = {k: v[1] for k, v in RV.items()}; q.update(params or {})
    p = dict(M.P); p.update(q)
    nid = M.build(p); ctrl = nid['LJ']
    ops.timeSeries('Linear', 1); ops.pattern('Plain', 1, 1)
    for s in 'LR': ops.load(nid[f'{s}T'], 0.0, -p['N_col'], 0.0)
    ops.constraints('Transformation'); ops.numberer('RCM'); ops.system('UmfPack')
    ops.test('NormUnbalance', 1e-3, 50); ops.algorithm('Newton')
    ops.integrator('LoadControl', 0.1); ops.analysis('Static'); ops.analyze(10)
    ops.loadConst('-time', 0.0); ops.wipeAnalysis()
    ops.timeSeries('Linear', 2); ops.pattern('Plain', 2, 2); ops.load(ctrl, 1.0, 0.0, 0.0)
    ops.constraints('Transformation'); ops.numberer('RCM'); ops.system('UmfPack')
    ops.test('NormUnbalance', 1e-2, 50); ops.algorithm('Newton'); ops.analysis('Static')
    umax = drift / 100 * p['Hdrift']; F = [0.0]; U = [0.0]; ok = True
    for i in range(1, int(round(umax / dstep)) + 1):
        if M.step(ctrl, i * dstep) != 0: ok = False; break
        U.append(i * dstep); F.append(ops.getLoadFactor(2) / 1e3)
    import numpy as np
    F = np.array(F); U = np.array(U)
    return dict(capacity=float(F.max()), F_1pct=float(np.interp(0.01 * p['Hdrift'], U, F)), u_end=float(U[-1]), converged=ok)
if __name__ == '__main__':
    print(run())
