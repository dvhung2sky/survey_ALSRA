"""578: ONE specimen (PRCJ1, post-cast interior joint), ONE analysis = MONOTONIC push of the column top to 100 mm (7.1 % drift) under constant axial load.
capacity = peak lateral force F (kN).
Failure mode: joint-panel shear strength (kappa*sqrt(fc), pinched Hysteretic) with bar-slip; peak lateral force before softening."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _load import load
import openseespy.opensees as ops, numpy as np
CAP_UNITS = 'kN'
M = load('578_precast_postcast_joints')
SP = 'PRCJ1'
RV = {
    'fc_cast': ('LN', 31.22, 0.12, 'Table 7 mean; CoV JCSS-type concrete'),
    'fc_pre': ('LN', 36.56, 0.10, 'Table 7 mean; CoV assumed (precast plant)'),
    'fy18': ('LN', 415.7, 0.06, 'READ D18; CoV JCSS-type rebar'),
    'fy10': ('LN', 407.8, 0.06, 'READ D10 waist bars; CoV JCSS-type rebar'),
    'n_ratio': ('N', 0.234, 0.10, 'READ axial ratio; gravity load CoV assumed'),
    'kappa': ('LN', 0.58, 0.15, 'CAL panel strength = model-error RV; CoV assumed'),
    'lam_s': ('LN', 4.0, 0.20, 'CAL slip multiplier = model-error RV; CoV assumed'),
}
def run(params=None, umax=100.0, dstep=0.5):
    q = {k: v[1] for k, v in RV.items()}; q.update(params or {})
    M.SPEC[SP].update(fc_cast=q['fc_cast'], fc_pre=q['fc_pre']); M.FY18 = q['fy18']; M.FY10 = q['fy10']; M.N_RATIO = q['n_ratio']
    cfg = dict(M.CFG); cfg['kappa'] = q['kappa']; cfg['lam_s'] = q['lam_s']
    T = M.build(SP, cfg)
    N = M.N_RATIO * q['fc_pre'] * M.BC * M.HC
    ops.timeSeries('Linear', 1); ops.pattern('Plain', 1, 1); ops.load(T['top'], 0., -N, 0.)
    ops.system('UmfPack'); ops.numberer('RCM'); ops.constraints('Transformation')
    ops.test(*M.CFG_TEST); ops.algorithm('Newton')
    ops.integrator('LoadControl', 0.1); ops.analysis('Static'); ok = ops.analyze(10) == 0; ops.loadConst('-time', 0.)
    ops.timeSeries('Linear', 2); ops.pattern('Plain', 2, 2); ops.load(T['top'], 1., 0., 0.)
    F = [0.0]; u = 0.0
    while ok and u < umax - 1e-9:
        if not M.step(T['top'], dstep): ok = False; break
        u += dstep; F.append(ops.getLoadFactor(2) / 1e3)
    return dict(capacity=float(max(F)), u_end=u, converged=bool(ok))
if __name__ == '__main__':
    print(run())
