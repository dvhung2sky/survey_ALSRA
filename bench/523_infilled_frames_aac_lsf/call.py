"""523: ONE specimen (AAC: RC frame + AAC infill, two compression struts), ONE analysis = MONOTONIC push of the actuator node to 25 mm at the right-beam LPDT (test peak at 7 mm).
capacity = peak lateral force (kN).
Failure mode: AAC diagonal-strut crushing (Concrete01 peak Vs/cos) with column-end hinging; capacity = peak lateral force, then strut softening."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _load import load
import openseespy.opensees as ops, numpy as np
CAP_UNITS = 'kN'
M = load('523_infilled_frames_aac_lsf')
RV = {
    'fc': ('LN', 30.15, 0.12, 'paper mean; CoV JCSS-type concrete (fcc scaled with fc)'),
    'fy': ('LN', 430., 0.06, 'paper mean; CoV JCSS-type rebar'),
    'fu': ('LN', 545., 0.05, 'paper mean; CoV assumed'),
    'N_ax': ('N', 0.1 * 30.15 * 250 * 250, 0.10, '0.1 fc Ag gravity; CoV assumed'),
    'Em': ('LN', 2195., 0.20, 'AAC masonry modulus (derived Table 3); CoV assumed'),
    'Vs': ('LN', 95e3, 0.25, 'CAL strut peak horizontal force (N) = AAC infill strength; CoV assumed'),
    'r': ('N', 0.2, 0.30, 'CAL residual strut ratio; assumed'),
}
def run(params=None, spec='AAC', umax=25.0, du=0.05):
    q = {k: v[1] for k, v in RV.items()}; q.update(params or {})
    fr = dict(M.FR); fr.update(fc=q['fc'], fcc=q['fc'] * 40.0 / 30.15, fy=q['fy'], fu=q['fu'], N=q['N_ax'])
    cal = dict(M.CAL); cal.update(Vs=q['Vs'], r=min(max(q['r'], 0.0), 0.9)); aac = dict(M.AAC); aac.update(Em=q['Em'])
    m = M.Model(spec, cal, fr, aac)
    F = [0.0]; U = [0.0]; u = 0.0; ok = True
    while u < umax:
        if not m.step(du): ok = False; break
        u += du; F.append(m.forces()[0] / 1e3); U.append(ops.nodeDisp(m.joint[M.L], 1))
    F = np.array(F)
    return dict(capacity=float(F.max()), u_at_peak=float(U[int(F.argmax())]), u_end=float(U[-1]), converged=bool(ok))
if __name__ == '__main__':
    print(run())
