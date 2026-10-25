"""python3 tab_s7.py [pf_suffix] -> RESULTS_S7.md: method x benchmark tables of mean |err| (worst) vs Pf_RB and vs the population
count, mean FE calls and mean wall time, from out/<id>/<method>_s<seed>[sfx].json (10 seeds expected per cell)."""
import json, glob, os, re, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SFX = sys.argv[1] if len(sys.argv) > 1 else ''
BEN = ['556', '635', '1311', '523', '578']
METH = ['mc', 'rand_krig', 'akmcs', 'egra', 'akmcs_h', 'akmcsi', 'akss', 'akis', 'bal', 'bal_lit', 'altab', 'beta_rb']
R = {}
for f in glob.glob(f'{HERE}/out/*/*_s*{SFX}.json'):
    j = json.load(open(f))
    if SFX == '' and re.search(r'_(pf[^_]*|b\d+)\.json$', f): continue
    R.setdefault((j['id'].split('_')[0], j['method']), []).append(j)
def cell(L, key):
    if not L: return '—'
    e = np.abs([j[key] for j in L]); n = '' if len(L) == 10 else f' [n={len(L)}]'
    return f'{e.mean():.1f} ({e.max():.0f}){n}'
out = [f'# Section 7 results{(" " + SFX) if SFX else " (Pf = 1e-3)"} — mean |error| % (worst of the seeds)\n']
for key, title in (('err_rb_pct_pf', 'vs the Rao–Blackwellised reference Pf_RB'), ('err_emp_pct_pf', 'vs the crude count of the same population')):
    out += [f'\n## {title}\n', '| method | ' + ' | '.join(BEN) + ' |', '|:--|' + '--:|' * len(BEN)]
    out += [f'| {m} | ' + ' | '.join(cell(R.get((b, m), []), key) for b in BEN) + ' |' for m in METH]
out += ['\n## mean FE calls / mean wall time per run (s)\n', '| method | ' + ' | '.join(BEN) + ' |', '|:--|' + '--:|' * len(BEN)]
out += [f'| {m} | ' + ' | '.join((lambda L: f"{np.mean([j['n_FE'] for j in L]):.0f} / {np.mean([j['wall_s'] for j in L]):.0f}" if L else '—')(R.get((b, m), [])) for b in BEN) + ' |' for m in METH]
ref = {b: next(iter(R[(b, m)] for m in METH if (b, m) in R), None) for b in BEN}
out += ['\n## references\n', '| benchmark | N | d | Pf_RB | CoV(Pf_RB) | count | count vs RB % |', '|:--|--:|--:|--:|--:|--:|--:|']
for b in BEN:
    if ref[b]: j = ref[b][0]; out.append(f"| {j['id']} | {j['N']} | {j['d']} | {j['pf_rb']:.3g} | {j['pf_rb_cov']:.3f} | {j['n_fail_emp']} | {(j['pf_emp'] / j['pf_rb'] - 1) * 100:+.1f} |")
open(f'{HERE}/RESULTS_S7{SFX}.md', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
