"""python3 summ.py -> mean/sd/worst |err vs Pf_RB| and vs the population count (%), mean n_FE, mean wall per benchmark x method from out/*/*.json"""
import json, glob, os, re, collections, numpy as np
R = collections.defaultdict(list)
for f in glob.glob(f'{os.path.dirname(os.path.abspath(__file__))}/out/*/*.json'):
    if re.search(r'_(pf[^_]*|b\d+)\.json$', f): continue
    j = json.load(open(f)); R[(j['id'][:4].rstrip('_'), j['method'])].append(j)
for (b, m), L in sorted(R.items()):
    e = np.abs([j['err_rb_pct_pf'] for j in L]); c = np.abs([j['err_emp_pct_pf'] for j in L])
    print(f"{b:5} {m:10} n={len(L):2}  |err| {e.mean():6.2f} sd {e.std():6.2f} max {e.max():6.2f} | vs count {c.mean():6.2f} max {c.max():6.2f}  FE {np.mean([j['n_FE'] for j in L]):5.0f}  wall {np.mean([j['wall_s'] for j in L]):6.0f}s")
