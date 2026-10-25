"""Data for fig_bench: mean |error| of P_f vs P_f,RB (%) over 10 seeds at 100, 200 and 300 FE calls, per benchmark and method,
from bench/out/<id>/<method>_s<seed>[_b<budget>].json.  Sequential Kriging methods (budget used only as the stop): the value at
n calls is read from the 300-call trajectory (last entry with n_FE <= n).  Other methods: separate runs at budgets 25/50/100/200.
Writes fig_bench_<bench>_<method>.dat (n err capped flag) and prints the table.  Values above YCAP are drawn at the cap."""
import json, glob, os, re, numpy as np
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bench', 'out')
BEN = ['556', '635', '1311', '523', '578']
TRAJ = {'akmcs', 'egra', 'akmcs_h', 'akmcsi', 'bal', 'bal_lit'}
SHOW = ['mc', 'rand_krig', 'akmcs', 'akmcsi', 'akss', 'akis', 'bal', 'altab', 'beta_rb']
NS, YCAP = (25, 50, 100, 200, 300), 300.0

def at(j, n):                                    # P_f the method reports after n calls (trajectory methods)
    t = [r for r in j['traj'] if r[0] <= n and r[1] is not None]
    return t[-1][1] if t else 0.0

def err(b, m, n):
    d = glob.glob(f'{B}/{b}_*')
    if not d: return (np.nan, 0)
    d = d[0]
    if m in TRAJ or n == 300:
        L = [json.load(open(f)) for f in glob.glob(f'{d}/{m}_s*.json') if not re.search(r'_(pf[^_]*|b\d+)\.json$', f)]
        e = [abs(at(j, n) / j['pf_rb'] - 1) * 100 for j in L] if m in TRAJ else [abs(j['err_rb_pct_pf']) for j in L]
    else:
        L = [json.load(open(f)) for f in glob.glob(f'{d}/{m}_s*_b{n}.json')]; e = [abs(j['err_rb_pct_pf']) for j in L]
    return (float(np.mean(e)), len(e)) if e else (np.nan, 0)

def floor(b):                                   # |count / Pf_RB - 1| (%): the error left to a count-based readout
    f = glob.glob(f'{B}/{b}_*/altab_s0.json')
    return abs((j := json.load(open(f[0])))['pf_emp'] / j['pf_rb'] - 1) * 100 if f else 1.0

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    open('fig_bench_floor.tex', 'w').write(''.join(f'\\expandafter\\def\\csname floor{b}\\endcsname{{{max(floor(b), 0.06):.3g}}}\n' for b in BEN))
    for b in BEN:
        for m in SHOW:
            rows = ['n err capped seeds']
            for n in NS:
                v, k = err(b, m, n)
                if k: rows.append(f'{n} {min(v, YCAP):.4g} {int(v > YCAP)} {k}')
            open(f'fig_bench_{b}_{m}.dat', 'w').write('\n'.join(rows) + '\n')
            print(b, f'{m:10}', ' '.join(r.split()[1] + ('*' if r.split()[2] == '1' else '') + f'[{r.split()[3]}]' for r in rows[1:]))
