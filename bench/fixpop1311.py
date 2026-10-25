"""Re-evaluate the rows of 1311 pop_merged.npz flagged ok=False under call.py's corrected convergence rule (v2b); rewrite cap/ok in place.
python fixpop1311.py <workers>"""
import os, sys, numpy as np, multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import popgen
if __name__ == '__main__':
    f = os.path.abspath(f'{HERE}/1311_corroded_psc_girders/pop_merged.npz')       # absolute: call.py chdirs
    Z = dict(np.load(f, allow_pickle=True)); nm = [str(s) for s in Z['names']]; bad = np.where(~Z['ok'].astype(bool))[0]
    if len(sys.argv) > 2: bad = bad[:int(sys.argv[2])]   # probe mode: first k rows only, no write
    with mp.get_context('spawn').Pool(int(sys.argv[1]), initializer=popgen._init, initargs=('1311_corroded_psc_girders',)) as pool:
        R = pool.map(popgen._one, [dict(zip(nm, Z['X'][i])) for i in bad], chunksize=20)
    for i, (cap, ok, t) in zip(bad, R): Z['cap'][i], Z['ok'][i], Z['t_call'][i] = cap, ok, t
    if len(sys.argv) > 2:
        print('probe', [round(float(Z['cap'][i]), 1) for i in bad]); sys.exit()
    np.savez(f, **Z); print(f'rechecked {len(bad)}: now ok {int(sum(r[1] for r in R))}, still bad {int(sum(not r[1] for r in R))}; total ok {int(Z["ok"].sum())}/{len(Z["ok"])}')
