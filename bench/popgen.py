"""Monte Carlo population of FE capacities for one benchmark.  python3 popgen.py <id> <N> <seed> <workers> [out.npz]
Draws N structural samples from the wrapper's RV table, runs call.run() once per sample (one OpenSees process per worker,
1 thread each), saves params, capacity, converged flag and per-call wall time. Demand is closed-form and applied later."""
import os, sys, time, json, importlib.util
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np, multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rv

def _load(cid):
    spec = importlib.util.spec_from_file_location('call', f'{HERE}/{cid}/call.py'); C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C); return C

_C = None
def _init(cid):
    global _C; sys.dont_write_bytecode = True; _C = _load(cid)

def _one(p):
    t = time.perf_counter()
    try: r = _C.run(p); cap, ok = float(r['capacity']), bool(r['converged'])
    except Exception: cap, ok = float('nan'), False
    return cap, ok, time.perf_counter() - t

if __name__ == '__main__':
    cid, N, seed, W = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    out = os.path.abspath(sys.argv[5]) if len(sys.argv) > 5 else f'{HERE}/{cid}/pop_N{N}_s{seed}.npz'   # before _load: 1311 call.py chdirs
    C = _load(cid); rng = np.random.default_rng(seed); names = list(C.RV)
    P = [rv.draw(C.RV, rng) for _ in range(N)]
    t0 = time.time()
    with mp.get_context('spawn').Pool(W, initializer=_init, initargs=(cid,)) as pool:
        res = pool.map(_one, P, chunksize=max(1, N // (W * 20)))
    wall = time.time() - t0
    cap, ok, tc = (np.array(v) for v in zip(*res))
    np.savez(out, X=np.array([[p[k] for k in names] for p in P]), names=names, cap=cap, ok=ok, t_call=tc)
    print(json.dumps(dict(id=cid, N=N, workers=W, wall_s=round(wall, 1), calls_per_s=round(N / wall, 2),
                          median_call_s=round(float(np.median(tc)), 3), conv=f'{int(ok.sum())}/{N}',
                          cap_mean=float(np.nanmean(cap[ok])), cap_cov=float(np.nanstd(cap[ok]) / np.nanmean(cap[ok])))), flush=True)
