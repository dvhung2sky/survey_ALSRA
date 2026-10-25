"""Merge population chunks: python3 merge_pop.py <id>  ->  <id>/pop_merged.npz (X, names, cap, ok, t_call)."""
import sys, glob, numpy as np, os
HERE = os.path.dirname(os.path.abspath(__file__)); cid = sys.argv[1]
fs = sorted(glob.glob(f"{HERE}/pop_chunks/{cid}_c*.npz"), key=lambda f: int(f.rsplit('_c', 1)[1][:-4]))
assert fs, f"no chunks for {cid}"
D = [np.load(f, allow_pickle=True) for f in fs]
out = dict(X=np.vstack([d["X"] for d in D]), names=D[0]["names"], cap=np.concatenate([d["cap"] for d in D]),
           ok=np.concatenate([d["ok"] for d in D]), t_call=np.concatenate([d["t_call"] for d in D]))
np.savez(f"{HERE}/{cid}/pop_merged.npz", **out)
print(cid, "chunks", len(fs), "N", len(out["cap"]), "converged", int(out["ok"].sum()), "median s/call", round(float(np.median(out["t_call"])), 3))
