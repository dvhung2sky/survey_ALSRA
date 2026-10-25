"""Smoke checks for al_run.py on the pilot population:  python3 test_al.py [/tmp/pilot578.npz]"""
import sys, numpy as np; sys.path.insert(0, '.')
import al_run as A
pop = sys.argv[1] if len(sys.argv) > 1 else '/tmp/pilot578.npz'
W = A.World('578', 1e-3, pop)
assert abs(W.pf_rb - 1e-3) < 1e-9, W.pf_rb                                   # bisection hits the target
u = W.Zx[5]; x = W.rv.to_x(u); assert np.allclose([x[n] for n in W.names], W.X[5], rtol=1e-9)   # u <-> x round trip
assert np.allclose(W.Zx.mean(0), 0, atol=0.1) and np.allclose(W.Zx.std(0), 1, atol=0.1)         # standard-normal coordinates
assert np.allclose(A.norm.cdf(-W.beta), W.p_i)                                                # beta = -Phi^-1(P(S>R)) exactly
idx = A.init_design(W, 0); assert idx == A.init_design(W, 0) and len(set(idx)) == A.N0       # paired initial design
gp = A.fit_gp(W.Z[idx], W.g[idx]); P = W.Z[:50]; mu, sd = A.gp_pred(gp, P)
c = A.post_cov(gp, P, P); assert np.allclose(np.sqrt(np.maximum(np.diag(c), 0)), sd, rtol=1e-4, atol=1e-6), (np.sqrt(np.diag(c))[:3], sd[:3])  # post_cov matches predict
print('OK', W.cid, 'N', W.N, 's_med', round(W.s_med, 3), 'pf_emp', W.pf_emp)
