"""AL-SRA method comparison harness.  python3 al_run.py <id> <method> <seed> [--pf 1e-3] [--budget 300] [--pop path]
Writes out/<id>/<method>_s<seed>.json (suffix _pf<target> when --pf != 1e-3).  Formulas and provenance: HARNESS.md.

Limit state g = R(x_s) - S, S lognormal CoV 0.30 independent; s_med bisected so Pf_RB = mean_i P(S > R_i) = target.
Surrogate input u = (standard-normal coordinates of x_s, z_S); g is expressed in units of s_med.
Budget = TOTAL FE labels (12 initial + acquired).  Methods: mc akmcs egra akmcs_h akmcsi akss akis bal altab beta_rb rand_krig.

TABICL GUARD: altab and beta_rb load TabICL (several GB with 1e5-row readouts); NEVER run two TabICL jobs at once
(the 64 GB laptop rebooted twice).  A blocking flock on /tmp/tabicl_al.lock enforces one at a time; also use guard/run_queue.py.
"""
import os, sys, glob, json, time, argparse, resource, importlib.util
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.dont_write_bytecode = True
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import norm
from scipy.spatial import cKDTree
from scipy.linalg import solve_triangular
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
N0 = 12
CKPT = os.environ.get("TABICL_CKPT") or next((p for p in (f"{ROOT}/tabicl-regressor-v2-20260212.ckpt", f"{os.path.dirname(HERE)}/tabicl-regressor-v2-20260212.ckpt") if os.path.exists(p)), f"{ROOT}/tabicl-regressor-v2-20260212.ckpt")
SIG_S = float(np.sqrt(np.log(1 + 0.30 ** 2)))          # S lognormal, CoV 0.30
METHODS = "mc akmcs egra akmcs_h akmcsi akss akis bal bal_lit altab beta_rb rand_krig".split()


# ------------------------------------------------------------------ world
def find_dir(cid):
    m = sorted(glob.glob(f"{HERE}/{cid}*/call.py")); assert m, f"no benchmark {cid}"
    return os.path.dirname(m[0])

def load_call(cid):
    d = find_dir(cid); sys.path.insert(0, HERE)
    spec = importlib.util.spec_from_file_location(f"call_{os.path.basename(d)}", f"{d}/call.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

class World:
    """Population + exact limit state.  Z = [u_x (d_s), z_S]; g in units of s_med; g<=0 is failure."""
    def __init__(self, cid, pf, pop, zseed=12345):
        self.dir = find_dir(cid); self.cid = os.path.basename(self.dir)
        pop = os.path.abspath(pop) if pop else f"{self.dir}/pop_N100000_s0.npz"; self.pop_path = pop
        D = np.load(pop, allow_pickle=True)
        self.names = [str(s) for s in D["names"]]
        keep = np.asarray(D["ok"], bool) & np.isfinite(D["cap"]) & (D["cap"] > 0)
        self.n_dropped = int((~keep).sum())            # non-converged / failed FE rows are dropped (see HARNESS.md)
        self.X, self.cap = D["X"][keep], D["cap"][keep]
        self.N, self.ds = self.X.shape; self.d = self.ds + 1
        try: self.mod = load_call(self.cid)              # needs the calibrated model.py (CALIB_DIR); only live FE calls use it
        except Exception as e:                            # without it: RV table from rv.json, population lookups only
            self.mod = None; self.mod_error = repr(e)[:200]
            self.mod_rv = {k: tuple(v) for k, v in json.load(open(f"{self.dir}/rv.json")).items()}
        self.rv = RV_TABLE(self.mod, self.names, getattr(self, "mod_rv", None))
        self.Zx = self.rv.to_u(self.X)
        # s_med: Pf_RB(s) = mean Phi((ln s - ln R)/sig_S) = target
        lr = np.log(self.cap); lo, hi = lr.min() - 12 * SIG_S, lr.max() + 12 * SIG_S
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if norm.cdf((mid - lr) / SIG_S).mean() < pf: lo = mid
            else: hi = mid
        self.s_med = float(np.exp(0.5 * (lo + hi)))
        self.pf_target = pf
        self.p_i = norm.cdf((np.log(self.s_med) - lr) / SIG_S)         # P(S > R_i), exact given R
        self.pf_rb = float(self.p_i.mean())
        self.zS = np.random.default_rng(zseed).standard_normal(self.N)  # one demand draw per population sample
        self.Z = np.column_stack([self.Zx, self.zS])
        self.g = self.cap / self.s_med - np.exp(SIG_S * self.zS)       # g / s_med
        self.pf_emp = float((self.g <= 0).mean()); self.n_fail = int((self.g <= 0).sum())
        self.beta = (lr - np.log(self.s_med)) / SIG_S                  # = -Phi^-1(P(S>R)) exactly
    def rb_cov(self):
        return float(self.p_i.std() / np.sqrt(self.N) / self.pf_rb)

class RV_TABLE:
    """Invert the LN/N transforms of rv.draw (same constants), without importing OpenSees."""
    def __init__(self, mod, names, rv=None):
        self.RV = mod.RV if mod is not None else rv
        assert list(self.RV) == names, (list(self.RV), names)
        self.names = names
    def to_u(self, X):
        U = np.empty_like(X, dtype=float)
        for k, n in enumerate(self.names):
            dist, m, c, _ = self.RV[n]
            if dist == "LN": s = np.sqrt(np.log(1 + c * c)); U[:, k] = (np.log(X[:, k] / m) + s * s / 2) / s
            else: U[:, k] = (X[:, k] / m - 1) / c
        return U
    def to_x(self, u):
        out = {}
        for k, n in enumerate(self.names):
            dist, m, c, _ = self.RV[n]
            if dist == "LN": s = np.sqrt(np.log(1 + c * c)); out[n] = m * np.exp(-s * s / 2 + s * u[k])
            else: out[n] = m * (1 + c * u[k])
        return out

class Live:
    """Live FE oracle for points outside the population (AK-SS, AK-IS).  Counts calls."""
    def __init__(self, W):
        self.W, self.mod, self.n, self.n_bad, self.t = W, W.mod, 0, 0, 0.0
    def g(self, u):
        t = time.perf_counter()
        try:
            r = self.mod.run(self.W.rv.to_x(u[:self.W.ds])); cap = float(r["capacity"])
            if not np.isfinite(cap) or not r.get("converged", True): cap = np.nan; self.n_bad += 1
        except Exception: cap = np.nan; self.n_bad += 1
        self.n += 1; self.t += time.perf_counter() - t
        return cap / self.W.s_med - float(np.exp(SIG_S * u[self.W.ds]))   # NaN = not converged: caller drops the point


# ------------------------------------------------------------------ Kriging (ak_mcs.py logic: ARD Matern 5/2, interpolating, normalize_y)
def _kernel(d):
    return C(1.0, (1e-3, 1e3)) * Matern(np.ones(d), [(1e-2, 1e3)] * d, nu=2.5)

def fit_gp(X, y, prev=None):
    """Same settings as ak_mcs.py (alpha 1e-10, 0 restarts).  Warm start from the previous optimum (speed); jitter fallback."""
    k = _kernel(X.shape[1]) if prev is None else prev.kernel_.clone_with_theta(prev.kernel_.theta)
    for a in (1e-10, 1e-8, 1e-6):
        try:
            return GaussianProcessRegressor(kernel=k, alpha=a, normalize_y=True, n_restarts_optimizer=0, random_state=0).fit(X, y)
        except Exception: k = _kernel(X.shape[1])
    raise RuntimeError("GP fit failed")

def gp_pred(gp, P, chunk=20000):
    mu = np.empty(len(P)); sd = np.empty(len(P))
    for i in range(0, len(P), chunk):
        mu[i:i + chunk], sd[i:i + chunk] = gp.predict(P[i:i + chunk], return_std=True)
    return mu, np.maximum(sd, 1e-12)

def post_cov(gp, A, B):
    """Posterior covariance (original y units) between rows of A and B."""
    LA = solve_triangular(gp.L_, gp.kernel_(gp.X_train_, A), lower=True)
    LB = LA if B is A else solve_triangular(gp.L_, gp.kernel_(gp.X_train_, B), lower=True)
    return (gp.kernel_(A, B) - LA.T @ LB) * float(np.ravel(gp._y_train_std)[0]) ** 2


# ------------------------------------------------------------------ learning functions
def eff(mu, s, k=2.0):                       # EFF, Bichon et al. 2008, eps = k sigma  (run_acq_variants.py)
    e = k * s; a, b, c = -mu / s, (-e - mu) / s, (e - mu) / s
    return (mu * (2 * norm.cdf(a) - norm.cdf(b) - norm.cdf(c)) - s * (2 * norm.pdf(a) - norm.pdf(b) - norm.pdf(c))
            + e * (norm.cdf(c) - norm.cdf(b)))

def hfun(mu, s):                             # H, Lv et al. 2015: |int_{-2s}^{2s} -ln f(y) f(y) dy|, f = N(mu, s^2)
    a, b = -(2 * s + mu) / s, (2 * s - mu) / s
    h = (np.log(np.sqrt(2 * np.pi) * s) + 0.5) * (norm.cdf(b) - norm.cdf(a)) - 0.5 * (b * norm.pdf(b) - a * norm.pdf(a))
    return np.abs(h)


# ------------------------------------------------------------------ result helper
def rel(p, ref): return float((p - ref) / ref * 100) if ref > 0 else float("nan")

def init_design(W, seed): return [int(i) for i in np.random.default_rng(seed).choice(W.N, N0, replace=False)]


# ------------------------------------------------------------------ 1 mc, 11 rand_krig
def run_mc(W, seed, B, a):
    idx = np.random.default_rng(seed + 31).choice(W.N, B, replace=False)
    pf = float((W.g[idx] <= 0).mean())
    return dict(pf=pf, n_FE=B, traj=[[B, pf]])

def run_rand_krig(W, seed, B, a):
    rng = np.random.default_rng(seed + 31); idx = init_design(W, seed)
    rest = np.setdiff1d(np.arange(W.N), idx); idx += [int(i) for i in rng.choice(rest, B - N0, replace=False)]
    gp = fit_gp(W.Z[idx], W.g[idx]); mu, _ = gp_pred(gp, W.Z); pf = float((mu <= 0).mean())
    return dict(pf=pf, n_FE=B, traj=[[B, pf]])


# ------------------------------------------------------------------ 2-5 Kriging AL on the population (akmcs, egra, akmcs_h, akmcsi)
def run_krig_al(W, seed, B, a, method):
    batch = a.batch or (8 if method == "akmcsi" else 1)
    idx = init_design(W, seed); gp = None; traj = []; stop = None
    # epsilon = 2 sigma, EFF/H and the EFF threshold are evaluated in units of the median capacity scale c (from the 12 initial labels:
    # R/s_med = g + exp(sigma_S z_S)), i.e. on the standardised margin g' = (R - S)/c.  ponytail: c fixed per run.
    c = float(np.median(W.g[idx] + np.exp(SIG_S * W.zS[idx])))
    # AK-MCS (Echard 2011) candidate pool: seeded prefix of the population, doubled when min U>=2 but CoV(Pf_hat)>0.05
    perm = np.random.default_rng(seed + 5).permutation(W.N) if method == "akmcs" else np.arange(W.N)
    nc = min(W.N, a.nc0) if (method == "akmcs" and a.nc0 > 0) else W.N; enl = []; exhausted = False
    while True:
        act = perm[:nc]; gp = fit_gp(W.Z[idx], W.g[idx], gp); mu, sd = gp_pred(gp, W.Z[act])
        mask = np.isin(act, idx)
        U = np.abs(mu) / sd; U[mask] = np.inf
        pf = float((mu <= 0).mean()); traj.append([len(idx), pf])
        cov = float(np.sqrt((1 - pf) / (pf * nc))) if pf > 0 else float("inf")      # CoV of the MC estimate on the pool
        if method == "egra": sc = eff(mu / c, sd / c); sc[mask] = 0.0; crit = sc.max() < 1e-3
        elif method == "akmcs_h": sc = hfun(mu / c, sd / c); sc[mask] = -1.0; crit = U.min() >= 2.0     # H has no verified own threshold -> U>=2
        else: crit = U.min() >= 2.0
        if method == "akmcs" and crit and cov > 0.05:
            if nc < W.N: nc = min(W.N, 2 * nc); enl.append([len(idx), nc]); continue      # enlarge pool, re-test
            exhausted = True
        if stop is None and crit: stop = dict(n_FE_stop=len(idx), pf_stop=pf, n_cand_stop=nc, cov_pf_stop=cov)
        if len(idx) >= B: break
        nb = min(batch, B - len(idx))
        if method in ("akmcs",): new = [int(np.argmin(U))]
        elif method == "egra": new = [int(np.argmax(sc))] if sc.max() > 0 else [int(np.argmin(U))]    # stall guard
        elif method == "akmcs_h": new = [int(np.argmax(sc))]
        else:                                                         # akmcsi: Lelievre 2018 k-means on the U<2 pool
            from sklearn.cluster import KMeans
            pool = np.nonzero(U < 2.0)[0]
            if len(pool) < 8 * nb: pool = np.argsort(U)[:8 * nb]
            if len(pool) <= nb: new = [int(i) for i in pool[:nb]]
            else:
                lab = KMeans(nb, n_init=3, random_state=seed).fit_predict(W.Z[act[pool]]); new = []
                for k in range(nb):
                    m = pool[lab == k]
                    if len(m): new.append(int(m[np.argmin(U[m])]))
        new = [int(act[j]) for j in new]
        idx += [i for i in dict.fromkeys(new) if i not in set(idx)] or [int(act[np.argmin(U)])]
    if nc < W.N: pf_cand = pf; pf = float((gp_pred(gp, W.Z)[0] <= 0).mean())      # truncated-at-budget result on the FULL population
    out = dict(pf=pf, n_FE=len(idx), traj=traj, stop_rule={"egra": "max EFF/c<1e-3", "akmcs_h": "min U>=2"}.get(method, "min U>=2"), cap_scale_c=c)
    if method == "akmcs":
        out.update(n_cand_final=nc, enlargements=enl, enlargement_exhausted=bool(exhausted), cov_pf_final_pool=cov)
        if nc < W.N: out["pf_cand_budget"] = pf_cand
        if stop is None: out["note"] = "min U>=2 (with CoV<=0.05 or pool exhausted) never reached within budget"
    if stop: out.update(stop)
    return out


# ------------------------------------------------------------------ 8 bal
def run_bal(W, seed, B, a, look=True):
    idx = init_design(W, seed); gp = None; traj = []; stop = None; mask = np.zeros(W.N, bool); K, M = 24, 2000
    while True:
        gp = fit_gp(W.Z[idx], W.g[idx], gp); mu, sd = gp_pred(gp, W.Z)
        pi = norm.cdf(-mu / sd); v = pi * (1 - pi); mask[:] = False; mask[idx] = True; v[mask] = 0.0
        pf = float(pi.mean()); cov = float(np.sqrt(v.sum()) / W.N / max(pf, 1e-300)); traj.append([len(idx), pf, cov])
        if stop is None and cov < 0.05: stop = dict(n_FE_stop=len(idx), pf_stop=pf)
        if len(idx) >= B: break
        pool = np.argsort(-v)[:M]
        if look and v[pool[0]] > 0:                # one-step lookahead (Kriging believer) on a shortlist
            sh = pool[:K]; c = post_cov(gp, W.Z[pool], W.Z[sh])
            sd2 = np.sqrt(np.maximum(sd[pool][:, None] ** 2 - c ** 2 / sd[sh][None, :] ** 2, 1e-24))
            p2 = norm.cdf(-mu[pool][:, None] / sd2); red = (v[pool][:, None] - p2 * (1 - p2)).sum(0)
            j = int(sh[np.argmax(red)])
        else: j = int(np.argmax(v))                      # literal: max Phi(-U)(1-Phi(-U)) (same ranking as min U)
        idx.append(j)
    out = dict(pf=pf, n_FE=len(idx), traj=traj, stop_rule="posterior CoV(Pf)<0.05", pf_hard=float((mu <= 0).mean()), post_cov_final=cov)
    if stop: out.update(stop)
    return out


# ------------------------------------------------------------------ 6 akss
def _mma(rng, seeds, mu_fn, b, n_total, w=1.0):
    """Modified Metropolis (Au & Beck 2001) conditional on mu_fn(u) <= b; chains from `seeds`, Ns total states."""
    nc, d = seeds.shape; L = int(np.ceil(n_total / nc)); cur = seeds.copy(); out = [cur.copy()]
    for _ in range(L - 1):
        xi = cur + rng.uniform(-w, w, cur.shape)
        acc = rng.random(cur.shape) < np.exp(-0.5 * (xi ** 2 - cur ** 2)); cand = np.where(acc, xi, cur)
        ok = mu_fn(cand) <= b; cur = np.where(ok[:, None], cand, cur); out.append(cur.copy())
    return np.vstack(out)[:n_total]

def run_akss(W, seed, B, a):
    rng = np.random.default_rng(seed + 77); live = Live(W); p0 = 0.1; Ns = min(a.n_ss, W.N); cap_level = a.ss_cap
    idx = init_design(W, seed); TX = list(W.Z[idx]); TY = list(W.g[idx]); gp = None
    n_live_extra = 0; sel = rng.choice(W.N, Ns, replace=False)
    S, lab = W.Z[sel].copy(), {i: W.g[j] for i, j in enumerate(sel)}      # level 0 = population subset; labels are cache hits
    lab_fn = lambda k, u: lab[k] if k in lab else live.g(u)
    k_level, complete, traj, converged_all = 0, False, [], True; bad = set()
    while not complete:
        added = 0; hit = False
        while True:
            gp = fit_gp(np.array(TX), np.array(TY), gp); mu, sd = gp_pred(gp, S)
            bq = max(float(np.quantile(mu, p0)), 0.0)                      # next-level threshold (0 = final level)
            traj.append([len(TX), None])
            if os.environ.get('DBG'): print('lvl',k_level,'n',len(TX),'bq',round(bq,3),'minU',round(float((np.abs(mu-bq)/sd).min()),2),'mu_min',round(float(mu.min()),3),'sd_med',round(float(np.median(sd)),3))
            U = np.abs(mu - bq) / sd; U[list(bad)] = np.inf   # ponytail: bad state stays in the level sample (0.2% effect)
            if U.min() >= 2.0: break
            if added >= cap_level or len(TX) >= B: converged_all = False; break
            j = int(np.argmin(U)); y = lab_fn(j, S[j]); added += 1
            if not np.isfinite(y): bad.add(j); continue
            TX.append(S[j].copy()); TY.append(y); lab[j] = y
        if bq <= 0.0 or k_level >= 8:
            pf = float(p0 ** k_level * (mu <= 0).mean()); complete = True; traj[-1][1] = pf; break
        seeds = S[mu <= bq]; mu_fn = lambda U_: gp_pred(gp, U_)[0]
        S = _mma(rng, seeds, mu_fn, bq, Ns); lab = {}; bad = set(); k_level += 1
    out = dict(pf=pf, n_FE=len(TX), traj=traj, levels=k_level + 1, n_live=live.n, n_live_nonconv=live.n_bad,
               t_live_s=round(live.t, 2), stop_rule="all levels min U>=2 (p0=0.1)")
    if converged_all: out.update(n_FE_stop=len(TX), pf_stop=pf)
    return out


# ------------------------------------------------------------------ 7 akis
def _form(gp, rng, d):
    from scipy.optimize import minimize
    f = lambda u: float(gp.predict(u[None, :])[0])
    r = minimize(lambda u: u @ u, np.zeros(d), jac=lambda u: 2 * u, method="SLSQP", bounds=[(-7, 7)] * d,
                 constraints=[{"type": "eq", "fun": f}], options=dict(maxiter=100, ftol=1e-8))
    if r.success and abs(f(r.x)) < 1e-2 * max(1.0, abs(f(np.zeros(d)))) and r.x @ r.x > 1e-6: return r.x, True
    P = rng.standard_normal((20000, d)); P *= (rng.uniform(2.5, 4.5, 20000) / np.linalg.norm(P, axis=1))[:, None]
    m = gp.predict(P); j = int(np.argmin(np.where(m <= 0, np.linalg.norm(P, axis=1), 1e9 + m)))   # fallback: lowest-norm predicted failure
    return P[j], False

def run_akis(W, seed, B, a):
    rng = np.random.default_rng(seed + 77); live = Live(W); d = W.d
    idx = init_design(W, seed); TX = list(W.Z[idx]); TY = list(W.g[idx]); gp = None; ustar = None; form_ok = False
    for it in range(6):                                                    # active FORM: refine the Kriging at the design point
        gp = fit_gp(np.array(TX), np.array(TY), gp); ustar, form_ok = _form(gp, rng, d)
        if len(TX) >= B: break
        gt = live.g(ustar)
        if not np.isfinite(gt): break
        TX.append(ustar.copy()); TY.append(gt)
        if abs(gt) < 0.05: break
    gp = fit_gp(np.array(TX), np.array(TY), gp)
    ustar, form_ok = _form(gp, rng, d)
    S = ustar + rng.standard_normal((a.n_is, d)); lw = -S @ ustar + 0.5 * ustar @ ustar    # ln phi(u)/h(u), h = N(u*, I)
    wgt = np.exp(lw); stop = None; traj = []; good = np.ones(len(S), bool)
    while True:
        gp = fit_gp(np.array(TX), np.array(TY), gp); mu, sd = gp_pred(gp, S)
        pf = float(np.mean(((mu <= 0) * wgt)[good])); U = np.abs(mu) / sd; U[~good] = np.inf; traj.append([len(TX), pf])
        if stop is None and U.min() >= 2.0: stop = dict(n_FE_stop=len(TX), pf_stop=pf)
        if len(TX) >= B or U.min() >= 2.0: break
        j = int(np.argmin(U)); y = live.g(S[j])
        if np.isfinite(y): TX.append(S[j].copy()); TY.append(y)
        else: good[j] = False
    out = dict(pf=pf, n_FE=len(TX), traj=traj, n_live=live.n, n_live_nonconv=live.n_bad, t_live_s=round(live.t, 2),
               beta_ustar=float(np.linalg.norm(ustar)), form_converged=bool(form_ok), stop_rule="IS-sample min U>=2")
    if stop: out.update(stop)
    return out


# ------------------------------------------------------------------ 9 altab, 10 beta_rb  (TabICL)
def tabicl():
    import fcntl
    global _LOCK
    if os.environ.get("TABICL_LOCK", "1") != "0":                  # laptop default: one TabICL job at a time; TABICL_LOCK=0 on a GPU box
        _LOCK = open("/tmp/tabicl_al.lock", "w"); fcntl.flock(_LOCK, fcntl.LOCK_EX)
    import torch; torch.set_num_threads(1)                                                      # torch 2.11 multi-thread segfault
    from tabicl import TabICLRegressor
    dev = os.environ.get("DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
    return TabICLRegressor(n_estimators=1, device=dev, model_path=CKPT,
                           allow_auto_download=False, kv_cache=False)

def run_tab(W, seed, B, a, method):
    REG = tabicl(); CHUNK = 50_000                                         # < 65536 query rows (silent Pf=0 above it)
    X = W.Z if method == "altab" else W.Zx; y = W.g if method == "altab" else W.beta
    batch = a.batch or 8; CAND = a.cand or int(min(W.N, np.ceil(20 / a.pf)))      # expected failures in the candidate set >= 20
    pred = lambda P, al: [REG.predict(P[i:i + 6000], output_type=["mean", "quantiles"], alphas=al) for i in range(0, len(P), 6000)]
    idx = init_design(W, seed); crng = np.random.default_rng(seed + 999)    # A8's candidate stream
    while len(idx) < B:
        REG.fit(X[idx], y[idx]); cand = crng.choice(W.N, min(CAND, W.N), replace=False); Pc = X[cand]
        o = pred(Pc, [0.05, 0.95]); mu = np.concatenate([q["mean"] for q in o]); q = np.concatenate([q["quantiles"] for q in o])
        sA = np.maximum((q[:, 1] - q[:, 0]) / 2, 1e-12); s = np.maximum((q[:, 1] - q[:, 0]) / 3.29, 1e-12)
        sc = sA / (np.abs(mu) + 1e-12) if method == "altab" else eff(mu, s)                 # U (A8) | EFF (run_acq_variants)
        sc = sc.astype(float); sc[np.isin(cand, idx)] = -1.0
        rho = cKDTree(X[idx]).query(Pc, k=1)[0] + 1e-9; n0 = len(idx)
        for _ in range(min(batch, B - len(idx))):
            j = int(np.argmax(sc))
            if sc[j] <= 0: break
            idx.append(int(cand[j])); dd = np.linalg.norm(Pc - Pc[j], axis=1)
            sc *= 1.0 - np.exp(-0.5 * (dd / rho[j]) ** 2); sc[j] = -1.0       # local penalisation
        if len(idx) == n0:                                                    # stall guard (all scores underflowed)
            free = np.setdiff1d(cand, idx); idx.append(int(free[np.argmax(cKDTree(X[idx]).query(X[free], k=1)[0])]))
    REG.fit(X[idx], y[idx])
    med = np.concatenate([REG.predict(X[i:i + CHUNK], output_type=["mean", "quantiles"], alphas=[0.5])["quantiles"][:, 0]
                          for i in range(0, W.N, CHUNK)])
    pf = float((med <= 0).mean()) if method == "altab" else float(norm.cdf(-med).mean())   # hard indicator | RB mean Phi(-beta_hat)
    return dict(pf=pf, n_FE=len(idx), traj=[[len(idx), pf]], readout="median<=0" if method == "altab" else "mean Phi(-beta_med)")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("id"); ap.add_argument("method", choices=METHODS); ap.add_argument("seed", type=int)
    ap.add_argument("--pf", type=float, default=1e-3); ap.add_argument("--budget", type=int, default=300); ap.add_argument("--pop")
    ap.add_argument("--batch", type=int, default=0); ap.add_argument("--cand", type=int, default=0)
    ap.add_argument("--n_ss", type=int, default=2000); ap.add_argument("--ss_cap", type=int, default=60); ap.add_argument("--n_is", type=int, default=3000)
    ap.add_argument("--nc0", type=int, default=10000); ap.add_argument("--out_dir"); a = ap.parse_args()
    t_load = time.time(); W = World(a.id, a.pf, a.pop); t_load = time.time() - t_load
    t0 = time.time(); m = a.method
    if m == "mc": r = run_mc(W, a.seed, a.budget, a)
    elif m == "rand_krig": r = run_rand_krig(W, a.seed, a.budget, a)
    elif m in ("akmcs", "egra", "akmcs_h", "akmcsi"): r = run_krig_al(W, a.seed, a.budget, a, m)
    elif m in ("bal", "bal_lit"): r = run_bal(W, a.seed, a.budget, a, look=(m == "bal"))
    elif m == "akss": r = run_akss(W, a.seed, a.budget, a)
    elif m == "akis": r = run_akis(W, a.seed, a.budget, a)
    else: r = run_tab(W, a.seed, a.budget, a, m)
    wall = time.time() - t0
    for k in ("pf", "pf_stop"):
        if r.get(k) is not None:
            r["err_rb_pct_" + k] = rel(r[k], W.pf_rb); r["err_emp_pct_" + k] = rel(r[k], W.pf_emp)
    rec = dict(id=W.cid, method=m, seed=a.seed, pf_target=a.pf, budget=a.budget, pop=W.pop_path, N=W.N, d=W.d, n_dropped_rows=W.n_dropped,
               s_med=W.s_med, pf_rb=W.pf_rb, pf_rb_cov=W.rb_cov(), pf_emp=W.pf_emp, n_fail_emp=W.n_fail,
               wall_s=round(wall, 2), load_s=round(t_load, 2), peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1), **r)
    od = os.path.abspath(a.out_dir) if a.out_dir else f"{HERE}/out/{W.cid}"; os.makedirs(od, exist_ok=True)
    sfx = "" if abs(a.pf - 1e-3) < 1e-12 else f"_pf{a.pf:g}"
    if a.budget != 300: sfx += f"_b{a.budget}"                  # budget runs must not overwrite the 300-call results
    json.dump(rec, open(f"{od}/{m}_s{a.seed}{sfx}.json", "w"), default=float)
    print(json.dumps({k: v for k, v in rec.items() if k != "traj"}, default=float))

if __name__ == "__main__":
    main()
