"""Composable active-learning path engine for the 11-axis design space of the RESS survey.

    python3 engine.py --db 578 --path '{"surr":"gp","acq":"eff"}' --seed 0 --budget 300 [--pf 1e-3] [--out r.json] [--progress p.jsonl]

A path = one option per axis (AXES below, same cells as the design-space artifact); unspecified axes take DEF.
Every option is a small function or class in this file; check(path, db) lists the combinations that are invalid.
Databases = the five S7 populations (../bench/<id>/pop_merged.npz).  A label at a DB row is a lookup (it still counts as one FE
call); a label anywhere else (subset-simulation states, IS samples, LHS points, low-fidelity runs) is a LIVE OpenSees call through
../bench/<id>/call.py.  References = Pf_RB and the crude count of the same DB (../bench/HARNESS.md); the limit state, the demand
draw and the 12-point initial design are those of bench/al_run.py, so a preset path reproduces the S7 setting.
"""
import os, sys, json, time, argparse
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); BENCH = os.path.join(os.path.dirname(HERE), "bench")
sys.path.insert(0, BENCH)
import al_run as H                      # World, Live conventions, fit_gp, gp_pred, post_cov, eff, hfun, tabicl, init_design
import numpy as np
from scipy.stats import norm, chi2
from scipy.spatial import cKDTree
from sklearn.gaussian_process import GaussianProcessRegressor

N0, SIG = H.N0, H.SIG_S
LF_FACTOR, LF_COST = 4.0, 0.25          # low fidelity = 4x coarser load/displacement step; nominal cost = 1/4 of an HF call

DBS = {
    "556": dict(dir="556_xbraced_steel_frame", name="X-braced steel frame with ring damper", lf=("dstep", 0.5)),
    "635": dict(dir="635_corrugated_web_steel_frames", name="Corrugated-web steel frame, column removal", lf=None),
    "1311": dict(dir="1311_corroded_psc_girders", name="Corroded post-tensioned concrete girder", lf=("dstep", 0.1)),
    "523": dict(dir="523_infilled_frames_aac_lsf", name="AAC-infilled light-steel frame", lf=("du", 0.05)),
    "578": dict(dir="578_precast_postcast_joints", name="Precast beam-column joint with post-cast zone", lf=("dstep", 0.5)),
}
def pop_path(db): return f"{BENCH}/{DBS[db]['dir']}/pop_merged.npz"
def live_ok(db):
    """Live FE calls need the calibrated model.py of the benchmark (CALIB_DIR); the populations alone serve every lookup."""
    import _load; return os.path.exists(f"{_load.CALIB}/{DBS[db]['dir']}/model.py")

# --------------------------------------------------------------------------------------------- the box (cells of the artifact)
def _o(k, l, m, s, na=None): return dict(k=k, l=l, m=m, s=s, na=na)
AXES = [
 dict(id="tgt", t="Regression target", sub="What the surrogate is fitted on", opts=[
    _o("margin", "Margin g(x)", "A1-A10", "code"), _o("tg", "Compressed T(g)", "g/(1+|g|)", "code"),
    _o("beta", "Conditional beta(x_s)", "capacity side; input x_s only", "code"),
    _o("othr", "Other transforms", "sign(g) log(1+|g|), experimental", "open")]),
 dict(id="surr", t="Surrogate", sub="Model class", opts=[
    _o("gp", "GP / Kriging", "ARD Matern 5/2", "code"), _o("xgb", "XGBoost", "300 trees, depth 4", "code"),
    _o("tabicl", "TabICL v2", "in-context; one job at a time", "code"),
    _o("pck", "PC-Kriging", "two-stage: LARS PCE trend + GP", "lit"), _o("pce", "PCE", "LARS, Hermite degree 2-3", "lit"),
    _o("svmc", "SVM classifier", "sign only (EDSD)", "lit"), _o("svr", "SVR regressor", "RBF", "lit"),
    _o("nn", "Deep NN", "MLP 64x64 (no physics loss)", "lit"), _o("rsm", "Quadratic response surface", "no cross terms (1990)", "lit"),
    _o("none", "None: direct sampling", "true model only", "lit"), _o("agn", "Surrogate-agnostic", "MFIS: GP builds the proposal only", "lit")]),
 dict(id="read", t="Readout (estimator)", sub="How the final surrogate becomes one number", opts=[
    _o("hard", "Hard indicator", "mean 1(mu<=0)", "code"), _o("med", "Median quantile", "q50<=0 (= mean if no quantiles)", "code"),
    _o("bag", "Bag + smooth", "5 bootstrap fits, vote", "code"), _o("rb", "Rao-Blackwellised", "demand integrated exactly", "code"),
    _o("isw", "IS-weighted", "sum w 1 / n", "lit"), _o("bayes", "Bayesian posterior", "mean Phi(-mu/sd); interval = surrogate part only", "lit")]),
 dict(id="uq", t="UQ source", sub="Where the width comes from", opts=[
    _o("gpsig", "GP posterior sd", "GP, PC-Kriging", "code"), _o("kfold", "K-fold conformal", "w = q rho, 90%", "code"),
    _o("native", "TabICL native", "(q95-q05)/3.29", "code"), _o("cqra", "CQR additive", "+ Q_c", "code"),
    _o("cqrm", "CQR multiplicative", "x (1+lambda)", "code"), _o("boot", "Bootstrap ensemble", "5 refits", "lit"),
    _o("bq", "BQ posterior", "GP sd; Pf as a random variable", "lit"), _o("na", "n/a: no width", "geometric rules only", "lit")]),
 dict(id="acq", t="Acquisition / learning function", sub="Which point to evaluate next", opts=[
    _o("amb", "Sign ambiguity (U)", "argmin |mu|/sd", "code"), _o("ucb", "UCB  w x amb", "sd exp(-U^2/2)", "code"),
    _o("eff", "EFF", "Bichon 2008", "code"), _o("cont", "Contour improvement", "Ranjan 2008, alpha 2", "lit"),
    _o("maxmin", "Max-min distance", "EDSD, near-boundary", "lit"), _o("effc", "EFF + CQR", "EFF on the CQR band", "code"),
    _o("hls", "H (Lv 2015)", "information entropy", "code"), _o("pbw", "Probability-band width", "beta target only", "code"),
    _o("rand", "Random", "no search", "code"), _o("erf", "REIF", "folded normal, re-derived (unverified form)", "lit"),
    _o("bqv", "Pf-variance reduction", "one-step lookahead (GP)", "lit"), _o("naq", "n/a: no learning loop", "one-shot random design", "lit"),
    _o("rl", "Bandit portfolio", "UCB1 over U/EFF/H/UCB, experimental", "open")]),
 dict(id="batch", t="Batch and candidate pool", sub="How many points, chosen from where", opts=[
    _o("single", "Single point, full pool", "AK-MCS", "code"), _o("b8", "BATCH = 8", "local penalisation", "code"),
    _o("cand", "CAND subsample", "8 pts from max(6000, 20/Pf) rows", "code"),
    _o("trunc", "Truncated candidate region", "|u| >= beta_hat - 1", "lit"), _o("kmeans", "K-means batch", "AK-MCSi, 8 clusters", "code"),
    _o("parreg", "Parallel adaptive region", "8 pts inside U<2", "lit"), _o("adapt", "Adaptive batch size", "q = #(U<2)/25, 1-16, experimental", "open")]),
 dict(id="pop", t="Population and sampling", sub="How the candidate population is drawn", opts=[
    _o("mc", "Crude Monte Carlo", "the database rows", "code"), _o("ss", "Subset simulation", "p0 0.1, Ns 2000, live", "lit"),
    _o("isc", "Classical IS", "N(u*, I), live", "lit"), _o("rad", "Radial IS", "ARBIS chi tail, live", "lit"),
    _o("meta", "meta-IS", "h ~ pi phi, live", "lit"), _o("ce", "CE-IS", "Gaussian mixture, live", "lit"),
    _o("sis", "Sequential IS", "tempered, pCN, live", "lit"), _o("line", "Line sampling", "500 lines, live", "lit"),
    _o("strat", "Stratified / LHS", "LHS in u-space, live", "lit"), _o("nis", "Neural IS", "RealNVP flow, live", "lit"),
    _o("ttis", "Transport-map IS", "tensor train", "lit", na="needs a squared tensor-train density library; none is installed")]),
 dict(id="stop", t="Stopping criterion", sub="When the loop ends", opts=[
    _o("budget", "Fixed FE budget", "all S7 arms", "code"), _o("lft", "Learning-fn. threshold", "min U>=2 | max EFF<1e-3", "lit"),
    _o("pfcov", "Pf CoV", "CoV < 0.05", "lit"), _o("err", "Error-rate bound", "ESC-type bound < 5%", "lit"),
    _o("conf", "Conformal coverage", "90% interval width < 10% of Pf", "lit")]),
 dict(id="fid", t="Fidelity", sub="How many model levels the loop uses", opts=[
    _o("sf", "Single fidelity", "the database", "code"), _o("cok", "Co-Kriging (AR1)", "LF = 4x coarser step, live", "lit"),
    _o("rom", "Hierarchical Kriging", "LF Kriging as HF trend", "lit"), _o("adapf", "Adaptive fidelity", "level by variance share, experimental", "open")]),
 dict(id="dim", t="Dimension handling", sub="What the surrogate sees", opts=[
    _o("full", "Full input space", "all RVs", "code"), _o("fidr", "Failure-informed DR", "Uribe 2021 style", "lit"),
    _o("assub", "Active subspace", "gradients of an auxiliary GP", "lit"), _o("pls", "PLS", "3 components", "lit"),
    _o("ae", "Learned latent space", "MLP autoencoder, experimental", "open")]),
 dict(id="ls", t="Limit-state composition", sub="What the surrogate represents", opts=[
    _o("composed", "Composed g", "one surrogate", "code"),
    _o("percomp", "Per-component surrogates", "", "lit", na="the databases store one composed capacity per row; no component responses"),
    _o("disjoint", "Multiple disjoint regions", "multi-centre IS (isc, ce)", "lit"),
    _o("adaptmode", "Adaptive mode identification", "", "open", na="no established method, and the databases hold one failure mode")]),
]
DEF = dict(tgt="margin", surr="gp", read="hard", uq="gpsig", acq="amb", batch="single", pop="mc", stop="budget", fid="sf", dim="full", ls="composed")
SD_FREE_ACQ = {"rand", "maxmin", "naq"}
LIVE_POPS = {"ss", "isc", "rad", "meta", "ce", "sis", "line", "strat", "nis"}

PRESETS = {   # S7 harness methods (bench/al_run.py) and artifact arms; notes say where the preset differs from the original
 "A1 (AK-MCS, fixed budget)": ({}, "the default path: GP, U, single point, crude MC, no stopping rule"),
 "AK-MCS (S7 akmcs)": (dict(stop="lft"), "no population enlargement: the whole DB is the pool"),
 "EGRA (S7 egra)": (dict(acq="eff", stop="lft"), ""),
 "AK-MCS+H (S7 akmcs_h)": (dict(acq="hls", stop="lft"), ""),
 "AK-MCSi (S7 akmcsi)": (dict(batch="kmeans", stop="lft"), ""),
 "AK-SS (S7 akss)": (dict(pop="ss", stop="lft"), "level 0 from DB rows, later levels live"),
 "AK-IS (S7 akis)": (dict(pop="isc", stop="lft"), "design point by ray search, not SLSQP"),
 "BAL (S7 bal)": (dict(acq="bqv", uq="bq", read="bayes", stop="pfcov"), ""),
 "Random Kriging (S7 rand_krig)": (dict(acq="naq"), ""),
 "Crude MC (S7 mc)": (dict(surr="none", uq="na", acq="naq"), ""),
 "AL-TabICL (S7 altab)": (dict(surr="tabicl", uq="native", batch="cand", read="med"), "hard readout on the median, as in S7"),
 "beta + RB (S7 beta_rb)": (dict(tgt="beta", surr="tabicl", uq="native", acq="eff", batch="cand", read="rb"), ""),
 "A5": (dict(surr="tabicl", uq="kfold", acq="ucb", batch="cand", read="med"), ""),
 "A8": (dict(surr="tabicl", uq="native", batch="cand", read="med"), "= S7 altab"),
 "A11": (dict(tgt="tg", surr="tabicl", uq="cqra", acq="ucb", batch="cand", read="med"), ""),
 "EFF+CQR": (dict(tgt="beta", surr="tabicl", uq="cqra", acq="effc", batch="cand", read="rb"), ""),
 "A12-pc (pbw)": (dict(tgt="beta", surr="tabicl", uq="native", acq="pbw", batch="cand", read="rb"), ""),
 "EDSD (SVM, max-min)": (dict(surr="svmc", uq="na", acq="maxmin", read="hard"), ""),
 "Meta-IS": (dict(pop="meta", read="isw", stop="lft"), ""),
 "MFIS": (dict(surr="agn", pop="ce", read="isw"), "half the budget builds the GP, half is true-model IS"),
 "Co-Kriging AK-MCS": (dict(fid="cok"), "costs counted in HF-equivalent calls"),
}

def opt(axis, k):
    a = next(a for a in AXES if a["id"] == axis); return next(o for o in a["opts"] if o["k"] == k)

def check(path, db, detail=False):
    """-> (errors, warnings) for a path on a database; detail=True gives errors as (message, axes involved)."""
    p = {**DEF, **path}; E, Wn = [], []
    e = lambda m, *ax: E.append((m, ax))
    for a in AXES:
        if p[a["id"]] not in [o["k"] for o in a["opts"]]: e(f"{a['t']}: unknown option {p[a['id']]}", a["id"]); continue
        if opt(a["id"], p[a["id"]])["na"]: e(f"{opt(a['id'], p[a['id']])['l']}: {opt(a['id'], p[a['id']])['na']}", a["id"])
    s, u, q, r, P, st = p["surr"], p["uq"], p["acq"], p["read"], p["pop"], p["stop"]
    has_sd = u != "na"
    out = lambda: (E if detail else [m for m, _ in E], Wn)
    if s == "none":
        if u != "na": e("No surrogate: set UQ = n/a", "surr", "uq")
        if q != "naq": e("No surrogate: set acquisition = n/a (no learning loop)", "surr", "acq")
        if P not in ("mc", "strat", "ss", "isc", "rad", "line", "ce"): e("No surrogate: population must be MC, LHS, SS, IS, radial IS, line sampling or CE-IS", "surr", "pop")
        if p["tgt"] != "margin": e("No surrogate: target = margin", "surr", "tgt")
        if r not in ("hard", "isw"): e("No surrogate: readout = hard or IS-weighted", "surr", "read")
        if p["dim"] != "full": e("No surrogate: full input space", "surr", "dim")
        if p["fid"] != "sf": e("No surrogate: single fidelity", "surr", "fid")
        if st != "budget": e("No surrogate: only the fixed budget applies", "surr", "stop")
        return out()
    if s == "agn":
        if P not in ("isc", "rad", "meta", "ce", "sis", "nis"): e("Surrogate-agnostic (MFIS) needs an IS-type population: IS, radial, meta-IS, CE, SIS or neural IS", "surr", "pop")
        if u not in ("gpsig", "bq"): e("Surrogate-agnostic uses a GP internally: UQ = GP posterior sd or BQ", "surr", "uq")
        if r != "isw": e("Surrogate-agnostic: the estimator is true-model IS; readout = IS-weighted", "surr", "read")
        if p["fid"] != "sf": e("Surrogate-agnostic: single fidelity", "surr", "fid")
        if p["dim"] != "full": e("Surrogate-agnostic: full input space", "surr", "dim")
    if u in ("gpsig", "bq") and s not in ("gp", "pck", "agn"): e("GP posterior sd / BQ need a GP or PC-Kriging surrogate", "uq", "surr")
    if u in ("native", "cqra", "cqrm") and s != "tabicl": e("Native / CQR bands need TabICL quantiles", "uq", "surr")
    if s == "svmc":
        if p["tgt"] == "beta": e("SVM classifier on the beta target would classify R vs s_med, not failure", "surr", "tgt")
        if u not in ("na", "boot"): e("SVM classifier: UQ = n/a or bootstrap", "surr", "uq")
        if r not in ("hard", "isw", "bag", "med"): e("SVM classifier: readout hard, median, bag or IS-weighted", "surr", "read")
    if not has_sd:
        if q not in SD_FREE_ACQ: e(f"{opt('acq', q)['l']} needs a width; with UQ = n/a use random, max-min or n/a", "acq", "uq")
        if r == "bayes": e("Bayesian readout needs a width", "read", "uq")
        if st in ("lft", "err"): e("This stopping rule needs a width", "stop", "uq")
        if p["batch"] in ("kmeans", "parreg", "adapt"): e("This batch rule needs U = |mu|/sd", "batch", "uq")
        if P in ("meta", "sis", "nis"): e("meta-IS / SIS / neural IS build pi = Phi(-mu/sd): needs a width", "pop", "uq")
    if q == "effc" and u not in ("cqra", "cqrm"): e("EFF + CQR needs UQ = CQR additive or multiplicative", "acq", "uq")
    if q == "bqv" and (s not in ("gp", "pck") or u not in ("gpsig", "bq")): e("Pf-variance lookahead needs the GP posterior covariance", "acq", "surr", "uq")
    if q == "pbw" and p["tgt"] != "beta": e("Probability-band width is defined on the beta target", "acq", "tgt")
    if r == "rb" and P not in ("mc", "strat"): e("RB readout integrates the demand under N(0,1): only with MC or LHS populations", "read", "pop")
    if r == "isw" and P in ("mc", "strat") and s != "agn": Wn.append("IS-weighted on an unweighted population = hard indicator (all weights 1)")
    if P in ("ss", "line"):
        if st not in ("budget", "lft"): e("Subset / line sampling: stopping = budget or learning-function threshold", "pop", "stop")
        if p["batch"] not in ("single", "b8"): e("Subset / line sampling: batch single or 8", "pop", "batch")
        Wn.append(f"{opt('pop', P)['l']} has its own estimator; the readout axis is ignored")
    if not live_ok(db):
        why = "needs live OpenSees calls, and the calibrated model.py of this benchmark is not installed (set CALIB_DIR; the models are available from the corresponding author)"
        if P in LIVE_POPS: e(f"{opt('pop', P)['l']} {why}", "pop")
        if s == "agn": e(f"Surrogate-agnostic IS {why}", "surr")
        if p["fid"] != "sf": e(f"Multi-fidelity {why}", "fid")
    if P in LIVE_POPS or s == "none" and P != "mc":
        Wn.append(f"labels off the database are live OpenSees calls (about {DB_TCALL.get(db, 1):.1f} s each)")
    if p["ls"] == "disjoint" and P not in ("isc", "ce"): e("Multiple disjoint regions acts on the IS / CE proposal: population = IS or CE-IS", "ls", "pop")
    if p["fid"] != "sf":
        if DBS[db]["lf"] is None: e(f"Database {db} has no step parameter to coarsen: no low-fidelity level", "fid")
        if s != "gp": e("Multi-fidelity needs the GP surrogate", "fid", "surr")
        if u not in ("gpsig", "bq"): e("Multi-fidelity needs GP sd", "fid", "uq")
        if P != "mc": e("Multi-fidelity needs the MC population", "fid", "pop")
        if p["dim"] != "full": e("Multi-fidelity needs the full input space", "fid", "dim")
        if q in ("bqv", "naq", "rl"): e("Multi-fidelity: acquisition U, UCB, EFF, H, contour, REIF, max-min or random", "fid", "acq")
    if q == "naq" and (st != "budget" or p["batch"] != "single"): Wn.append("no learning loop: batch and stopping are ignored")
    if s == "tabicl": Wn.append("TabICL runs one job at a time (RAM guard); several minutes per run")
    return out()

def repair(path, db, axis):
    """Keep path[axis] and change as few other axes as needed to clear the errors it brings (greedy, <= 4 rounds).
    -> (new path, [(axis, old, new)], remaining errors).  Prefers the default option, then options in the codebase."""
    p = {**DEF, **path}; changes = []
    for _ in range(4):
        E = check(p, db, True)[0]
        if not E: break
        best = None
        for _, ax in E:
            for b in ax:
                if b == axis or any(c[0] == b for c in changes): continue
                for o in next(a for a in AXES if a["id"] == b)["opts"]:
                    if o["na"] or o["k"] == p[b]: continue
                    E2 = [m for m, _ in check({**p, b: o["k"]}, db, True)[0]]; n = len(E2)
                    key = (n, sum(m in E2 for m, _ in E), o["k"] != DEF[b], o["k"] != "native", o["s"] != "code")   # native: TabICL's own band
                    if best is None or key < best[0]: best = (key, b, o["k"])
        if best is None or best[0][0] > len(E): break                   # equal count allowed: chains like EFF+CQR -> CQR -> TabICL
        changes.append((best[1], p[best[1]], best[2])); p[best[1]] = best[2]
    return p, changes, [m for m, _ in check(p, db, True)[0]]

def option_state(path, db):
    """{axis: {option: {"changes": [...]} | {"blocked": reason}}} for options that are not free to pick on this path."""
    p = {**DEF, **path}; base = {m for m, _ in check(p, db, True)[0]}; out = {}
    for a in AXES:
        for o in a["opts"]:
            if o["k"] == p[a["id"]] or o["na"]: continue
            q = {**p, a["id"]: o["k"]}
            if not [m for m, _ in check(q, db, True)[0] if m not in base]: continue
            q2, ch, rest = repair(q, db, a["id"])
            out.setdefault(a["id"], {})[o["k"]] = {"blocked": rest[0]} if rest else {"changes": ch}
    return out

DB_TCALL = {"556": 0.93, "635": 1.66, "1311": 3.40, "523": 0.64, "578": 0.37}


# --------------------------------------------------------------------------------------------- oracle
class Oracle:
    """HF label: DB lookup at a row (1 FE, no wall time) or a live call; LF label: live call with a 4x coarser step."""
    def __init__(s, W, db):
        s.W, s.lf_kw = W, DBS[db]["lf"]; s.n_lookup = s.n_live = s.n_bad = s.n_lf = s.n_lf_bad = 0; s.t_live = s.t_lf = 0.0
    def _run(s, u, kw):
        try:
            r = s.W.mod.run(s.W.rv.to_x(u[:s.W.ds]), **kw); cap = float(r["capacity"])
            if not np.isfinite(cap) or cap <= 0 or not r.get("converged", True): return np.nan
        except Exception: return np.nan
        return cap / s.W.s_med - float(np.exp(SIG * u[s.W.ds]))          # g in units of s_med, as al_run.World
    def hf(s, u, row=-1):
        if row >= 0: s.n_lookup += 1; return float(s.W.g[row])
        t = time.perf_counter(); g = s._run(u, {}); s.t_live += time.perf_counter() - t; s.n_live += 1; s.n_bad += not np.isfinite(g)
        return g
    def lf(s, u):
        k, base = s.lf_kw; t = time.perf_counter(); g = s._run(u, {k: base * LF_FACTOR}); s.t_lf += time.perf_counter() - t
        s.n_lf += 1; s.n_lf_bad += not np.isfinite(g); return g
    def cost(s): return s.n_lookup + s.n_live + LF_COST * s.n_lf


# --------------------------------------------------------------------------------------------- targets
def target_y(k, Z, g):
    if k == "margin": return g
    if k == "tg": return g / (1 + np.abs(g))
    if k == "othr": return np.sign(g) * np.log1p(np.abs(g))
    return np.log(np.maximum(g + np.exp(SIG * Z[:, -1]), 1e-300)) / SIG        # beta(x_s) = ln(R/s_med)/sigma_S


# --------------------------------------------------------------------------------------------- surrogates
def _chunk(f, P, n=20000): return np.concatenate([f(P[i:i + n]) for i in range(0, len(P), n)]) if len(P) else np.zeros(0)

def _multi_idx(d, p):
    from itertools import combinations_with_replacement as cwr
    return [t for k in range(1, p + 1) for t in cwr(range(d), k)]

def _herm(X, idx, p):
    from math import factorial
    Hk = [np.ones_like(X), X]
    for k in range(2, p + 1): Hk.append(X * Hk[k - 1] - (k - 1) * Hk[k - 2])
    F = np.ones((len(X), len(idx)))
    for j, t in enumerate(idx):
        for v in set(t): k = t.count(v); F[:, j] *= Hk[k][:, v] / np.sqrt(factorial(k))
    return F

class GPm:
    sd = True
    def __init__(s, seed, kern=None): s.seed, s.kern, s.m = seed, kern, None
    def fit(s, X, y):
        if s.kern is not None:                                    # clone: fixed hyper-parameters (OOF / bootstrap refits)
            for a in (1e-8, 1e-6, 1e-4):
                try: s.m = GaussianProcessRegressor(kernel=s.kern, alpha=a, normalize_y=True, optimizer=None).fit(X, y); return s
                except Exception: pass
        prev = s.m if s.m is not None and s.m.X_train_.shape[1] == X.shape[1] else None
        s.m = H.fit_gp(X, y, prev); return s
    def clone(s): return GPm(s.seed, s.m.kernel_ if s.m is not None else None)
    def predict(s, P): mu, sd = H.gp_pred(s.m, P); return dict(mu=mu, sd=sd)

_REG, _OWNER = None, None
class Tab:
    sd = True; quant = True
    def __init__(s, seed): s.seed = seed
    def fit(s, X, y):
        global _REG, _OWNER
        if _REG is None: _REG = H.tabicl()
        _REG.fit(X, y); _OWNER = s; s.X, s.y = X, y; return s
    def clone(s): return Tab(s.seed)
    def predict(s, P):
        global _OWNER
        if _OWNER is not s: _REG.fit(s.X, s.y); _OWNER = s
        o = [_REG.predict(P[i:i + 6000], output_type=["mean", "quantiles"], alphas=[0.05, 0.5, 0.95]) for i in range(0, len(P), 6000)]
        q = np.concatenate([a["quantiles"] for a in o]); mu = np.concatenate([a["mean"] for a in o])
        return dict(mu=mu.astype(float), q05=q[:, 0], q50=q[:, 1], q95=q[:, 2], sd=np.maximum((q[:, 2] - q[:, 0]) / 3.29, 1e-12))

class XGB:
    sd = False
    def __init__(s, seed): s.seed = seed
    def fit(s, X, y):
        from xgboost import XGBRegressor
        s.m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, n_jobs=1, random_state=s.seed, verbosity=0).fit(X, y); return s
    def clone(s): return XGB(s.seed)
    def predict(s, P): return dict(mu=s.m.predict(P).astype(float))

class PCE:
    sd = False
    def __init__(s, seed, p=3): s.seed, s.p = seed, p
    def fit(s, X, y):
        from sklearn.linear_model import LassoLarsCV, Ridge
        s.pp = s.p if len(X) >= 40 else 2; s.idx = _multi_idx(X.shape[1], s.pp); F = _herm(X, s.idx, s.pp)
        try: s.m = LassoLarsCV(cv=min(5, len(X)), max_iter=500).fit(F, y)
        except Exception: s.m = Ridge(1e-6).fit(F, y)
        return s
    def clone(s): return PCE(s.seed, s.p)
    def predict(s, P): return dict(mu=_chunk(lambda A: s.m.predict(_herm(A, s.idx, s.pp)), P))

class PCK:
    """Two-stage PC-Kriging: sparse PCE trend (degree 2, LARS), then a GP on the residuals (Schobi et al. use universal Kriging)."""
    sd = True
    def __init__(s, seed, kern=None): s.seed, s.t, s.g = seed, PCE(seed, 2), GPm(seed, kern)
    def fit(s, X, y): s.t.fit(X, y); s.g.fit(X, y - s.t.predict(X)["mu"]); return s
    def clone(s): return PCK(s.seed, s.g.m.kernel_)
    def predict(s, P): a = s.t.predict(P)["mu"]; b = s.g.predict(P); return dict(mu=a + b["mu"], sd=b["sd"])

class SVMC:
    sd = False
    def __init__(s, seed): s.seed = seed
    def fit(s, X, y):
        from sklearn.svm import SVC
        lab = y > 0; s.const = None
        if lab.all() or (~lab).all(): s.const = 1.0 if lab.all() else -1.0
        else: s.m = SVC(C=1e3, gamma="scale").fit(X, lab)
        return s
    def clone(s): return SVMC(s.seed)
    def predict(s, P): return dict(mu=np.full(len(P), s.const) if s.const is not None else _chunk(s.m.decision_function, P))

class _Std:                                       # standardised-target wrapper for SVR / MLP
    sd = False
    def __init__(s, seed): s.seed = seed
    def fit(s, X, y): s.a, s.b = y.mean(), y.std() + 1e-12; s.m = s.make().fit(X, (y - s.a) / s.b); return s
    def clone(s): return type(s)(s.seed)
    def predict(s, P): return dict(mu=_chunk(s.m.predict, P) * s.b + s.a)
class SVRm(_Std):
    def make(s): from sklearn.svm import SVR; return SVR(C=100.0, epsilon=0.01, gamma="scale")
class NNm(_Std):
    def make(s):
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor((64, 64), activation="tanh", alpha=1e-4, max_iter=3000, learning_rate_init=1e-3, random_state=s.seed)

class RSM:
    sd = False
    def __init__(s, seed): s.seed = seed
    def _F(s, X): return np.column_stack([np.ones(len(X)), X, X * X])
    def fit(s, X, y):
        F = s._F(X); s.c = np.linalg.solve(F.T @ F + 1e-8 * np.eye(F.shape[1]), F.T @ y); return s
    def clone(s): return RSM(s.seed)
    def predict(s, P): return dict(mu=s._F(P) @ s.c)

SURR = dict(gp=GPm, tabicl=Tab, xgb=XGB, pce=PCE, pck=PCK, svmc=SVMC, svr=SVRm, nn=NNm, rsm=RSM, agn=GPm)


# --------------------------------------------------------------------------------------------- UQ sources
def cq(scores, alpha=0.1):
    """Split-conformal quantile ceil((n+1)(1-alpha))/n."""
    s = np.sort(np.asarray(scores)); n = len(s); return float(s[min(n, int(np.ceil((n + 1) * (1 - alpha)))) - 1])

def oof(model, X, y, K=5):
    from sklearn.model_selection import KFold
    out = {}; K = min(K, len(X))
    for tr, va in KFold(K, shuffle=True, random_state=0).split(X):
        o = model.clone().fit(X[tr], y[tr]).predict(X[va])
        for k, v in o.items(): out.setdefault(k, np.empty(len(X)))[va] = v
    return out

class UQ:
    def __init__(s, k, seed): s.k, s.seed, s.q, s.members, s.r = k, seed, None, None, None
    def calibrate(s, model, X, y, refresh):
        s.X = X
        if not (refresh or s.q is None and s.members is None): return
        if s.k in ("kfold", "cqra", "cqrm"):
            o = oof(model, X, y); s.r = np.abs(y - o["mu"])
            if s.k == "kfold": s.q = cq(s.r / (cKDTree(X).query(X, k=2)[0][:, 1] + 1e-9))
            elif s.k == "cqra": s.q = cq(np.maximum(o["q05"] - y, y - o["q95"]))
            else: s.q = cq(np.abs(y - o["q50"]) / np.maximum((o["q95"] - o["q05"]) / 2, 1e-12))
        if s.k == "boot":
            rng = np.random.default_rng(s.seed + len(X))
            s.members = [model.clone().fit(X[b], y[b]) for b in (rng.integers(0, len(X), len(X)) for _ in range(5))]
    def sd(s, o, P):
        k = s.k
        if k in ("gpsig", "bq"): return o["sd"]
        if k == "native": return o["sd"]
        if k == "kfold": return np.maximum(s.q * (cKDTree(s.X).query(P, k=1)[0] + 1e-9) / 1.645, 1e-12)
        if k == "cqra": return np.maximum((o["q95"] - o["q05"] + 2 * s.q) / 3.29, 1e-12)
        if k == "cqrm": return np.maximum(s.q * (o["q95"] - o["q05"]) / 2 / 1.645, 1e-12)
        if k == "boot":
            M = np.array([m.predict(P)["mu"] for m in s.members]); return np.maximum(M.std(0), 1e-9 * (np.abs(M).mean() + 1))
        return None


# --------------------------------------------------------------------------------------------- learning functions (all >= 0, argmax)
def ranjan(mu, s, a=2.0):
    """Ranjan et al. 2008 expected improvement for the contour g = 0, eps = a s (closed form, checked by quadrature in test_engine)."""
    d = -mu / s; t1, t2 = d - a, d + a
    return s * s * ((a * a - d * d - 1) * (norm.cdf(t2) - norm.cdf(t1)) - 2 * d * (norm.pdf(t2) - norm.pdf(t1))
                    + t2 * norm.pdf(t2) - t1 * norm.pdf(t1))

def reif(mu, s):
    """E[max(0, |mu| - |G|)], G ~ N(mu, s^2): expected improvement of the folded normal |G| toward the limit state.
    ponytail: re-derived from the REIF abstract (Zhang, Wang & Sorensen 2019); check the paper's closed form before citing."""
    m = np.abs(mu)
    def seg(a, b, c, k):                                   # int_a^b (c + k g) N(g; m, s^2) dg
        ta, tb = (a - m) / s, (b - m) / s; dP = norm.cdf(tb) - norm.cdf(ta); dp = norm.pdf(tb) - norm.pdf(ta)
        return c * dP + k * (m * dP - s * dp)
    return np.maximum(seg(0, m, m, -1) + seg(-m, 0, m, 1), 0.0)

def utility(k, y, sd, P, Xtr, rng, c):
    if k == "amb": return sd / (np.abs(y) + 1e-300)                  # same ranking as min U, no underflow (A8)
    if k == "ucb": return sd * np.exp(-0.5 * (y / sd) ** 2)
    if k in ("eff", "effc"): return H.eff(y / c, sd / c)
    if k == "cont": return ranjan(y / c, sd / c)
    if k == "hls": return H.hfun(y / c, sd / c)
    if k == "erf": return reif(y / c, sd / c)
    if k == "pbw": return norm.cdf(-(y - 1.645 * sd)) - norm.cdf(-(y + 1.645 * sd))
    if k == "maxmin":                                               # EDSD: farthest from the data among the 2% closest to the boundary
        a = np.abs(y); near = a <= np.quantile(a, min(1.0, max(0.02, 50 / len(a)))); u = np.zeros(len(y))
        u[near] = cKDTree(Xtr).query(P[near], k=1)[0]; return u
    return rng.random(len(y))                                       # rand / naq


# --------------------------------------------------------------------------------------------- dimension handling
class DR:
    def __init__(s, k, seed): s.k, s.seed, s.Wp, s.pls, s.enc, s.gp = k, seed, None, None, None, None
    def tf(s, X):
        if s.k == "full" or (s.Wp is None and s.pls is None and s.enc is None): return X
        if s.k == "pls": return s.pls.transform(X)
        if s.k == "ae": W0, b0, W1, b1 = s.enc; return np.tanh(np.tanh(X @ W0 + b0) @ W1 + b1)
        return X @ s.Wp
    def update(s, X, y, Xpop, rng):
        d = X.shape[1]
        if s.k == "pls":
            from sklearn.cross_decomposition import PLSRegression
            s.pls = PLSRegression(min(3, d, len(X) - 1), scale=False).fit(X, y)
        elif s.k == "ae" and s.enc is None:                       # unsupervised, once, on 20000 population inputs
            from sklearn.neural_network import MLPRegressor
            A = Xpop[rng.choice(len(Xpop), min(20000, len(Xpop)), replace=False)]; r = min(3, d - 1)
            m = MLPRegressor((32, r, 32), activation="tanh", max_iter=300, random_state=s.seed).fit(A, A)
            s.enc = (m.coefs_[0], m.intercepts_[0], m.coefs_[1], m.intercepts_[1])
        elif s.k in ("assub", "fidr"):                            # gradients of an auxiliary full-space Kriging mean
            s.gp = H.fit_gp(X, y, s.gp if s.gp is not None and s.gp.X_train_.shape[1] == d else None)
            A = Xpop[rng.choice(len(Xpop), min(400, len(Xpop)), replace=False)]; h = 1e-3
            G = np.column_stack([(s.gp.predict(A + h * e) - s.gp.predict(A - h * e)) / (2 * h) for e in np.eye(d)])
            wt = np.ones(len(A))
            if s.k == "fidr":                                     # weight phi(t)^2/Phi(t): |grad log I_eps|^2 under h* ~ I_eps phi
                s0 = 0.2 * y.std() + 1e-12; t = -s.gp.predict(A) / s0; wt = norm.pdf(t) ** 2 / np.maximum(norm.cdf(t), 1e-300)
            Cm = (G * wt[:, None]).T @ G / max(wt.sum(), 1e-300); ev, V = np.linalg.eigh(Cm); ev, V = ev[::-1], V[:, ::-1]
            r = int(np.searchsorted(np.cumsum(ev) / max(ev.sum(), 1e-300), 0.95) + 1); s.Wp = V[:, :max(1, min(r, d))]
    def dims(s, d):
        if s.k == "full": return d
        return s.pls.n_components if s.pls is not None else (s.Wp.shape[1] if s.Wp is not None else (s.enc[2].shape[1] if s.enc else d))


# --------------------------------------------------------------------------------------------- populations
class Pop:
    """Points Z (full u-space incl. z_S), DB row or -1 (live), IS weights (None = 1), optional per-point level shift (SS)."""
    def __init__(s, Z, rows=None, w=None, shift=None, est=None):
        s.Z, s.w, s.shift, s.est = Z, w, shift, est; s.rows = np.full(len(Z), -1) if rows is None else rows
        s.used = np.zeros(len(Z), bool)

def wmean(v, w): return float(np.mean(v if w is None else v * w))

def _rays(gf, d, rng, n=1000, rounds=4):
    """Root of the predicted margin along rays from the origin; directions refined around the best rays. -> roots (k x d) sorted by norm."""
    D = rng.standard_normal((n, d)); best = None; t = np.linspace(0, 8, 41)
    for _ in range(rounds):
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        m = gf((D[:, None, :] * t[None, :, None]).reshape(-1, d))[0].reshape(len(D), len(t))
        hit = m <= 0; first = np.where(hit.any(1), hit.argmax(1), -1); r = np.full(len(D), np.inf)
        for i in np.nonzero(first > 0)[0]:
            j = first[i]; r[i] = t[j - 1] + (t[j] - t[j - 1]) * m[i, j - 1] / (m[i, j - 1] - m[i, j])
        r[first == 0] = 0.0
        R = D[np.isfinite(r)] * r[np.isfinite(r)][:, None]
        best = R if best is None else np.vstack([best, R])
        if len(best): best = best[np.argsort(np.linalg.norm(best, axis=1))][:200]
        top = best[:20] if len(best) else rng.standard_normal((20, d))
        D = np.repeat(top, n // len(top) + 1, 0)[:n]; D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9) + 0.3 * rng.standard_normal((n, d))
    if best is None or not len(best): return rng.standard_normal((1, d)) / np.sqrt(d) * 3.0       # no predicted failure: a 3-sigma guess
    return best

def _lognorm(U): return -0.5 * (U * U).sum(1) - 0.5 * U.shape[1] * np.log(2 * np.pi)

def build_isc(R, gf, rng, disjoint):
    roots = _rays(gf, R.d, rng); C = roots[:1]
    if disjoint and len(roots) >= 6:
        from sklearn.cluster import KMeans
        lab = KMeans(3, n_init=3, random_state=0).fit_predict(roots / np.linalg.norm(roots, axis=1, keepdims=True).clip(1e-9))
        C = np.array([roots[lab == k][0] for k in range(3) if (lab == k).any()])
        C = C[np.linalg.norm(C, axis=1) <= np.linalg.norm(C, axis=1).min() + 1.5]
    R.label_live(C[0])                                               # AK-IS: the true model at the design point
    n = 10000; U = C[rng.integers(0, len(C), n)] + rng.standard_normal((n, R.d))
    lh = np.logaddexp.reduce([_lognorm(U - c) for c in C], axis=0) - np.log(len(C))
    return Pop(U, w=np.exp(_lognorm(U) - lh))

def build_rad(R, gf, rng):
    r = 0.95 * float(np.linalg.norm(_rays(gf, R.d, rng)[0])); n = 20000; d = R.d
    D = rng.standard_normal((n, d)); D /= np.linalg.norm(D, axis=1, keepdims=True)
    rad = np.sqrt(chi2.isf(rng.random(n) * chi2.sf(r * r, d), d))
    return Pop(D * rad[:, None], w=np.full(n, chi2.sf(r * r, d)))

def _sir_target(gf, rng, d, M=50000, s0=1.5):
    """Samples of q0 = N(0, s0^2 I) with weights pi(u) phi(u)/q0(u), pi = Phi(-mu/sd)."""
    Q = s0 * rng.standard_normal((M, d)); m, sd = gf(Q); pi = norm.cdf(-m / sd)
    la = np.log(np.maximum(pi, 1e-300)) + _lognorm(Q) - (_lognorm(Q / s0) - d * np.log(s0)); a = np.exp(la - la.max())
    return Q, a, la, pi

def build_meta(R, gf, rng):
    Q, a, la, _ = _sir_target(gf, rng, R.d); peps = float(np.mean(np.exp(la)))
    U = Q[rng.choice(len(Q), 10000, p=a / a.sum())]; m, sd = gf(U); pi = norm.cdf(-m / sd)
    return Pop(U, w=peps / np.maximum(pi, 1e-300))

def build_ce(R, gf, rng, disjoint, n=5000):
    from sklearn.mixture import GaussianMixture
    k = 3 if disjoint else 1; gm = None
    for _ in range(12):
        U = rng.standard_normal((n, R.d)) if gm is None else gm.sample(n)[0]
        lh = _lognorm(U) if gm is None else gm.score_samples(U)
        m = gf(U)[0]; b = max(float(np.quantile(m, 0.1)), 0.0); e = m <= b
        w = np.exp(_lognorm(U[e]) - lh[e]); E = U[e][rng.choice(e.sum(), 2000, p=w / w.sum())]
        gm = GaussianMixture(k, reg_covar=1e-4, random_state=0).fit(E)
        if b == 0: break
    U = gm.sample(10000)[0]; rng.shuffle(U)
    return Pop(U, w=np.exp(_lognorm(U) - gm.score_samples(U)))

def build_sis(R, gf, rng, n=2000, delta=1.5):
    """Sequential IS (Papaioannou et al. 2016) on the surrogate: tempered smoothed indicator Phi(-m/sigma_j), pCN moves."""
    U = rng.standard_normal((n, R.d)); m = gf(U)[0]; sp = np.inf; logZ = np.log(0.5)   # Z_0 = E_phi[pi_0], pi_0 = 1/2
    lp = lambda m_, s_: np.full(len(m_), np.log(0.5)) if not np.isfinite(s_) else norm.logcdf(-m_ / s_)
    for _ in range(30):
        cv = lambda s_: (lambda r: r.std() / max(r.mean(), 1e-300))(np.exp(lp(m, s_) - lp(m, sp)))
        lo, hi = 1e-6 * (np.abs(m).max() + 1e-12), (np.abs(m).max() + 1e-12) * 1e3 if not np.isfinite(sp) else sp
        for _ in range(60):
            mid = np.sqrt(lo * hi)
            if cv(mid) > delta: lo = mid
            else: hi = mid
        s_ = hi; lw = lp(m, s_) - lp(m, sp); W = np.exp(lw - lw.max()); logZ += float(np.log(W.mean()) + lw.max())
        U = U[rng.choice(n, n, p=W / W.sum())]; m = gf(U)[0]
        for _ in range(3):                                   # pCN, reversible w.r.t. phi; accept on the tempered indicator
            V = 0.8 * U + 0.6 * rng.standard_normal(U.shape); mv = gf(V)[0]
            acc = np.log(rng.random(n)) < lp(mv, s_) - lp(m, s_); U[acc], m[acc] = V[acc], mv[acc]
        sp = s_; fin = (m <= 0) / np.exp(lp(m, s_))
        if fin.std() / max(fin.mean(), 1e-300) <= delta: break
    return Pop(U, w=np.exp(logZ - lp(gf(U)[0], sp)))

def build_nis(R, gf, rng):
    """Neural IS: a RealNVP flow fitted by maximum likelihood to SIR samples of h* ~ pi phi; 10% defensive N(0, I)."""
    import torch; torch.set_num_threads(1); torch.manual_seed(R.seed); d = R.d
    Q, a, _, _ = _sir_target(gf, rng, d); X = torch.tensor(Q[rng.choice(len(Q), 5000, p=a / a.sum())], dtype=torch.float32)
    masks = [torch.tensor([(i + j) % 2 for i in range(d)], dtype=torch.float32) for j in range(6)]
    nets = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.Tanh(), torch.nn.Linear(64, 2 * d)) for _ in masks])
    for n_ in nets: torch.nn.init.zeros_(n_[2].weight); torch.nn.init.zeros_(n_[2].bias)
    def fwd(x):                                          # x -> z, log|det|
        ld = torch.zeros(len(x))
        for mk, net in zip(masks, nets):
            st = net(x * mk); s_, t_ = torch.tanh(st[:, :d]) * (1 - mk), st[:, d:] * (1 - mk)
            x = x * mk + (1 - mk) * (x * torch.exp(s_) + t_); ld = ld + s_.sum(1)
        return x, ld
    def inv(z):
        for mk, net in zip(masks[::-1], nets[::-1]):
            st = net(z * mk); s_, t_ = torch.tanh(st[:, :d]) * (1 - mk), st[:, d:] * (1 - mk)
            z = z * mk + (1 - mk) * (z - t_) * torch.exp(-s_)
        return z
    logp = lambda x: (lambda z, ld: -0.5 * (z * z).sum(1) - 0.5 * d * np.log(2 * np.pi) + ld)(*fwd(x))
    opt_ = torch.optim.Adam(nets.parameters(), 1e-3)
    for _ in range(400): opt_.zero_grad(); loss = -logp(X).mean(); loss.backward(); opt_.step()
    with torch.no_grad():
        n = 10000; nf = int(0.9 * n); U = np.vstack([inv(torch.randn(nf, d)).numpy().astype(float), rng.standard_normal((n - nf, d))])
        lf = logp(torch.tensor(U, dtype=torch.float32)).numpy().astype(float)
    lh = np.logaddexp(np.log(0.9) + lf, np.log(0.1) + _lognorm(U))
    return Pop(U, w=np.exp(_lognorm(U) - lh))

def build_strat(R, rng):
    from scipy.stats import qmc
    n = min(R.W.N, 50000); U = norm.ppf(np.clip(qmc.LatinHypercube(R.d, seed=R.seed).random(n), 1e-12, 1 - 1e-12))
    return Pop(U)

def build_line(R, gf, rng, NL=500):
    a = _rays(gf, R.d, rng)[0]; a = a / np.linalg.norm(a); t = np.linspace(-3, 8, 45)
    X = rng.standard_normal((NL, R.d)); X -= (X @ a)[:, None] * a
    m = gf((X[:, None, :] + t[None, :, None] * a).reshape(-1, R.d))[0].reshape(NL, len(t))
    c = np.full(NL, np.inf); hit = m <= 0; first = np.where(hit.any(1), hit.argmax(1), -1)
    for i in np.nonzero(first > 0)[0]:
        j = first[i]; c[i] = t[j - 1] + (t[j] - t[j - 1]) * m[i, j - 1] / (m[i, j - 1] - m[i, j])
    c[first == 0] = t[0]; ok = np.isfinite(c); pf = float(norm.cdf(-c).mean())
    P = Pop(X[ok] + c[ok][:, None] * a if ok.any() else X[:1] + 3 * a); P.est = pf; return P

def _mma(rng, seeds, mfn, b, n_total, w=1.0):            # modified Metropolis, as al_run._mma
    nc = len(seeds); L = int(np.ceil(n_total / nc)); cur = seeds.copy(); out = [cur.copy()]
    for _ in range(L - 1):
        xi = cur + rng.uniform(-w, w, cur.shape); acc = rng.random(cur.shape) < np.exp(-0.5 * (xi ** 2 - cur ** 2))
        cand = np.where(acc, xi, cur); ok = mfn(cand) <= b; cur = np.where(ok[:, None], cand, cur); out.append(cur.copy())
    return np.vstack(out)[:n_total]

def build_ss(R, gf, rng, Ns=2000, p0=0.1):
    rows = rng.choice(R.W.N, Ns, replace=False); S = R.W.Z[rows]; Zs, Rs, Bs = [], [], []; k = 0
    while True:
        m = gf(S)[0]; b = max(float(np.quantile(m, p0)), 0.0); Zs.append(S); Rs.append(rows); Bs.append(np.full(len(S), b))
        if b == 0 or k >= 8: pf = float(p0 ** k * (m <= 0).mean()); break
        S = _mma(rng, S[m <= b], lambda U: gf(U)[0], b, Ns); rows = np.full(Ns, -1); k += 1
    P = Pop(np.vstack(Zs), np.concatenate(Rs), shift=np.concatenate(Bs)); P.est = pf; return P


# --------------------------------------------------------------------------------------------- the run
class Run:
    def __init__(s, db, path, seed, budget, pf, prog=None):
        s.p = {**DEF, **path}; s.db, s.seed, s.B, s.pft = db, seed, budget, pf
        s.rng = np.random.default_rng(seed + 2024); s.W = H.World(db, pf, pop_path(db)); s.O = Oracle(s.W, db)
        s.ds, s.d = s.W.ds, s.W.d; s.prog = open(prog, "a") if prog else None; s.t0 = time.time()
        s.TZ, s.TG, s.lab_rows = [], [], set(); s.traj = []; s.notes = []; s.nfit = 0
        s.model = SURR.get(s.p["surr"], GPm)(seed); s.uq = UQ(s.p["uq"], seed); s.dr = DR(s.p["dim"], seed)
        s.LZ, s.LG, s.LGd = [], [], {}                                          # low-fidelity data
    # ---- io
    def log(s, **k):
        k.update(t=round(time.time() - s.t0, 1), n=round(s.O.cost(), 2))
        if s.prog: s.prog.write(json.dumps(k, default=float) + "\n"); s.prog.flush()
    # ---- labels
    def label_row(s, j):
        if j in s.lab_rows: return
        s.lab_rows.add(int(j)); s.TZ.append(s.W.Z[j]); s.TG.append(s.O.hf(None, int(j)))
    def label_live(s, u):
        if s.O.cost() + 1 > s.B + 1e-9: return False
        g = s.O.hf(u)
        if np.isfinite(g): s.TZ.append(np.array(u, float)); s.TG.append(g); return True
        return False                                                            # non-converged: dropped (HARNESS lesson)
    def label(s, P, j):
        P.used[j] = True
        return s.label_row(int(P.rows[j])) if P.rows[j] >= 0 else s.label_live(P.Z[j])
    # ---- model
    def xb(s, Z): return Z[:, :s.ds] if s.p["tgt"] == "beta" else Z
    def xin(s, Z): return s.dr.tf(s.xb(Z))
    def margin(s, mu, Z): return mu - Z[:, -1] if s.p["tgt"] == "beta" else mu
    def fit(s):
        Z, g = np.array(s.TZ), np.array(s.TG); y = target_y(s.p["tgt"], Z, g)
        if s.p["dim"] != "full" and (s.nfit % 5 == 0): s.dr.update(s.xb(Z), y, s.xb(s.W.Z), s.rng)
        X = s.xin(Z); s.Xtr, s.ytr = X, y
        if s.p["fid"] != "sf": s.fit_mf(Z, y)
        else: s.model.fit(X, y)
        s.uq.calibrate(s.model, X, y, refresh=(s.nfit % 5 == 0)); s.nfit += 1
    def pred(s, Z):
        P = s.xin(Z); o = s.model.predict(P); sd = s.uq.sd(o, P)
        return o["mu"], sd, o.get("q50", o["mu"]), o
    def gf(s, Z):                                                               # margin-equivalent mean, sd (samplers)
        mu, sd, _, _ = s.pred(Z); return s.margin(mu, Z), (sd if sd is not None else np.full(len(Z), 1e-3 * (np.abs(mu).std() + 1e-12)))
    # ---- multi-fidelity (GP only): AR1 co-Kriging or hierarchical Kriging on a 4x coarser step
    def fit_mf(s, Z, y):
        LZ = np.array(s.LZ); XL, yL = s.xin(LZ), target_y(s.p["tgt"], LZ, np.array(s.LG)); X = s.xin(Z)
        s.gL = H.fit_gp(XL, yL, getattr(s, "gL", None)); muL = s.gL.predict(X)
        if s.p["fid"] in ("cok", "adapf"):                                      # rho by least squares on LF values at the HF points
            yLh = np.array([target_y(s.p["tgt"], z[None], np.array([s.LGd[tuple(np.round(z, 12))]]))[0] if tuple(np.round(z, 12)) in s.LGd else m
                            for z, m in zip(Z, muL)])
        else: yLh = muL                                                         # hierarchical Kriging: LF Kriging mean as trend
        s.rho = float(np.dot(yLh, y) / max(np.dot(yLh, yLh), 1e-300))
        s.gD = H.fit_gp(X, y - s.rho * yLh, getattr(s, "gD", None)); R = s
        class MF:
            sd = True
            def predict(_, P):
                a, sa = H.gp_pred(R.gL, P); b, sb = H.gp_pred(R.gD, P)
                v = sb ** 2 + (R.rho ** 2 * sa ** 2 if R.p["fid"] != "rom" else 0)
                return dict(mu=R.rho * a + b, sd=np.sqrt(v), sL=abs(R.rho) * sa, sD=sb)
            def clone(_): return GPm(R.seed, R.gD.kernel_)
        s.model = MF()
    def label_lf(s, z):
        if s.O.cost() + LF_COST > s.B + 1e-9: return
        g = s.O.lf(z)
        if np.isfinite(g): s.LZ.append(np.array(z, float)); s.LG.append(g); s.LGd[tuple(np.round(z, 12))] = g
    # ---- readout
    def readout(s, P):
        if P.est is not None: return dict(pf=float(P.est))
        k = s.p["read"]; Z = P.Z; mu, sd, q50, o = s.pred(Z); beta = s.p["tgt"] == "beta"; out = {}; s._full = (mu, sd, o)
        if k in ("hard", "isw"): v = s.margin(mu, Z) <= 0
        elif k == "med": v = s.margin(q50, Z) <= 0
        elif k == "bag":
            mem = s.uq.members or [s.model.clone().fit(s.Xtr[b], s.ytr[b]) for b in (s.rng.integers(0, len(s.Xtr), len(s.Xtr)) for _ in range(5))]
            Pin = s.xin(Z); v = np.mean([s.margin(m.predict(Pin)["mu"], Z) <= 0 for m in mem], axis=0)
        elif k == "rb":
            if beta: v = norm.cdf(-q50)                                        # P(S > R | x_s) = Phi(-beta), exact
            else:                                       # demand integrated exactly: root z* of mu(x_s, z) = 0 on a z grid, P = Phi(-z*)
                zg = np.array([-2, 0, 1.5, 2.5, 3.25, 4, 5, 6.5]); M = np.empty((len(Z), len(zg)))   # ponytail: assumes g falls with demand
                for i, z in enumerate(zg): Zq = Z.copy(); Zq[:, -1] = z; M[:, i] = s.pred(Zq)[0]
                hit = M <= 0; f = np.where(hit.any(1), hit.argmax(1), -1); zs = np.full(len(Z), np.inf); zs[f == 0] = -np.inf
                i = np.nonzero(f > 0)[0]; j = f[i]; a, b = M[i, j - 1], M[i, j]
                zs[i] = zg[j - 1] + (zg[j] - zg[j - 1]) * a / (a - b); v = norm.cdf(-zs)
        else:                                                                   # bayes
            pi = norm.cdf(-mu / np.sqrt(1 + sd ** 2)) if beta else norm.cdf(-s.margin(mu, Z) / sd); v = pi
            n = len(Z); w2 = 1 if P.w is None else P.w ** 2; var = float(np.sum(w2 * pi * (1 - pi)) / n ** 2)
            out["pf_ci95"] = [wmean(pi, P.w) - 1.96 * np.sqrt(var), wmean(pi, P.w) + 1.96 * np.sqrt(var)]
        out["pf"] = wmean(v.astype(float), P.w); s._last = (mu, sd, Z, P); return out
    # ---- stopping
    def stop_fires(s, y_c, sd_c, ut, pf, P):
        k = s.p["stop"]
        if k == "budget" or pf is None: return False
        if k == "lft":
            if s.p["acq"] in ("eff", "effc"): return float(ut.max()) < 1e-3
            return float((np.abs(y_c) / sd_c).min()) >= 2.0
        mu, sd, Z, _ = s._last; m = s.margin(mu, Z); w = P.w; n = len(Z)
        if k == "pfcov":
            vmc = float(np.var((m <= 0) * (1 if w is None else w)) / n)
            vs = 0.0 if sd is None else float(np.sum((1 if w is None else w ** 2) * (lambda p_: p_ * (1 - p_))(norm.cdf(-m / sd))) / n ** 2)
            return pf > 0 and np.sqrt(vmc + vs) / pf < 0.05
        if k == "err":                                                          # ESC-type bound (Wang & Shafieezadeh 2019 pattern)
            pw = norm.cdf(-np.abs(m) / sd); f = m <= 0; Nf = float(f.sum())
            Sf = pw[f].sum() + 1.96 * np.sqrt((pw[f] * (1 - pw[f])).sum()); Ss = pw[~f].sum() + 1.96 * np.sqrt((pw[~f] * (1 - pw[~f])).sum())
            eps = max(abs(Nf / max(Nf - Sf, 1e-12) - 1), abs(Nf / (Nf + Ss) - 1)) if Nf > 0 else np.inf
            return eps < 0.05
        if k == "conf":                                                         # conformal band on the margin, 90%
            r = s.uq.r if s.uq.r is not None else np.abs(s.ytr - oof(s.model, s.Xtr, s.ytr)["mu"]); q = cq(r)
            lo, hi = wmean((m + q <= 0).astype(float), w), wmean((m - q <= 0).astype(float), w)
            return pf > 0 and (hi - lo) <= 0.1 * pf
        return False
    # ---- candidates and batch
    def candidates(s, P):
        free = np.nonzero(~P.used & ~np.isin(P.rows, list(s.lab_rows)))[0] if len(s.lab_rows) else np.nonzero(~P.used)[0]
        b = s.p["batch"]
        if b == "cand": free = s.rng.choice(free, min(len(free), max(6000, int(np.ceil(20 / s.pft)))), replace=False)
        if b == "trunc" and s.traj:
            bh = -norm.ppf(min(max(s.traj[-1][1] or 0, 1 / len(P.Z)), 0.5)); keep = np.linalg.norm(P.Z[free], axis=1) >= bh - 1
            free = free[keep] if keep.sum() >= 10 else free
        return free
    def choose(s, ut, U, Pc, nmax):
        b = s.p["batch"]; q = 1 if b in ("single", "trunc") else 8
        if b == "adapt": q = int(np.clip(round((U < 2).sum() / 25), 1, 16))
        q = max(1, min(q, nmax))
        if b == "kmeans":
            from sklearn.cluster import KMeans
            pool = np.nonzero(U < 2)[0]
            if len(pool) < 8 * q: pool = np.argsort(U)[:8 * q]
            if len(pool) <= q: return list(pool[:q])
            lab = KMeans(q, n_init=3, random_state=s.seed).fit_predict(Pc[pool])
            return [int(pool[lab == k][np.argmin(U[pool[lab == k]])]) for k in range(q) if (lab == k).any()]
        if b == "parreg":
            reg = U < 2; ut = np.where(reg, ut, 0.0) if reg.sum() >= q else ut
        if q == 1: return [int(np.argmax(ut))] if ut.max() > 0 else []
        rho = cKDTree(s.Xtr).query(Pc, k=1)[0] + 1e-9; u = ut.astype(float).copy(); out = []
        for _ in range(q):                                                      # local penalisation (A5-A12)
            j = int(np.argmax(u))
            if u[j] <= 0: break
            out.append(j); dd = np.linalg.norm(Pc - Pc[j], axis=1); u *= 1 - np.exp(-0.5 * (dd / rho[j]) ** 2); u[j] = -1
        return out
    # ---- bqv lookahead (al_run.run_bal) on the candidate set
    def bqv_pick(s, y, sd, Pc, q):
        gp = s.model.m if isinstance(s.model, GPm) else s.model.g.m
        pi = norm.cdf(-y / sd); v = pi * (1 - pi); pool = np.argsort(-v)[:2000]; sh = pool[:24]
        if v[pool[0]] <= 0: return [int(pool[0])]
        c = H.post_cov(gp, Pc[pool], Pc[sh]); sd2 = np.sqrt(np.maximum(sd[pool][:, None] ** 2 - c ** 2 / sd[sh][None, :] ** 2, 1e-24))
        p2 = norm.cdf(-y[pool][:, None] / sd2); red = (v[pool][:, None] - p2 * (1 - p2)).sum(0)
        return [int(sh[j]) for j in np.argsort(-red)[:q]]
    # ---- population factory
    def make_pop(s):
        k, gf = s.p["pop"], s.gf; dj = s.p["ls"] == "disjoint"
        if k == "mc": return Pop(s.W.Z, np.arange(s.W.N))
        if k == "strat": return build_strat(s, s.rng)
        if k == "isc": return build_isc(s, gf, s.rng, dj)
        if k == "rad": return build_rad(s, gf, s.rng)
        if k == "meta": return build_meta(s, gf, s.rng)
        if k == "ce": return build_ce(s, gf, s.rng, dj)
        if k == "sis": return build_sis(s, gf, s.rng)
        if k == "nis": return build_nis(s, gf, s.rng)
        if k == "line": return build_line(s, gf, s.rng)
        return build_ss(s, gf, s.rng)
    def rebuild_every(s): return {"mc": 10 ** 9, "strat": 10 ** 9, "line": 1, "ss": 10}.get(s.p["pop"], 25)

    # ---- main loop
    def run(s):
        p = s.p
        if p["surr"] == "none": return s.finish(s.direct())
        if p["surr"] == "agn": return s.finish(s.mfis())
        for j in H.init_design(s.W, s.seed): s.label_row(j)
        if p["fid"] != "sf":
            for z in s.W.Z[s.rng.choice(s.W.N, 3 * N0, replace=False)]: s.label_lf(z)
            for z in s.TZ: s.label_lf(z)
        if p["acq"] == "naq":                                                   # one-shot design, one fit
            P = s.make_pop() if p["pop"] in ("mc", "strat") else None; s.fit()
            P = P or s.make_pop(); free = s.candidates(P); s.rng.shuffle(free)
            for j in free:
                if s.O.cost() >= s.B: break
                s.label(P, j)
            s.fit(); P = s.make_pop() if p["pop"] not in ("mc", "strat") else P
            r = s.readout(P); s.traj.append([s.O.cost(), r["pf"]]); s.log(pf=r["pf"]); return s.finish(r)
        s.fit(); P = s.make_pop(); last_build = len(s.TZ); stop = None; stalls = 0
        arms = ["amb", "eff", "hls", "ucb"]; bandit = np.zeros((4, 2)); arm = None
        while True:
            Pc_idx = s.candidates(P)
            if not len(Pc_idx): break
            big = isinstance(s.model, Tab) and len(P.Z) > 20000
            r = s.readout(P) if (not big or s.nfit % 5 == 1) else None        # TabICL on >20k rows: readout every 5th refit
            if r is not None:
                s.traj.append([s.O.cost(), r["pf"]]); s.log(pf=r["pf"])
                if arm is not None and len(s.traj) > 1:                         # bandit reward = relative change of Pf
                    bandit[arm] += [abs(s.traj[-1][1] - s.traj[-2][1]) / max(s.traj[-2][1], 1.0 / len(P.Z)), 1]
            Zc = P.Z[Pc_idx]
            if r is not None and P.est is None: mu, sd, o = s._full[0][Pc_idx], (None if s._full[1] is None else s._full[1][Pc_idx]), {k: v[Pc_idx] for k, v in s._full[2].items()}
            else: mu, sd, _, o = s.pred(Zc)
            Pc = s.xin(Zc)
            # acquisition variable: the fitted target (boundary 0) on MC/LHS, as in S7 (beta_rb: EFF at beta = 0);
            # the margin-equivalent on IS-type populations; margin minus the level threshold in subset simulation
            if P.shift is not None: y = s.margin(mu, Zc) - P.shift[Pc_idx]
            elif p["pop"] in ("mc", "strat"): y = mu
            else: y = s.margin(mu, Zc)
            sdc = sd if sd is not None else np.ones(len(y)); c = float(np.median(np.abs(s.ytr))) + 1e-12
            k = p["acq"]
            if k == "rl":
                n_ = bandit[:, 1]; arm = int(np.argmin(n_)) if (n_ == 0).any() else int(np.argmax(bandit[:, 0] / n_ + np.sqrt(2 * np.log(n_.sum()) / n_)))
                k = arms[arm]
            ut = utility(k, y, sdc, Pc, s.Xtr, s.rng, c) if k != "bqv" else None
            if stop is None and p["stop"] != "budget":
                if r is None and p["stop"] == "lft" and s.stop_fires(y, sdc, ut if ut is not None else np.zeros(1), 0.0, P): r = s.readout(P)
                if r is not None and s.stop_fires(y, sdc, ut if ut is not None else np.zeros(1), r["pf"], P):
                    stop = dict(n_FE_stop=s.O.cost(), pf_stop=r["pf"]); s.log(msg=f"stopping rule fired at {s.O.cost():.0f} FE"); break
            if s.O.cost() >= s.B - 1e-9: break
            U = np.abs(y) / sdc; nmax = int(max(1, s.B - s.O.cost()))
            pick = s.bqv_pick(y, sdc, Pc, 1 if p["batch"] == "single" else 8) if k == "bqv" else s.choose(ut, U, Pc, nmax)
            if not pick: pick = [int(np.argmax(cKDTree(s.Xtr).query(Pc, k=1)[0]))]   # stall guard: farthest candidate from the data
            c0 = s.O.cost()
            lf_only = p["fid"] == "adapf" and len(s.TZ) > N0 and o["sL"][pick[0]] > o["sD"][pick[0]]   # experimental level choice
            for j in pick:
                if lf_only: P.used[Pc_idx[j]] = True; s.label_lf(Zc[j])
                else: s.label(P, int(Pc_idx[j]))
            if p["fid"] != "sf":
                if p["fid"] != "rom" and not lf_only:
                    for j in pick: s.label_lf(Zc[j])                            # nested design for AR1 co-Kriging
                for j in (np.argsort(-ut)[len(pick):len(pick) + 3] if ut is not None else []): s.label_lf(Zc[j])
            stalls = stalls + 1 if s.O.cost() == c0 else 0
            if stalls >= 3: s.notes.append("stopped: three refits without a new label"); break
            s.fit()
            if len(s.TZ) - last_build >= s.rebuild_every(): P = s.make_pop(); last_build = len(s.TZ)
        if p["pop"] in ("ss", "line", "isc", "rad", "meta", "ce", "sis", "nis"): P = s.make_pop()     # final estimator on the final surrogate
        out = s.readout(P); s.traj.append([s.O.cost(), out["pf"]]); s.log(pf=out["pf"], msg="final")
        if stop: out.update(stop)
        return s.finish(out)

    # ---- surrogate-free samplers (true model; DB rows are lookups, everything else live)
    def direct(s):
        k = s.p["pop"]; B = s.B; d = s.d; rng = s.rng; O = s.O
        g = lambda u: O.hf(u)
        if k == "mc":
            rows = np.random.default_rng(s.seed + 31).choice(s.W.N, B, replace=False); O.n_lookup += B
            return dict(pf=float((s.W.g[rows] <= 0).mean()))
        if k == "strat":
            from scipy.stats import qmc
            U = norm.ppf(np.clip(qmc.LatinHypercube(d, seed=s.seed).random(B), 1e-12, 1 - 1e-12)); v = np.array([g(u) for u in U])
            return dict(pf=float((v[np.isfinite(v)] <= 0).mean()))
        if k == "ss":
            Ns, p0 = max(50, B // 4), 0.1; rows = rng.choice(s.W.N, Ns, replace=False); S = s.W.Z[rows]; G = s.W.g[rows]; O.n_lookup += Ns; lev = 0
            while True:
                b = max(float(np.quantile(G, p0)), 0.0)
                if b == 0: return dict(pf=float(p0 ** lev * (G <= 0).mean()), levels=lev + 1)
                sd_ = np.argsort(G)[:max(1, int(p0 * Ns))]; S, G = S[sd_], G[sd_]; outS, outG = [S], [G]; cur, cg = S.copy(), G.copy()
                while sum(len(a) for a in outS) < Ns:
                    if O.cost() >= B: s.notes.append("subset simulation truncated by the budget: Pf from an incomplete level"); return dict(pf=float(p0 ** (lev + 1) * (np.concatenate(outG) <= 0).mean()), levels=lev + 2)
                    xi = cur + rng.uniform(-1, 1, cur.shape); acc = rng.random(cur.shape) < np.exp(-0.5 * (xi ** 2 - cur ** 2)); cand = np.where(acc, xi, cur)
                    gc = np.array([g(u) for u in cand]); ok = np.isfinite(gc) & (gc <= b); cur[ok], cg[ok] = cand[ok], gc[ok]; outS.append(cur.copy()); outG.append(cg.copy())
                S, G = np.vstack(outS)[:Ns], np.concatenate(outG)[:Ns]; lev += 1
        u = np.zeros(d)                                                         # HL-RF FORM, forward differences (h = 0.1)
        for _ in range(5):
            if O.cost() + d + 1 > B / 2: break
            g0 = g(u); gr = np.array([(g(u + 0.1 * e) - g0) / 0.1 for e in np.eye(d)])
            if not (np.isfinite(g0) and np.isfinite(gr).all()): break
            un = (gr @ u - g0) / (gr @ gr) * gr
            if np.linalg.norm(un - u) < 1e-2: u = un; break
            u = un
        beta = float(np.linalg.norm(u)); n = int(B - O.cost()); s.notes.append(f"FORM beta = {beta:.2f}")
        if n < 4: return dict(pf=float(norm.cdf(-beta)), note="FORM only")
        if k == "isc":
            U = u + rng.standard_normal((n, d)); v = np.array([g(x) for x in U]); ok = np.isfinite(v)
            return dict(pf=float(np.mean(((v <= 0) * np.exp(-U @ u + 0.5 * u @ u))[ok])))
        if k == "rad":
            r = 0.9 * beta; D = rng.standard_normal((n, d)); D /= np.linalg.norm(D, axis=1, keepdims=True)
            U = D * np.sqrt(chi2.isf(rng.random(n) * chi2.sf(r * r, d), d))[:, None]; v = np.array([g(x) for x in U]); ok = np.isfinite(v)
            return dict(pf=float(chi2.sf(r * r, d) * (v[ok] <= 0).mean()))
        if k == "line":                                                         # 3-point root per line (secant), ~4 calls per line
            a = u / max(beta, 1e-9); NL = max(1, n // 4); c = []
            for _ in range(NL):
                x = rng.standard_normal(d); x -= (x @ a) * a; t0, t1 = beta, beta + 1.0; f0, f1 = g(x + t0 * a), g(x + t1 * a)
                for _ in range(2):
                    if not (np.isfinite(f0) and np.isfinite(f1)) or f1 == f0: break
                    t0, t1, f0 = t1, t1 - f1 * (t1 - t0) / (f1 - f0), f1; f1 = g(x + t1 * a)
                if np.isfinite(t1): c.append(t1)
            return dict(pf=float(norm.cdf(-np.array(c)).mean()), lines=len(c))
        # ce: single Gaussian, rho = 0.1, n per level = B/6
        m_, C_ = np.zeros(d), np.eye(d); nl = max(20, B // 6)
        while True:
            m_left = int(B - O.cost())
            Ln = min(nl, m_left); U = rng.multivariate_normal(m_, C_, Ln); v = np.array([g(x) for x in U]); ok = np.isfinite(v); U, v = U[ok], v[ok]
            lw = _lognorm(U) - (lambda D_: -0.5 * np.einsum("ij,jk,ik->i", D_, np.linalg.inv(C_), D_) - 0.5 * np.log(np.linalg.det(2 * np.pi * C_)))(U - m_)
            b = max(float(np.quantile(v, 0.1)), 0.0)
            if b == 0 or O.cost() + nl > B:
                return dict(pf=float(np.mean((v <= 0) * np.exp(lw))))
            e = v <= b; w = np.exp(lw[e]); m_ = (w[:, None] * U[e]).sum(0) / w.sum(); D_ = U[e] - m_
            C_ = (w[:, None, None] * np.einsum("ij,ik->ijk", D_, D_)).sum(0) / w.sum() + 1e-6 * np.eye(d)

    def mfis(s):
        """MFIS (Peherstorfer et al. 2016): half the budget trains the GP with the chosen acquisition on the DB, the GP builds the
        biasing density, the other half is true-model IS from it, so surrogate error costs variance, not bias."""
        B, s.B = s.B, s.B // 2; p = dict(s.p); s.p.update(surr="gp", pop="mc", read="hard")
        for j in H.init_design(s.W, s.seed): s.label_row(j)
        s.fit(); M = Pop(s.W.Z, np.arange(s.W.N))
        while s.O.cost() < s.B:
            idx = s.candidates(M); Zc = s.W.Z[idx]; mu, sd, _, _ = s.pred(Zc); c = float(np.median(np.abs(s.ytr))) + 1e-12
            ut = utility(p["acq"] if p["acq"] not in ("bqv", "naq", "rl") else "amb", mu, sd, s.xin(Zc), s.Xtr, s.rng, c)
            for j in s.choose(ut, np.abs(mu) / sd, s.xin(Zc), int(s.B - s.O.cost())) or [int(np.argmax(ut))]: s.label_row(int(idx[j]))
            s.fit(); s.traj.append([s.O.cost(), None])
        s.p.update(pop=p["pop"]); s.B = B; P = s.make_pop(); s.p = p
        n = int(B - s.O.cost()); J = s.rng.choice(len(P.Z), min(n, len(P.Z)), replace=False); v, wv = [], []
        for j in J:
            gv = s.O.hf(P.Z[j])
            if np.isfinite(gv): v.append(gv <= 0); wv.append(P.w[j])
        v, wv = np.array(v, float), np.array(wv)
        pf = float(np.mean(v * wv)); se = float(np.std(v * wv) / np.sqrt(len(v)))
        return dict(pf=pf, pf_ci95=[pf - 1.96 * se, pf + 1.96 * se], n_is=len(v))

    def finish(s, r):
        W = s.W; pf = r["pf"]
        rel = lambda a, b: float((a - b) / b * 100) if b > 0 else float("nan")
        out = dict(db=s.db, db_name=DBS[s.db]["name"], path=s.p, seed=s.seed, budget=s.B, pf_target=s.pft, N=W.N, d=W.d,
                   pf_rb=W.pf_rb, pf_rb_cov=W.rb_cov(), pf_count=W.pf_emp, n_fail_count=W.n_fail,
                   err_rb_pct=rel(pf, W.pf_rb), err_count_pct=rel(pf, W.pf_emp), n_FE=round(s.O.cost(), 2),
                   n_lookup=s.O.n_lookup, n_live=s.O.n_live, n_live_nonconv=s.O.n_bad, t_live_s=round(s.O.t_live, 1),
                   n_LF=s.O.n_lf, n_LF_nonconv=s.O.n_lf_bad, t_LF_s=round(s.O.t_lf, 1), n_train=len(s.TZ),
                   input_dims=s.dr.dims(s.xb(s.W.Z[:1]).shape[1]) if s.p["dim"] != "full" else None,
                   traj=s.traj, notes=s.notes, wall_s=round(time.time() - s.t0, 1), **r)
        if "pf_stop" in r: out["err_rb_pct_stop"] = rel(r["pf_stop"], W.pf_rb)
        return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--db", required=True, choices=list(DBS)); ap.add_argument("--path", default="{}")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--budget", type=int, default=300); ap.add_argument("--pf", type=float, default=1e-3)
    ap.add_argument("--out"); ap.add_argument("--progress"); a = ap.parse_args()
    path = json.loads(a.path); E, Wn = check(path, a.db)
    if E: print(json.dumps(dict(error=E))); sys.exit(2)
    r = Run(a.db, path, a.seed, a.budget, a.pf, a.progress).run(); r["warnings"] = Wn
    s = json.dumps(r, default=float)
    if a.out: open(a.out, "w").write(s)
    print(json.dumps({k: v for k, v in r.items() if k != "traj"}, default=float))

if __name__ == "__main__":
    main()
