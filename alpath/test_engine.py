"""Checks: closed forms vs quadrature, conformal quantile, then a smoke run of every implemented cell (one axis changed from a
valid base path) on one database at a small budget.  python3 test_engine.py [db] [budget]   (run it through guard/run_guarded.py)"""
import sys, json, time
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import engine as E

def formulas():
    for mu, s in [(0.3, 0.5), (-1.2, 0.4), (2.0, 1.5), (0.0, 0.2)]:
        f = lambda y: norm.pdf(y, mu, s)
        ref = quad(lambda y: max((2 * s) ** 2 - y * y, 0) * f(y), -2 * s, 2 * s)[0]
        assert abs(E.ranjan(np.array(mu), np.array(s)) - ref) < 1e-6 * max(1, ref), (mu, s)
        ref = quad(lambda y: max(abs(mu) - abs(y), 0) * f(y), -abs(mu) - 1e-12, abs(mu) + 1e-12)[0]
        assert abs(E.reif(np.array(mu), np.array(s)) - ref) < 1e-6, (mu, s, E.reif(np.array(mu), np.array(s)), ref)
    assert E.cq(np.arange(1, 11)) == 10 and E.cq(np.arange(1, 101)) == 91
    print("formulas ok")

BASES = {   # a valid base per axis, so each cell is tested in a context where it is allowed
    "tgt": {}, "read": {}, "batch": {}, "stop": {}, "dim": {}, "ls": dict(pop="isc"),
    "surr": dict(uq="boot"), "uq": dict(surr="tabicl", batch="cand"), "acq": {}, "pop": {}, "fid": {},
}
FIX = {   # per-cell adjustments needed for validity
    ("surr", "gp"): dict(uq="gpsig"), ("surr", "pck"): dict(uq="gpsig"), ("surr", "tabicl"): dict(uq="native", batch="cand"),
    ("surr", "svmc"): dict(uq="na", acq="maxmin"), ("surr", "none"): dict(uq="na", acq="naq"), ("surr", "agn"): dict(uq="gpsig", pop="ce", read="isw"),
    ("uq", "gpsig"): dict(surr="gp", batch="single"), ("uq", "bq"): dict(surr="gp", batch="single"), ("uq", "na"): dict(surr="xgb", acq="rand", batch="single"),
    ("uq", "kfold"): dict(surr="gp", batch="single"), ("uq", "boot"): dict(surr="xgb", batch="single"),
    ("acq", "effc"): dict(surr="tabicl", uq="cqra", batch="cand"), ("acq", "pbw"): dict(tgt="beta"),
    ("read", "isw"): dict(pop="isc"), ("ls", "composed"): {}, ("stop", "conf"): dict(uq="kfold"),
}

def smoke(db, B):
    rows = []
    for a in E.AXES:
        for o in a["opts"]:
            if o["na"]: rows.append((a["id"], o["k"], "n/a", o["na"])); continue
            path = {**BASES[a["id"]], a["id"]: o["k"], **FIX.get((a["id"], o["k"]), {})}
            err, _ = E.check(path, db)
            if err: rows.append((a["id"], o["k"], "INVALID", "; ".join(err))); continue
            t = time.time()
            try:
                r = E.Run(db, path, 0, B, 1e-3).run()
                assert np.isfinite(r["pf"]) and r["pf"] >= 0 and r["n_FE"] <= B + 1e-6, r
                rows.append((a["id"], o["k"], "ok", f"pf {r['pf']:.2e} err_rb {r['err_rb_pct']:+.0f}% FE {r['n_FE']} live {r['n_live']} LF {r['n_LF']} {time.time() - t:.0f}s"))
            except Exception as e:
                import traceback; rows.append((a["id"], o["k"], "FAIL", repr(e)[:300])); traceback.print_exc()
            print(*rows[-1], flush=True)
    return rows

if __name__ == "__main__":
    formulas()
    db = sys.argv[1] if len(sys.argv) > 1 else "578"; B = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    if only: E.AXES = [a for a in E.AXES if a["id"] in only]
    rows = smoke(db, B)
    bad = [r for r in rows if r[2] in ("FAIL", "INVALID")]
    print(json.dumps(dict(n=len(rows), ok=sum(r[2] == "ok" for r in rows), bad=bad), indent=1))
    sys.exit(1 if bad else 0)
