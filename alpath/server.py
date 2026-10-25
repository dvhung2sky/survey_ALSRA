"""Local GUI for the AL path engine.  python3 server.py [--port 8765]  ->  http://127.0.0.1:8765

Pick a database and one cell per axis, press Run: each seed runs `engine.py` under guard/run_guarded.py (RSS cap 12 GB),
ONE job at a time (laptop RAM rule; TabICL jobs rebooted it before).  Results land in runs/<job>_s<seed>.json.
"""
import os, sys, json, glob, time, threading, subprocess, signal, argparse, uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import engine as E
import numpy as np

RUNS = os.path.join(HERE, "runs"); os.makedirs(RUNS, exist_ok=True)
GUARD = next((p for p in (os.path.join(os.path.dirname(HERE), "guard", "run_guarded.py"), os.path.join(ROOT, "guard", "run_guarded.py")) if os.path.exists(p)), "guard/run_guarded.py")
S7_METHODS = "mc rand_krig akmcs egra akmcs_h akmcsi akss akis bal bal_lit altab beta_rb".split()

def s7_tables():
    """Per database: the 12 S7 methods at 300 calls, 10 seeds (bench/out); mean |err| vs Pf_RB, own-stop error and calls."""
    dbs, s7 = [], {}
    for k, v in E.DBS.items():
        d = os.path.join(E.BENCH, "out", v["dir"]); ref = json.load(open(f"{d}/akmcs_s0.json")); s7[k] = {}
        dbs.append(dict(id=k, name=v["name"], dir=v["dir"], N=ref["N"], d=ref["d"], pf_rb_cov=ref["pf_rb_cov"], n_fail=ref["n_fail_emp"],
                        lf=v["lf"] is not None, t_call=E.DB_TCALL[k]))
        for m in S7_METHODS:
            R = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/{m}_s[0-9].json"))]
            if not R: continue
            e = [abs(r["err_rb_pct_pf"]) for r in R]; st = [abs(r["err_rb_pct_pf_stop"]) for r in R if "err_rb_pct_pf_stop" in r]
            s7[k][m] = dict(n=len(R), err=float(np.mean(e)), err_max=float(np.max(e)), fe=float(np.mean([r["n_FE"] for r in R])),
                            wall=float(np.mean([r["wall_s"] for r in R])), err_stop=float(np.mean(st)) if st else None,
                            fe_stop=float(np.mean([r["n_FE_stop"] for r in R if "n_FE_stop" in r])) if st else None)
    return dbs, s7

DBS, S7 = s7_tables()
META = dict(axes=E.AXES, default=E.DEF, presets={k: dict(path={**E.DEF, **v[0]}, note=v[1]) for k, v in E.PRESETS.items()},
            dbs=DBS, s7=S7, lf_cost=E.LF_COST)

JOBS, QUEUE, LOCK = {}, [], threading.Lock()

def worker():
    while True:
        with LOCK: jid = QUEUE[0] if QUEUE else None
        if jid is None: time.sleep(0.5); continue
        J = JOBS[jid]; J["status"] = "running"
        for S in J["seeds"]:
            if J["status"] == "cancelled": break
            S["status"] = "running"; base = os.path.join(RUNS, f"{jid}_s{S['seed']}")
            cmd = [sys.executable, GUARD, "--rss-gb", "12", "--free-gb", "8", "--wall-s", "7200", "--log", base + ".log", "--",
                   sys.executable, os.path.join(HERE, "engine.py"), "--db", J["db"], "--path", json.dumps(J["path"]), "--seed", str(S["seed"]),
                   "--budget", str(J["budget"]), "--pf", str(J["pf"]), "--out", base + ".json", "--progress", base + ".prog.jsonl"]
            S["proc"] = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, cwd=HERE)
            out = S["proc"].communicate()[0].decode(errors="replace").strip().splitlines()
            S["guard"] = out[-1] if out else ""; S.pop("proc", None)
            if os.path.exists(base + ".json"): S["status"] = "done"; S["result"] = json.load(open(base + ".json"))
            else:
                S["status"] = "cancelled" if J["status"] == "cancelled" else "failed"
                S["error"] = "".join(open(base + ".log", errors="replace").readlines()[-25:]) if os.path.exists(base + ".log") else S["guard"]
        if J["status"] != "cancelled": J["status"] = "done"
        with LOCK: QUEUE.pop(0)

def progress(jid, seed, n=400):
    f = os.path.join(RUNS, f"{jid}_s{seed}.prog.jsonl")
    if not os.path.exists(f): return []
    L = open(f).read().splitlines(); return [json.loads(x) for x in L[-n:] if x.strip()]

def history():
    out = []
    for f in sorted(glob.glob(os.path.join(RUNS, "*_s*.json")), key=os.path.getmtime, reverse=True)[:60]:
        try: r = json.load(open(f))
        except Exception: continue
        out.append({k: r.get(k) for k in ("db", "path", "seed", "budget", "pf_target", "pf", "err_rb_pct", "err_count_pct", "n_FE", "n_live", "wall_s", "n_FE_stop", "err_rb_pct_stop")}
                   | dict(file=os.path.basename(f)))
    return out

class Hd(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, obj, code=200, ctype="application/json"):
        b = obj if isinstance(obj, bytes) else json.dumps(obj, default=float).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def body(self): return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"): return self.send(open(os.path.join(HERE, "index.html"), "rb").read(), ctype="text/html; charset=utf-8")
        if p == "/logo.png": return self.send(open(os.path.join(HERE, "logo.png"), "rb").read(), ctype="image/png")
        if p == "/api/meta": return self.send(META)
        if p == "/api/runs": return self.send(history())
        if p.startswith("/api/job/"):
            jid = p.rsplit("/", 1)[1]; J = JOBS.get(jid)
            if not J:                                             # finished job from an earlier server session: rebuild from runs/
                R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RUNS, f"{os.path.basename(jid)}_s*.json")))]
                if not R: return self.send(dict(error="no such job"), 404)
                J = dict(status="done", db=R[0]["db"], path=R[0]["path"], budget=R[0]["budget"], pf=R[0]["pf_target"],
                         seeds=[dict(seed=r["seed"], status="done", result=r) for r in R])
            with LOCK: qpos = QUEUE.index(jid) if jid in QUEUE else -1
            return self.send(dict(id=jid, status=J["status"], queue_pos=qpos, db=J["db"], path=J["path"], budget=J["budget"], pf=J["pf"],
                                  seeds=[{k: v for k, v in S.items() if k != "proc"} | dict(progress=progress(jid, S["seed"]))
                                         for S in J["seeds"]]))
        self.send(dict(error="not found"), 404)
    def do_POST(self):
        p = self.path; b = self.body()
        if p == "/api/check":
            db, path = b.get("db", "578"), b.get("path", {}); e, w = E.check(path, db, True)
            return self.send(dict(errors=[dict(msg=m, axes=list(a)) for m, a in e], warnings=w, options=E.option_state(path, db)))
        if p == "/api/run":
            db, path = b["db"], {**E.DEF, **b["path"]}; e, _ = E.check(path, db)
            if e: return self.send(dict(error=e), 400)
            seeds = [int(x) for x in b.get("seeds", [0])][:20]; budget = int(b.get("budget", 300)); pf = float(b.get("pf", 1e-3))
            if not (12 < budget <= 2000) or pf not in (1e-2, 1e-3, 1e-4): return self.send(dict(error=["budget 13-2000, Pf 1e-2 / 1e-3 / 1e-4"]), 400)
            jid = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
            JOBS[jid] = dict(status="queued", db=db, path=path, budget=budget, pf=pf, seeds=[dict(seed=s, status="queued") for s in seeds])
            with LOCK: QUEUE.append(jid)
            return self.send(dict(id=jid))
        if p.startswith("/api/cancel/"):
            J = JOBS.get(p.rsplit("/", 1)[1])
            if not J: return self.send(dict(error="no such job"), 404)
            J["status"] = "cancelled"
            for S in J["seeds"]:
                if S.get("proc"):
                    try: os.killpg(os.getpgid(S["proc"].pid), signal.SIGKILL)
                    except ProcessLookupError: pass
            with LOCK:
                jid = p.rsplit("/", 1)[1]
                if jid in QUEUE and QUEUE[0] != jid: QUEUE.remove(jid)
            return self.send(dict(ok=True))
        self.send(dict(error="not found"), 404)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765); a = ap.parse_args()
    threading.Thread(target=worker, daemon=True).start()
    print(f"AL path GUI on http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Hd).serve_forever()
