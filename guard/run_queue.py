"""Sequential job queue: skip-if-done, memory preflight, circuit breaker, RUNTIME PROJECTION and tail trimming. Never parallel.
jobs file: one job per line:  <done-file or -> | <command>
usage: python3 guard/run_queue.py jobs.txt --est-s 150 --stop-at 10:00 [--rss-gb 12 --free-gb 12 --wall-s 600 --max-fail 2 --max-hours 2]
Projection: pending x est-s (then median of observed durations). Refuses to start if projected finish > stop-at or > max-hours;
after every job re-projects and stops early (trims the tail) if the projected finish would pass stop-at. Trimmed jobs are listed.
"""
import argparse, datetime as dt, json, os, shlex, statistics, subprocess, sys, time
import psutil
ap = argparse.ArgumentParser(); ap.add_argument("jobs"); ap.add_argument("--est-s", type=float, required=True); ap.add_argument("--stop-at", required=True)
ap.add_argument("--rss-gb", default="12"); ap.add_argument("--free-gb", default="12"); ap.add_argument("--wall-s", default="600")
ap.add_argument("--max-fail", type=int, default=2); ap.add_argument("--swap-gb", default="3"); ap.add_argument("--max-hours", type=float, default=2.0); a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
jobs = []
for line in open(a.jobs):
    line = line.strip()
    if line and not line.startswith("#"):
        done, cmd = [s.strip() for s in line.split("|", 1)]
        if done == "-" or not os.path.exists(done): jobs.append((done, cmd))
now = dt.datetime.now(); h, m = map(int, a.stop_at.split(":")); stop = now.replace(hour=h, minute=m, second=0, microsecond=0)
proj = len(jobs) * a.est_s
print(f"PROJECTION: {len(jobs)} pending x {a.est_s:.0f}s = {proj/60:.0f} min -> finish ~{(now + dt.timedelta(seconds=proj)).strftime('%H:%M')} (stop-at {a.stop_at}, max {a.max_hours} h)", flush=True)
if proj > a.max_hours * 3600:
    print(f"REFUSED: projected {proj/3600:.1f} h exceeds --max-hours {a.max_hours}. Cut the job list or revise the strategy.", flush=True); sys.exit(3)
durs, fails, trimmed = [], 0, []
for i, (done, cmd) in enumerate(jobs):
    est = statistics.median(durs) if durs else a.est_s
    eta = dt.datetime.now() + dt.timedelta(seconds=est * (len(jobs) - i))
    if eta > stop and i > 0:
        trimmed = [c for _, c in jobs[i:]]; print(f"TRIM: projected finish {eta.strftime('%H:%M')} > {a.stop_at}; dropping {len(trimmed)} remaining jobs", flush=True); break
    if dt.datetime.now() >= stop: trimmed = [c for _, c in jobs[i:]]; print("stop-at reached", flush=True); break
    for _ in range(60):
        if psutil.virtual_memory().available > (float(a.free_gb) + 4) * (1 << 30): break
        time.sleep(5)
    else: print("ABORT: memory never freed up", flush=True); break
    for attempt in range(3):          # a kill caused by SYSTEM-wide memory (other apps) is retried after the system recovers; own-RSS / wall kills are not
        r = subprocess.run([sys.executable, os.path.join(HERE, "run_guarded.py"), "--rss-gb", a.rss_gb, "--free-gb", a.free_gb, "--swap-gb", a.swap_gb, "--wall-s", a.wall_s, "--", *shlex.split(cmd)],
                           capture_output=True, text=True)
        last = (r.stdout.strip().splitlines() or ["{}"])[-1]
        if r.returncode == 97 and ("system available" in last or "swap" in last) and attempt < 2:
            print(f"[{i+1}/{len(jobs)}] system-memory kill (attempt {attempt+1}); waiting for the system to recover, then retry: {last[:120]}", flush=True)
            for _ in range(120):
                if psutil.virtual_memory().available > (float(a.free_gb) + 8) * (1 << 30): break
                time.sleep(10)
            continue
        break
    try: info = json.loads(last); durs.append(info.get("secs", est))
    except Exception: info = {}
    print(f"[{i+1}/{len(jobs)}] rc={r.returncode} {last[:230]}", flush=True)
    fails = 0 if r.returncode == 0 else fails + 1
    if fails >= a.max_fail: print(f"CIRCUIT BREAKER: {fails} consecutive failures", flush=True); break
print("QUEUE END; trimmed:", trimmed[:20], flush=True)
