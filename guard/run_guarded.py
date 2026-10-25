"""Run ONE command under a memory/time watchdog (macOS has no cgroups, so we poll).
Kills the whole process group if: process-tree RSS > --rss-gb, OR system available memory < --free-gb,
OR swap used grows by > --swap-gb since start, OR wall time > --wall-s. Prints one JSON summary line.
usage: python3 guard/run_guarded.py [--rss-gb 12] [--free-gb 10] [--swap-gb 3] [--wall-s 1800] [--log FILE] -- cmd ...
exit: 0 ok, 97 killed by guard, else the child's exit code.
"""
import argparse, json, os, signal, subprocess, sys, time
import psutil

ap = argparse.ArgumentParser(); ap.add_argument("--rss-gb", type=float, default=12); ap.add_argument("--free-gb", type=float, default=10)
ap.add_argument("--swap-gb", type=float, default=3); ap.add_argument("--wall-s", type=float, default=1800); ap.add_argument("--log")
ap.add_argument("cmd", nargs=argparse.REMAINDER); a = ap.parse_args()
cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
GB = 1 << 30
env = dict(os.environ, OMP_NUM_THREADS="1", PYTORCH_MPS_HIGH_WATERMARK_RATIO=os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.4"),
           PYTORCH_MPS_LOW_WATERMARK_RATIO=os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.3"))
out = open(a.log, "ab") if a.log else None
t0 = time.time(); swap0 = psutil.swap_memory().used
p = subprocess.Popen(cmd, stdout=out or None, stderr=subprocess.STDOUT if out else None, env=env, start_new_session=True)
peak, minfree, reason = 0.0, 1e9, None
def kill(r):
    global reason; reason = r
    try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except ProcessLookupError: pass
while p.poll() is None:
    try:
        ps = psutil.Process(p.pid); rss = ps.memory_info().rss + sum(c.memory_info().rss for c in ps.children(recursive=True))
    except psutil.Error: rss = 0
    free = psutil.virtual_memory().available; swap = psutil.swap_memory().used - swap0
    peak = max(peak, rss / GB); minfree = min(minfree, free / GB)
    if rss > a.rss_gb * GB: kill(f"rss {rss/GB:.1f}GB > {a.rss_gb}")
    elif free < a.free_gb * GB: kill(f"system available {free/GB:.1f}GB < {a.free_gb}")
    elif swap > a.swap_gb * GB: kill(f"swap grew {swap/GB:.1f}GB > {a.swap_gb}")
    elif time.time() - t0 > a.wall_s: kill(f"wall {time.time()-t0:.0f}s > {a.wall_s}")
    if reason: break
    time.sleep(0.5)
p.wait()
print(json.dumps(dict(cmd=" ".join(cmd)[:120], status="killed: " + reason if reason else ("ok" if p.returncode == 0 else f"exit {p.returncode}"),
                      peak_rss_gb=round(peak, 2), min_available_gb=round(minfree, 1), secs=round(time.time() - t0))), flush=True)
sys.exit(97 if reason else p.returncode)
