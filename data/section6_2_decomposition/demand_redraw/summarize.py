"""Summarise the demand-redraw experiment: margin+indicator U-function arm over 4 population draws (draw 0 = paper's population)."""
import json, glob, csv, statistics as st, math
ROOT = os.environ.get("ALSRA_ROOT", ".")   # folder holding the original experiment scripts
RB = 0.0029142227928298354
rows = {r["arm"]: r for r in csv.DictReader(open(f"{ROOT}/repro_pcjointD3/results/accuracy_16arms_pf3e-03.csv"))}
beta_u = [abs(float(rows["A8-RB"][f"seed{i}_err_pct"])) for i in range(5)]          # independent of the demand draw
draws = {0: [(i, float(rows["A8"][f"seed{i}_err_pct"])) for i in range(5)]}
counts = {0: 3.02}
for f in sorted(glob.glob(f"{ROOT}/survey_ALSRA/exp/out/redraw[1-9]_seed*.json")):
    d = json.load(open(f)); draws.setdefault(d["redraw"], []).append((d["seed"], d["err_vs_rb_pct"])); counts[d["redraw"]] = d["count_vs_rb_pct"]
print(f"draw  count-vs-RB(%)  n  mean|err| vs RB(%)  sd   (beta+RB U-function arm: {st.mean(beta_u):.2f})")
allabs = []
for r in sorted(draws):
    a = [abs(e) for _, e in sorted(draws[r])]; allabs += a
    print(f"{r:4d}  {counts[r]:+11.2f}  {len(a)}  {st.mean(a):14.2f}  {st.pstdev(a):4.2f}")
m = st.mean(allabs); per_draw = [st.mean([abs(e) for _, e in draws[r]]) for r in sorted(draws) if len(draws[r]) == 5]
print(f"pooled over {len(allabs)} runs / {len(draws)} draws: mean |err| {m:.2f}%  -> gap vs beta arm {m - st.mean(beta_u):.2f} points")
if len(per_draw) > 1:
    s = st.stdev(per_draw); n = len(per_draw); t = {2: 12.71, 3: 4.30, 4: 3.18}[n]
    print(f"per-draw means {[round(x, 2) for x in per_draw]}; mean {st.mean(per_draw):.2f}, 95% t-interval over draws "
          f"{st.mean(per_draw) - t * s / math.sqrt(n):.2f}-{st.mean(per_draw) + t * s / math.sqrt(n):.2f}")
