"""Field-wide title-based count of AL structural-reliability papers (Crossref). Run: python3 trend_scan.py
Finding: with cursor paging Crossref ignores query.bibliographic, so the cursor pass enumerates EVERY
journal-article of each ISSN in 2011-2026 (exhaustive within the journal). The 5 queries are run as
offset-paged relevance searches (top 1000 each) only as a recall cross-check against the enumeration."""
import json, os, re, time, csv, random, html, urllib.request, urllib.parse, collections
W = os.path.dirname(os.path.abspath(__file__)); RAW = f"{W}/trend_raw"; os.makedirs(RAW, exist_ok=True)
J = {"RESS":"0951-8320","StructSafety":"0167-4730","PEM":"0266-8920","SMO":"1615-147X","CMAME":"0045-7825",
     "MSSP":"0888-3270","CompStruct":"0045-7949","EngStruct":"0141-0296","ASCE-ASME-A":"2376-7642",
     "ASCE-ASME-B":"2332-9017","JMechDes":"1050-0472"}
Q = ["active learning reliability","adaptive Kriging reliability","learning function reliability",
     "Kriging failure probability","surrogate active learning failure probability"]
BASE = "https://api.crossref.org/journals/%s/works?filter=from-pub-date:2011-01-01,until-pub-date:2026-12-31,type:journal-article&rows=1000&select=DOI,title,issued,container-title"
def get(url, cache):
    if os.path.exists(cache): return json.load(open(cache))
    for k in range(8):
        time.sleep(0.3)
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent":"survey-trend-scan/1.0"}), timeout=90))
            json.dump(d, open(cache, "w")); return d
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500: time.sleep(5*2**k); continue
            raise
        except Exception: time.sleep(5*2**k)
    raise RuntimeError(url)
def recs(d):
    for it in d["message"]["items"]:
        t = html.unescape(" ".join(it.get("title") or [])); y = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
        if t and y: yield it["DOI"].lower(), t, y
allr, qr = {}, {}   # doi -> (journal, title, year)
for jn, issn in J.items():
    cur, p = "*", 0
    while True:
        d = get(BASE % issn + "&cursor=" + urllib.parse.quote(cur), f"{RAW}/{issn}_cur{p:03d}.json")
        its = list(recs(d))
        for doi, t, y in its: allr[doi] = (jn, t, y)
        nxt = d["message"].get("next-cursor"); p += 1
        if not its or not nxt: break
        cur = nxt
    for qi, q in enumerate(Q):
        d = get(BASE % issn + "&query.bibliographic=" + urllib.parse.quote_plus(q), f"{RAW}/{issn}_q{qi}.json")
        for doi, t, y in recs(d): qr[doi] = (jn, t, y)
    print(jn, sum(1 for v in allr.values() if v[0] == jn), flush=True)
AL = re.compile(r"active[- ]learning|adaptive[- ]kriging|adaptive surrogate|learning function|sequential (sampling|design)|adaptive sampling|Bayesian active|enrichment", re.I)
ALc = re.compile(r"AK-[A-Z]|AK[A-Z]{2,}")
RE = re.compile(r"reliabilit|failure probabilit|rare event|limit state|fragility", re.I)
keep = lambda t: bool((AL.search(t) or ALc.search(t)) and RE.search(t))
kept = {d: v for d, v in allr.items() if keep(v[1])}
rej = {d: v for d, v in allr.items() if d not in kept}
qmiss = {d: v for d, v in qr.items() if keep(v[1]) and d not in allr}
cnt = collections.Counter((v[0], v[2]) for v in kept.values())
with open(f"{W}/trend_counts.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["journal", "year", "n"])
    for (j, y), n in sorted(cnt.items(), key=lambda x: (list(J).index(x[0][0]), x[0][1])): w.writerow([j, y, n])
# coverage
R = os.path.dirname(W) + "/refs"
bib = open(f"{R}/references.bib").read()
entries = re.split(r"\n(?=@)", bib); key2doi = {}
for e in entries:
    m = re.match(r"@\w+\{([^,]+),", e); dm = re.search(r"doi\s*=\s*[{\"]([^}\"]+)", e)
    if m and dm: key2doi[m.group(1)] = dm.group(1).strip().lower()
bibdois = set(key2doi.values())
corp = {r["key"] for r in csv.DictReader(open(f"{R}/corpus_map.csv"))}
corpdois = {key2doi[k] for k in corp if k in key2doi}
cited = {d for d in kept if d in bibdois}; incorp = {d for d in kept if d in corpdois}
def yrs(S, a=2021): return sum(1 for d in S if kept[d][2] >= a)
cy = collections.Counter(kept[d][2] for d in cited); ky = collections.Counter(v[2] for v in kept.values())
json.dump({"cited": sorted(cited)}, open(f"{W}/trend_cited.json", "w"))
with open(f"{W}/trend_cited_by_year.csv", "w") as f:
    f.write("year,n_cited\n"); [f.write(f"{y},{cy.get(y,0)}\n") for y in range(2011, 2027)]
random.seed(1)
def sample(D, n=30): return random.sample(sorted(D.items()), min(n, len(D)))
with open(f"{W}/trend_scan_check.md", "w") as f:
    f.write("# Trend scan precision check (seed 1)\n\n## 30 random KEPT titles\n\n")
    for d, v in sample(kept): f.write(f"- [{v[0]} {v[2]}] {v[1]}\n")
    f.write("\n## 30 random REJECTED titles (all journals, uniform over rejected pool)\n\n")
    for d, v in sample(rej): f.write(f"- [{v[0]} {v[2]}] {v[1]}\n")
    f.write("\n## 30 random REJECTED titles from relevance queries only (enriched pool, near-misses)\n\n")
    qrej = {d: v for d, v in qr.items() if d not in kept}
    for d, v in sample(qrej): f.write(f"- [{v[0]} {v[2]}] {v[1]}\n")
tot = collections.Counter(v[0] for v in kept.values())
S = open(f"{W}/trend_stats.json", "w")
json.dump({"n_all": len(allr), "n_kept": len(kept), "n_query_unique": len(qr), "query_kept_not_in_enum": len(qmiss),
           "per_journal_all": collections.Counter(v[0] for v in allr.values()), "per_journal_kept": tot,
           "per_year_kept": sorted(ky.items()), "bib_dois": len(bibdois), "corpus_dois": len(corpdois),
           "cited": len(cited), "cited_2021_26": yrs(cited), "kept_2021_26": yrs(kept), "in_corpus": len(incorp),
           "in_corpus_2021_26": yrs(incorp), "cited_by_year": sorted(cy.items())}, S, indent=1); S.close()
print(open(f"{W}/trend_stats.json").read())
