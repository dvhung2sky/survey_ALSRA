# Field-wide trend scan: active-learning structural-reliability papers, 2011-2026

Script: `facts/trend_scan.py` (raw JSON cached in `facts/trend_raw/`, 132 files). Run date 2026-10-19; 2026 is a partial year.
Outputs: `trend_counts.csv` (journal, year, n), `trend_cited_by_year.csv`, `trend_cited.json`, `trend_stats.json`, `trend_scan_check.md`.

## Method
- Crossref REST API, `/journals/<ISSN>/works`, filter `from-pub-date:2011-01-01,until-pub-date:2026-12-31,type:journal-article`, `select=DOI,title,issued,container-title`, rows=1000, no e-mail, 0.3 s sleep, 429/5xx backoff.
- Journals (ISSN): RESS 0951-8320; Structural Safety 0167-4730; Probab. Eng. Mech. (PEM) 0266-8920; Struct. Multidiscip. Optim. (SMO) 1615-147X; CMAME 0045-7825; MSSP 0888-3270; Computers & Structures 0045-7949; Engineering Structures 0141-0296; ASCE-ASME J. Risk Uncertainty Eng. Syst. Part A 2376-7642; Part B 2332-9017; J. Mechanical Design 1050-0472.
- Deviation from the brief, found in testing: with `cursor=*` Crossref ignores `query.bibliographic` (deep paging follows deposit order). So the cursor pass enumerates every journal article of each ISSN in the window (60,353 records), which makes the five queries redundant for coverage. The five queries ("active learning reliability", "adaptive Kriging reliability", "learning function reliability", "Kriging failure probability", "surrogate active learning failure probability") were still run as offset-paged relevance searches (top 1000 each, 9,096 unique records) as a cross-check: 0 title-matching records were found by the queries that were not in the enumeration.
- Title filter (both must match): AL = `active[- ]learning | adaptive[- ]kriging | adaptive surrogate | learning function | sequential (sampling|design) | adaptive sampling | Bayesian active | enrichment` (case-insensitive) or `AK-[A-Z] | AK[A-Z]{2,}` (case-sensitive); RE = `reliabilit | failure probabilit | rare event | limit state | fragility` (case-insensitive). Year = first element of `issued`. DOIs lower-cased.

## Results
Kept: **291** papers of 60,353 journal articles. Per journal: RESS 100, SMO 52, Structural Safety 31, PEM 30, CMAME 30, MSSP 17, Computers & Structures 10, ASCE-ASME Part A 9, Engineering Structures 5, J. Mech. Design 5, ASCE-ASME Part B 2.

Per-year total (all journals): 2011 1; 2012 1; 2013 1; 2014 2; 2015 1; 2016 2; 2017 3; 2018 10; 2019 15; 2020 33; 2021 18; 2022 30; 2023 37; 2024 38; 2025 46; 2026 53 (partial, to 19 Oct). 2018-2026 holds 280 of 291 (96%). Per journal x year: `trend_counts.csv`. The 2020 peak (33) then 2021 dip (18) is real in the data (RESS 12 -> 5, SMO 10 -> 4, Structural Safety 7 -> 2); cause not investigated (possible mix of special issues and Crossref issue-date assignment).

## Coverage by the review
- `references.bib` has 349 DOIs; corpus_map.csv maps 134 corpus keys to 134 bib DOIs.
- Kept papers that the review cites (DOI in bib): **93 of 291 (32%)**; 2021-2026: **66 of 222 (30%)**. By year cited: 2011 1, 2014 1, 2016 2, 2017 1, 2018 5, 2019 4, 2020 13, 2021 7, 2022 9, 2023 8, 2024 15, 2025 8, 2026 19.
- Kept papers that are in the coded corpus (corpus DOIs): 21 (all 2021-2026). The corpus is a convenience set and was not built from these journals alone, so the low overlap says the corpus is not a sample of this title-defined population.

## Precision check (`trend_scan_check.md`, by eye, seed 1, n=30 each)
- 30 random kept titles: 0 clear false positives; 4 borderline where the paper is about reliability-based design optimisation or MDO with an adaptive-sampling/surrogate component rather than AL reliability analysis (e.g. RBDO for a hydraulic pipeline with adaptive sampling region; hybrid adaptive surrogate + sampling for MDO). Estimated precision about 87-100% (26/30 firm, 30/30 lenient); the interval from n=30 is wide (roughly +/-10 points).
- 30 random rejected titles drawn uniformly from 60,062 rejected records: 0 false negatives (all are off-topic), which only bounds the global rate (<10%) and says little about recall.
- Targeted recall probe: 389 rejected titles contain a surrogate-type term (kriging, Gaussian process, surrogate, metamodel, neural, PCE, SVM, machine learning) plus a reliability term but no AL term. A random 15 of these: about 5 are AL-type in substance judging from the title (e.g. "surrogate-based time-dependent reliability ... reducing the candidate sample pool"; "reliability analysis ... dependent Kriging predictions and parallel learning strategy"; "adaptive multi-fidelity ensemble of surrogates"), the rest are plain surrogates, degradation or fragility fits. So roughly 100-150 further AL papers may be missed in these 11 journals; recall of the title filter is plausibly 65-75%, and the counts are a lower bound. This estimate is from a 15-title sample and a title-only judgement.

## Limitations
- Title-only: no abstract or full-text check; AL papers without the keywords are missed (see above), RBDO/MDO papers with "adaptive sampling" can slip in.
- Journals are not exhaustive: conference papers, venues such as Computers and Geotechnics, Aerospace Science and Technology, other publishers' mechanics journals and arXiv and other journals publishing AL reliability work are not scanned. Engineering Structures (19,030 articles) contributes only 5, so the filter is not dominated by the largest journal.
- Crossref `issued` year can be the online-first date, and some journals' records are deposited late; 2026 is partial.
- Query recall of Crossref relevance search is unknown; the conclusion rests on the full enumeration, whose completeness equals Crossref's coverage of these ISSNs (total-results per ISSN were taken as returned; not independently verified against the publishers).
- Coverage-of-review uses DOI equality only; bib entries without DOI (and DOIs differing in case/format beyond lower-casing) would be undercounted as not cited.
