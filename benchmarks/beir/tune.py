"""Sets glossdex's text profile ONCE, on development splits only, then freezes it.

    python tune.py          # writes text_profile.json (refuses to overwrite a frozen one)

Development data: three collections that are NOT test collections, from other domains (SciDocs:
scientific papers; ArguAna: debate arguments; CQADupStack webmasters: technical Q&A), each with
no-answer probes drawn from another development collection. No document, query or label of the
test collections (SciFact, NFCorpus, FiQA) is seen here: the settings are chosen once, on
content they will not be tested on.

What may change: only the result-set decision (noise floor, evidence gate, band fraction,
headroom, minimum results). Ranking stays exactly as shipped, so nDCG is unaffected. The objective
is the same task score the report uses (F2 on answerable queries, 1/0 on probes), averaged with
equal weight per collection. Ties go to the setting closest to the shipped profile.

The chosen profile is written with the SHA-256 of the dev query lists and a UTC timestamp; after
that, evaluate.py test applies it unchanged to every test collection.
"""
import datetime
import hashlib
import itertools
import json
import os
import sys
from dataclasses import asdict, replace

import evaluate as ev

GRID = {
    "noise_floor": [0.20, 0.24, 0.28, 0.32],
    "gate": [0.32, 0.36, 0.40, 0.44, 0.48, 0.52, 0.56, 0.60],
    "frac": [0.60, 0.70, 0.80, 0.90],
    "headroom": [0.20, 0.30, 0.40],
    "min_results": [1, 2],
}
OUT = os.path.join(ev.HERE, "text_profile.json")


def distance(fields):
    base = asdict(ev.EMBEDDINGGEMMA)
    return sum(abs(fields[k] - base[k]) / (max(GRID[k]) - min(GRID[k])) for k in fields)


def main():
    if os.path.exists(OUT) and "--force" not in sys.argv:
        sys.exit(f"{OUT} is frozen; it was set once. Delete it by hand only if the dev protocol changed.")
    names = list(ev.SPLITS["dev"])
    cols = {n: ev.Collection(n) for n in set(names) | {ev.PROBE_FROM[n] for n in names}}
    sets = {n: ev.query_set(cols[n], "dev", cols[ev.PROBE_FROM[n]]) for n in names}
    keys = list(GRID)
    combos = [dict(zip(keys, c)) for c in itertools.product(*(GRID[k] for k in keys))]
    profs = [replace(ev.EMBEDDINGGEMMA, **f) for f in combos]
    # Query-major: each query is ranked once and its keyword evidence (which depends only on the
    # query) is computed once, then all settings are evaluated on it. Same results as looping over
    # settings, just without recomputing the same lookups 768 times.
    rows = [{n: {} for n in names} for _ in combos]
    for n in names:
        lex = cols[n].index.lexical
        originals = {m: getattr(lex, m) for m in ("bm25", "idf_coverage", "items_matching_all_terms",
                                                  "items_matching_at_least")}
        for qi, q in enumerate(sets[n]):
            memo = {}
            for m, fn in originals.items():
                setattr(lex, m, (lambda fn, m: lambda *a: memo[(m, a)] if (m, a) in memo
                                 else memo.setdefault((m, a), fn(*a)))(fn, m))
            cache = {}
            for ci, prof in enumerate(profs):
                rows[ci][n].update(ev.evaluate_glossdex(cols[n], [q], prof, cache))
            if qi % 250 == 0:
                print(f"{n}: {qi}/{len(sets[n])} queries", flush=True)
        for m, fn in originals.items():
            setattr(lex, m, fn)
    results = []
    for fields, r in zip(combos, rows):
        per = {n: ev.summarize(r[n], sets[n]) for n in names}
        score = sum(per[n]["task"] for n in names) / len(names)
        results.append((score, -distance(fields), fields, per))
    results.sort(key=lambda r: (round(r[0], 4), r[1]), reverse=True)
    best = results[0]
    shipped = next(r for r in results if all(r[2][k] == getattr(ev.EMBEDDINGGEMMA, k) for k in keys))
    qhash = hashlib.sha256(json.dumps({n: [q[0] for q in sets[n]] for n in names}, sort_keys=True).encode()).hexdigest()
    doc = {"fields": best[2], "frozen_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "dev_queries_sha256": qhash, "dev": {n: ev.SPLITS["dev"][n] for n in names}, "probes": {n: ev.PROBE_FROM[n] for n in names},
           "objective": "mean over collections of task score (F2 answerable, 1/0 probes)",
           "dev_task_text": best[0], "dev_task_shipped": shipped[0], "grid": GRID,
           "dev_detail": {"text": best[3], "shipped": shipped[3]},
           "top10": [[r[0], r[2]] for r in results[:10]]}
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
    print(f"\nshipped profile on dev: {shipped[0]:.4f}\ntext profile on dev:    {best[0]:.4f}  {best[2]}\nfrozen -> {OUT}")


if __name__ == "__main__":
    main()
