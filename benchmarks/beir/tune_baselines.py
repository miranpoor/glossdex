"""Chooses the score-based baselines' settings ONCE, on the development collections only, then
freezes them, under the same protocol as tune.py.

    python tune_baselines.py      # writes baseline_settings.json (refuses to overwrite a frozen one)

Baselines (all on the same EmbeddingGemma-300M single-vector cosine scores as "dense"):
  dense · threshold          show every document with cosine >= t (at most 50)
  dense · top-k + threshold  show the first k documents with cosine >= t
  adaptive-k                 Taguchi et al., EMNLP 2025; only its buffer B is chosen here (B = 5 as
                             published is also reported, untuned)

Objective: the report's task score (F2 on answerable queries, 1/0 on probes), averaged with equal
weight per development collection. Ties go to the first setting in grid order. No document,
query or label of a test collection is seen here.
"""
import datetime
import hashlib
import json
import os
import sys

import evaluate as ev

T_GRID = [round(0.20 + 0.01 * i, 2) for i in range(61)]   # 0.20 .. 0.80
K_GRID = [5, 10]
B_GRID = list(range(0, 11))
OUT = os.path.join(ev.HERE, "baseline_settings.json")


def best(cols, sets, names, systems):
    """systems: [(fields, f(scores) -> how many)] -> (score, fields, per-collection task)."""
    rows = {n: ev.score_sorted_systems(cols[n], sets[n], {i: fn for i, (_, fn) in enumerate(systems)})
            for n in names}
    out = None
    for i, (fields, _) in enumerate(systems):
        per = {n: ev.summarize(rows[n][i], sets[n])["task"] for n in names}
        score = sum(per.values()) / len(per)
        if out is None or score > out[0] + 1e-12:
            out = (score, fields, per)
    return out


def main():
    if os.path.exists(OUT) and "--force" not in sys.argv:
        sys.exit(f"{OUT} is frozen; it was set once. Delete it by hand only if the dev protocol changed.")
    names = list(ev.SPLITS["dev"])
    cols = {n: ev.Collection(n) for n in set(names) | {ev.PROBE_FROM[n] for n in names}}
    sets = {n: ev.query_set(cols[n], "dev", cols[ev.PROBE_FROM[n]]) for n in names}
    thr = best(cols, sets, names, [({"threshold_t": t}, lambda sc, t=t: ev.n_threshold(sc, t)) for t in T_GRID])
    tk = best(cols, sets, names, [({"topk_threshold_k": k, "topk_threshold_t": t},
                                   lambda sc, k=k, t=t: ev.n_threshold(sc, t, k)) for k in K_GRID for t in T_GRID])
    ak = best(cols, sets, names, [({"adaptive_k_B": b}, lambda sc, b=b: ev.n_adaptive_k(sc, b)) for b in B_GRID])
    qhash = hashlib.sha256(json.dumps({n: [q[0] for q in sets[n]] for n in names}, sort_keys=True).encode()).hexdigest()
    doc = {"fields": {**thr[1], **tk[1], **ak[1]},
           "frozen_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "dev_queries_sha256": qhash, "objective": "mean over dev collections of task score (F2 answerable, 1/0 probes)",
           "grid": {"t": T_GRID, "k": K_GRID, "B": B_GRID},
           "dev_task": {"dense · threshold": thr[0], "dense · top-k + threshold": tk[0], "adaptive-k": ak[0]},
           "dev_detail": {"dense · threshold": thr[2], "dense · top-k + threshold": tk[2], "adaptive-k": ak[2]}}
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
    print(json.dumps(doc["fields"]), json.dumps(doc["dev_task"]), f"frozen -> {OUT}", sep="\n")


if __name__ == "__main__":
    main()
