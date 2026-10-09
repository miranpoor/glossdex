"""glossdex core benchmark on public text collections (BEIR): what each system SHOWS.

    python evaluate.py test            # the frozen text profile on the three test collections

Systems, all on the same EmbeddingGemma-300M vectors (build_index.py):
  glossdex   multi-vector scoring + the evidence-gated result set, glossdex.scoring.select, with
             the text profile set once on the development collections (tune.py) and then frozen
  dense      single vector per document, cosine ranking            -> top-5, top-10
  bm25       BM25 (bm25s, English stopwords, Snowball stemmer)     -> top-5, top-10
  hybrid     reciprocal rank fusion of dense and bm25 (k = 60)     -> top-5, top-10
  dense · threshold          every document with cosine >= t (at most 50, glossdex's flood guard)
  dense · top-k + threshold  the first k documents with cosine >= t
  adaptive-k                 Taguchi et al., EMNLP 2025: cut the sorted cosine scores at their
                             largest gap (searched over the top 90% of documents), then add B more;
                             as published (B = 5) and with B chosen on dev
  t, k and B are chosen on the development collections only (tune_baselines.py ->
  baseline_settings.json), exactly as glossdex's text profile is (tune.py -> text_profile.json).

Queries: the split's queries with relevance judgments, plus NO-ANSWER probes: queries from an
unrelated collection (finance questions against biomedical corpora, biomedical claims against the
finance corpus), which the corpus cannot answer.

Metrics per query (same as the photo benchmark): recall, precision, F2, wrong, shown; for probes,
whether nothing was shown. Task score = F2 on answerable queries, 1/0 for a correct empty answer
on probes. Ranking: nDCG@10 and Recall@10 on the full ranking. Note: BEIR judgments are sparse, so
precision is a lower bound for every system alike.
"""
import json
import math
import os
import random
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from glossdex.profiles import EMBEDDINGGEMMA, Profile  # noqa: E402
from glossdex.scoring import ScoringIndex, Unit, select  # noqa: E402

DATA, CACHE, OUT = (os.path.join(HERE, d) for d in ("data", "cache", "results"))
# Development and test use DIFFERENT collections: no test collection (or any split of one) is
# seen while choosing settings.
SPLITS = {"test": {"scifact": "test", "nfcorpus": "test", "fiqa": "test"},
          "dev": {"scidocs": "test", "arguana": "test", "webmasters": "test"}}
PROBE_FROM = {"scifact": "fiqa", "nfcorpus": "fiqa", "fiqa": "scifact",
              "scidocs": "arguana", "arguana": "webmasters", "webmasters": "arguana"}
PROBE_SPLIT = {"dev": {"webmasters": "test", "arguana": "test"}, "test": {"fiqa": "test", "scifact": "test"}}
# ArguAna's queries are themselves documents of its corpus; BEIR ignores that self-match, and so do we.
SELF_MATCH = {"arguana"}
N_PROBES = 50
SEED = 7


def qrels(name, split):
    rel = {}
    for ln in open(os.path.join(DATA, name, "qrels", f"{split}.tsv"), encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if p[0] == "query-id" or len(p) < 3 or int(p[2]) <= 0:
            continue
        rel.setdefault(p[0], set()).add(p[1])
    return rel


class Collection:
    def __init__(self, name):
        z = np.load(os.path.join(CACHE, f"{name}.npz"))
        m = json.load(open(os.path.join(CACHE, f"{name}.json"), encoding="utf-8"))
        self.name, self.doc_ids, self.doc_texts = name, m["doc_ids"], m["doc_texts"]
        self.pos = {d: i for i, d in enumerate(self.doc_ids)}
        self.qvec = dict(zip(m["query_ids"], z["query_vec"].astype(np.float32)))
        self.qtext = dict(zip(m["query_ids"], m["query_texts"]))
        units = [Unit(int(i), int(t), txt, v) for i, t, txt, v in
                 zip(z["unit_item"], z["unit_type"], m["unit_text"], z["unit_vec"].astype(np.float32))]
        self.index = ScoringIndex(units)
        dv = z["doc_vec"].astype(np.float32)
        self.doc_vec = dv / np.linalg.norm(dv, axis=1, keepdims=True)
        import bm25s
        import Stemmer
        self._stem = Stemmer.Stemmer("english")
        self.bm25 = bm25s.BM25()
        self.bm25.index(bm25s.tokenize(self.doc_texts, stopwords="en", stemmer=self._stem, show_progress=False), show_progress=False)

    def dense_rank(self, qv):
        q = qv / np.linalg.norm(qv)
        s = self.doc_vec @ q
        return [int(i) for i in np.argsort(-s)[:100]]

    def dense_sorted(self, qv, own=None):
        """Every document, best first, with its cosine score (the excluded self-match removed)."""
        q = qv / np.linalg.norm(qv)
        s = self.doc_vec @ q
        if own is not None:
            s = s.copy()
            s[own] = -np.inf
        order = np.argsort(-s)
        if own is not None:
            order = order[:-1]
        return order, s[order]

    def bm25_rank(self, text):
        import bm25s
        toks = bm25s.tokenize([text], stopwords="en", stemmer=self._stem, show_progress=False)
        res, _ = self.bm25.retrieve(toks, k=100, show_progress=False)
        return [int(i) for i in res[0]]


def rrf(*rankings, k=60):
    s = {}
    for r in rankings:
        for pos, d in enumerate(r):
            s[d] = s.get(d, 0.0) + 1.0 / (k + pos + 1)
    return [d for d, _ in sorted(s.items(), key=lambda x: -x[1])]


def shown_metrics(shown, rel, no_answer):
    n = len(shown)
    if no_answer:
        return {"task": 1.0 if n == 0 else 0.0, "shown": n, "rejected": 1.0 if n == 0 else 0.0}
    tp = len(set(shown) & rel)
    rec = tp / len(rel)
    prec = tp / n if n else None
    f2 = 0.0 if tp == 0 else 5 * prec * rec / (4 * prec + rec)
    return {"task": f2, "F2": f2, "recall": rec, "precision": prec, "wrong": n - tp, "shown": n}


def ranking_metrics(ranked, rel):
    hits = [d in rel for d in ranked[:10]]
    dcg = sum(1 / math.log2(i + 2) for i, h in enumerate(hits) if h)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(10, len(rel))))
    return {"nDCG@10": dcg / idcg, "R@10": sum(hits) / len(rel)}


def query_set(col, split_name, probe_col):
    """[(qid, text, vec, relevant doc indices, no_answer)]"""
    split = SPLITS[split_name][col.name]
    rel = qrels(col.name, split)
    pos = {d: i for i, d in enumerate(col.doc_ids)}
    out = [(q, col.qtext[q], col.qvec[q], {pos[d] for d in ds if d in pos}, False)
           for q, ds in sorted(rel.items()) if q in col.qvec]
    out = [x for x in out if x[3]]
    # Probes: queries from an unrelated collection, embedded with the same model (its cache).
    pq = sorted(q for q in qrels(probe_col.name, PROBE_SPLIT[split_name][probe_col.name]) if q in probe_col.qvec)
    random.Random(f"{SEED}:{split_name}:{col.name}").shuffle(pq)
    out += [(f"probe:{q}", probe_col.qtext[q], probe_col.qvec[q], set(), True) for q in pq[:N_PROBES]]
    return out


def self_match(col, qid):
    return col.pos.get(qid) if col.name in SELF_MATCH else None


# Speed only, exact: every result-set rule looks only at items whose score or base score is at
# least min(frac, pad_min_frac) x the best base score (band >= frac x top; pads need
# base >= pad_min_frac x top). Dropping the rest first changes no output for any profile with both
# fractions >= CANDIDATE_FRAC, and makes selection independent of collection size.
CANDIDATE_FRAC = 0.6


def candidates(ranked):
    if not ranked:
        return ranked
    cut = CANDIDATE_FRAC * max(r.base_score for r in ranked)
    return [r for r in ranked if r.score >= cut or r.base_score >= cut]


def evaluate_glossdex(col, qs, profile, ranked_cache=None):
    assert min(profile.frac, profile.pad_min_frac) >= CANDIDATE_FRAC
    rows = {}
    for qid, text, qv, rel, na in qs:
        hit = ranked_cache.get(qid) if ranked_cache is not None else None
        if hit is None:
            ranked = col.index.rank(qv, text, profile)
            own = self_match(col, qid)
            if own is not None:
                ranked = [r for r in ranked if r.item_id != own]
            hit = ([r.item_id for r in ranked[:10]], candidates(ranked))
            if ranked_cache is not None:
                ranked_cache[qid] = hit
        top10, cand = hit
        shown, _ = select(col.index, qv, text, profile, limit=50, ranked=cand)
        r = shown_metrics([s.item_id for s in shown], rel, na)
        if not na:
            r.update(ranking_metrics(top10, rel))
        rows[qid] = r
    return rows


def evaluate_baselines(col, qs):
    sys_rows = {}
    for qid, text, qv, rel, na in qs:
        own = self_match(col, qid)
        d = [i for i in col.dense_rank(qv) if i != own]
        b = [i for i in col.bm25_rank(text) if i != own]
        h = rrf(d, b)
        for name, r in (("dense", d), ("bm25", b), ("hybrid", h)):
            for k in (5, 10):
                m = shown_metrics(r[:k], rel, na)
                if not na:
                    m.update(ranking_metrics(r, rel))
                sys_rows.setdefault(f"{name} · top{k}", {})[qid] = m
    return sys_rows


FLOOD_GUARD = 50  # glossdex shows at most 50; the threshold baseline gets the same cap


def n_threshold(scores, t, k=None):
    """How many to show: documents with cosine >= t (scores descending), at most k (or 50)."""
    n = int(np.searchsorted(-scores, -t, side="right"))
    return min(n, FLOOD_GUARD if k is None else k)


def n_adaptive_k(scores, B):
    """Adaptive-k (Taguchi et al., EMNLP 2025): the largest drop between consecutive sorted scores,
    searched within the top 90% of documents, then B more. Never returns zero."""
    lim = max(2, int(0.9 * len(scores)))
    gaps = scores[:lim - 1] - scores[1:lim]
    return min(len(scores), int(np.argmax(gaps)) + 1 + B)


def score_sorted_systems(col, qs, systems):
    """systems: {name: f(scores) -> how many}. Shown sets cut from the full dense ranking."""
    rows = {name: {} for name in systems}
    for qid, text, qv, rel, na in qs:
        order, sc = col.dense_sorted(qv, self_match(col, qid))
        top10 = [int(i) for i in order[:10]]
        for name, how_many in systems.items():
            m = shown_metrics([int(i) for i in order[:how_many(sc)]], rel, na)
            if not na:
                m.update(ranking_metrics(top10, rel))
            rows[name][qid] = m
    return rows


def baseline_systems():
    """The score-based baselines with their frozen settings (tune_baselines.py)."""
    p = os.path.join(HERE, "baseline_settings.json")
    if not os.path.exists(p):
        sys.exit("baseline_settings.json is missing: run tune_baselines.py first")
    f = json.load(open(p, encoding="utf-8"))["fields"]
    t, (tk_k, tk_t), B = f["threshold_t"], (f["topk_threshold_k"], f["topk_threshold_t"]), f["adaptive_k_B"]
    return {
        "dense · threshold": lambda sc: n_threshold(sc, t),
        f"dense · top{tk_k} + threshold": lambda sc: n_threshold(sc, tk_t, tk_k),
        "adaptive-k · B=5 (published)": lambda sc: n_adaptive_k(sc, 5),
        f"adaptive-k · B={B} (set on dev)": lambda sc: n_adaptive_k(sc, B),
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(rows, qs):
    ans = [q[0] for q in qs if not q[4]]
    noa = [q[0] for q in qs if q[4]]
    return {"task": mean([rows[q]["task"] for q in ans + noa]),
            "recall": mean([rows[q]["recall"] for q in ans]), "precision": mean([rows[q]["precision"] for q in ans]),
            "F2": mean([rows[q]["F2"] for q in ans]), "wrong": mean([rows[q]["wrong"] for q in ans]),
            "shown": mean([rows[q]["shown"] for q in ans]), "rejected": mean([rows[q]["rejected"] for q in noa]),
            "answerable_empty": mean([1.0 if rows[q]["shown"] == 0 else 0.0 for q in ans]),
            "nDCG@10": mean([rows[q].get("nDCG@10") for q in ans]), "R@10": mean([rows[q].get("R@10") for q in ans]),
            "n_answerable": len(ans), "n_probes": len(noa)}


def text_profile():
    p = os.path.join(HERE, "text_profile.json")
    if not os.path.exists(p):
        sys.exit("text_profile.json is missing: run tune.py first")
    d = json.load(open(p, encoding="utf-8"))
    return replace(EMBEDDINGGEMMA, name="embeddinggemma-300m (text profile, set once on dev)", **d["fields"])


def main():
    split_name = sys.argv[1] if len(sys.argv) > 1 else "test"
    names = list(SPLITS[split_name])
    cols = {n: Collection(n) for n in set(names) | {PROBE_FROM[n] for n in names}}
    os.makedirs(OUT, exist_ok=True)
    report = {}
    for n in names:
        col = cols[n]
        qs = query_set(col, split_name, cols[PROBE_FROM[n]])
        rows = evaluate_baselines(col, qs)
        rows.update(score_sorted_systems(col, qs, baseline_systems()))
        rows["glossdex"] = evaluate_glossdex(col, qs, text_profile())
        report[n] = {k: summarize(v, qs) for k, v in rows.items()}
        with open(os.path.join(OUT, f"{split_name}-{n}.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"queries": [[q[0], q[1], sorted(q[3]), q[4]] for q in qs], "rows": rows}, f)
        print(f"\n== {n} ({split_name}): {report[n]['dense · top5']['n_answerable']} answerable, {N_PROBES} probes")
        for k, s in sorted(report[n].items(), key=lambda kv: -kv[1]["task"]):
            print(f"  {k:28s} task {100*s['task']:5.1f}  F2 {100*s['F2']:5.1f}  prec {100*s['precision']:5.1f}  "
                  f"rec {100*s['recall']:5.1f}  wrong {s['wrong']:5.2f}  shown {s['shown']:5.2f}  "
                  f"nothing-on-probes {100*s['rejected']:5.1f}  nDCG@10 {100*s['nDCG@10']:5.1f}")
    with open(os.path.join(OUT, f"summary-{split_name}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
