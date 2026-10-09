"""Embeds a BEIR text collection for the glossdex core benchmark, once, and caches it.

    python build_index.py scifact nfcorpus fiqa

For every document: glossdex's text-only describer (its own sentences become its phrases, via
glossdex.phrases.parse_phrases), then EmbeddingGemma-300M in DOCUMENT mode for the whole document
(the description unit) and for each phrase. Queries (every split, in QUERY mode) are embedded too.
The same document vectors serve the dense single-vector baseline, so every system scores the
same embeddings and differs only in how it ranks and how many it shows.

Output: cache/<collection>.npz  (unit vectors, unit item ids, unit types, doc vectors, query vectors)
        cache/<collection>.json (doc ids, unit texts, query ids and texts)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from glossdex.phrases import parse_phrases  # noqa: E402
from glossdex.embedders import SentenceTransformersEmbedder  # noqa: E402

DATA = os.path.join(HERE, "data")
CACHE = os.path.join(HERE, "cache")
MAX_CHARS = 1500  # glossdex passage size; longer documents are truncated to their first passage


def load(name):
    docs = [json.loads(l) for l in open(os.path.join(DATA, name, "corpus.jsonl"), encoding="utf-8")]
    queries = {q["_id"]: q["text"] for q in (json.loads(l) for l in open(os.path.join(DATA, name, "queries.jsonl"), encoding="utf-8"))}
    return docs, queries


def build(name, emb):
    os.makedirs(CACHE, exist_ok=True)
    docs, queries = load(name)
    doc_ids, doc_texts, unit_item, unit_type, unit_text = [], [], [], [], []
    for i, d in enumerate(docs):
        text = ((d.get("title") or "").strip() + ". " if d.get("title") else "") + (d.get("text") or "").strip()
        text = text[:MAX_CHARS]
        doc_ids.append(d["_id"])
        doc_texts.append(text)
        unit_item.append(i), unit_type.append(0), unit_text.append(text)
        for p in parse_phrases(text):
            unit_item.append(i), unit_type.append(1), unit_text.append(p)
    print(f"{name}: {len(docs)} docs, {len(unit_text)} units, {len(queries)} queries")
    vecs = emb.embed_documents(unit_text)
    doc_rows = [k for k, t in enumerate(unit_type) if t == 0]
    qids = sorted(queries)
    qvecs = np.stack([emb.embed_query(queries[q]) for q in qids])
    np.savez_compressed(os.path.join(CACHE, f"{name}.npz"), unit_vec=vecs.astype(np.float16),
                        unit_item=np.array(unit_item, np.int32), unit_type=np.array(unit_type, np.int8),
                        doc_vec=vecs[doc_rows].astype(np.float16), query_vec=qvecs.astype(np.float16))
    with open(os.path.join(CACHE, f"{name}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"doc_ids": doc_ids, "doc_texts": doc_texts, "unit_text": unit_text,
                   "query_ids": qids, "query_texts": [queries[q] for q in qids]}, f)


def main():
    emb = SentenceTransformersEmbedder(device="cuda")
    for name in sys.argv[1:] or ["scifact", "nfcorpus", "fiqa"]:
        if os.path.exists(os.path.join(CACHE, f"{name}.npz")):
            print(f"{name}: cached")
            continue
        build(name, emb)


if __name__ == "__main__":
    main()
