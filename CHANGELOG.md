# Changelog

## 0.1.0

First public release.

- Indexing: local describers (Ollama vision and text models, or text-only), phrase parsing,
  whole-gloss and per-phrase embeddings (Ollama or sentence-transformers), and an incremental
  SQLite store.
- Search: blended multi-vector scoring with a rarity-weighted keyword-coverage boost, and the
  evidence-gated result set: noise floor, evidence gate, floor-anchored band, coverage-gap
  tightening and conditional padding. One strictness control.
- Threshold profiles for EmbeddingGemma-300M, one per kind of content: `embeddinggemma-300m` for
  glosses written by a model, and `embeddinggemma-300m/passages` (`EMBEDDINGGEMMA_PASSAGES`) for
  raw text indexed as written (band 0.90, evidence gate 0.52, minimum 1 result). `Glossdex`
  selects the passages profile automatically with the text-only describer; `profile_for()` takes
  a `content` argument (`"descriptions"`, the default, or `"passages"`).
- `Glossdex.context()` for retrieval-augmented generation that can return "not found".
- CLI (`index`, `search`, `explain`, `serve`, `doctor`) and a local web UI with a step-by-step
  explanation and a fixed top-k comparison.
- Benchmark on three public BEIR collections (SciFact, NFCorpus, FiQA) against dense, BM25 and
  hybrid search at fixed cutoffs, a similarity threshold (alone and capped at top 5) and
  Adaptive-k, with scripts and results in `benchmarks/beir/`. Every setting, including the
  baselines' (`tune_baselines.py`), is chosen on development collections and frozen.
