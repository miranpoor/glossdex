"""Profiles: per embedding model, and per kind of content.

Every threshold in the result-set stages is a number in ONE embedding model's score units.
A noise floor of 0.28 means "unrelated" for EmbeddingGemma with this blend and pooling; under a
different model it means something else. So the thresholds travel with the model, not with the
collection: a profile is derived once per embedding model and reused across collections.

Within one model, two kinds of content behave differently and get their own result-set values:

- ``descriptions``: short glosses written by a describer model (images, and documents described
  by a model). The default.
- ``passages``: raw text indexed as is (the text-only describer), where each passage is up to
  about 1,500 characters and its sentences are its phrases. Longer items match more queries a
  little, so the band is narrower and the evidence gate higher. Ranking is unchanged.

Using an embedding model without its own profile still works, but the thresholds are then only
a guess and glossdex says so.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional


@dataclass(frozen=True)
class Profile:
    name: str
    # scoring
    doc_weight: float = 0.8  # weight of the whole-description vector vs pooled phrases
    tau: float = 16.0  # soft-maximum temperature over phrase similarities
    coverage_boost: float = 1.0  # keyword-coverage boost weight
    coverage_exponent: float = 3.0  # partial coverage is weak, full coverage is strong
    boost_floor: float = 0.28  # no boost for items the embedding already judged unrelated
    # result set
    frac: float = 0.70  # band fraction (also the default strictness)
    noise_floor: float = 0.28  # best match below this: nothing in the collection is like it
    headroom: float = 0.30  # band measured from the noise floor
    gate: float = 0.36  # evidence gate on the best match plus keyword evidence
    gate_bm25: float = 0.10  # weight of in-band BM25 evidence toward the gate
    gate_all_terms: float = 0.10  # bonus when one item contains every content word
    gap_tighten: float = 0.20  # band tightening per unit of coverage gap
    min_results: int = 2  # pad a shorter list up to this, with near misses only
    pad_min_terms: int = 1  # a pad must share this many content words with the query
    pad_min_frac: float = 0.60  # ...and score at least this fraction of the best match
    max_results: int = 50  # a flood guard, not a relevance filter
    calibrated: bool = True

    def at_strictness(self, u: float) -> "Profile":
        """One control: ``u`` sets the band fraction; every suppressing threshold scales with it.

        At the default (``u == frac``) everything is at full strength. Toward 0 the floor,
        headroom and gate shrink linearly to zero, so "show me more" can always reach the
        ranked items.
        """
        u = min(1.0, max(0.0, float(u)))
        scale = min(1.0, max(0.0, u / self.frac)) if self.frac > 0 else 1.0
        return replace(
            self,
            frac=u,
            noise_floor=self.noise_floor * scale,
            headroom=self.headroom * scale,
            gate=self.gate * scale,
        )

    def describe(self) -> str:
        """Every field that changes behavior, so logged results stay comparable."""
        return (
            f"{self.name} | doc={self.doc_weight} softmax(tau={self.tau}) "
            f"cov={self.coverage_boost}idf^{self.coverage_exponent}@>={self.boost_floor} | "
            f"rel>={self.frac}(min={self.min_results},max={self.max_results},"
            f"floor={self.noise_floor:.4g},headroom={self.headroom:.4g},"
            f"gate={self.gate:.4g}+{self.gate_bm25}bm25+{self.gate_all_terms}all,"
            f"padKw>={self.pad_min_terms},padSem>={self.pad_min_frac},"
            f"gapTighten={self.gap_tighten})"
        )


#: EmbeddingGemma-300M (768 dims), query/document task prompts, on generated descriptions. The
#: configuration that ships in the Memoyad app.
EMBEDDINGGEMMA = Profile(name="embeddinggemma-300m")

#: EmbeddingGemma-300M on raw text passages. Same ranking; result-set values chosen once on three
#: public collections (SciDocs, ArguAna, CQADupStack-webmasters) and then evaluated, unchanged, on
#: SciFact, NFCorpus and FiQA (see benchmarks/beir/).
EMBEDDINGGEMMA_PASSAGES = replace(EMBEDDINGGEMMA, name="embeddinggemma-300m/passages",
                                  frac=0.90, gate=0.52, min_results=1)

DESCRIPTIONS = "descriptions"
PASSAGES = "passages"

PROFILES: Dict[str, Dict[str, Profile]] = {
    "embeddinggemma-300m": {DESCRIPTIONS: EMBEDDINGGEMMA, PASSAGES: EMBEDDINGGEMMA_PASSAGES},
}


def profile_for(key: str, override: Optional[Profile] = None, content: str = DESCRIPTIONS) -> Profile:
    """The profile for an embedding model key and kind of content (``descriptions`` or
    ``passages``), or EmbeddingGemma's values for that content marked uncalibrated."""
    if override is not None:
        return override
    if content not in (DESCRIPTIONS, PASSAGES):
        raise ValueError(f"content must be {DESCRIPTIONS!r} or {PASSAGES!r}, not {content!r}")
    p = PROFILES.get(key, {}).get(content)
    if p is not None:
        return p
    base = PROFILES["embeddinggemma-300m"][content]
    return replace(base, name=f"{key} (uncalibrated: using {base.name} values)", calibrated=False)
