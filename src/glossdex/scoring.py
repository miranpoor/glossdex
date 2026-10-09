"""Scoring and the evidence-gated result set.

Two separate concerns, kept separate on purpose:

* **Ranking** (:meth:`ScoringIndex.rank`): a semantic score per item from its whole-description
  vector blended with a soft maximum over its phrase vectors, plus a capped, rarity-weighted
  keyword-coverage boost.
* **How many to show** (:func:`select`): every threshold is derived from the best *semantic*
  score, so keyword evidence can never raise a threshold; the evidence gate can permit a list but
  never lengthens one. The stages are

  1. noise-floor gate      - the best match is too weak to mean anything: show nothing
  2. evidence gate         - a higher bar that keyword evidence from the shown set can help clear
  3. floor-anchored band   - keep items within a ratio of the best, measured from the noise floor
  4. coverage-gap tighten  - if one item accounts for the query far better than the rest, cut harder
  5. conditional pad       - top up a too-short list only with near misses that share query words
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Optional, Sequence, Set

import numpy as np

from .lexical import LexicalIndex, content_terms
from .profiles import Profile

UNIT_DESCRIPTION = 0  # the whole description
UNIT_PHRASE = 1  # one phrase


@dataclass
class Unit:
    """One embedded retrieval unit of an item."""

    item_id: int
    type: int
    text: str
    vec: np.ndarray


@dataclass
class ScoredItem:
    item_id: int
    score: float  # ranking score: semantic + keyword boost
    base_score: float  # semantic score only; every threshold is anchored to this
    best_text: str = ""
    coverage: float = 0.0
    reason: str = ""  # "band" or "pad", filled in by select()


@dataclass
class Trace:
    """What the result-set stages decided for one query, in order. Rendered by the UI."""

    profile: str = ""
    query_terms: List[str] = field(default_factory=list)
    top: Optional[float] = None
    noise_floor: float = 0.0
    passed_floor: Optional[bool] = None
    gate_threshold: float = 0.0
    gate_bonus: float = 0.0
    passed_gate: Optional[bool] = None
    band_threshold: Optional[float] = None
    band_count: int = 0
    coverage_gap: Optional[float] = None
    vocabulary_absent: bool = False
    tightened_frac: Optional[float] = None
    tightened_threshold: Optional[float] = None
    padded: int = 0
    shown: int = 0
    verdict: str = ""

    def to_dict(self) -> Dict:
        return dict(self.__dict__)


def pool(sims: np.ndarray, tau: float) -> float:
    """Soft maximum ``m + ln(sum(exp(tau * (s - m)))) / tau``; ``tau <= 0`` means plain max."""
    if sims.size == 0:
        return 0.0
    m = float(sims.max())
    if tau <= 0:
        return m
    return m + math.log(float(np.exp((sims.astype(np.float64) - m) * tau).sum())) / tau


def band_threshold(top: float, frac: float, profile: Profile) -> float:
    """``B(top) = max(f * top, F + h * (top - F))``: a ratio measured from the noise floor."""
    return max(top * frac, profile.noise_floor + profile.headroom * (top - profile.noise_floor))


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class ScoringIndex:
    """Vectors L2-normalized once and grouped by item, plus the lexical index."""

    def __init__(self, units: Sequence[Unit], lexical_texts: Mapping[int, Sequence[str]] = None,
                 lexical: Optional[LexicalIndex] = None):
        self.item_ids: List[int] = []
        self._offsets: List[int] = []
        self._is_doc: np.ndarray
        rows: List[np.ndarray] = []
        texts: List[str] = []
        types: List[int] = []
        by_item: Dict[int, List[Unit]] = {}
        for u in units:
            by_item.setdefault(u.item_id, []).append(u)
        dims = None
        for item_id, us in by_item.items():
            self.item_ids.append(item_id)
            self._offsets.append(len(rows))
            for u in us:
                v = np.asarray(u.vec, dtype=np.float32)
                if dims is None:
                    dims = v.shape[0]
                if v.shape[0] != dims:
                    raise ValueError("mixed embedding dimensions; re-index with one embedding model")
                rows.append(_l2(v))
                texts.append(u.text)
                types.append(u.type)
        self.dims = dims or 0
        self._matrix = np.vstack(rows) if rows else np.zeros((0, 0), dtype=np.float32)
        self._texts = texts
        self._types = np.asarray(types, dtype=np.int32)
        self._offsets.append(len(rows))
        if lexical is None:
            src = lexical_texts or {i: [u.text for u in us] for i, us in by_item.items()}
            lexical = LexicalIndex.build(src)
        self.lexical = lexical

    def __len__(self) -> int:
        return len(self.item_ids)

    # -- ranking ---------------------------------------------------------------------------

    def rank(self, query_vec: np.ndarray, query_text: str, profile: Profile) -> List[ScoredItem]:
        """Every item, scored and sorted by ranking score (best first). No cutoff."""
        if not self.item_ids:
            return []
        q = _l2(np.asarray(query_vec, dtype=np.float32))
        if q.shape[0] != self.dims:
            raise ValueError(f"query has {q.shape[0]} dims, index has {self.dims}")
        sims = self._matrix @ q
        coverage = self.lexical.idf_coverage(query_text) if query_text else {}
        w = profile.doc_weight
        out: List[ScoredItem] = []
        for i, item_id in enumerate(self.item_ids):
            a, b = self._offsets[i], self._offsets[i + 1]
            s = sims[a:b]
            t = self._types[a:b]
            doc_idx = np.flatnonzero(t == UNIT_DESCRIPTION)
            phrase_pos = np.flatnonzero(t != UNIT_DESCRIPTION)
            phrase_sims = s[phrase_pos]
            # Report the best PHRASE when there is one: it says why the item matched.
            best = a + (int(phrase_pos[np.argmax(phrase_sims)]) if phrase_pos.size
                        else int(np.argmax(s)))
            if w <= 0:
                semantic = pool(s, profile.tau)
            else:
                phrase_score = pool(phrase_sims if phrase_sims.size else s, profile.tau)
                semantic = (w * float(s[doc_idx[0]]) + (1 - w) * phrase_score
                            if doc_idx.size else phrase_score)
            cov = coverage.get(item_id, 0.0)
            boost = 0.0
            if semantic >= profile.boost_floor and cov > 0:
                boost = profile.coverage_boost * cov ** profile.coverage_exponent
            out.append(ScoredItem(item_id, semantic + boost, semantic, self._texts[best], cov))
        out.sort(key=lambda r: r.score, reverse=True)
        return out

    # -- lexical evidence ------------------------------------------------------------------

    def gate_bonus(self, query_text: str, ranked: List[ScoredItem], profile: Profile,
                   frac: float) -> float:
        """Keyword evidence toward the evidence gate: BM25 from items the band would show, plus the
        all-terms bonus if any item in the collection contains every content term."""
        if profile.gate <= 0 or not query_text:
            return 0.0
        graded = 0.0
        if profile.gate_bm25 > 0 and ranked:
            band = band_threshold(max(r.base_score for r in ranked), frac, profile)
            in_band = {r.item_id for r in ranked if r.base_score >= band}
            content = " ".join(content_terms(query_text))
            if content:
                scores = self.lexical.bm25(content)
                best = max((v for k, v in scores.items() if k in in_band), default=0.0)
                graded = profile.gate_bm25 * best
        all_terms = profile.gate_all_terms if (
            profile.gate_all_terms > 0 and self.lexical.items_matching_all_terms(query_text)
        ) else 0.0
        return graded + all_terms

    def vocabulary_absent(self, query_text: str) -> bool:
        """True when the query has content words and no item matches any of them."""
        if not content_terms(query_text):
            return False
        return not self.lexical.idf_coverage(query_text)

    def coverage_gap(self, query_text: str, ids: Set[int]) -> float:
        """``(C1 - C2) / C1`` over the given items: 0 when several items tie."""
        cov = sorted((v for k, v in self.lexical.idf_coverage(query_text).items() if k in ids),
                     reverse=True)
        if not cov or cov[0] <= 0:
            return 0.0
        second = cov[1] if len(cov) > 1 else 0.0
        return min(1.0, max(0.0, (cov[0] - second) / cov[0]))


def _cut(ranked: List[ScoredItem], frac: float, profile: Profile, gate_bonus: float,
         pad_eligible: Optional[Set[int]], limit: int, trace: Optional[Trace]) -> List[ScoredItem]:
    if not ranked:
        return []
    top = max(r.base_score for r in ranked)
    if trace is not None:
        trace.top = top
        trace.passed_floor = top >= profile.noise_floor
    if top < profile.noise_floor:
        return []
    if profile.gate > 0:
        passed = top + gate_bonus >= profile.gate
        if trace is not None:
            trace.gate_bonus = gate_bonus
            trace.passed_gate = passed
        if not passed:
            return []
    threshold = band_threshold(top, frac, profile)
    above = [replace(r, reason="band") for r in ranked if r.score >= threshold]
    if trace is not None:
        trace.band_threshold = threshold
        trace.band_count = len(above)
    if len(above) >= profile.min_results:
        return above[: profile.max_results][:limit]
    have = {r.item_id for r in above}
    pads: List[ScoredItem] = []
    for r in ranked:
        if len(pads) >= profile.min_results - len(above):
            break
        if r.item_id in have:
            continue
        if pad_eligible is not None and r.item_id not in pad_eligible:
            continue
        if r.base_score < top * profile.pad_min_frac or r.base_score < profile.noise_floor:
            continue
        pads.append(replace(r, reason="pad"))
    return (above + pads)[: profile.max_results][:limit]


def select(index: ScoringIndex, query_vec: np.ndarray, query_text: str, profile: Profile,
           limit: int = 50, strictness: Optional[float] = None,
           ranked: Optional[List[ScoredItem]] = None) -> "tuple[List[ScoredItem], Trace]":
    """Rank, gate and cut in one call. Returns the shown items and the trace of every decision."""
    p = profile.at_strictness(strictness) if strictness is not None else profile
    trace = Trace(profile=p.describe(), query_terms=content_terms(query_text),
                  noise_floor=p.noise_floor, gate_threshold=p.gate)
    if ranked is None:
        ranked = index.rank(query_vec, query_text, p)
    if not ranked:
        trace.verdict = "The index is empty."
        return [], trace

    bonus = index.gate_bonus(query_text, ranked, p, p.frac)
    pad_eligible = (index.lexical.items_matching_at_least(query_text, p.pad_min_terms)
                    if p.pad_min_terms > 0 and query_text else None)
    shown = _cut(ranked, p.frac, p, bonus, pad_eligible, limit, trace)

    if p.gap_tighten > 0 and len(shown) >= 2 and query_text:
        ids = {r.item_id for r in shown}
        gap = index.coverage_gap(query_text, ids)
        trace.coverage_gap = gap
        effective = gap
        if gap <= 0 and index.vocabulary_absent(query_text):
            trace.vocabulary_absent = True
            effective = 1.0
        if effective > 0:
            tighter = min(p.frac + p.gap_tighten * effective, 0.95)
            trace.tightened_frac = tighter
            sub = Trace()
            shown = _cut(ranked, tighter, p, bonus, pad_eligible, limit, sub)
            trace.tightened_threshold = sub.band_threshold

    trace.padded = sum(1 for r in shown if r.reason == "pad")
    trace.shown = len(shown)
    trace.verdict = _verdict(trace, p)
    return shown, trace


def _verdict(t: Trace, p: Profile) -> str:
    if t.passed_floor is False:
        return (f"Nothing shown: the best match ({t.top:.3f}) is below the noise floor "
                f"({p.noise_floor:.3f}), so the collection has nothing like this.")
    if t.passed_gate is False:
        return (f"Nothing shown: the best match ({t.top:.3f}) plus keyword evidence "
                f"({t.gate_bonus:.3f}) is below the evidence gate ({p.gate:.3f}).")
    parts = [f"{t.band_count} item(s) within the band (score >= {t.band_threshold:.3f})."]
    if t.tightened_frac is not None:
        why = ("no item shares any query word" if t.vocabulary_absent
               else f"one item covers the query far better than the rest (gap {t.coverage_gap:.2f})")
        parts.append(f"Band tightened to {t.tightened_threshold:.3f} because {why}.")
    if t.padded:
        parts.append(f"{t.padded} near miss(es) added that share query words.")
    parts.append(f"Showing {t.shown}.")
    return " ".join(parts)


def top_k(ranked: List[ScoredItem], k: int) -> List[ScoredItem]:
    """The fixed top-k most search tools show, for side-by-side comparison."""
    return [replace(r, reason="top-k") for r in ranked[:k]]
