"""The result-set stages, on items whose semantic scores are set exactly.

Each item has one phrase vector ``cos(a) e0 + sin(a) e_i`` and no whole-description vector,
so its semantic score against the query ``e0`` is exactly ``cos(a)``.
"""

import math

import numpy as np
import pytest

from glossdex.profiles import EMBEDDINGGEMMA as P
from glossdex.scoring import ScoringIndex, Unit, band_threshold, pool, select

DIMS = 64


def unit(i, score, text):
    v = np.zeros(DIMS, dtype=np.float32)
    v[0] = score
    v[i] = math.sqrt(max(0.0, 1 - score * score))
    return Unit(i, 1, text, v)


def build(items):
    """items: list of (semantic score, text); ids start at 1."""
    return ScoringIndex([unit(i, s, t) for i, (s, t) in enumerate(items, 1)])


Q = np.eye(DIMS, dtype=np.float32)[0]


def run(index, query, **kw):
    shown, trace = select(index, Q, query, P, **kw)
    return [r.item_id for r in shown], trace


def test_pool_is_a_soft_maximum():
    s = np.array([0.5, 0.2], dtype=np.float32)
    assert pool(s, 16) == pytest.approx(0.5 + math.log(1 + math.exp(-16 * 0.3)) / 16, abs=1e-6)
    assert pool(np.array([0.4]), 16) == pytest.approx(0.4, abs=1e-7)
    assert pool(s, 0) == pytest.approx(0.5)


def test_band_matches_worked_example():
    # Spec FIG. 7: top 0.514 -> band 0.360; tightened fraction 0.85 -> 0.437.
    assert band_threshold(0.514, 0.70, P) == pytest.approx(0.3598, abs=1e-4)
    assert band_threshold(0.514, 0.85, P) == pytest.approx(0.4369, abs=1e-4)
    # The floor-anchored term only ever raises the bar, and crosses the ratio at 0.49.
    assert band_threshold(0.30, 0.70, P) > 0.30 * 0.70
    assert band_threshold(0.60, 0.70, P) == pytest.approx(0.42)


def test_noise_floor_shows_nothing():
    idx = build([(0.25, "A red car."), (0.22, "A blue door.")])
    ids, t = run(idx, "giraffe")
    assert ids == [] and t.passed_floor is False
    assert "noise floor" in t.verdict


def test_evidence_gate_needs_keywords_for_a_weak_top():
    idx = build([(0.33, "Michael Smith."), (0.30, "A microwave oven.")])
    ids, t = run(idx, "helicopter")  # weak and uncorroborated
    assert ids == [] and t.passed_gate is False
    ids, t = run(idx, "michael")  # same weak semantic score, but the word is there
    assert t.passed_gate is True and ids[0] == 1


def test_padding_requires_shared_words_and_a_near_score():
    # top 0.60 -> band 0.42. Item 2 shares one word, stays below the band even with its small
    # keyword boost, and scores 0.40 >= 0.6*0.60: padded. Item 3 shares nothing: never padded.
    idx = build([(0.60, "Red roses in a vase."), (0.40, "A red tulip."),
                 (0.41, "A kitchen sink.")])
    ids, t = run(idx, "red roses")
    assert ids == [1, 2] and t.padded == 1
    # A one-word query whose next item shares nothing is not padded (the microwave case).
    idx = build([(0.60, "Michael Smith, passport."), (0.41, "A microwave oven.")])
    ids, _ = run(idx, "michael")
    assert ids == [1]


def test_coverage_gap_tightens_when_one_item_answers():
    # One item has every word; a cluster matches only "son" (via "boy").
    items = [(0.514, "A boy drinking from a milk bottle.")]
    items += [(0.40 - 0.002 * i, f"A boy playing {i}.") for i in range(10)]
    ids, t = run(build(items), "my son with milk bottle")
    assert t.band_count == 11
    assert t.tightened_frac == pytest.approx(min(0.95, 0.70 + 0.20 * t.coverage_gap))
    assert ids[0] == 1 and len(ids) <= 2


def test_ties_leave_the_band_alone():
    items = [(0.55, "A massive waterfall."), (0.53, "Waterfall with mist."),
             (0.50, "A waterfall at dusk.")]
    ids, t = run(build(items), "waterfall")
    assert ids == [1, 2, 3]
    assert t.coverage_gap == 0 and t.tightened_frac is None


def test_absent_vocabulary_counts_as_maximal_gap():
    # Band at 0.70 is 0.331 and keeps all three; no item shares a query word, so the gap counts
    # as 1, the fraction rises to 0.90 (band 0.405), and nothing qualifies to pad.
    items = [(0.45, "A green field."), (0.40, "A grey sky."), (0.38, "A city street.")]
    ids, t = run(build(items), "helicopter landing")
    assert t.band_count == 3
    assert t.vocabulary_absent and t.tightened_frac == pytest.approx(0.90)
    assert ids == [1]


def test_loosest_strictness_disables_every_suppression():
    idx = build([(0.25, "A red car."), (0.22, "A blue door.")])
    ids, t = run(idx, "giraffe", strictness=0.0)
    assert ids == [1, 2]
    assert t.noise_floor == 0.0


def test_keyword_boost_never_raises_the_band():
    # Item 2 contains every query word and is boosted far above item 1, but the band is anchored
    # to the best SEMANTIC score (0.60 -> 0.42), so item 3 (no query words) still makes the band.
    items = [(0.60, "A dog asleep."), (0.50, "Golden retriever puppy on a sofa."),
             (0.45, "A sleeping animal.")]
    shown, t = select(build(items), Q, "puppy sofa", P)
    assert t.top == pytest.approx(0.60, abs=1e-6)
    assert shown[0].item_id == 2 and shown[0].base_score < shown[0].score
    assert t.band_threshold == pytest.approx(0.42)
    assert t.band_count == 3


def test_whole_description_blend():
    v_doc = np.zeros(DIMS, dtype=np.float32); v_doc[0] = 0.5; v_doc[1] = math.sqrt(0.75)
    v_ph = np.zeros(DIMS, dtype=np.float32); v_ph[0] = 0.3; v_ph[2] = math.sqrt(0.91)
    idx = ScoringIndex([Unit(7, 0, "whole", v_doc), Unit(7, 1, "phrase", v_ph)])
    r = idx.rank(Q, "", P)[0]
    assert r.base_score == pytest.approx(0.8 * 0.5 + 0.2 * 0.3, abs=1e-6)
