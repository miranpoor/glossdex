"""Profile selection: per embedding model, and per kind of content."""

import numpy as np
import pytest

from glossdex import EMBEDDINGGEMMA, EMBEDDINGGEMMA_PASSAGES, Glossdex, Profile, TextOnlyDescriber
from glossdex.profiles import DESCRIPTIONS, PASSAGES, profile_for


class GemmaKeyEmbedder:
    """Claims EmbeddingGemma's score units; vectors are irrelevant to profile selection."""

    name = "stub"
    profile_key = "embeddinggemma-300m"

    def embed_documents(self, texts):
        return np.ones((len(texts), 8), dtype=np.float32)

    def embed_query(self, text):
        return np.ones(8, dtype=np.float32)


class StubDescriber:
    name = "stub-vlm"

    def describe(self, item):
        raise AssertionError("not called in these tests")


def test_descriptions_is_the_default():
    assert profile_for("embeddinggemma-300m") is EMBEDDINGGEMMA
    assert profile_for("embeddinggemma-300m", content=DESCRIPTIONS) is EMBEDDINGGEMMA


def test_passages_profile_changes_only_the_result_set():
    p = profile_for("embeddinggemma-300m", content=PASSAGES)
    assert p is EMBEDDINGGEMMA_PASSAGES
    assert (p.frac, p.gate, p.min_results) == (0.90, 0.52, 1)
    for field in ("doc_weight", "tau", "coverage_boost", "coverage_exponent", "boost_floor"):
        assert getattr(p, field) == getattr(EMBEDDINGGEMMA, field)  # ranking is unchanged


def test_unknown_model_is_uncalibrated_for_either_content():
    for content, base in ((DESCRIPTIONS, EMBEDDINGGEMMA), (PASSAGES, EMBEDDINGGEMMA_PASSAGES)):
        p = profile_for("some-other-model", content=content)
        assert not p.calibrated
        assert (p.frac, p.gate) == (base.frac, base.gate)


def test_unknown_content_is_rejected():
    with pytest.raises(ValueError):
        profile_for("embeddinggemma-300m", content="images")


def test_override_wins():
    mine = Profile(name="mine", gate=0.1)
    assert profile_for("embeddinggemma-300m", mine, PASSAGES) is mine


def test_engine_picks_passages_for_text_only_indexing(tmp_path):
    gx = Glossdex.for_folder(str(tmp_path), describer=TextOnlyDescriber(), embedder=GemmaKeyEmbedder())
    assert gx.profile is EMBEDDINGGEMMA_PASSAGES


def test_engine_picks_descriptions_for_a_describer_model(tmp_path):
    gx = Glossdex.for_folder(str(tmp_path), describer=StubDescriber(), embedder=GemmaKeyEmbedder())
    assert gx.profile is EMBEDDINGGEMMA
