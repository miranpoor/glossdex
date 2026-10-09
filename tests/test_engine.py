"""End to end on text files, with no model: sentences as phrases, a hashing embedder."""

import hashlib
import os

import numpy as np

from glossdex import Glossdex, Profile, TextOnlyDescriber
from glossdex.lexical import content_terms


class HashEmbedder:
    """Bag of content words hashed into 256 dims. Deterministic and good enough to test wiring."""

    name = "hash"
    profile_key = "hash"

    def _vec(self, text):
        v = np.zeros(256, dtype=np.float32)
        for w in content_terms(text):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1.0
        v[255] += 0.3  # a shared component, like real embedders have
        return v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


PROFILE = Profile(name="hash-test", noise_floor=0.2, gate=0.25)

DOCS = {
    "hydro-june.txt": "Maple Grove Hydro electricity bill. Billing period May 1 to May 31. "
                      "Amount due $96.80. Due June 19.",
    "dentist.txt": "Dental appointment reminder. Dr. Patel. Cleaning on October 14 at 3 pm.",
    "recipe.md": "Banana bread recipe. Three ripe bananas. Bake for one hour.",
}


def make(tmp_path):
    for name, text in DOCS.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return Glossdex.for_folder(str(tmp_path), describer=TextOnlyDescriber(),
                               embedder=HashEmbedder(), profile=PROFILE)


def test_index_search_and_nothing_found(tmp_path):
    gx = make(tmp_path)
    rep = gx.index()
    assert rep.added == 3 and not rep.failed
    res = gx.search("electricity bill")
    assert res.found and res.hits[0].title == "hydro-june.txt"
    res = gx.search("helicopter landing")
    assert not res.found
    assert "Nothing shown" in res.trace.verdict


def test_reindex_is_incremental(tmp_path):
    gx = make(tmp_path)
    gx.index()
    rep = gx.index()
    assert rep.added == 0 and rep.unchanged == 3
    os.remove(tmp_path / "recipe.md")
    (tmp_path / "dentist.txt").write_text("Dental invoice. Amount due $120.", encoding="utf-8")
    os.utime(tmp_path / "dentist.txt", (1, 1))
    rep = gx.index()
    assert rep.removed == 1 and rep.added == 1
    assert gx.search("banana bread").found is False


def test_context_for_rag(tmp_path):
    gx = make(tmp_path)
    gx.index()
    ctx = gx.context("when is my dental cleaning")
    assert ctx.found and "[1] dentist.txt" in ctx.text
    assert not gx.context("tax return 2019")


def test_compare_top_k_always_returns_k(tmp_path):
    gx = make(tmp_path)
    gx.index()
    assert len(gx.search("helicopter landing", compare_top_k=2).hits) == 2
