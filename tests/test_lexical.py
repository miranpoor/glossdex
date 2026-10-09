from glossdex.lexical import (DEFAULT_VOCABULARY, LexicalIndex, content_terms, stem, tokenize)


def test_tokenize_keeps_numbers_and_drops_single_chars():
    assert tokenize("Invoice #A-1234, due 2026-06-19 a") == ["invoice", "1234", "due", "2026", "06", "19"]


def test_content_terms_removes_stopwords_and_duplicates():
    assert content_terms("a photo of my son with his son") == ["son"]
    assert content_terms("tv on top of stand") == ["tv", "stand"]


def test_stem_is_conservative():
    assert stem("held") == "hold"
    assert stem("roses") == "rose"
    assert stem("babies") == "baby"
    assert stem("glass") == "glass"
    assert stem("standing") == "standing"  # never merged with "stand"


def test_synonym_classes_are_not_transitive():
    wife = DEFAULT_VOCABULARY.interchangeable("wife")
    mother = DEFAULT_VOCABULARY.interchangeable("mother")
    assert "woman" in wife and "woman" in mother
    assert "mother" not in wife and "wife" not in mother
    boy = DEFAULT_VOCABULARY.interchangeable("boy")
    assert "child" in boy and "girl" not in boy


def _index():
    return LexicalIndex.build({
        1: ["A baby is being held by one of the women."],
        2: ["A massive waterfall.", "Water spray."],
        3: ["Electricity bill.", "Amount due $96.80."],
        4: ["A man standing in a kitchen."],
    })


def test_all_terms_through_synonyms():
    lx = _index()
    assert lx.items_matching_all_terms("my son held by my wife") == {1}
    assert lx.items_matching_all_terms("niagara falls") == set()
    assert lx.items_matching_all_terms("falls") == {2}


def test_idf_coverage_is_absolute_and_bounded():
    lx = _index()
    cov = lx.idf_coverage("waterfall")
    assert set(cov) == {2}
    assert abs(cov[2] - 1.0) < 1e-9  # a full match on a term in one item is 1.0
    half = lx.idf_coverage("waterfall helicopter")
    assert abs(half[2] - 0.5) < 1e-9  # the absent term still counts in the ceiling


def test_bm25_charges_absent_terms():
    lx = _index()
    full = lx.bm25("electricity")[3]
    partial = lx.bm25("electricity helicopter")[3]
    assert partial < full <= 1.0


def test_items_matching_at_least_counts_query_terms_once():
    lx = _index()
    assert lx.items_matching_at_least("baby women", 2) == {1}
    assert lx.items_matching_at_least("kid child baby", 3) == {1}
    assert lx.items_matching_at_least("kitchen helicopter", 2) == set()
