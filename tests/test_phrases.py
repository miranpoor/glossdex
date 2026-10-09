from glossdex.phrases import parse_phrases


def test_splits_lines_and_sentences_and_strips_noise():
    raw = ("Sure, here is the description:\n"
           "- The image shows a golden retriever.\n"
           "1. Green grass. Bright sunlight.\n"
           "* A red ball.\n"
           "A red ball.\n")
    assert parse_phrases(raw) == ["A golden retriever.", "Green grass.", "Bright sunlight.",
                                  "A red ball."]


def test_drops_trailing_fragment_only_when_truncated():
    raw = "A kitchen.\nA woman cooking.\nA pot of"
    assert parse_phrases(raw, truncated=True) == ["A kitchen.", "A woman cooking."]
    assert parse_phrases(raw)[-1] == "A pot of"


def test_drops_too_short_phrases():
    assert parse_phrases("Dog.\nA dog on a couch.") == ["A dog on a couch."]
