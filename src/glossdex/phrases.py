"""Turn a model's free-text description into clean, standalone phrases (retrieval units)."""

from __future__ import annotations

import re
from typing import List

_LIST_MARKER = re.compile(r"^\s*([-*•]|\d+[.)])\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_TERMINATORS = ".!?"

#: Intro, outro and refusal lines, dropped entirely.
CHATTER_STARTS = ("sure", "okay", "ok,", "here is", "here are", "certainly", "of course",
                  "i can", "i'll")

#: Leading filler, stripped so each phrase begins with its content.
NOISE_PREFIXES = (
    "the image shows", "the image depicts", "the image contains", "the image features",
    "this image shows", "this image depicts", "the photo shows", "the picture shows",
    "the document shows", "this document is", "this is an image of", "this is a photo of",
    "this is a picture of", "this is", "i see", "it appears", "there is", "there are",
    "we can see", "you can see",
)

MIN_PHRASE_CHARS = 5


def parse_phrases(text: str, truncated: bool = False) -> List[str]:
    """Split on line breaks, then sentences; strip markers, chatter and filler; de-duplicate.

    Does not cap the number of phrases: once tokens are generated, every extra phrase is an
    extra retrieval unit. When generation was cut short (``truncated``) a trailing phrase that
    did not finish its sentence is dropped, because a fragment is a poor retrieval unit.
    """
    candidates: List[str] = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        candidates.extend(s for s in _SENTENCE_END.split(line.strip()) if s.strip())

    if truncated and candidates:
        tail = candidates[-1].rstrip()
        if not tail or tail[-1] not in _TERMINATORS:
            candidates.pop()

    out: List[str] = []
    seen = set()
    for raw in candidates:
        s = _LIST_MARKER.sub("", raw.strip()).strip().strip("*").strip()
        if not s:
            continue
        lower = s.lower()
        if lower.startswith(CHATTER_STARTS) or lower.endswith(":") or "sorry," in lower:
            continue
        for prefix in NOISE_PREFIXES:
            if lower.startswith(prefix):
                cleaned = s[len(prefix):].lstrip(" ,:")
                if len(cleaned) > 3:
                    s = cleaned[0].upper() + cleaned[1:]
                break
        if len(s) <= MIN_PHRASE_CHARS:
            continue
        key = s.lower().rstrip(". ")
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out
