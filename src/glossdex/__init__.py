"""glossdex: glossed locally, indexed by meaning, gated by evidence."""

from .describers import Describer, OllamaDescriber, TextOnlyDescriber
from .embedders import Embedder, OllamaEmbedder, SentenceTransformersEmbedder
from .engine import Context, Glossdex, Hit, IndexReport, SearchResult
from .phrases import parse_phrases
from .profiles import EMBEDDINGGEMMA, EMBEDDINGGEMMA_PASSAGES, Profile
from .scoring import ScoringIndex, Trace, Unit, select
from .sources import FolderSource, Item

__version__ = "0.1.0"

__all__ = [
    "Glossdex", "SearchResult", "Hit", "Context", "IndexReport",
    "Describer", "OllamaDescriber", "TextOnlyDescriber",
    "Embedder", "OllamaEmbedder", "SentenceTransformersEmbedder",
    "FolderSource", "Item", "Profile", "EMBEDDINGGEMMA", "EMBEDDINGGEMMA_PASSAGES",
    "ScoringIndex", "Unit", "Trace", "select", "parse_phrases",
]
