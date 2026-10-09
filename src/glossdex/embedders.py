"""Embedding adapters: text -> vector, in a document mode and a query mode.

``profile_key`` names the model whose score units the vectors are in; it selects the
thresholds in :mod:`glossdex.profiles`. Two adapters that run the same model (for example
EmbeddingGemma through Ollama or through sentence-transformers) share a key.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import List, Protocol, Sequence

import numpy as np

from .describers import DEFAULT_OLLAMA

#: EmbeddingGemma's task prompts for retrieval (from its model card).
GEMMA_QUERY_PREFIX = "task: search result | query: "
GEMMA_DOCUMENT_PREFIX = "title: none | text: "


class Embedder(Protocol):
    name: str
    profile_key: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class EmbedderError(RuntimeError):
    pass


class OllamaEmbedder:
    """EmbeddingGemma (or any embedding model) served by a local Ollama."""

    def __init__(self, model: str = "embeddinggemma", host: str = DEFAULT_OLLAMA,
                 profile_key: str = None, query_prefix: str = None, document_prefix: str = None,
                 timeout: float = 120.0, batch: int = 32):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.batch = batch
        is_gemma = model.split(":")[0] == "embeddinggemma"
        self.profile_key = profile_key or ("embeddinggemma-300m" if is_gemma else model)
        self.query_prefix = query_prefix if query_prefix is not None else (
            GEMMA_QUERY_PREFIX if is_gemma else "")
        self.document_prefix = document_prefix if document_prefix is not None else (
            GEMMA_DOCUMENT_PREFIX if is_gemma else "")
        self.name = f"ollama:{model}"

    def _embed(self, texts: List[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch):
            payload = {"model": self.model, "input": texts[i:i + self.batch]}
            req = urllib.request.Request(f"{self.host}/api/embed",
                                         data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                raise EmbedderError(f"Ollama returned {e.code}: "
                                    f"{e.read().decode('utf-8', 'replace')[:300]}") from e
            except urllib.error.URLError as e:
                raise EmbedderError(f"Cannot reach Ollama at {self.host}. Is it running?") from e
            out.extend(data["embeddings"])
        return np.asarray(out, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.query_prefix + text])[0]


class SentenceTransformersEmbedder:
    """EmbeddingGemma through sentence-transformers (``pip install glossdex[st]``).

    ``google/embeddinggemma-300m`` is gated on Hugging Face: accept its license on the model
    page and run ``hf auth login`` once.
    """

    def __init__(self, model: str = "google/embeddinggemma-300m", device: str = None,
                 profile_key: str = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise EmbedderError("pip install 'glossdex[st]' to use sentence-transformers") from e
        self._model = SentenceTransformer(model, device=device)
        self.profile_key = profile_key or (
            "embeddinggemma-300m" if "embeddinggemma" in model else model)
        self.name = f"st:{model}"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        encode = getattr(self._model, "encode_document", self._model.encode)
        return np.asarray(encode(list(texts)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        encode = getattr(self._model, "encode_query", self._model.encode)
        return np.asarray(encode([text])[0], dtype=np.float32)
