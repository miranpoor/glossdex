"""The glossdex engine: index a collection, search it, hand context to a language model."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .describers import Describer, DescriberError, OllamaDescriber, TextOnlyDescriber
from .embedders import Embedder, OllamaEmbedder
from .lexical import LexicalIndex
from .profiles import DESCRIPTIONS, PASSAGES, Profile, profile_for
from .scoring import (UNIT_DESCRIPTION, UNIT_PHRASE, ScoredItem, ScoringIndex, Trace, Unit,
                      select, top_k)
from .sources import FolderSource, Item, Source
from .store import Store

INDEX_DIR = ".glossdex"


@dataclass
class Hit:
    id: int
    title: str
    path: str
    kind: str
    score: float
    semantic: float
    coverage: float
    matched: str  # the retrieval unit that matched best
    reason: str  # "band", "pad" or "top-k"
    text: str = ""  # a passage's text
    phrases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return dict(self.__dict__)


@dataclass
class SearchResult:
    query: str
    hits: List[Hit]
    trace: Trace
    seconds: float

    @property
    def found(self) -> bool:
        return bool(self.hits)

    def to_dict(self) -> Dict:
        return {"query": self.query, "hits": [h.to_dict() for h in self.hits],
                "trace": self.trace.to_dict(), "seconds": self.seconds}


@dataclass
class IndexReport:
    added: int = 0
    unchanged: int = 0
    removed: int = 0
    failed: List[str] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class Context:
    """Passages for a language model, or an explicit "nothing found"."""

    found: bool
    passages: List[Hit]
    text: str  # ready to paste into a prompt, with [n] source references

    def __bool__(self) -> bool:
        return self.found


class Glossdex:
    """Glossed locally, indexed by meaning, gated by evidence.

    >>> gx = Glossdex.for_folder("~/Documents/scans")
    >>> gx.index()
    >>> result = gx.search("electricity bill from June")
    >>> result.found, [h.title for h in result.hits]
    """

    def __init__(self, db_path: str, describer: Optional[Describer] = None,
                 embedder: Optional[Embedder] = None, profile: Optional[Profile] = None,
                 source: Optional[Source] = None):
        self.store = Store(db_path)
        self.describer = describer if describer is not None else OllamaDescriber()
        self.embedder = embedder if embedder is not None else OllamaEmbedder()
        # Raw text indexed as is gets the passages profile; model-written glosses the default.
        content = PASSAGES if self.describer.name == TextOnlyDescriber.name else DESCRIPTIONS
        self.profile = profile_for(self.embedder.profile_key, profile, content)
        self.source = source
        self._index: Optional[ScoringIndex] = None
        self._index_gen = -1
        self._index_lock = threading.Lock()
        self._items: Dict[int, object] = {}
        self.progress: Dict = {"running": False, "done": 0, "total": 0, "current": "",
                               "errors": 0}

    @classmethod
    def for_folder(cls, folder: str, **kwargs) -> "Glossdex":
        """An engine whose index lives in ``<folder>/.glossdex/index.db``."""
        folder = os.path.abspath(os.path.expanduser(folder))
        source = kwargs.pop("source", None) or FolderSource(folder)
        return cls(os.path.join(folder, INDEX_DIR, "index.db"), source=source, **kwargs)

    # -- indexing --------------------------------------------------------------------------

    def _check_embedder(self) -> None:
        key = self.store.get_meta("embedder")
        if key is None:
            self.store.set_meta("embedder", self.embedder.profile_key)
        elif key != self.embedder.profile_key:
            raise RuntimeError(
                f"This index was built with embedding model '{key}', not "
                f"'{self.embedder.profile_key}'. Vectors from different models cannot be "
                f"compared; rebuild with `glossdex index --rebuild`.")

    def _embed_units(self, desc) -> List[tuple]:
        texts = ([desc.text] if desc.text else []) + desc.phrases
        if not texts:
            return []
        vecs = self.embedder.embed_documents(texts)
        types = ([UNIT_DESCRIPTION] if desc.text else []) + [UNIT_PHRASE] * len(desc.phrases)
        return list(zip(types, texts, vecs))

    def index(self, source: Optional[Source] = None, rebuild: bool = False,
              on_item: Optional[Callable[[Item, Optional[str]], None]] = None) -> IndexReport:
        """Describe and embed every new or changed item; drop items that disappeared.

        Embedding and storing item k overlaps with describing item k+1 (at most one embedding
        outstanding), since the two usually run on different hardware.
        """
        source = source or self.source
        if source is None:
            raise ValueError("no source: pass one, or use Glossdex.for_folder()")
        if rebuild:
            self.store.clear()
        self._check_embedder()
        start = time.time()
        report = IndexReport()
        known = self.store.versions()
        items = list(source.items())
        seen = {it.key for it in items}
        stale = [k for k in known if k not in seen]
        self.store.delete_keys(stale)
        report.removed = len(stale)
        todo = [it for it in items if known.get(it.key) != it.version]
        report.unchanged = len(items) - len(todo)
        self.progress.update(running=True, done=0, total=len(todo), current="", errors=0)

        pending: Optional[tuple] = None  # (item, desc, future)

        def finish(p: tuple) -> None:
            item, desc, fut = p
            try:
                units = fut.result()
                self.store.put(item, desc.text, desc.model, desc.seconds, units)
                report.added += 1
                err = None
            except Exception as e:  # keep going; report at the end
                report.failed.append(f"{item.key}: {e}")
                self.progress["errors"] += 1
                err = str(e)
            self.progress["done"] += 1
            if on_item:
                on_item(item, err)

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                for item in todo:
                    self.progress["current"] = item.title
                    try:
                        desc = self.describer.describe(item)
                    except DescriberError as e:
                        if pending:
                            finish(pending)
                            pending = None
                        if "Cannot reach" in str(e):
                            raise
                        report.failed.append(f"{item.key}: {e}")
                        self.progress["errors"] += 1
                        self.progress["done"] += 1
                        if on_item:
                            on_item(item, str(e))
                        continue
                    fut: Future = pool.submit(self._embed_units, desc)
                    if pending:
                        finish(pending)
                    pending = (item, desc, fut)
                if pending:
                    finish(pending)
        finally:
            self.progress.update(running=False, current="")
        report.seconds = time.time() - start
        return report

    # -- searching -------------------------------------------------------------------------

    def _scoring_index(self) -> ScoringIndex:
        with self._index_lock:
            if self._index is None or self._index_gen != self.store.generation:
                gen = self.store.generation
                items = self.store.items()
                units = [Unit(i, t, txt, v) for i, t, txt, v in self.store.units()]
                texts: Dict[int, List[str]] = {}
                for u in units:
                    texts.setdefault(u.item_id, []).append(u.text)
                for item_id, it in items.items():
                    if it.text and item_id in texts:
                        texts[item_id].append(it.text)  # exact identifiers match lexically
                self._index = ScoringIndex(units, lexical=LexicalIndex.build(texts))
                self._items = items
                self._index_gen = gen
            return self._index

    def _hit(self, r: ScoredItem) -> Hit:
        it = self._items[r.item_id]
        return Hit(id=r.item_id, title=it.title, path=it.path, kind=it.kind, score=r.score,
                   semantic=r.base_score, coverage=r.coverage, matched=r.best_text,
                   reason=r.reason, text=it.text or "")

    def search(self, query: str, strictness: Optional[float] = None, limit: int = 50,
               compare_top_k: int = 0) -> SearchResult:
        """The evidence-gated result set for ``query``; may be empty, by design.

        ``strictness`` (0..1, default the profile's 0.70): lower shows more; at 0 every
        suppressing rule is off and the ranking is cut only by the ratio.
        ``compare_top_k`` > 0 returns a fixed top-k instead, for comparison.
        """
        start = time.time()
        index = self._scoring_index()
        query = query.strip()
        if not query or len(index) == 0:
            return SearchResult(query, [], Trace(verdict="The index is empty." if not len(index)
                                                  else "Empty query."), 0.0)
        qv = self.embedder.embed_query(query)
        p = self.profile.at_strictness(strictness) if strictness is not None else self.profile
        ranked = index.rank(qv, query, p)
        if compare_top_k > 0:
            shown, trace = top_k(ranked, compare_top_k), Trace(
                profile=f"fixed top-{compare_top_k}",
                verdict=f"Fixed top-{compare_top_k}: always shows {compare_top_k}, "
                        f"whether or not anything matches.")
            trace.shown = len(shown)
        else:
            shown, trace = select(index, qv, query, p, limit=limit, ranked=ranked)
        if not self.profile.calibrated:
            trace.verdict += " (Thresholds are not calibrated for this embedding model.)"
        return SearchResult(query, [self._hit(r) for r in shown], trace, time.time() - start)

    def explain(self, query: str, n: int = 20) -> List[Hit]:
        """The full ranking before any cutoff, for diagnosing a miss."""
        index = self._scoring_index()
        if not len(index):
            return []
        ranked = index.rank(self.embedder.embed_query(query), query, self.profile)
        return [self._hit(r) for r in ranked[:n]]

    def context(self, question: str, max_chars: int = 6000,
                strictness: Optional[float] = None) -> Context:
        """Context for retrieval-augmented generation, sized by the evidence, not a fixed k.

        If nothing qualifies, ``found`` is False: tell the model so, or answer "not found"
        without calling it.
        """
        result = self.search(question, strictness=strictness)
        passages, blocks, used = [], [], 0
        for i, h in enumerate(result.hits, 1):
            body = h.text or self.store.get(h.id).description or ""
            block = f"[{i}] {h.title}\n{body.strip()}"
            if used + len(block) > max_chars and passages:
                break
            passages.append(h)
            blocks.append(block)
            used += len(block)
        return Context(bool(passages), passages, "\n\n".join(blocks))

    def item(self, item_id: int):
        return self.store.get(item_id)

    def stats(self) -> Dict:
        return {"items": self.store.count(), "profile": self.profile.describe(),
                "calibrated": self.profile.calibrated, "describer": self.describer.name,
                "embedder": self.embedder.name, "db": self.store.path}
