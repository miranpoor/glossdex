"""A single-file SQLite store: items, their glosses and their vectors.

Each item's analysis is written in one transaction that replaces any previous analysis of the
item, so re-indexing is idempotent and an interrupted run never leaves half an item behind.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY,
    key TEXT UNIQUE NOT NULL,
    path TEXT NOT NULL,
    kind TEXT NOT NULL,
    title TEXT,
    text TEXT,
    version TEXT,
    description TEXT,
    model TEXT,
    seconds REAL,
    indexed_at REAL
);
CREATE TABLE IF NOT EXISTS units (
    item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    type INTEGER NOT NULL,
    text TEXT NOT NULL,
    vec BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS units_item ON units(item_id);
"""


@dataclass
class StoredItem:
    id: int
    key: str
    path: str
    kind: str
    title: str
    text: str
    version: str
    description: str
    model: str
    seconds: float


class Store:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.executescript(SCHEMA)
        self.generation = 0  # bumped on every write; the engine rebuilds its index when it moves

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # -- meta ------------------------------------------------------------------------------

    def get_meta(self, k: str) -> Optional[str]:
        with self._lock:
            row = self._db.execute("SELECT v FROM meta WHERE k = ?", (k,)).fetchone()
        return row[0] if row else None

    def set_meta(self, k: str, v: str) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO meta(k, v) VALUES (?, ?)", (k, v))

    # -- items -----------------------------------------------------------------------------

    def versions(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._db.execute("SELECT key, version FROM items"))

    def put(self, item, description: str, model: str, seconds: float,
            units: Sequence[tuple]) -> int:
        """Replace an item's analysis atomically. ``units`` is ``[(type, text, vector), ...]``."""
        with self._lock, self._db:
            self._db.execute("DELETE FROM items WHERE key = ?", (item.key,))
            cur = self._db.execute(
                "INSERT INTO items(key, path, kind, title, text, version, description, model,"
                " seconds, indexed_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (item.key, item.path, item.kind, item.title, item.text, item.version,
                 description, model, seconds, time.time()))
            item_id = cur.lastrowid
            self._db.executemany(
                "INSERT INTO units(item_id, type, text, vec) VALUES (?,?,?,?)",
                [(item_id, t, txt, np.asarray(v, dtype=np.float32).tobytes())
                 for t, txt, v in units])
        self.generation += 1
        return item_id

    def delete_keys(self, keys: Sequence[str]) -> None:
        if not keys:
            return
        with self._lock, self._db:
            self._db.executemany("DELETE FROM items WHERE key = ?", [(k,) for k in keys])
        self.generation += 1

    def clear(self) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM items")
            self._db.execute("DELETE FROM meta")
        self.generation += 1

    def items(self) -> Dict[int, StoredItem]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, key, path, kind, title, text, version, description, model, seconds"
                " FROM items").fetchall()
        return {r[0]: StoredItem(*r) for r in rows}

    def get(self, item_id: int) -> Optional[StoredItem]:
        with self._lock:
            r = self._db.execute(
                "SELECT id, key, path, kind, title, text, version, description, model, seconds"
                " FROM items WHERE id = ?", (item_id,)).fetchone()
        return StoredItem(*r) if r else None

    def phrases(self, item_id: int) -> List[str]:
        with self._lock:
            return [r[0] for r in self._db.execute(
                "SELECT text FROM units WHERE item_id = ? AND type = 1 ORDER BY rowid",
                (item_id,))]

    def units(self) -> Iterator[tuple]:
        with self._lock:
            rows = self._db.execute(
                "SELECT item_id, type, text, vec FROM units ORDER BY item_id, rowid").fetchall()
        for item_id, t, text, blob in rows:
            yield item_id, t, text, np.frombuffer(blob, dtype=np.float32)

    def count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM items").fetchone()[0]
