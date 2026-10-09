"""Content-source adapters: enumerate the items of a collection.

The built-in :class:`FolderSource` walks a folder: each image is one item; each text, Markdown
or PDF file is split into passages at paragraph boundaries, and each passage is one item.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterator, List, Protocol

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
TEXT_EXTS = {".txt", ".md", ".markdown", ".rst", ".csv", ".log"}
PDF_EXTS = {".pdf"}


@dataclass
class Item:
    key: str  # stable identity: path, plus "#n" for the n-th passage of a document
    path: str
    kind: str  # "image" or "passage"
    version: str  # changes when the source changes (mtime and size)
    title: str = ""
    text: str = ""  # a passage's own text; empty for images
    segment: int = 0


class Source(Protocol):
    def items(self) -> Iterator[Item]: ...


def split_passages(text: str, max_chars: int = 1500) -> List[str]:
    """Group paragraphs into passages of at most ``max_chars`` (a long paragraph is cut)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: List[str] = []
    cur = ""
    for p in paras:
        while len(p) > max_chars:
            cut = p.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 2 else max_chars
            if cur:
                out.append(cur)
                cur = ""
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if cur and len(cur) + len(p) + 2 > max_chars:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        out.append(cur)
    return out


def _read_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in PDF_EXTS:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise RuntimeError("pip install 'glossdex[pdf]' to index PDF files")
        reader = PdfReader(path)
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


class FolderSource:
    def __init__(self, root: str, images: bool = True, documents: bool = True,
                 max_chars: int = 1500):
        self.root = os.path.abspath(root)
        self.images = images
        self.documents = documents
        self.max_chars = max_chars

    def files(self) -> List[str]:
        out = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for name in sorted(filenames):
                if name.startswith("."):
                    continue
                ext = os.path.splitext(name)[1].lower()
                if (self.images and ext in IMAGE_EXTS) or (
                        self.documents and ext in TEXT_EXTS | PDF_EXTS):
                    out.append(os.path.join(dirpath, name))
        return out

    def items(self) -> Iterator[Item]:
        for path in self.files():
            st = os.stat(path)
            version = f"{int(st.st_mtime)}:{st.st_size}"
            rel = os.path.relpath(path, self.root).replace("\\", "/")
            ext = os.path.splitext(path)[1].lower()
            if ext in IMAGE_EXTS:
                yield Item(key=rel, path=path, kind="image", version=version, title=rel)
                continue
            try:
                passages = split_passages(_read_text(path), self.max_chars)
            except RuntimeError:
                continue
            for i, passage in enumerate(passages):
                yield Item(key=f"{rel}#{i}", path=path, kind="passage", version=version,
                           title=rel if len(passages) == 1 else f"{rel} (part {i + 1})",
                           text=passage, segment=i)
