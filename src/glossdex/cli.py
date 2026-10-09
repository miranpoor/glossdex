"""Command line: ``glossdex index|search|explain|serve|doctor``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

from . import __version__
from .describers import DEFAULT_OLLAMA, OllamaDescriber, TextOnlyDescriber
from .embedders import OllamaEmbedder
from .engine import Glossdex


def _engine(args) -> Glossdex:
    if args.describer == "text-only":
        describer = TextOnlyDescriber()
    else:
        describer = OllamaDescriber(model=args.vision_model, text_model=args.text_model,
                                    host=args.ollama)
    if args.embedder == "st":
        from .embedders import SentenceTransformersEmbedder
        embedder = SentenceTransformersEmbedder()
    else:
        embedder = OllamaEmbedder(model=args.embed_model, host=args.ollama)
    return Glossdex.for_folder(args.folder, describer=describer, embedder=embedder)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("folder", help="the folder to index or search")
    p.add_argument("--ollama", default=os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA))
    p.add_argument("--vision-model", default="gemma4:e4b",
                   help="Ollama model that describes images (default: gemma4:e4b)")
    p.add_argument("--text-model", default=None,
                   help="Ollama model that lists facts of documents (default: the vision model)")
    p.add_argument("--embed-model", default="embeddinggemma")
    p.add_argument("--describer", choices=["ollama", "text-only"], default="ollama",
                   help="text-only: no language model; a document's sentences are its phrases")
    p.add_argument("--embedder", choices=["ollama", "st"], default="ollama",
                   help="st: sentence-transformers (pip install 'glossdex[st]')")


def cmd_index(args) -> int:
    gx = _engine(args)

    def on_item(item, err):
        p = gx.progress
        status = "FAILED " + err if err else "ok"
        print(f"[{p['done']}/{p['total']}] {item.title}  {status}", flush=True)

    rep = gx.index(rebuild=args.rebuild, on_item=on_item)
    print(f"\nIndexed {rep.added}, unchanged {rep.unchanged}, removed {rep.removed}, "
          f"failed {len(rep.failed)} in {rep.seconds:.1f}s. Index: {gx.store.path}")
    return 1 if rep.failed and not rep.added else 0


def _one_line(text: str, width: int = 100) -> str:
    s = " ".join(text.split())
    return s if len(s) <= width else s[: width - 1] + "…"


def _print_result(res, show_trace: bool) -> None:
    if not res.found:
        print("Nothing found.")
    for i, h in enumerate(res.hits, 1):
        extra = "  (near miss)" if h.reason == "pad" else ""
        words = f"  words {h.coverage:.0%}" if h.coverage > 0 else ""
        print(f"{i:2d}. match {h.semantic:.3f}{words}  {h.title}{extra}\n"
              f"      matched: {_one_line(h.matched)}")
    if show_trace:
        print("\n" + res.trace.verdict)


def cmd_search(args) -> int:
    gx = _engine(args)
    res = gx.search(args.query, strictness=args.strictness, compare_top_k=args.top_k)
    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        _print_result(res, not args.quiet)
    return 0


def cmd_explain(args) -> int:
    gx = _engine(args)
    res = gx.search(args.query, strictness=args.strictness)
    shown = {h.id for h in res.hits}
    print(res.trace.verdict + "\n\nFull ranking before the cutoff:")
    for i, h in enumerate(gx.explain(args.query, args.n), 1):
        mark = "SHOWN" if h.id in shown else "     "
        print(f"{i:2d}. {mark} score {h.score:.4f}  semantic {h.semantic:.4f}  "
              f"coverage {h.coverage:.3f}  {h.title}\n      matched: {_one_line(h.matched)}")
    print(f"\nProfile: {res.trace.profile}")
    return 0


def cmd_serve(args) -> int:
    from .server import serve
    serve(_engine(args), host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_doctor(args) -> int:
    ok = True
    print(f"glossdex {__version__}, Python {sys.version.split()[0]}")
    try:
        with urllib.request.urlopen(f"{args.ollama}/api/tags", timeout=5) as r:
            models = {m["name"] for m in json.loads(r.read())["models"]}
        print(f"Ollama at {args.ollama}: running")
        for m in (args.vision_model, args.embed_model):
            have = any(x == m or x.split(":")[0] == m for x in models) or f"{m}:latest" in models
            print(f"  {m}: {'installed' if have else 'MISSING, run: ollama pull ' + m}")
            ok &= have
    except Exception as e:
        print(f"Ollama at {args.ollama}: NOT reachable ({e}). Install from https://ollama.com")
        ok = False
    try:
        import PIL  # noqa: F401
        print("Pillow: installed")
    except ImportError:
        print("Pillow: missing (images are sent at full size)")
    try:
        import pypdf  # noqa: F401
        print("pypdf: installed (PDF files will be indexed)")
    except ImportError:
        print("pypdf: not installed (PDFs skipped; pip install 'glossdex[pdf]')")
    print("Ready." if ok else "Not ready: fix the lines above.")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="glossdex", description=(
        "Glossed locally, indexed by meaning, gated by evidence. Private semantic search over "
        "a folder of images and documents that shows only what the evidence supports."))
    ap.add_argument("--version", action="version", version=f"glossdex {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="describe and embed new or changed files")
    _common(p)
    p.add_argument("--rebuild", action="store_true", help="discard the index and start over")
    p.set_defaults(fn=cmd_index)

    for name, fn, helptext in (("search", cmd_search, "search the index"),
                               ("explain", cmd_explain, "show the ranking and every decision")):
        p = sub.add_parser(name, help=helptext)
        _common(p)
        p.add_argument("query")
        p.add_argument("--strictness", type=float, default=None,
                       help="0..1; lower shows more (default 0.70)")
        if name == "search":
            p.add_argument("--top-k", type=int, default=0,
                           help="show a fixed top-k instead, for comparison")
            p.add_argument("--json", action="store_true")
            p.add_argument("--quiet", action="store_true")
        else:
            p.add_argument("-n", type=int, default=15)
        p.set_defaults(fn=fn)

    p = sub.add_parser("serve", help="open the local web UI")
    _common(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8484)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("doctor", help="check that Ollama and the models are ready")
    p.add_argument("--ollama", default=os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA))
    p.add_argument("--vision-model", default="gemma4:e4b")
    p.add_argument("--embed-model", default="embeddinggemma")
    p.set_defaults(fn=cmd_doctor)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
