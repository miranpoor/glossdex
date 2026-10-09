"""A local web UI over the engine. Standard library only; binds to 127.0.0.1 by default."""

from __future__ import annotations

import io
import json
import mimetypes
import threading
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import parse_qs, urlparse

from .engine import Glossdex


def _ui_html() -> bytes:
    return resources.files("glossdex").joinpath("static/index.html").read_bytes()


def make_handler(gx: Glossdex):
    state = {"thread": None, "last": None}

    @lru_cache(maxsize=512)
    def thumb(path: str, size: int) -> tuple:
        try:
            from PIL import Image, ImageOps
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((size, size))
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=82)
                return buf.getvalue(), "image/jpeg"
        except Exception:
            with open(path, "rb") as f:
                return f.read(), mimetypes.guess_type(path)[0] or "application/octet-stream"

    def run_index(rebuild: bool) -> None:
        try:
            rep = gx.index(rebuild=rebuild)
            state["last"] = {"added": rep.added, "unchanged": rep.unchanged,
                             "removed": rep.removed, "failed": rep.failed[:20],
                             "seconds": round(rep.seconds, 1)}
        except Exception as e:
            state["last"] = {"error": str(e)}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(json.dumps(obj).encode("utf-8"), "application/json", code)

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    return self._send(_ui_html(), "text/html; charset=utf-8")
                if url.path == "/api/stats":
                    return self._json({**gx.stats(), "progress": gx.progress,
                                       "last": state["last"]})
                if url.path == "/api/search":
                    s = q.get("s")
                    res = gx.search(q.get("q", ""), strictness=float(s) if s else None,
                                    compare_top_k=int(q.get("k", "0") or 0))
                    return self._json(res.to_dict())
                if url.path == "/api/item":
                    it = gx.item(int(q["id"]))
                    if it is None:
                        return self._json({"error": "not found"}, 404)
                    return self._json({"id": it.id, "title": it.title, "kind": it.kind,
                                       "path": it.path, "text": it.text,
                                       "description": it.description, "model": it.model,
                                       "seconds": it.seconds, "phrases": gx.store.phrases(it.id)})
                if url.path == "/api/thumb":
                    it = gx.item(int(q["id"]))  # only files that are in the index are served
                    if it is None or it.kind != "image":
                        return self._json({"error": "not found"}, 404)
                    body, ctype = thumb(it.path, int(q.get("size", "360")))
                    return self._send(body, ctype)
                return self._json({"error": "not found"}, 404)
            except Exception as e:
                return self._json({"error": str(e)}, 500)

        def do_POST(self):  # noqa: N802
            url = urlparse(self.path)
            if url.path == "/api/index":
                t = state["thread"]
                if t is not None and t.is_alive():
                    return self._json({"started": False, "reason": "already running"})
                rebuild = parse_qs(url.query).get("rebuild", ["0"])[0] == "1"
                state["last"] = None
                t = threading.Thread(target=run_index, args=(rebuild,), daemon=True)
                state["thread"] = t
                t.start()
                return self._json({"started": True})
            return self._json({"error": "not found"}, 404)

    return Handler


def serve(gx: Glossdex, host: str = "127.0.0.1", port: int = 8484,
          open_browser: bool = True) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(gx))
    url = f"http://{host}:{port}/"
    print(f"glossdex is running at {url}  (Ctrl+C to stop)\nIndex: {gx.store.path}")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
