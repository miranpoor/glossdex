"""Description-generator adapters: content item -> a gloss (a list of independent phrases).

Any local model works. The built-in adapters talk to Ollama over its local HTTP API; nothing
leaves the machine. Write your own by implementing :class:`Describer`.
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Protocol

from .phrases import parse_phrases
from .sources import Item

DEFAULT_OLLAMA = "http://localhost:11434"

IMAGE_PROMPT = (
    "Describe this image for a search index as up to {n} short phrases.\n"
    "Put each phrase on its own line and end it with a period.\n"
    "One element per phrase: the main subject, what people or animals are doing, objects, "
    "setting, colors, mood. Do not join two elements into one phrase.\n"
    "If the image is a document, receipt, bill, sign or screen: name its type and copy its key "
    "text (names, organizations, dates, amounts, identifiers).\n"
    "Only state what is visible. No introduction, no conclusion, no markdown."
)

DOCUMENT_PROMPT = (
    "List the key facts of this document as up to {n} concise, independent phrases, one fact "
    "per line, each ending with a period: document type, persons or parties named, "
    "organization, dates, amounts, identifiers, subject, findings or decisions, and requested "
    "actions. Use the document's own words for names, numbers and dates. Do not add "
    "information that is not in the document. No intro or outro.\n\nDocument:\n{text}"
)


@dataclass
class Description:
    text: str  # the raw model output (the whole description)
    phrases: List[str]
    model: str
    seconds: float


class Describer(Protocol):
    name: str

    def describe(self, item: Item) -> Description: ...


class DescriberError(RuntimeError):
    pass


def _post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        raise DescriberError(f"Ollama returned {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise DescriberError(
            f"Cannot reach Ollama at {url}. Is it running? (`ollama serve`)") from e


def _image_b64(path: str, max_side: int) -> str:
    try:
        from PIL import Image, ImageOps
    except ImportError:  # pragma: no cover
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class OllamaDescriber:
    """Describes images with a local vision-language model and text with the same (or another) model.

    Defaults to Gemma 4 E4B, which reads both images and text.
    """

    def __init__(self, model: str = "gemma4:e4b", text_model: Optional[str] = None,
                 host: str = DEFAULT_OLLAMA, max_phrases: int = 7, max_side: int = 1024,
                 timeout: float = 600.0, image_prompt: str = IMAGE_PROMPT,
                 document_prompt: str = DOCUMENT_PROMPT):
        self.model = model
        self.text_model = text_model or model
        self.host = host.rstrip("/")
        self.max_phrases = max_phrases
        self.max_side = max_side
        self.timeout = timeout
        self.image_prompt = image_prompt
        self.document_prompt = document_prompt
        self.name = f"ollama:{self.model}"

    def _generate(self, model: str, prompt: str, images: Optional[List[str]] = None) -> dict:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": 60 * self.max_phrases + 120},
        }
        if images:
            payload["images"] = images
        try:
            return _post(f"{self.host}/api/generate", payload, self.timeout)
        except DescriberError as e:
            if "think" in str(e):  # models without a thinking switch reject the field
                payload.pop("think")
                return _post(f"{self.host}/api/generate", payload, self.timeout)
            raise

    def describe(self, item: Item) -> Description:
        start = time.time()
        if item.kind == "image":
            res = self._generate(self.model, self.image_prompt.format(n=self.max_phrases),
                                 [_image_b64(item.path, self.max_side)])
            model = self.model
        else:
            res = self._generate(self.text_model, self.document_prompt.format(
                n=self.max_phrases, text=item.text))
            model = self.text_model
        text = (res.get("response") or "").strip()
        truncated = res.get("done_reason") == "length"
        return Description(text, parse_phrases(text, truncated), model, time.time() - start)


class TextOnlyDescriber:
    """No model at all: an item's own sentences are its phrases. Text and documents only.

    Useful to try glossdex on documents in seconds, and for text that is already terse
    (notes, tickets, logs). Images are skipped.
    """

    name = "text-only"

    def describe(self, item: Item) -> Description:
        if item.kind == "image":
            raise DescriberError("text-only describer cannot describe images")
        start = time.time()
        text = item.text.strip()
        return Description(text, parse_phrases(text), self.name, time.time() - start)
