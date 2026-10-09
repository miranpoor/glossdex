"""Answer questions from a folder with a local model, and say "not found" when it isn't there.

The number of passages handed to the model is decided by the evidence, not a fixed k. When the
result set is empty the model is never called, so it cannot invent an answer.

    python examples/rag_ollama.py "When is the school field trip?"
"""

import json
import os
import sys
import urllib.request

from glossdex import Glossdex, TextOnlyDescriber

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("GLOSSDEX_ANSWER_MODEL", "gemma4:e4b")


def ask_ollama(prompt: str) -> str:
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                         "options": {"temperature": 0}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["response"].strip()


def main() -> None:
    question = " ".join(sys.argv[1:]) or "When is the school field trip?"
    gx = Glossdex.for_folder(os.path.join(HERE, "sample-docs"), describer=TextOnlyDescriber())
    gx.index()
    ctx = gx.context(question)
    if not ctx:
        print("Not found in your documents.")
        return
    prompt = (f"Answer the question using only the sources below, and cite them like [1].\n\n"
              f"{ctx.text}\n\nQuestion: {question}\nAnswer:")
    print(ask_ollama(prompt))
    print("\nSources: " + "; ".join(f"[{i}] {h.title}" for i, h in enumerate(ctx.passages, 1)))


if __name__ == "__main__":
    main()
