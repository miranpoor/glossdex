"""Index the sample documents and run a few searches, including ones with no answer.

    ollama pull embeddinggemma
    python examples/quickstart.py            # documents only: no vision model needed
    python examples/quickstart.py --describe # let gemma4:e4b write each document's gloss
"""

import os
import sys

from glossdex import Glossdex, OllamaDescriber, TextOnlyDescriber

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, "sample-docs")

QUERIES = [
    "how much is the electricity bill",
    "when does my lease end",
    "my son's doctor visit",          # "son" reaches "Jordan Morgan, age 7" only by meaning
    "laptop serial number",
    "flight to Portugal",
    "charitable giving tax receipt",
    "dog vaccination record",         # not in the folder
    "mortgage statement",             # not in the folder
]


def main() -> None:
    describer = OllamaDescriber() if "--describe" in sys.argv else TextOnlyDescriber()
    gx = Glossdex.for_folder(FOLDER, describer=describer)
    report = gx.index(rebuild=True)
    print(f"Indexed {report.added} passages in {report.seconds:.1f}s\n")
    for q in QUERIES:
        res = gx.search(q)
        found = ", ".join(h.title for h in res.hits) or "nothing found"
        print(f"{q!r:40} -> {found}")
        print(f"{'':43}{res.trace.verdict}\n")


if __name__ == "__main__":
    main()
