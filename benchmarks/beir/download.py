"""Downloads every collection of the core benchmark (BEIR format) into data/.

    python download.py

Test collections: SciFact, NFCorpus, FiQA.
Development collections, used only to choose the text profile (tune.py): SciDocs (scientific
papers), ArguAna (debate arguments) and CQADupStack webmasters (technical Q&A). None of them is a
test collection.
"""
import io
import os
import urllib.request
import zipfile

from huggingface_hub import hf_hub_download

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
UKP = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


def from_ukp(name):
    if os.path.exists(os.path.join(DATA, name, "corpus.jsonl")):
        return
    print("downloading", name)
    with urllib.request.urlopen(UKP.format(name)) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(DATA)


def from_hf(name, repo):
    dst = os.path.join(DATA, name)
    if os.path.exists(os.path.join(dst, "corpus.jsonl")):
        return
    print("downloading", name)
    for f in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        p = hf_hub_download(repo, f, repo_type="dataset")
        os.makedirs(os.path.dirname(os.path.join(dst, f)), exist_ok=True)
        with open(p, "rb") as src, open(os.path.join(dst, f), "wb") as out:
            out.write(src.read())


if __name__ == "__main__":
    for name in ("scifact", "nfcorpus", "fiqa", "scidocs", "arguana"):
        from_ukp(name)
    from_hf("webmasters", "mteb/cqadupstack-webmasters")
    for n in ("scifact", "nfcorpus", "fiqa", "scidocs", "arguana", "webmasters"):
        d = os.path.join(DATA, n)
        print(n, sum(1 for _ in open(os.path.join(d, "corpus.jsonl"), encoding="utf-8")), "docs,",
              sum(1 for _ in open(os.path.join(d, "qrels", "test.tsv"), encoding="utf-8")) - 1, "judgments")
