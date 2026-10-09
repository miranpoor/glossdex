"""Download 40 sample photos (Open Images V7, CC BY 2.0) into examples/sample-images/photos/.

The photos are not stored in this repository; this script fetches them from the Open Images
mirror and writes examples/sample-images/ATTRIBUTION.md. Standard library only.

    python examples/get_sample_images.py
"""

import csv
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
LIST = os.path.join(HERE, "sample-images", "images.csv")
OUT = os.path.join(HERE, "sample-images", "photos")
URL = "https://open-images-dataset.s3.amazonaws.com/validation/{id}.jpg"


def fetch(row):
    path = os.path.join(OUT, f"{row['image_id']}.jpg")
    if os.path.exists(path):
        return None
    try:
        with urllib.request.urlopen(URL.format(id=row["image_id"]), timeout=60) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
        return None
    except Exception as e:  # report and continue
        return f"{row['image_id']}: {e}"


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(LIST, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with ThreadPoolExecutor(max_workers=8) as ex:
        errors = [e for e in ex.map(fetch, rows) if e]
    # Next to the photos folder, not inside it, so it is not indexed with them.
    with open(os.path.join(HERE, "sample-images", "ATTRIBUTION.md"), "w", encoding="utf-8") as f:
        f.write("# Attribution\n\nPhotos from Open Images V7, each licensed CC BY 2.0 by its "
                "author. Not modified.\n\n")
        for r in rows:
            f.write(f"- `{r['image_id']}.jpg`: \"{r['title']}\" by [{r['author']}]"
                    f"({r['author_url']}), [source]({r['source_page']}), "
                    f"[CC BY 2.0]({r['license']})\n")
    for e in errors:
        print("FAILED", e)
    print(f"{len(rows) - len(errors)} of {len(rows)} photos in {OUT}")


if __name__ == "__main__":
    main()
