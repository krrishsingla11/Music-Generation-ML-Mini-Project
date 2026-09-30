"""Download piano MIDI files from piano-midi.de into data/raw/<composer>/.

Usage:
    python scripts/download_data.py                 # default composers
    python scripts/download_data.py bach chopin     # specific composers

If a composer page layout differs, download the .mid files manually
into data/raw/<composer>/ - the rest of the pipeline only needs the files.
"""
import re
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

BASE = "https://www.piano-midi.de/"
DEFAULT = ["bach", "beethoven", "brahms", "chopin", "mozart", "schubert", "haydn", "debussy"]
OUT = Path(__file__).resolve().parent.parent / "data" / "raw"
HEADERS = {"User-Agent": "Mozilla/5.0 (student ML project)"}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def main() -> None:
    composers = sys.argv[1:] or DEFAULT
    for comp in composers:
        page = urljoin(BASE, f"{comp}.htm")
        try:
            html = fetch(page).decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"[skip] {comp}: could not load {page} ({e})")
            continue
        links = sorted(set(re.findall(r'href="([^"]+\.mid)"', html, flags=re.I)))
        print(f"{comp}: {len(links)} midi links")
        dest = OUT / comp
        dest.mkdir(parents=True, exist_ok=True)
        for rel in links:
            url = urljoin(page, rel)
            f = dest / Path(rel).name
            if f.exists():
                continue
            try:
                f.write_bytes(fetch(url))
                time.sleep(0.3)  # be polite to the server
            except Exception as e:
                print(f"  [fail] {url} ({e})")
    total = len(list(OUT.rglob("*.mid")))
    print(f"Done. {total} .mid files in {OUT}")


if __name__ == "__main__":
    main()
