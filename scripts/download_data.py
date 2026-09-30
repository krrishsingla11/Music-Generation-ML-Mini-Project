"""Crawl piano-midi.de and download all .mid files into data/raw/.
Usage: python scripts/download_data.py
"""
import re
import ssl
import time
import urllib.request
from pathlib import Path
from urllib.parse import urljoin, urlparse

OUT = Path(__file__).resolve().parent.parent / "data" / "raw"
HEADERS = {"User-Agent": "Mozilla/5.0 (student ML project)"}
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
START = ["https://www.piano-midi.de/midi_files.htm", "http://www.piano-midi.de/"]
HOST = "piano-midi.de"


def fetch(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
        return r.read()


def links(html, base):
    out = []
    for h in re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I):
        u = urljoin(base, h.split("#")[0])
        if HOST in urlparse(u).netloc:
            out.append(u)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seen, queue, midis = set(), [], set()
    for s in START:
        try:
            fetch(s)
            queue.append((s, 0))
            break
        except Exception as e:
            print(f"[start fail] {s} ({e})")
    while queue:
        url, depth = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            html = fetch(url).decode("utf-8", errors="ignore")
        except Exception:
            continue
        for u in links(html, url):
            low = u.lower()
            if low.endswith(".mid") or low.endswith(".midi"):
                midis.add(u)
            elif low.endswith((".htm", ".html")) and depth < 2 and u not in seen:
                queue.append((u, depth + 1))
        print(f"crawled {len(seen)} pages, {len(midis)} midi links", end="\r")
    print(f"\nFound {len(midis)} midi files. Downloading...")
    for i, u in enumerate(sorted(midis), 1):
        parts = urlparse(u).path.strip("/").split("/")
        comp = parts[-2] if len(parts) >= 2 else "misc"
        dest = OUT / comp
        dest.mkdir(exist_ok=True)
        f = dest / parts[-1]
        if f.exists():
            continue
        try:
            f.write_bytes(fetch(u))
            time.sleep(0.2)
        except Exception as e:
            print(f"[fail] {u} ({e})")
        if i % 50 == 0:
            print(f"  {i}/{len(midis)}")
    print(f"Done. {len(list(OUT.rglob('*.mid')))} .mid files in {OUT}")


if __name__ == "__main__":
    main()