"""Download MAESTRO v3 (MIDI only, ~60 MB) into data/raw/maestro/."""
import urllib.request, zipfile
from pathlib import Path

URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)
zpath = OUT / "maestro.zip"

if not zpath.exists():
    print("Downloading...")
    urllib.request.urlretrieve(URL, zpath)
print("Extracting...")
with zipfile.ZipFile(zpath) as z:
    z.extractall(OUT / "maestro")
print(len(list((OUT / "maestro").rglob("*.mid*"))), "midi files")