"""Download MAESTRO v3 (MIDI only, ~60 MB) into data/raw/maestro/."""
import urllib.request, zipfile, ssl
from pathlib import Path

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)
zpath = OUT / "maestro.zip"

if not zpath.exists():
    print("Downloading...")
    with urllib.request.urlopen(URL, context=ctx) as response, open(zpath, 'wb') as out_file:
        out_file.write(response.read())
print("Extracting...")
with zipfile.ZipFile(zpath) as z:
    z.extractall(OUT / "maestro")
print(len(list((OUT / "maestro").rglob("*.mid*"))), "midi files")
