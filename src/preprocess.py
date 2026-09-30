"""MIDI files -> token arrays with a piece-level 80/10/10 split.

Usage:
    python -m src.preprocess --max-files 200
Outputs (data/processed/): train.npy, val.npy, test.npy, meta.json
Also writes outputs/midi/roundtrip_check.mid (decoded from the first train piece)
so you can listen and confirm the tokenizer is lossless-ish.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.tokens import VOCAB_SIZE, midi_to_tokens, tokens_to_midi

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--max-files", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = sorted(list(Path(args.raw).rglob("*.mid")) + list(Path(args.raw).rglob("*.midi")))
    print(f"found {len(files)} midi files")
    if not files:
        raise SystemExit("no MIDI files found - check data/raw/")
    random.Random(args.seed).shuffle(files)
    files = files[: args.max_files]

    n = len(files)
    n_tr, n_va = int(0.8 * n), int(0.1 * n)
    splits = {
        "train": files[:n_tr],
        "val": files[n_tr : n_tr + n_va],
        "test": files[n_tr + n_va :],
    }

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"vocab_size": VOCAB_SIZE, "splits": {}}
    first_train = None

    for name, fl in splits.items():
        arrs, used = [], []
        for f in tqdm(fl, desc=name):
            try:
                toks = midi_to_tokens(f)
            except Exception as e:
                print(f"  [skip] {f.name}: {e}")
                continue
            if len(toks) < 200:
                continue
            arrs.append(np.array(toks, dtype=np.int16))
            used.append(f.name)
            if name == "train" and first_train is None:
                first_train = toks
        data = np.concatenate(arrs) if arrs else np.zeros(0, dtype=np.int16)
        np.save(out_dir / f"{name}.npy", data)
        meta["splits"][name] = {"files": used, "n_files": len(used), "n_tokens": int(len(data))}
        print(f"{name}: {len(used)} pieces, {len(data):,} tokens")

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    if first_train:
        chk = ROOT / "outputs" / "midi"
        chk.mkdir(parents=True, exist_ok=True)
        tokens_to_midi(first_train[:3000], chk / "roundtrip_check.mid")
        print(f"wrote {chk / 'roundtrip_check.mid'} (listen to it!)")


if __name__ == "__main__":
    main()
