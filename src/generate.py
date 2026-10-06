"""Generate music and write a .mid file.

Usage:
    python -m src.generate --model lstm   --tokens 1500 --temp 1.0 --topk 20
    python -m src.generate --model markov --tokens 1500 --temp 1.0

The LSTM uses models/lstm.npz (pure numpy, no torch needed) if it exists,
otherwise it falls back to models/lstm.pt (needs torch).
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from src.tokens import VOCAB_SIZE, tokens_to_midi

ROOT = Path(__file__).resolve().parent.parent


def _sample(logits, temp, topk, rng):
    lg = np.asarray(logits, dtype=np.float64) / max(temp, 1e-3)
    if topk:
        kth = np.sort(lg)[-topk]
        lg = np.where(lg < kth, -1e9, lg)
    lg -= lg.max()
    p = np.exp(lg)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def _generate_numpy(npz_path, seed, n, temp, topk, rng):
    from src.numpy_lstm import NumpyLSTM

    m = NumpyLSTM(npz_path)
    state, logits = m.init_state(), None
    for t in seed:
        logits, state = m.step(int(t), state)
    out = []
    for _ in range(n):
        t = _sample(logits, temp, topk, rng)
        out.append(t)
        logits, state = m.step(t, state)
    return out


def _generate_torch(ckpt_path, seed, n, temp, topk, rng):
    import torch
    from src.model import MusicLSTM

    ck = torch.load(ckpt_path, map_location="cpu")
    model = MusicLSTM(**ck["cfg"])
    model.load_state_dict(ck["state"])
    model.eval()
    out = []
    with torch.no_grad():
        logits, state = model(torch.tensor([list(map(int, seed))]))
        for _ in range(n):
            t = _sample(logits[0, -1].numpy(), temp, topk, rng)
            out.append(t)
            logits, state = model(torch.tensor([[t]]), state)
    return out


def generate_lstm(ckpt_path, seed, n, temp=1.0, topk=20, rng=None):
    rng = rng or np.random.default_rng()
    npz = Path(ckpt_path).with_suffix(".npz")
    if npz.exists():
        return _generate_numpy(npz, seed, n, temp, topk, rng)
    return _generate_torch(ckpt_path, seed, n, temp, topk, rng)


def generate_markov(pkl_path, seed, n, temp=1.0, rng=None):
    with open(pkl_path, "rb") as f:
        m = pickle.load(f)
    return m.sample(list(map(int, seed)), n, temp, rng)


def get_seed(length=32, seed_rng=None):
    test = np.load(ROOT / "data" / "processed" / "test.npy")
    rng = seed_rng or np.random.default_rng()
    i = int(rng.integers(0, len(test) - length))
    return test[i : i + length]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm", "markov"], default="lstm")
    ap.add_argument("--tokens", type=int, default=1500)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seed = get_seed()
    if args.model == "lstm":
        toks = generate_lstm(ROOT / "models" / "lstm.pt", seed, args.tokens, args.temp, args.topk)
    else:
        toks = generate_markov(ROOT / "models" / "markov.pkl", seed, args.tokens, args.temp)
    out = Path(args.out) if args.out else ROOT / "outputs" / "midi" / f"{args.model}_{int(time.time())}.mid"
    out.parent.mkdir(parents=True, exist_ok=True)
    tokens_to_midi(toks, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
