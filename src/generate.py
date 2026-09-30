"""Generate music and write a .mid file.

Usage:
    python -m src.generate --model lstm   --tokens 1500 --temp 1.0 --topk 20
    python -m src.generate --model markov --tokens 1500 --temp 1.0
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from src.tokens import VOCAB_SIZE, tokens_to_midi

ROOT = Path(__file__).resolve().parent.parent


def generate_lstm(ckpt_path, seed, n, temp=1.0, topk=20, rng=None):
    import torch
    from src.model import MusicLSTM

    rng = rng or np.random.default_rng()
    ck = torch.load(ckpt_path, map_location="cpu")
    model = MusicLSTM(**ck["cfg"])
    model.load_state_dict(ck["state"])
    model.eval()
    out = []
    with torch.no_grad():
        x = torch.tensor([list(map(int, seed))])
        logits, state = model(x)
        for _ in range(n):
            lg = logits[0, -1] / max(temp, 1e-3)
            if topk:
                v, _ = torch.topk(lg, topk)
                lg = torch.where(lg < v[-1], torch.full_like(lg, -1e9), lg)
            p = torch.softmax(lg, -1).numpy().astype(np.float64)
            p /= p.sum()
            t = int(rng.choice(len(p), p=p))
            out.append(t)
            logits, state = model(torch.tensor([[t]]), state)
    return out


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
