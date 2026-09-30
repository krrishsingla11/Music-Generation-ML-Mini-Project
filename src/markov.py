"""Baseline: n-gram (Markov chain) model over the same event tokens.

Usage:
    python -m src.markov --order 2
Trains on data/processed/train.npy, reports val/test next-token accuracy + perplexity,
saves models/markov.pkl.
"""
import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.tokens import VOCAB_SIZE

ROOT = Path(__file__).resolve().parent.parent
V = VOCAB_SIZE


class MarkovModel:
    def __init__(self, order=2):
        self.order = order
        # tables[k] maps a k-token context (as an int) -> Counter of next tokens
        self.tables = [defaultdict(Counter) for _ in range(order + 1)]

    @staticmethod
    def _key(ctx):
        k = 0
        for t in ctx:
            k = k * V + int(t)
        return k

    def fit(self, data):
        data = data.astype(np.int64).tolist()
        for k in range(self.order + 1):
            tab = self.tables[k]
            for i in range(k, len(data)):
                tab[self._key(data[i - k : i])][data[i]] += 1
        return self

    def dist(self, ctx):
        """Probability vector over next token. Uses the longest seen context (backoff),
        mixed 90/10 with the unigram distribution and add-one smoothed."""
        ctx = list(ctx)[-self.order :] if self.order else []
        uni = self.tables[0][0]
        uni_tot = sum(uni.values())
        p_uni = np.full(V, 1.0)
        for t, c in uni.items():
            p_uni[t] += c
        p_uni /= uni_tot + V
        for k in range(min(self.order, len(ctx)), 0, -1):
            c = self.tables[k].get(self._key(ctx[len(ctx) - k :]))
            if c:
                tot = sum(c.values())
                p = np.zeros(V)
                for t, n in c.items():
                    p[t] = n / tot
                return 0.9 * p + 0.1 * p_uni
        return p_uni

    def evaluate(self, data, max_tokens=30000):
        data = data[:max_tokens].astype(np.int64)
        correct, nll, n = 0, 0.0, 0
        for i in range(self.order, len(data)):
            p = self.dist(data[i - self.order : i])
            correct += int(p.argmax() == data[i])
            nll -= np.log(p[data[i]] + 1e-12)
            n += 1
        return {"acc": correct / n, "loss": nll / n, "ppl": float(np.exp(nll / n))}

    def sample(self, seed, n, temp=1.0, rng=None):
        rng = rng or np.random.default_rng()
        out = list(seed)
        for _ in range(n):
            p = self.dist(out[-self.order :]) ** (1.0 / temp)
            p /= p.sum()
            out.append(int(rng.choice(V, p=p)))
        return out[len(seed) :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=2)
    args = ap.parse_args()
    d = ROOT / "data" / "processed"
    train, val, test = (np.load(d / f"{s}.npy") for s in ("train", "val", "test"))
    print(f"training order-{args.order} Markov model on {len(train):,} tokens...")
    from src.markov import MarkovModel as _M  # so the pickle references src.markov, not __main__
    m = _M(args.order).fit(train)
    print("val :", m.evaluate(val))
    print("test:", m.evaluate(test))
    (ROOT / "models").mkdir(exist_ok=True)
    with open(ROOT / "models" / "markov.pkl", "wb") as f:
        pickle.dump(m, f)
    print("saved models/markov.pkl")


if __name__ == "__main__":
    main()
