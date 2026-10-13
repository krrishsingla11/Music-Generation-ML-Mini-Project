"""Fair evaluation: Markov baseline vs LSTM on the SAME targets.

Protocol
  * the whole test split (data/processed/test.npy) is cut into non-overlapping
    windows of 256 tokens (the incomplete tail is dropped);
  * inside every window, every token except the first is a prediction target;
  * the LSTM sees the window so far (state reset at each window start), the
    Markov model sees at most its last `order` tokens of the same window;
  * identical target tokens for every model => numbers are directly comparable.

Needs NO torch: the LSTM runs from models/lstm.npz (numpy).

Usage:
    python -m src.evaluate            # unigram, order-2 Markov, LSTM
    python -m src.evaluate --order3   # also order-3 Markov (needs more RAM)

Writes outputs/results.json, docs/results.md and updates the results block in
README.md (between <!-- RESULTS:START --> and <!-- RESULTS:END -->).
"""
import argparse
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np

from src.markov import MarkovModel
from src.tokens import VOCAB_SIZE

ROOT = Path(__file__).resolve().parent.parent
V = VOCAB_SIZE


def make_windows(data, w):
    n = len(data) // w
    return data[: n * w].reshape(n, w).astype(np.int64)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def lstm_scores(npz_path, win, batch=128):
    z = np.load(npz_path)
    L, H = int(z["layers"]), int(z["hidden"])
    w_ih = [z[f"w_ih_{k}"] for k in range(L)]
    w_hh = [z[f"w_hh_{k}"] for k in range(L)]
    b = [z[f"b_ih_{k}"] + z[f"b_hh_{k}"] for k in range(L)]
    emb, out_w, out_b = z["embed"], z["out_w"], z["out_b"]
    nll, correct, count = 0.0, 0, 0
    for s in range(0, len(win), batch):
        x = win[s : s + batch]
        B = len(x)
        h = [np.zeros((B, H), np.float32) for _ in range(L)]
        c = [np.zeros((B, H), np.float32) for _ in range(L)]
        for t in range(win.shape[1] - 1):
            inp = emb[x[:, t]]
            for k in range(L):
                g = inp @ w_ih[k].T + h[k] @ w_hh[k].T + b[k]
                i, f, gg, o = np.split(g, 4, axis=1)
                c[k] = _sig(f) * c[k] + _sig(i) * np.tanh(gg)
                h[k] = _sig(o) * np.tanh(c[k])
                inp = h[k]
            logits = inp @ out_w.T + out_b
            y = x[:, t + 1]
            m = logits.max(1)
            lse = m + np.log(np.exp(logits - m[:, None]).sum(1))
            nll += float((lse - logits[np.arange(B), y]).sum(dtype=np.float64))
            correct += int((logits.argmax(1) == y).sum())
            count += B
    return nll / count, correct / count, count


def markov_scores(m, win):
    uni = m.tables[0][0]
    p_uni = np.ones(V)
    for t, n in uni.items():
        p_uni[t] += n
    p_uni /= sum(uni.values()) + V
    cache = {}
    nll, correct, count = 0.0, 0, 0
    for row in win.tolist():
        for i in range(1, len(row)):
            ctx = tuple(row[max(0, i - m.order) : i])
            info = cache.get(ctx)
            if info is None:
                c = None
                for k in range(len(ctx), 0, -1):
                    cc = m.tables[k].get(m._key(ctx[len(ctx) - k :]))
                    if cc:
                        c = cc
                        break
                if c:
                    tot = sum(c.values())
                    p = 0.1 * p_uni
                    for t, n in c.items():
                        p[t] += 0.9 * n / tot
                    info = (c, tot, int(p.argmax()))
                else:
                    info = (None, 0, int(p_uni.argmax()))
                cache[ctx] = info
            c, tot, am = info
            y = row[i]
            py = p_uni[y] if c is None else 0.9 * c.get(y, 0) / tot + 0.1 * p_uni[y]
            nll -= np.log(py + 1e-12)
            correct += int(am == y)
            count += 1
    return nll / count, correct / count, count


def unigram_scores(train, win):
    p = np.bincount(train.astype(np.int64), minlength=V) + 1.0
    p /= p.sum()
    y = win[:, 1:].reshape(-1)
    return float(-np.log(p[y]).mean()), float((y == p.argmax()).mean()), len(y)


def row(name, loss, acc):
    return {"name": name, "loss": float(loss), "ppl": float(np.exp(loss)), "acc": float(acc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--order3", action="store_true")
    ap.add_argument("--lstm", default=str(ROOT / "models" / "lstm.npz"))
    args = ap.parse_args()

    d = ROOT / "data" / "processed"
    train = np.load(d / "train.npy")
    test = np.load(d / "test.npy")
    win = make_windows(test, args.window)
    n_targets = len(win) * (args.window - 1)
    print(f"test tokens: {len(test):,}  windows: {len(win):,}  targets per model: {n_targets:,}")

    pkl = ROOT / "models" / "markov.pkl"
    if pkl.exists():
        try:
            with open(pkl, "rb") as f:
                order = pickle.load(f).order
            print(f"models/markov.pkl is order {order}", "" if order == 2 else
                  "  <-- the demo/survey use this file; run `python -m src.markov --order 2` to make it order 2")
        except Exception as e:
            print("could not read models/markov.pkl:", e)

    rows = []
    loss, acc, n = unigram_scores(train, win)
    rows.append(row("Unigram (no context)", loss, acc))
    print(rows[-1])

    for order in (2, 3) if args.order3 else (2,):
        t0 = time.time()
        print(f"fitting order-{order} Markov on {len(train):,} tokens...")
        m = MarkovModel(order).fit(train)
        loss, acc, n = markov_scores(m, win)
        rows.append(row(f"Markov (order {order})", loss, acc))
        print(rows[-1], f"({time.time() - t0:.0f}s)")
        del m

    t0 = time.time()
    if not Path(args.lstm).exists():
        raise SystemExit(f"missing {args.lstm} - see README (export_lstm_npz.py)")
    loss, acc, n = lstm_scores(args.lstm, win)
    rows.append(row("LSTM (2 layers, 256 hidden, 8000 steps)", loss, acc))
    print(rows[-1], f"({time.time() - t0:.0f}s)")

    res = {"protocol": {"test_tokens": int(len(test)), "window": args.window,
                        "windows": int(len(win)), "targets_per_model": int(n_targets),
                        "train_tokens": int(len(train))}, "models": rows}
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "results.json").write_text(json.dumps(res, indent=2))

    table = ["| Model | Test loss | Perplexity | Accuracy |", "|---|---:|---:|---:|"]
    for r in rows:
        table.append(f"| {r['name']} | {r['loss']:.4f} | {r['ppl']:.2f} | {r['acc']*100:.2f}% |")
    note = (f"Held-out test pieces, {len(win):,} non-overlapping windows of {args.window} tokens; "
            f"all models are scored on the same {n_targets:,} target tokens "
            f"(`python -m src.evaluate`).")
    block = "\n".join(table) + "\n\n" + note
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "results.md").write_text("# Results\n\n" + block + "\n")

    readme = ROOT / "README.md"
    if readme.exists():
        txt = readme.read_text()
        new = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->",
                     lambda _: f"<!-- RESULTS:START -->\n{block}\n<!-- RESULTS:END -->", txt, flags=re.S)
        if new != txt:
            readme.write_text(new)
            print("README results block updated")
    print("\n" + block)


if __name__ == "__main__":
    main()
