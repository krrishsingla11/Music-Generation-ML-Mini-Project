"""Train the LSTM on event tokens.

Usage:
    python -m src.train_lstm --steps 200      # quick smoke test
    python -m src.train_lstm --steps 3000     # real run (use a GPU / Colab if CPU is slow)
Saves best checkpoint to models/lstm.pt, log to outputs/lstm_log.json, plot to outputs/plots/lstm_loss.png
"""
import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.model import MusicLSTM
from src.tokens import VOCAB_SIZE

ROOT = Path(__file__).resolve().parent.parent


def get_batch(data, batch, seq, rng, device):
    idx = rng.integers(0, len(data) - seq - 1, size=batch)
    x = np.stack([data[i : i + seq] for i in idx]).astype(np.int64)
    y = np.stack([data[i + 1 : i + seq + 1] for i in idx]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def evaluate(model, data, seq, device, n_windows=256, batch=64):
    model.eval()
    rng = np.random.default_rng(0)  # same windows every time
    loss_sum, correct, count = 0.0, 0, 0
    lossf = nn.CrossEntropyLoss(reduction="sum")
    for _ in range(n_windows // batch):
        x, y = get_batch(data, batch, seq, rng, device)
        logits, _ = model(x)
        loss_sum += lossf(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        correct += (logits.argmax(-1) == y).sum().item()
        count += y.numel()
    model.train()
    loss = loss_sum / count
    return {"loss": loss, "ppl": math.exp(loss), "acc": correct / count}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    d = ROOT / "data" / "processed"
    train, val, test = (np.load(d / f"{s}.npy") for s in ("train", "val", "test"))
    model = MusicLSTM(VOCAB_SIZE, hidden=args.hidden, layers=args.layers).to(device)
    print("params:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(1)
    log, best, t0 = [], float("inf"), time.time()
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "outputs" / "plots").mkdir(parents=True, exist_ok=True)

    run_loss = 0.0
    for step in range(1, args.steps + 1):
        x, y = get_batch(train, args.batch, args.seq, rng, device)
        logits, _ = model(x)
        loss = lossf(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run_loss += loss.item()
        if step % args.eval_every == 0 or step == args.steps:
            n = args.eval_every if step % args.eval_every == 0 else step % args.eval_every
            v = evaluate(model, val, args.seq, device)
            rec = {"step": step, "train_loss": run_loss / n, "val_loss": v["loss"],
                   "val_ppl": v["ppl"], "val_acc": v["acc"], "sec": time.time() - t0}
            log.append(rec)
            run_loss = 0.0
            print(f"step {step}/{args.steps} train {rec['train_loss']:.3f} val {v['loss']:.3f} "
                  f"ppl {v['ppl']:.1f} acc {v['acc']:.3f}  ({rec['sec']:.0f}s)")
            if v["loss"] < best:
                best = v["loss"]
                torch.save({"cfg": model.cfg, "state": model.state_dict()}, ROOT / "models" / "lstm.pt")

    # final test score with the best checkpoint
    ck = torch.load(ROOT / "models" / "lstm.pt", map_location=device)
    model.load_state_dict(ck["state"])
    t = evaluate(model, test, args.seq, device)
    print("TEST:", t)
    (ROOT / "outputs" / "lstm_log.json").write_text(json.dumps({"log": log, "test": t}, indent=2))

    plt.figure(figsize=(6, 4))
    plt.plot([r["step"] for r in log], [r["train_loss"] for r in log], label="train")
    plt.plot([r["step"] for r in log], [r["val_loss"] for r in log], label="val")
    plt.xlabel("step"); plt.ylabel("cross-entropy"); plt.legend(); plt.title("LSTM training")
    plt.tight_layout(); plt.savefig(ROOT / "outputs" / "plots" / "lstm_loss.png", dpi=150)


if __name__ == "__main__":
    main()
