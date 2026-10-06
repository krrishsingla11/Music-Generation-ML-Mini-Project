"""Export an LSTM checkpoint (.pt) to a plain .npz so the demo can run without torch.

Run this ONCE on a machine where torch works (Google Colab, a friend's laptop).
Standalone: does not need the rest of the repo.

    python export_lstm_npz.py lstm.pt lstm.npz

It also checks the numpy forward pass against torch and prints the max difference
(should be ~1e-5 or smaller).
"""
import sys

import numpy as np
import torch
import torch.nn as nn


class MusicLSTM(nn.Module):  # same structure as src/model.py
    def __init__(self, vocab_size, emb=128, hidden=256, layers=2, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x, state=None):
        h, state = self.lstm(self.embed(x), state)
        return self.out(self.drop(h)), state


def sig(x):
    return 1 / (1 + np.exp(-x))


def np_forward(z, tokens):
    L, H = int(z["layers"]), int(z["hidden"])
    h = [np.zeros(H, np.float32) for _ in range(L)]
    c = [np.zeros(H, np.float32) for _ in range(L)]
    for t in tokens:
        x = z["embed"][t]
        for k in range(L):
            g = z[f"w_ih_{k}"] @ x + z[f"w_hh_{k}"] @ h[k] + z[f"b_ih_{k}"] + z[f"b_hh_{k}"]
            i, f, gg, o = np.split(g, 4)
            c[k] = sig(f) * c[k] + sig(i) * np.tanh(gg)
            h[k] = sig(o) * np.tanh(c[k])
            x = h[k]
    return z["out_w"] @ x + z["out_b"]


def main(src, dst):
    ck = torch.load(src, map_location="cpu")
    cfg = ck["cfg"]
    model = MusicLSTM(**cfg)
    model.load_state_dict(ck["state"])
    model.eval()
    sd = {k: v.numpy().astype(np.float32) for k, v in model.state_dict().items()}
    arrays = dict(layers=cfg["layers"], hidden=cfg["hidden"],
                  embed=sd["embed.weight"], out_w=sd["out.weight"], out_b=sd["out.bias"])
    for k in range(cfg["layers"]):
        arrays[f"w_ih_{k}"] = sd[f"lstm.weight_ih_l{k}"]
        arrays[f"w_hh_{k}"] = sd[f"lstm.weight_hh_l{k}"]
        arrays[f"b_ih_{k}"] = sd[f"lstm.bias_ih_l{k}"]
        arrays[f"b_hh_{k}"] = sd[f"lstm.bias_hh_l{k}"]
    np.savez(dst, **arrays)
    print("saved", dst)

    # verify numpy forward == torch forward
    z = np.load(dst)
    toks = np.random.default_rng(0).integers(0, cfg["vocab_size"], 60).tolist()
    with torch.no_grad():
        ref = model(torch.tensor([toks]))[0][0, -1].numpy()
    diff = np.abs(ref - np_forward(z, toks)).max()
    print("max abs difference vs torch:", diff, "(OK)" if diff < 1e-3 else "(PROBLEM - tell Krrish)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lstm.pt", sys.argv[2] if len(sys.argv) > 2 else "lstm.npz")
