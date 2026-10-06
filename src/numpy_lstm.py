"""Pure-numpy LSTM inference (no torch import), loaded from models/lstm.npz.
Weights are exported once from lstm.pt by scripts/export_lstm_npz.py.
Gate order matches PyTorch: input, forget, cell(g), output.
"""
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyLSTM:
    def __init__(self, path):
        z = np.load(path)
        self.layers = int(z["layers"])
        self.hidden = int(z["hidden"])
        self.embed = z["embed"]
        self.out_w, self.out_b = z["out_w"], z["out_b"]
        self.w_ih = [z[f"w_ih_{k}"] for k in range(self.layers)]
        self.w_hh = [z[f"w_hh_{k}"] for k in range(self.layers)]
        self.b = [z[f"b_ih_{k}"] + z[f"b_hh_{k}"] for k in range(self.layers)]

    def init_state(self):
        z = lambda: [np.zeros(self.hidden, dtype=np.float32) for _ in range(self.layers)]
        return z(), z()

    def step(self, tok, state):
        h, c = list(state[0]), list(state[1])
        x = self.embed[int(tok)]
        for k in range(self.layers):
            g = self.w_ih[k] @ x + self.w_hh[k] @ h[k] + self.b[k]
            i, f, gg, o = np.split(g, 4)
            c[k] = _sigmoid(f) * c[k] + _sigmoid(i) * np.tanh(gg)
            h[k] = _sigmoid(o) * np.tanh(c[k])
            x = h[k]
        return self.out_w @ x + self.out_b, (h, c)
