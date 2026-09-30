import torch
import torch.nn as nn


class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, emb=128, hidden=256, layers=2, dropout=0.2):
        super().__init__()
        self.cfg = dict(vocab_size=vocab_size, emb=emb, hidden=hidden, layers=layers, dropout=dropout)
        self.embed = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x, state=None):
        h, state = self.lstm(self.embed(x), state)
        return self.out(self.drop(h)), state
