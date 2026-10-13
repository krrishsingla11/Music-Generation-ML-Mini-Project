# Results

| Model | Test loss | Perplexity | Accuracy |
|---|---:|---:|---:|
| Unigram (no context) | 4.4810 | 88.32 | 11.89% |
| Markov (order 2) | 3.8966 | 49.23 | 18.58% |
| LSTM (2 layers, 256 hidden, 8000 steps) | 2.3982 | 11.00 | 32.17% |

Held-out test pieces, 1,419 non-overlapping windows of 256 tokens; all models are scored on the same 361,845 target tokens (`python -m src.evaluate`).
