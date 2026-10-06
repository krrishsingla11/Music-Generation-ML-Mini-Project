# Markov baseline record

The baseline was trained on the same event-token representation and split used by the LSTM: 3,240,519 training tokens, 386,786 validation tokens, and 363,359 test tokens.

| Model | Validation accuracy | Validation perplexity | Test accuracy | Test perplexity |
|---|---:|---:|---:|---:|
| Order 2 | 0.1782 | 52.13 | 0.1971 | 51.87 |
| Order 3 | 0.1495 | 113.68 | 0.1731 | 104.73 |

The order-2 checkpoint is the model used for the Markov listening samples. The order-3 run is recorded for comparison and is not the selected sampling model.
