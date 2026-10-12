# Generating Music with Machine Learning

**UE24CS352A Machine Learning mini-project - Team C-14**  
Krrish Singla (PES1UG24AM141) and Kasissnu Ssinha (PES1UG24AM130)

## Problem and objective

This project investigates whether a machine-learning model can learn local musical
structure from piano MIDI performances and generate a new sequence of music. We
frame music generation as next-token prediction: a MIDI performance is converted
to a sequence of note-on, note-off, and time-shift tokens, and a model predicts the
next token repeatedly. The project compares a simple order-2 Markov model with an
LSTM neural sequence model.

## Dataset and representation

We use classical piano performances from the MAESTRO MIDI dataset. MIDI files are
split at the piece level into training, validation, and test sets to reduce leakage
between pieces. The final processed data contains 3,240,519 training tokens,
386,786 validation tokens, and 363,359 test tokens.

The vocabulary has 356 events: 128 note-on tokens, 128 note-off tokens, and 100
time-shift tokens representing 10-1000 ms intervals. This event representation
preserves pitch and timing, and the decoder converts generated tokens back into
standard MIDI files.

## Models and experiment

The baseline is an order-2 Markov model. It estimates the next-token distribution
from the preceding two tokens, backs off to shorter contexts when necessary, and
uses smoothed unigram probabilities for unseen contexts.

The main model is a PyTorch LSTM with an embedding layer, two recurrent layers,
dropout, and a linear output layer over the 356-token vocabulary. It is trained
with cross-entropy loss and Adam optimization. Both models use the same processed
tokens and data split. The LSTM final run used 8,000 training steps and was run on
CPU.

## Results

| Model | Test loss | Test perplexity | Test accuracy |
|---|---:|---:|---:|
| Order-2 Markov | 3.9487 | 51.87 | 19.71% |
| LSTM | 2.3667 | 10.66 | 32.85% |

The LSTM performs better on every reported next-token metric. Its lower perplexity
indicates that it assigns higher probability to the held-out token sequence. The
Markov model remains a useful baseline because it is simple, fast to train, and
easy to interpret, but exact short contexts are sparse in polyphonic music. The
LSTM can generalize patterns beyond an exact two-token lookup.

![LSTM training loss](../outputs/plots/lstm_loss.png)

Generated samples are written as MIDI files in `outputs/midi/` and can be played
in a MIDI-compatible application or through the Streamlit demo. A blind listening
survey has been prepared with human, Markov, and LSTM clips; its results are still
pending and should be reported separately once responses are collected.

## Conclusion and limitations

The completed experiment shows that the LSTM is a stronger next-token predictor
than the order-2 Markov baseline on this dataset. This does not automatically
prove that it creates better music: musical quality also depends on repetition,
harmony, phrasing, dynamics, and listener preference. The current representation
does not explicitly model velocity, key, meter, or rich multi-instrument structure,
which limits expressive realism. Future work could add velocity and chord features,
improve the evaluation protocol, and compare generated samples through a larger
listener study.
