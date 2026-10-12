# Generating Music with Machine Learning

UE24CS352A Machine Learning mini-project (Team C-14, Problem 22).
The project compares a count-based order-2 Markov baseline with a PyTorch LSTM
for next-token prediction and piano MIDI generation.

## Approach
1. **Baseline:** order-2 Markov / n-gram model over note events
2. **Main model:** LSTM (PyTorch) predicting the next event token

Data: classical piano MIDI from the MAESTRO MIDI dataset.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data
```bash
python scripts/download_maestro.py
```
Files land in `data/raw/maestro/`.

## Run
```bash
python -m src.preprocess --raw data/raw/maestro --max-files 200
python -m src.markov --order 2
python -m src.train_lstm --steps 8000
python -m src.generate --model markov --tokens 1000
python -m src.generate --model lstm --tokens 1000
streamlit run app.py
```

The final experiments use the same event vocabulary and piece-level split:
3,240,519 training tokens, 386,786 validation tokens, and 363,359 test tokens.

## Final results

| Model | Test loss | Test perplexity | Test accuracy |
|---|---:|---:|---:|
| Order-2 Markov | 3.9487 | 51.87 | 19.71% |
| LSTM | 2.3667 | 10.66 | 32.85% |

The LSTM gives the better next-token metrics. Listening samples are in
`outputs/midi/`; mathematical metrics alone do not fully measure musicality.

## Structure
```
data/raw/        downloaded MIDI (not committed)
data/processed/  tokenized data (not committed)
src/             preprocessing, models, training, generation, evaluation
scripts/         helper scripts (data download)
notebooks/       experiments
outputs/         generated MIDI and plots
models/          saved checkpoints (not committed)
docs/             baseline notes, survey materials, and project write-up
```

## Team
- Krrish Singla (PES1UG24AM141)
- Kasissnu Ssinha (PES1UG24AM130)

## Current limitations

The representation models pitch, note-off events, and timing, but not expressive
velocity or a full multi-instrument arrangement. The listening-survey results are
still pending; the survey materials are in `docs/`.
