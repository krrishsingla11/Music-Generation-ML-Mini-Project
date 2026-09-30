# Generating Music with Machine Learning

UE24CS352A Machine Learning mini-project (Team C-14, Problem 22).
Based on the Stanford CS229 (2018) report "Generating music with Machine Learning".

## Approach
1. **Baseline:** Markov / n-gram model over note events
2. **Main model:** LSTM (PyTorch) predicting the next note event (pitch + duration)
3. **Extension (optional):** small Transformer

Data: classical piano MIDI from piano-midi.de.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data
```bash
python scripts/download_data.py
```
Files land in `data/raw/<composer>/`.

## Run
_(filled in as the pipeline gets built)_
```bash
python -m src.preprocess     # MIDI -> note events, vocab, train/val/test split
python -m src.train_markov   # baseline
python -m src.train_lstm     # main model
python -m src.generate       # write .mid files to outputs/midi/
streamlit run app.py         # demo
```

## Structure
```
data/raw/        downloaded MIDI (not committed)
data/processed/  tokenized data (not committed)
src/             preprocessing, models, training, generation, evaluation
scripts/         helper scripts (data download)
notebooks/       experiments
outputs/         generated MIDI and plots
models/          saved checkpoints (not committed)
```

## Team
- Krrish Singla (PES1UG24AM141)
- Kasissnu Ssinha (PES1UG24AM130)
