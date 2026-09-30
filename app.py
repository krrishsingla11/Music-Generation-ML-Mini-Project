"""Streamlit demo:  streamlit run app.py"""
import io
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import streamlit as st

from src.audio import synthesize, to_wav_bytes
from src.generate import generate_lstm, generate_markov, get_seed
from src.tokens import tokens_to_midi

ROOT = Path(__file__).resolve().parent
LSTM_PATH = ROOT / "models" / "lstm.pt"
MARKOV_PATH = ROOT / "models" / "markov.pkl"

st.set_page_config(page_title="ML Music Generator", page_icon="🎹", layout="centered")
st.title("🎹 Generating Music with Machine Learning")
st.caption("UE24CS352A mini-project · Markov baseline vs LSTM trained on MAESTRO piano MIDI")

models = []
if LSTM_PATH.exists():
    models.append("LSTM")
if MARKOV_PATH.exists():
    models.append("Markov baseline")
if not models:
    st.error("No trained models found. Run `python -m src.train_lstm` and/or `python -m src.markov` first.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", models)
    n_tokens = st.slider("Length (tokens)", 200, 4000, 1500, step=100)
    temp = st.slider("Temperature", 0.3, 1.5, 1.0, step=0.05,
                     help="Low = safe/repetitive, high = adventurous/chaotic")
    topk = st.slider("Top-k (LSTM only)", 5, 100, 20) if model_name == "LSTM" else 0
    seed_val = st.number_input("Random seed", 0, 99999, 42)
    st.divider()
    log = ROOT / "outputs" / "lstm_log.json"
    if log.exists():
        t = json.loads(log.read_text()).get("test")
        if t:
            st.subheader("LSTM test scores")
            st.metric("Perplexity", f"{t['ppl']:.1f}")
            st.metric("Next-token accuracy", f"{t['acc']*100:.1f}%")

if st.button("Generate music", type="primary"):
    rng = np.random.default_rng(int(seed_val))
    try:
        seed = get_seed(32, rng)
    except Exception:
        seed = np.array([60], dtype=np.int16)  # fallback: start from a single note-on
    with st.spinner("Generating..."):
        t0 = time.time()
        if model_name == "LSTM":
            toks = generate_lstm(LSTM_PATH, seed, n_tokens, temp, topk, rng)
        else:
            toks = generate_markov(MARKOV_PATH, seed, n_tokens, temp, rng)
        tmp = ROOT / "outputs" / "midi" / "_demo.mid"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        pm = tokens_to_midi(toks, tmp)
        audio = synthesize(pm)
        st.session_state["result"] = dict(
            wav=to_wav_bytes(audio), midi=tmp.read_bytes(), pm=pm,
            info=f"{model_name} · {len(pm.instruments[0].notes)} notes · "
                 f"{len(audio)/22050:.0f}s · generated in {time.time()-t0:.1f}s")

res = st.session_state.get("result")
if res:
    st.success(res["info"])
    st.audio(res["wav"], format="audio/wav")
    notes = res["pm"].instruments[0].notes
    if notes:
        fig, ax = plt.subplots(figsize=(8, 3))
        for n in notes:
            ax.plot([n.start, n.end], [n.pitch, n.pitch], lw=2, color="#4f46e5")
        ax.set_xlabel("time (s)"); ax.set_ylabel("MIDI pitch"); ax.set_title("Piano roll")
        st.pyplot(fig)
    st.download_button("Download .mid", res["midi"], file_name="generated.mid", mime="audio/midi")
    st.caption("Playback uses a simple built-in synth; open the .mid in MuseScore/VLC for a real piano sound.")
else:
    st.info("Pick a model in the sidebar and press **Generate music**.")
