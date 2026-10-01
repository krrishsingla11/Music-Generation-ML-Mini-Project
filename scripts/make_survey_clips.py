import random
from pathlib import Path
import pretty_midi
from src.audio import synthesize, to_wav_bytes

ROOT = Path(__file__).resolve().parent.parent

def process(pm, out_wav):
    audio = synthesize(pm)
    wav_bytes = to_wav_bytes(audio)
    out_wav.write_bytes(wav_bytes)
    print(f"Wrote {out_wav}")

out_dir = ROOT / "outputs" / "survey_clips"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Markov clips
m1 = pretty_midi.PrettyMIDI(str(ROOT / "outputs/midi/markov_sample1.mid"))
m2 = pretty_midi.PrettyMIDI(str(ROOT / "outputs/midi/markov_sample2.mid"))
process(m1, out_dir / "clip_markov_1.wav")
process(m2, out_dir / "clip_markov_2.wav")

# 2. Real MAESTRO clips
maestro_files = list((ROOT / "data/raw/maestro").rglob("*.mid*"))
rng = random.Random(42)
r1_path = rng.choice(maestro_files)
r2_path = rng.choice(maestro_files)

# take first 15 seconds of real midi
def truncate(pm, seconds=15):
    out = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for n in pm.instruments[0].notes:
        if n.start < seconds:
            n.end = min(n.end, seconds)
            inst.notes.append(n)
    out.instruments.append(inst)
    return out

process(truncate(pretty_midi.PrettyMIDI(str(r1_path))), out_dir / "clip_real_1.wav")
process(truncate(pretty_midi.PrettyMIDI(str(r2_path))), out_dir / "clip_real_2.wav")

# 3. LSTM clips
l1 = pretty_midi.PrettyMIDI(str(ROOT / "outputs/midi/lstm_sample1.mid"))
l2 = pretty_midi.PrettyMIDI(str(ROOT / "outputs/midi/lstm_sample2.mid"))
process(l1, out_dir / "clip_lstm_1.wav")
process(l2, out_dir / "clip_lstm_2.wav")
