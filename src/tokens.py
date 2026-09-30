"""Event-based token vocabulary (Performance-RNN style) + MIDI <-> token conversion.

Tokens:
    0..127     NOTE_ON  pitch
    128..255   NOTE_OFF pitch
    256..355   TIME_SHIFT 10ms..1000ms (index 256 = 10ms, 355 = 1000ms)
Total vocab = 356.
"""
import pretty_midi

NOTE_ON = 0
NOTE_OFF = 128
SHIFT = 256
N_SHIFT = 100
STEP = 0.01  # seconds per time-shift unit
VOCAB_SIZE = SHIFT + N_SHIFT  # 356


def midi_to_tokens(path):
    pm = pretty_midi.PrettyMIDI(str(path))
    evs = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            evs.append((n.start, 1, n.pitch))  # note on
            evs.append((n.end, 0, n.pitch))    # note off (sorts before on at same time)
    evs.sort()
    tokens, prev = [], 0
    for t, kind, pitch in evs:
        tick = round(t / STEP)
        d = tick - prev
        while d > 0:
            s = min(d, N_SHIFT)
            tokens.append(SHIFT + s - 1)
            d -= s
        prev = tick
        tokens.append((NOTE_ON if kind else NOTE_OFF) + pitch)
    return tokens


def tokens_to_midi(tokens, out_path, velocity=80, program=0):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    t, active = 0.0, {}
    for tok in tokens:
        tok = int(tok)
        if tok >= SHIFT:
            t += (tok - SHIFT + 1) * STEP
        elif tok >= NOTE_OFF:
            p = tok - NOTE_OFF
            if p in active:
                start = active.pop(p)
                if t > start:
                    inst.notes.append(pretty_midi.Note(velocity, p, start, t))
        else:
            active.setdefault(tok - NOTE_ON, t)
    for p, start in active.items():  # close notes left hanging
        inst.notes.append(pretty_midi.Note(velocity, p, start, t + 0.3))
    pm.instruments.append(inst)
    pm.write(str(out_path))
    return pm
