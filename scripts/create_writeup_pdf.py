"""Create the final two-page project write-up PDF."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "music_generation_writeup.pdf"


def p(text, style):
    return Paragraph(text, style)


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom", parent=styles["Title"], fontSize=20, leading=24,
        alignment=TA_CENTER, textColor=colors.HexColor("#8f1239"), spaceAfter=6,
    )
    subtitle = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontSize=9, leading=12,
        alignment=TA_CENTER, textColor=colors.HexColor("#374151"), spaceAfter=12,
    )
    h1 = ParagraphStyle(
        "H1Custom", parent=styles["Heading2"], fontSize=12, leading=14,
        textColor=colors.HexColor("#8f1239"), spaceBefore=6, spaceAfter=4,
    )
    body = ParagraphStyle(
        "BodyCustom", parent=styles["BodyText"], fontSize=8.4, leading=10.5,
        spaceAfter=5, alignment=0, wordWrap="CJK",
    )
    small = ParagraphStyle(
        "Small", parent=body, fontSize=7.2, leading=8.8, textColor=colors.HexColor("#4b5563"),
    )
    cell = ParagraphStyle("Cell", parent=body, fontSize=7.7, leading=9, wordWrap="CJK")

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4, rightMargin=0.55 * inch, leftMargin=0.55 * inch,
        topMargin=0.48 * inch, bottomMargin=0.45 * inch,
        title="Generating Music with Machine Learning",
        author="Team C-14",
    )
    story = [
        p("Generating Music with Machine Learning", title),
        p("UE24CS352A Machine Learning mini-project - Team C-14 - Krrish Singla and Kasissnu Ssinha", subtitle),
        p("1. Problem and objective", h1),
        p(
            "This project investigates whether a machine-learning model can learn local musical structure from piano MIDI performances and generate new music. A MIDI performance is converted into a sequence of note-on, note-off, and time-shift tokens. A model predicts the next token repeatedly, and the generated sequence is converted back into a MIDI file. We compare a simple order-2 Markov model with an LSTM neural sequence model.", body,
        ),
        p("2. Dataset and representation", h1),
        p(
            "We use classical piano performances from the MAESTRO MIDI dataset. Files are split at the piece level into training, validation, and test sets. The final processed data contains 3,240,519 training tokens, 386,786 validation tokens, and 363,359 test tokens.", body,
        ),
        p(
            "The vocabulary contains 356 events: 128 note-on tokens, 128 note-off tokens, and 100 time-shift tokens representing 10-1000 ms intervals. This representation preserves pitch and timing, and the decoder converts tokens back into standard MIDI files.", body,
        ),
        p("3. Models and experiment", h1),
        p(
            "The baseline is an order-2 Markov model. It estimates the next-token distribution from the preceding two tokens, backs off to shorter contexts when necessary, and uses smoothed unigram probabilities for unseen contexts.", body,
        ),
        p(
            "The main model is a PyTorch LSTM with an embedding layer, two recurrent layers, dropout, and a linear output layer over the vocabulary. It is trained with cross-entropy loss and Adam optimization. Both models use the same token representation and data split. The final LSTM run used 8,000 training steps on CPU.", body,
        ),
        p("4. Results", h1),
    ]
    results = [
        [p("Model", cell), p("Test loss", cell), p("Perplexity", cell), p("Accuracy", cell)],
        [p("Order-2 Markov", cell), p("3.9487", cell), p("51.87", cell), p("19.71%", cell)],
        [p("LSTM", cell), p("2.3667", cell), p("10.66", cell), p("32.85%", cell)],
    ]
    table = Table(results, colWidths=[2.4 * inch, 1.25 * inch, 1.25 * inch, 1.25 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#8f1239")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f9fafb")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9ca3af")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [table, Spacer(1, 5), p(
        "The LSTM performs better on every reported next-token metric. Its lower perplexity means it assigns higher probability to held-out token sequences. The Markov model is simple and interpretable, but exact short contexts are sparse in polyphonic music; the LSTM can generalize patterns beyond an exact two-token lookup.", body,
    )]
    plot = ROOT / "outputs" / "plots" / "lstm_loss.png"
    if plot.exists():
        img = Image(str(plot), width=3.45 * inch, height=2.3 * inch)
        story += [img, p("Figure 1. LSTM training and validation cross-entropy.", small)]

    story += [PageBreak(), p("5. Generation and evaluation", h1), p(
        "Both models generate new token sequences that are decoded into MIDI files. The repository includes generated Markov and LSTM samples in outputs/midi/ and a Streamlit application for selecting a model, controlling generation length and temperature, listening to synthesized audio, viewing a piano roll, and downloading the MIDI result.", body,
    ), p(
        "The numerical evaluation measures next-token prediction, while music quality is not fully captured by loss or accuracy. A blind listening survey has been prepared with human, Markov, and LSTM clips. Survey results are pending and should be added after responses are collected.", body,
    ), p("6. Conclusion", h1), p(
        "The completed experiment shows that the LSTM is a stronger next-token predictor than the order-2 Markov baseline on this dataset: accuracy improves from 19.71% to 32.85%, and perplexity decreases from 51.87 to 10.66. This improvement does not automatically prove that the LSTM creates better music, because musical quality also depends on repetition, harmony, phrasing, dynamics, and listener preference.", body,
    ), p("7. Limitations and future work", h1), p(
        "The current representation models pitch, note-off events, and timing, but does not explicitly model expressive velocity, key, meter, or rich multi-instrument structure. The generated audio uses a simple piano-like synthesizer. Future work could add velocity and chord features, use a more consistent full-test evaluation protocol, and conduct a larger listener study. A Transformer could also be explored after the two-model comparison is complete.", body,
    ), p("8. Reproducibility", h1), p(
        "Install dependencies with pip install -r requirements.txt. Run python -m src.preprocess --raw data/raw/maestro --max-files 200, then python -m src.markov --order 2 and python -m src.train_lstm --steps 8000. Generate samples with python -m src.generate --model markov --tokens 1000 or --model lstm --tokens 1000. Launch the demonstration with streamlit run app.py.", body,
    ), Spacer(1, 8), p("Project repository structure and experiment details are documented in README.md.", small)]
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    build()
