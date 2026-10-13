"""Turn the exported Google Form responses into the survey results table.

    python -m scripts.analyze_survey path/to/responses.csv

Expects columns titled like  "Clip A: Human or machine?"  and
"Clip A: How musical is this clip? (1-5)"  (see docs/survey_form.md) and
docs/survey_answer_key.json from make_survey_clips.py.
Writes outputs/survey_results.json and docs/survey_results.md.
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(csv_path):
    key = json.loads((ROOT / "docs" / "survey_answer_key.json").read_text())
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    guesses, ratings, complete = defaultdict(list), defaultdict(list), 0
    for r in rows:
        ok = True
        for col, val in r.items():
            m = re.match(r"\s*Clip\s*([A-F])\b", col or "")
            if not m or val is None:
                continue
            src, v = key[m.group(1)], val.strip()
            if v.isdigit():
                ratings[src].append(int(v))
            elif v.lower() in ("human", "machine"):
                guesses[src].append(v.lower() == "machine")
            elif v == "":
                ok = False
        complete += ok
    out, lines = {"responses": len(rows), "complete_responses": complete, "sources": {}}, []
    lines = ["| Source | Votes | Judged machine | Mean musicality (1-5) |", "|---|---:|---:|---:|"]
    for src in dict.fromkeys(key.values()):
        g, rt = guesses[src], ratings[src]
        rate = sum(g) / len(g) if g else float("nan")
        mean = sum(rt) / len(rt) if rt else float("nan")
        out["sources"][src] = {"votes": len(g), "machine_rate": rate, "mean_musicality": mean}
        lines.append(f"| {src} | {len(g)} | {rate*100:.0f}% | {mean:.2f} |")
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "survey_results.json").write_text(json.dumps(out, indent=2))
    warn = "" if len(rows) >= 20 else f"\n\n**Only {len(rows)} responses (target: 20+).**"
    (ROOT / "docs" / "survey_results.md").write_text(
        f"# Listening survey results\n\n{len(rows)} responses ({complete} complete).\n\n"
        + "\n".join(lines) + warn + "\n")
    print("\n".join(lines), f"\n{len(rows)} responses", warn)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
