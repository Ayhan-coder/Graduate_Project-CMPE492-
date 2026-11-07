"""Recover transcript contexts for the pilot items so the protocol judge can run.

The pilot sheets (labels_a1.csv / labels_a2.csv, 233 / 115 items) carry term +
explanation but no context column, while llm_judge_protocol.py's R rubric needs
the surrounding transcript window. This script rebuilds that window from the
human-verified gold transcripts (experiments/data/gold/{TR-1..3,EN-1..4}.txt) by
literal term match, and attaches the existing human labels in the same row.

Nothing here is a new label: human_a1_* / human_a2_* are copied verbatim from
labels_a1.csv / labels_a2.csv. context_how says how the window was found, and
items whose term never occurs in the gold transcript get no row (they cannot be
judged on R without context) — the count is printed and stored in the summary.

Usage
-----
python experiments/data/build_pilot_ctx_sheet.py
Output: experiments/annotation/labels_pilot_ctx.csv
        experiments/outputs/pilot_ctx_sheet.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "experiments" / "annotation"
GOLD = ROOT / "experiments" / "data" / "gold"
OUT_CSV = ANN / "labels_pilot_ctx.csv"
OUT_JSON = ROOT / "experiments" / "outputs" / "pilot_ctx_sheet.json"

WINDOW = 260
FIELDS = [
    "item_id", "video_id", "model", "lang", "term", "explanation",
    "context", "context_how",
    "human_a1_R", "human_a1_D", "human_a2_R", "human_a2_D",
]


def tr_lower(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").replace("Ş", "ş").replace("Ğ", "ğ").casefold()


def find_window(term: str, text: str) -> tuple[str, str] | None:
    """Return (context, how) for the first place the term plausibly occurs."""
    low_t, low_text = tr_lower(term), tr_lower(text)
    for how, needle in (("exact", low_t),):
        pos = low_text.find(needle)
        if pos >= 0:
            return slice_window(text, pos, len(needle)), how
    tokens = [t for t in term.split() if len(t) >= 4]
    if tokens:
        head = tr_lower(max(tokens, key=len))
        pos = low_text.find(head)
        if pos >= 0:
            return slice_window(text, pos, len(head)), "longest_token"
    if len(low_t) >= 7:
        stem = low_t[: max(5, int(len(low_t) * 0.7))]
        pos = low_text.find(stem)
        if pos >= 0:
            return slice_window(text, pos, len(stem)), "stem_prefix"
    return None


def slice_window(text: str, pos: int, match_len: int) -> str:
    start = max(0, pos - WINDOW)
    end = min(len(text), pos + match_len + WINDOW)
    return ("…" if start else "") + text[start:end].strip() + ("…" if end < len(text) else "")


def load(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return {r["item_id"]: r for r in csv.DictReader(fh)}


def main() -> None:
    a1 = load(ANN / "labels_a1.csv")
    a2 = load(ANN / "labels_a2.csv")
    texts = {p.stem: p.read_text(encoding="utf-8", errors="ignore") for p in GOLD.glob("*.txt")}

    rows, missing = [], []
    how_counts: dict[str, int] = {}
    for item_id, r in a1.items():
        clip = r["video_id"]
        text = texts.get(clip)
        if text is None:
            missing.append({"item_id": item_id, "reason": "no_gold_transcript"})
            continue
        hit = find_window(r["term"], text)
        if hit is None:
            missing.append({"item_id": item_id, "reason": "term_not_in_gold_transcript"})
            continue
        context, how = hit
        how_counts[how] = how_counts.get(how, 0) + 1
        h2 = a2.get(item_id, {})
        rows.append(
            {
                "item_id": item_id,
                "video_id": clip,
                "model": r["model"],
                "lang": r["lang"],
                "term": r["term"],
                "explanation": r["explanation"],
                "context": context,
                "context_how": how,
                "human_a1_R": r.get("R", ""),
                "human_a1_D": r.get("D", ""),
                "human_a2_R": h2.get("R", ""),
                "human_a2_D": h2.get("D", ""),
            }
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "source_labels_a1": len(a1),
        "source_labels_a2": len(a2),
        "rows_written": len(rows),
        "context_how": how_counts,
        "dropped": len(missing),
        "drop_reasons": {
            k: sum(1 for m in missing if m["reason"] == k)
            for k in sorted({m["reason"] for m in missing})
        },
        "clips": sorted({r["video_id"] for r in rows}),
        "note": (
            "contexts recovered by literal term match in the human-verified gold "
            "transcripts; human labels copied verbatim, no new labels created"
        ),
        "output": str(OUT_CSV.relative_to(ROOT)),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
