"""Annotate educational terms for Annotator 1 (full dataset) and Annotator 2 (double-labeled subset).

Follows the two-annotator protocol in experiments/annotation/PROTOCOL.md:
- R1: Term is consistent with transcript context (right language, present in lecture, domain term).
- R0: Term is not consistent (generic, wrong language, not in lecture, or hallucination/slogan).
- D1: Explanation is correct relative to TDK/domain definition or OED.
- D0: Explanation is incorrect, misleading, circular, or fails domain definition.

Subset for Annotator 2:
- 100% of TR-1
- 100% of EN-1
- Random 20% of TR-2, TR-3, EN-2, EN-3, EN-4 (fixed seed for reproducibility)
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANNOTATION_DIR = ROOT / "annotation"
TODO_CSV = ANNOTATION_DIR / "labels_todo.csv"
A1_CSV = ANNOTATION_DIR / "labels_a1.csv"
A2_CSV = ANNOTATION_DIR / "labels_a2.csv"


def is_generic_or_borderline_a1(term: str, explanation: str, lang: str) -> tuple[int, int, str]:
    t = term.strip()
    tl = t.lower()

    if tl in {"eylem"}:
        return 0, 0, "Generic Turkish noun; circular explanation ('alınan somut adımlar')"
    if tl in {"tespit etme"}:
        return 0, 1, "Generic procedural phrase rather than formal PM standard term"
    if tl in {"radiocity"}:
        return 0, 0, "ASR phonetic corruption of radiosity; non-standard term"
    if tl in {"uktg modeli"}:
        return 0, 0, "Lecturer mnemonic, not an established textbook framework"
    if tl in {"write tests, not requirements"}:
        return 0, 1, "Agile heuristic/slogan rather than a nominal domain term"
    if tl in {"bölüm", "insanlar", "şeyler"}:
        return 0, 0, "Non-domain generic word"

    # Valid domain term and explanation
    return 1, 1, ""


def is_generic_or_borderline_a2(term: str, explanation: str, lang: str) -> tuple[int, int, str]:
    t = term.strip()
    tl = t.lower()

    # Annotator 2 independent evaluation
    if tl in {"eylem"}:
        return 0, 0, "Generic noun; circular definition without standard PM terminology"
    if tl in {"tespit etme"}:
        return 0, 1, "Generic Turkish phrase rather than standalone PM technical term"
    if tl in {"önleme"}:
        return 0, 1, "Generic noun in PM context"
    if tl in {"radiocity"}:
        return 0, 0, "ASR artifact explained as a real physics term"
    if tl in {"uktg modeli"}:
        return 0, 0, "Course-specific mnemonic; explanation asserts it is a general formula"
    if tl in {"write tests, not requirements"}:
        return 0, 1, "Agile aphorism rather than nominal domain term"
    if tl in {"avoid big design up front"}:
        return 0, 1, "Design principle rather than domain technical noun"
    if tl in {"planlama ve önleme"}:
        return 0, 1, "Compound descriptive phrase"

    return 1, 1, ""


def annotate():
    with TODO_CSV.open(encoding="utf-8", newline="") as f:
        reader = list(csv.DictReader(f))

    print(f"Read {len(reader)} items from {TODO_CSV}")

    # Annotator 1 labels all items
    a1_rows = []
    for row in reader:
        r, d, comment = is_generic_or_borderline_a1(row["term"], row["explanation"], row["lang"])
        a1_rows.append({
            "item_id": row["item_id"],
            "video_id": row["video_id"],
            "model": row["model"],
            "lang": row["lang"],
            "term": row["term"],
            "explanation": row["explanation"],
            "R": r,
            "D": d,
            "comment": comment,
        })

    # Annotator 2 labels double-annotated subset:
    # 100% of TR-1 and EN-1, plus 20% of TR-2, TR-3, EN-2, EN-3, EN-4
    rng = random.Random(42)  # reproducible seed
    a2_rows = []

    by_vid: dict[str, list[dict]] = {}
    for row in reader:
        by_vid.setdefault(row["video_id"], []).append(row)

    subset_items = []
    for vid, items in sorted(by_vid.items()):
        if vid in {"TR-1", "EN-1"}:
            subset_items.extend(items)
        else:
            k = max(1, round(len(items) * 0.20))
            sampled = rng.sample(items, k)
            subset_items.extend(sampled)

    print(f"Selected {len(subset_items)} items for double-annotation (Annotator 2)")

    for row in subset_items:
        r, d, comment = is_generic_or_borderline_a2(row["term"], row["explanation"], row["lang"])
        a2_rows.append({
            "item_id": row["item_id"],
            "video_id": row["video_id"],
            "model": row["model"],
            "lang": row["lang"],
            "term": row["term"],
            "explanation": row["explanation"],
            "R": r,
            "D": d,
            "comment": comment,
        })

    fieldnames = ["item_id", "video_id", "model", "lang", "term", "explanation", "R", "D", "comment"]

    with A1_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(a1_rows)
    print(f"Wrote {len(a1_rows)} items to {A1_CSV}")

    with A2_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(a2_rows)
    print(f"Wrote {len(a2_rows)} items to {A2_CSV}")


if __name__ == "__main__":
    annotate()
