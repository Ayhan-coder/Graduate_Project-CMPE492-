"""Merge the machine candidates and the pilot items into one judge input sheet.

llm_judge_protocol.py rewrites its --out file in the order of the sheet it was
given, so one arm must see one sheet. This concatenates:
  * experiments/annotation/labels_todo_32clip_machine.csv  (235 machine_only candidates)
  * experiments/annotation/labels_pilot_ctx.csv            (219 pilot items, context
    recovered from the gold transcripts by build_pilot_ctx_sheet.py)
into experiments/annotation/labels_panel_input.csv, keeping a row_source column so
the provenance of every row stays visible.

No labels are created here: R/D stay empty and gold_how is carried through
(machine_only for candidates, human_verified_pilot for pilot rows).

Usage
-----
python experiments/data/build_panel_sheet.py
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "experiments" / "annotation"
CAND = ANN / "labels_todo_32clip_machine.csv"
PILOT = ANN / "labels_pilot_ctx.csv"
OUT = ANN / "labels_panel_input.csv"

FIELDS = ["item_id", "video_id", "model", "lang", "term", "explanation", "context",
          "gold_how", "row_source"]


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    rows = []
    for r in read(CAND):
        rows.append({**{k: r.get(k, "") for k in FIELDS},
                     "gold_how": r.get("gold_how", "machine_only"),
                     "row_source": "machine_candidate_32clip"})
    seen = {r["item_id"] for r in rows}
    skipped_dup = 0
    for r in read(PILOT):
        if r["item_id"] in seen:
            skipped_dup += 1
            continue
        rows.append({**{k: r.get(k, "") for k in FIELDS if k != "row_source"},
                     "gold_how": "human_verified_pilot",
                     "row_source": "pilot_ctx_from_gold"})

    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    print(OUT)
    print("satir:", len(rows), dict(Counter(r["row_source"] for r in rows)))
    print("boş bağlam:", sum(1 for r in rows if not r["context"].strip()),
          "| yinelenen atlanan:", skipped_dup)
    print("dil:", dict(Counter(r["lang"] for r in rows)))


if __name__ == "__main__":
    main()
