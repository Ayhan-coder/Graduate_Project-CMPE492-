"""Merge a form export into an annotation sheet.

The HTML form (make_annotation_form.py) exports only newly labeled rows as
labels_<annotator>_new.csv. This script merges them into labels_<annotator>.csv,
keeping the old file as .bak and refusing to silently overwrite a conflicting
non-empty label.

Usage
-----
python experiments/annotation/merge_labels.py --annotator a2 --new path/to/labels_a2_new.csv
python experiments/annotation/merge_labels.py --annotator a2 --new ... --overwrite
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "experiments" / "annotation"
FIELDS = ["item_id", "video_id", "model", "lang", "term", "explanation", "R", "D", "comment"]


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return [
            {k: (v or "").strip() for k, v in row.items() if k}
            for row in csv.DictReader(fh)
        ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--sheet", type=Path, default=ANN / "labels_todo_15clip.csv",
                    help="authoritative item order")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    target = ANN / f"labels_{args.annotator}.csv"
    existing = read(target) if target.exists() else []
    order = [r["item_id"] for r in read(args.sheet)]
    items = {r["item_id"]: r for r in read(args.sheet)}

    merged: dict[str, dict] = {}
    # rows already in the target but absent from --sheet (e.g. the pilot arm when
    # merging a different item sheet) keep their original position and content
    out_order = [r["item_id"] for r in existing]
    for r in existing:
        merged[r["item_id"]] = r
    out_order += [i for i in order if i not in merged]

    conflicts, added = [], 0
    for r in read(args.new):
        iid = r["item_id"]
        if iid not in items:
            print(f"atlandı (sayfada yok): {iid}")
            continue
        prev = merged.get(iid)
        row = {
            "item_id": iid,
            "video_id": items[iid]["video_id"],
            "model": items[iid]["model"],
            "lang": items[iid]["lang"],
            "term": items[iid]["term"],
            "explanation": items[iid]["explanation"],
            "R": r.get("R", ""),
            "D": r.get("D", ""),
            "comment": r.get("comment", ""),
        }
        if prev and (prev.get("R"), prev.get("D")) not in (("", ""), (row["R"], row["D"])) and not args.overwrite:
            conflicts.append((iid, prev.get("R"), prev.get("D"), row["R"], row["D"]))
            continue
        if not prev:
            added += 1
        merged[iid] = row

    if target.exists():
        shutil.copy2(target, target.with_suffix(".csv.bak"))

    out_rows = [merged[i] for i in out_order if i in merged]
    with target.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)

    labeled = sum(1 for r in out_rows if r["R"] != "" and r["D"] != "")
    print(f"{target}: {len(out_rows)} satır, {labeled} etiketli (yeni {added})")
    if conflicts:
        print(f"{len(conflicts)} çakışma atlandı (--overwrite ile ez):")
        for c in conflicts[:10]:
            print("  ", c)


if __name__ == "__main__":
    main()
