"""Term candidates for the 32-clip scope, labelled machine_only.

Same extractor as the 15-clip sheet (Qwen3.5-4B, prompt extract-v2), same
chunking and guards, but the input is the full-audio machine transcript and no
row claims a human annotation: R and D stay empty and gold_how=machine_only.

Outputs
-------
experiments/annotation/labels_todo_32clip_machine.csv
experiments/outputs/machine_terms_cache.json
experiments/outputs/machine_labels_32_summary.json

Usage
-----
python experiments/data/build_item_sheet_32_machine.py
python experiments/data/build_item_sheet_32_machine.py --ids TR-08 EN-05 --force

# 100+ expansion scope (same extractor; separate cache so the 32-clip artifacts stay byte-stable)
python experiments/data/build_item_sheet_32_machine.py \
  --manifest experiments/data/manifest_100plus.jsonl \
  --out experiments/annotation/labels_todo_100plus_machine.csv \
  --cache experiments/outputs/machine_terms_cache_100plus.json \
  --summary experiments/outputs/machine_labels_100plus_summary.json
# paths are relative to the repo root (run from there)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
ROOT = DATA_DIR.parent
sys.path.insert(0, str(DATA_DIR))

import build_item_sheet_15clip as B  # noqa: E402

MANIFEST = DATA_DIR / "manifest_32_machine.jsonl"
TRANSCRIPT_DIR = ROOT / "outputs" / "transcripts" / "machine_only"
OUT_CSV = ROOT / "annotation" / "labels_todo_32clip_machine.csv"
CACHE_JSON = ROOT / "outputs" / "machine_terms_cache.json"
SUMMARY_JSON = ROOT / "outputs" / "machine_labels_32_summary.json"

FIELDS = [
    "item_id", "video_id", "model", "lang", "term", "explanation",
    "context", "R", "D", "comment", "gold_how",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Machine-only term candidates for the 32-clip scope.")
    ap.add_argument("--ids", nargs="+", default=None)
    ap.add_argument("--force", action="store_true", help="ignore the extraction cache")
    ap.add_argument("--manifest", type=Path, default=MANIFEST,
                    help="machine manifest jsonl (default: the 32-clip scope)")
    ap.add_argument("--out", type=Path, default=OUT_CSV, help="candidate sheet to write")
    ap.add_argument("--cache", type=Path, default=CACHE_JSON,
                    help="extraction cache; keyed by clip id, so scopes may share it safely")
    ap.add_argument("--summary", type=Path, default=SUMMARY_JSON)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.ids:
        keep = set(args.ids)
        rows = [r for r in rows if r["id"] in keep]

    cache = json.loads(args.cache.read_text(encoding="utf-8")) if args.cache.exists() else {}
    out: list[dict] = []
    per_clip: list[dict] = []
    skipped: list[str] = []
    t_all = time.perf_counter()

    for row in rows:
        clip = row["id"]
        if row["audio_status"] != "present":
            skipped.append(clip)
            continue
        src = TRANSCRIPT_DIR / f"{clip}.json"
        if not src.exists():
            skipped.append(clip)
            continue

        rec = json.loads(src.read_text(encoding="utf-8"))
        text = B.clean_text(rec["full_text"])
        words = len(text.split())
        budget = max(4, min(24, words // 260))
        print(f"{clip}: {words} words, target {budget} terms", flush=True)
        t0 = time.perf_counter()
        items = B.extract_clip(clip, row["lang"], text, budget, args.force, cache)
        per_clip.append(
            {
                "id": clip,
                "lang": row["lang"],
                "words": words,
                "target_terms": budget,
                "items": len(items),
                "seconds": round(time.perf_counter() - t0, 1),
                "gold_how": "machine_only",
            }
        )
        for i, it in enumerate(items, 1):
            out.append(
                {
                    "item_id": f"{clip}-machine-{i:03d}",
                    "video_id": clip,
                    "model": B.MODEL_TAG,
                    "lang": row["lang"],
                    "term": it["term"],
                    "explanation": it["explanation"],
                    "context": it.get("context", ""),
                    "R": "",
                    "D": "",
                    "comment": f"machine-only candidate ({B.PROMPT_VERSION}); no human annotation",
                    "gold_how": "machine_only",
                }
            )
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)

    summary = {
        "clips_in_scope": len(rows),
        "clips_with_candidates": len({r["video_id"] for r in out}),
        "clips_skipped": skipped,
        "items_total": len(out),
        "gold_how": "machine_only",
        "extractor": B.LLM_MODEL,
        "prompt_version": B.PROMPT_VERSION,
        "asr_model": B.ASR_MODEL,
        "per_clip": per_clip,
        "wall_seconds": round(time.perf_counter() - t_all, 1),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {args.out} ({len(out)} items, {summary['clips_with_candidates']} clips)")
    if skipped:
        print(f"skipped (no audio / no transcript): {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
