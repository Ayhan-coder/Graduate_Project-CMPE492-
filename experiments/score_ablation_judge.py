"""Score the 2026 ASR checkpoint ablation rows with the LLM-as-a-Judge rubric.

This file originally always called judge_term_explanation(use_mock=True), which is a
deterministic keyword/circularity heuristic rather than a judge; those labels remain in
annotation/labels_ablation_extended.csv. The default path here makes real Qwen3.5-4B
rubric calls and writes to a separate, resumable labels file so the two methods stay
distinguishable in the audit trail. Pass --mock to reproduce the old heuristic.

Outputs:
    annotation/labels_ablation_real_judge.csv  one row per judged term-explanation pair
    outputs/ablation_judge.json                per (ASR model, language) R1%/D1% summary
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from llm_judge import judge_term_explanation

ROOT = Path(__file__).resolve().parent
ABLATION_PATH = ROOT / "outputs" / "ablation-all.jsonl"
LABELS_CSV = ROOT / "annotation" / "labels_ablation_real_judge.csv"
OUT_JSON = ROOT / "outputs" / "ablation_judge.json"
COLUMNS = ["item_id", "video_id", "model", "lang", "term", "explanation", "R", "D", "reason"]

TARGET_MODELS = [
    "selimc/whisper-large-v3-turbo-turkish",
    "Qwen/Qwen3-ASR-0.6B-hf",
    "nvidia/nemotron-3.5-asr-streaming-0.6b",
]


def load_pairs(path: Path, models: list[str]) -> list[dict]:
    pairs: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("asr_model") not in models:
            continue
        short = rec["asr_model"].split("/")[-1]
        for idx, item in enumerate(rec.get("terms", []), start=1):
            term = (item.get("term") or "").strip()
            expl = (item.get("explanation") or "").strip()
            if not term or not expl:
                continue
            pairs.append(
                {
                    "item_id": f"{rec['id']}-{short}-{idx:03d}",
                    "video_id": rec["id"],
                    "model": rec["asr_model"],
                    "lang": str(rec.get("lang", "")).lower()[:2],
                    "term": term,
                    "explanation": expl,
                }
            )
    return pairs


def read_done(path: Path) -> dict[str, tuple[int, int]]:
    if not path.exists():
        return {}
    done: dict[str, tuple[int, int]] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                done[row["item_id"]] = (int(row["R"]), int(row["D"]))
            except (KeyError, ValueError):
                continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=Path, default=ABLATION_PATH)
    parser.add_argument("--labels-csv", type=Path, default=LABELS_CSV)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--judge-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="use the keyword heuristic instead of real judge calls (reproduces labels_ablation_extended.csv)",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    pairs = load_pairs(args.ablation, TARGET_MODELS)
    done = read_done(args.labels_csv)
    todo = [p for p in pairs if p["item_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    mode = "keyword heuristic (use_mock=True)" if args.mock else f"real judge: {args.judge_model}"
    print(f"{len(pairs)} pairs, {len(done)} labelled, {len(todo)} to judge via {mode}", flush=True)

    args.labels_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.labels_csv.exists()
    with args.labels_csv.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        started = time.perf_counter()
        for n, p in enumerate(todo, start=1):
            r_val, d_val, reason = judge_term_explanation(
                term=p["term"],
                explanation=p["explanation"],
                topic=f"Lecture {p['video_id']}",
                lang=p["lang"],
                model_id=args.judge_model,
                use_mock=args.mock,
            )
            row = {k: p[k] for k in ("item_id", "video_id", "model", "lang", "term", "explanation")}
            row.update({"R": r_val, "D": d_val, "reason": reason.replace("\n", " ")[:300]})
            writer.writerow(row)
            fh.flush()
            done[p["item_id"]] = (r_val, d_val)
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} ({n / (time.perf_counter() - started):.2f}/s)", flush=True)

    cells: dict[str, dict] = defaultdict(lambda: {"terms": 0, "R1": 0, "D1": 0})
    clip_stats: dict[str, list[dict]] = defaultdict(list)
    for line in args.ablation.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("asr_model") in TARGET_MODELS:
                lang2 = str(rec.get("lang", "")).lower()[:2]
                clip_stats[f"{rec['asr_model']}|{lang2}"].append(rec)

    for p in pairs:
        labels = done.get(p["item_id"])
        if labels is None:
            continue
        key = f"{p['model']}|{p['lang']}"
        cells[key]["terms"] += 1
        cells[key]["R1"] += labels[0]
        cells[key]["D1"] += labels[1]

    for key, cell in cells.items():
        recs = clip_stats.get(key, [])
        lat = [float(r["latency_s"]) for r in recs if r.get("latency_s") is not None]
        cell["clips"] = len(recs)
        cell["emitted_terms_total"] = sum(int(r.get("n_terms") or 0) for r in recs)
        cell["mean_llm_latency_s"] = round(sum(lat) / len(lat), 2) if lat else None
        cell["r1_pct"] = round(cell["R1"] / cell["terms"] * 100, 1) if cell["terms"] else 0.0
        cell["d1_pct"] = round(cell["D1"] / cell["terms"] * 100, 1) if cell["terms"] else 0.0

    args.out_json.write_text(
        json.dumps(
            {
                "judge": mode,
                "source_ablation": args.ablation.name,
                "labels_csv": args.labels_csv.name,
                "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "cells": dict(cells),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(dict(cells), ensure_ascii=False, indent=2))
    print("wrote", args.out_json, "and", args.labels_csv)


if __name__ == "__main__":
    main()
