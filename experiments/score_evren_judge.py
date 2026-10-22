"""Score EVREN terminology outputs with the real local LLM-as-a-Judge.

Unlike score_ablation_judge.py, this never calls the judge with use_mock=True: every
R/D label here comes from an actual Qwen3.5-4B rubric call, and each labelled pair is
written to annotation/labels_evren_judge.csv so the reported percentages are auditable.

Usage:
    python score_evren_judge.py                     # judge both EVREN artifacts
    python score_evren_judge.py --limit 12          # smoke test
    python score_evren_judge.py --judge-model evren/qwen3.8-flash-next
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
LABELS_CSV = ROOT / "annotation" / "labels_evren_judge.csv"
OUT_JSON = ROOT / "outputs" / "evren_judge.json"
COLUMNS = ["item_id", "video_id", "model", "lang", "term", "explanation", "R", "D", "reason"]


def lang_code(raw: str) -> str:
    return "tr" if str(raw).lower().startswith("turk") else "en"


def load_pairs(artifacts: list[Path]) -> list[dict]:
    pairs: list[dict] = []
    for path in artifacts:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for idx, item in enumerate(rec.get("terms", []), start=1):
                term = (item.get("term") or "").strip()
                expl = (item.get("explanation") or "").strip()
                if not term or not expl:
                    continue
                pairs.append(
                    {
                        "item_id": f"{rec['id']}-{rec['model'].split('/')[-1]}-{idx:03d}",
                        "video_id": rec["id"],
                        "model": rec["model"],
                        "lang": lang_code(rec.get("lang", "")),
                        "term": term,
                        "explanation": expl,
                    }
                )
    return pairs


def read_done() -> dict[str, tuple[int, int]]:
    if not LABELS_CSV.exists():
        return {}
    done: dict[str, tuple[int, int]] = {}
    with LABELS_CSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                done[row["item_id"]] = (int(row["R"]), int(row["D"]))
            except (KeyError, ValueError):
                continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts",
        nargs="*",
        type=Path,
        default=sorted((ROOT / "outputs").glob("llm-*.jsonl"))[-2:],
        help="run_llm.py JSONL files to score (default: the two newest)",
    )
    parser.add_argument("--judge-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    pairs = load_pairs(args.artifacts)
    done = read_done()
    todo = [p for p in pairs if p["item_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(
        f"{len(pairs)} pairs total, {len(done)} already labelled, judging {len(todo)} "
        f"with {args.judge_model}",
        flush=True,
    )

    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LABELS_CSV.exists()
    with LABELS_CSV.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        t_start = time.perf_counter()
        for n, p in enumerate(todo, start=1):
            r_val, d_val, reason = judge_term_explanation(
                term=p["term"],
                explanation=p["explanation"],
                topic=f"Lecture {p['video_id']}",
                lang=p["lang"],
                model_id=args.judge_model,
            )
            writer.writerow(
                {
                    "item_id": p["item_id"],
                    "video_id": p["video_id"],
                    "model": p["model"],
                    "lang": p["lang"],
                    "term": p["term"],
                    "explanation": p["explanation"],
                    "R": r_val,
                    "D": d_val,
                    "reason": reason.replace("\n", " ")[:300],
                }
            )
            fh.flush()
            done[p["item_id"]] = (r_val, d_val)
            if n % 10 == 0 or n == len(todo):
                rate = n / (time.perf_counter() - t_start)
                print(f"  {n}/{len(todo)} judged ({rate:.2f}/s)", flush=True)

    stats: dict[str, dict] = defaultdict(lambda: {"terms": 0, "R1": 0, "D1": 0})
    for p in pairs:
        labels = done.get(p["item_id"])
        if labels is None:
            continue
        key = f"{p['model']}|{p['lang']}"
        stats[key]["terms"] += 1
        stats[key]["R1"] += labels[0]
        stats[key]["D1"] += labels[1]

    # Latency and per-clip term counts come from the extraction run itself, not from
    # the pair list, so a term-rich clip cannot dominate the mean.
    clip_rows: dict[str, list[dict]] = defaultdict(list)
    for path in args.artifacts:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                clip_rows[f"{rec['model']}|{lang_code(rec.get('lang', ''))}"].append(rec)

    for key, s in stats.items():
        lat = [float(r["latency_s"]) for r in clip_rows.get(key, []) if r.get("latency_s") is not None]
        s["r1_pct"] = round(s["R1"] / s["terms"] * 100, 1) if s["terms"] else 0.0
        s["d1_pct"] = round(s["D1"] / s["terms"] * 100, 1) if s["terms"] else 0.0
        s["mean_latency_s"] = round(sum(lat) / len(lat), 2) if lat else None
        s["latency_range_s"] = [round(min(lat), 2), round(max(lat), 2)] if lat else None
        s["clips"] = len(lat)
        s["clip_terms"] = {r["id"]: r["n_terms"] for r in clip_rows.get(key, [])}

    OUT_JSON.write_text(
        json.dumps(
            {
                "judge_model": args.judge_model,
                "judged_by": "real rubric calls (use_mock=False)",
                "artifacts": [str(p.name) for p in args.artifacts],
                "extraction_config": {"reasoning_effort": "none", "temperature": 0.0, "max_tokens": 4096},
                "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "cells": dict(stats),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))
    print("wrote", OUT_JSON, "and", LABELS_CSV)


if __name__ == "__main__":
    main()
