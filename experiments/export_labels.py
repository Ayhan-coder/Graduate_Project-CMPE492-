"""Turn an LLM or ablation JSONL into an annotation CSV (R/D empty)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["item_id", "video_id", "model", "lang", "term", "explanation", "R", "D", "comment"],
        )
        writer.writeheader()
        for line in args.input.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            model = row.get("model", "")
            asr = row.get("asr_model") or ""
            tag = f"{asr}|{model}" if asr else model
            for i, term in enumerate(row.get("terms") or [], start=1):
                n += 1
                writer.writerow(
                    {
                        "item_id": f"{row.get('id', 'x')}-{tag}-{i:03d}",
                        "video_id": row.get("id", ""),
                        "model": tag,
                        "lang": row.get("lang", ""),
                        "term": term.get("term", ""),
                        "explanation": term.get("explanation", ""),
                        "R": "",
                        "D": "",
                        "comment": "",
                    }
                )
    print(f"wrote {args.output} ({n} items)")


if __name__ == "__main__":
    main()
