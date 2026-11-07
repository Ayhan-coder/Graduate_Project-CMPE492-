"""Turn the verified 100-clip worklist into a machine manifest for ASR.

Only rows with audio_status=present become manifest rows, and every duration in the
output is the ffprobe-measured value, not the declared YouTube metadata. Rows that could
not be fetched stay in worklist_100plus.json with their error text so the gap is visible
instead of quietly shrinking the corpus.

Usage:
    python experiments/data/build_manifest_100plus.py

Output: experiments/data/manifest_100plus.jsonl  (same row shape as manifest_32_machine.jsonl)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WORKLIST = DATA / "worklist_100plus.json"
OUT = DATA / "manifest_100plus.jsonl"


def main() -> None:
    doc = json.loads(WORKLIST.read_text(encoding="utf-8"))
    rows = []
    for r in doc["rows"]:
        if r["audio_status"] != "present":
            continue
        wav = ROOT / r["audio_path"].replace("experiments/", "")
        if not wav.exists():
            continue
        dur = r["duration_measured_s"]
        rows.append({
            "id": r["id"],
            "lang": r["lang"],
            "domain": r["domain"],
            "title": r["title"],
            "channel": r["channel"],
            "youtube_id": r["youtube_id"],
            "duration_declared_s": r["duration_declared_s"],
            "duration_measured_s": dur,
            "duration_matches_declared": r.get("matches_declared"),
            "audio": str(wav.relative_to(ROOT)).replace("\\", "/"),
            "audio_status": "present",
            "gold_how": "machine_only",
            "reference": None,
            "reference_source": "machine_transcript",
            "transcript": f"outputs/transcripts/machine_only/{r['id']}.json",
            "notes": "100-clip expansion; discovered and fetched by exact video id, "
                     "duration measured with ffprobe",
        })

    rows.sort(key=lambda r: r["id"])
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    hours = sum(r["duration_measured_s"] or 0 for r in rows) / 3600
    dropped = [r for r in doc["rows"] if r["audio_status"] != "present"]
    print(f"manifest: {len(rows)} satir | olculen {hours:.2f} saat | "
          f"dil {dict(Counter(r['lang'] for r in rows))}")
    print(f"indirilemeyen {len(dropped)} kayit manifestoya alinmadi: "
          + ", ".join(f"{r['id']}({r['audio_status']})" for r in dropped))
    print(f"yazildi: {OUT}")


if __name__ == "__main__":
    main()
