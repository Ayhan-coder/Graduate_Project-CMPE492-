"""Build the 100-clip expansion worklist from the reviewed candidate pool.

Current measured state (30.10.2026, ffprobe over experiments/data/**.wav): 34 local WAVs,
16.76 h, 2 of them under 120 s (EN-12 117 s, EN-18 92 s) -> 32 clips carry usable audio.
To reach 100 clips with usable audio we need 68 new ones; the recorded download yield of
the existing corpus is 26/32 (81 %), so this worklist deliberately oversources to 100
candidates and the fetch step verifies each file after the fact.

Nothing here is gold. Every new row is written with gold_how=machine_only and
transcript.status=PENDING_MACHINE_ASR; the submission inventory (data_inventory.json) and
the 32-clip manifest are not touched.

Usage:
    python experiments/data/build_worklist_100plus.py --target 100
    python experiments/data/build_worklist_100plus.py --dry-run

Output: experiments/data/worklist_100plus.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments" / "data"
POOL = DATA / "candidates_100plus_reviewed.json"
INVENTORY = DATA / "data_inventory.json"
OUT = DATA / "worklist_100plus.json"

# measured floor for a clip to be worth transcribing
MIN_USABLE_S = 120


def existing_ids() -> tuple[set[str], dict[str, int]]:
    inv = json.loads(INVENTORY.read_text(encoding="utf-8"))
    ids = {r["id"] for r in inv["records"]}
    nxt = {"TR": 21, "EN": 21}
    for r in inv["records"]:
        pref, num = r["id"].split("-")
        nxt[pref] = max(nxt[pref], int(num) + 1)
    return ids, nxt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=100, help="candidates to enqueue")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pool = json.loads(POOL.read_text(encoding="utf-8"))
    keepers = [c for c in pool["keep"] if c["duration_s"] >= MIN_USABLE_S]
    used, nxt = existing_ids()

    # balance across domain x lang so no single topic dominates the expanded corpus
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for c in sorted(keepers, key=lambda c: (-c["duration_s"], c["title"])):
        by_cell[f'{c["lang"]}|{c["domain"]}'].append(c)

    picked: list[dict] = []
    cells = sorted(by_cell)
    while len(picked) < args.target and any(by_cell.values()):
        for cell in cells:
            lst = by_cell.get(cell) or []
            if not lst:
                continue
            picked.append(lst.pop(0))
            if len(picked) >= args.target:
                break

    rows = []
    for c in picked:
        pref = c["lang"].upper()
        clip_id = f"{pref}-{nxt[pref]:02d}"
        nxt[pref] += 1
        rows.append({
            "id": clip_id,
            "lang": c["lang"],
            "domain": c["domain"],
            "title": c["title"],
            "channel": c["channel"],
            "youtube_id": c["youtube_id"],
            "url": c["url"],
            "duration_declared_s": c["duration_s"],
            "duration_measured_s": None,
            "audio_status": "pending",
            "audio_path": f"experiments/data/audio/{clip_id}.wav",
            "gold_how": "machine_only",
            "transcript_status": "PENDING_MACHINE_ASR",
            "selection": "round-robin over domain x lang from candidates_100plus_reviewed.json",
        })

    out = {
        "note": "Worklist for the 100-clip expansion. duration_declared_s is the YouTube "
                "metadata value, which was shown to be unreliable in the existing corpus "
                "(18/32 records mismatched), so fetch_clips_100plus.py must measure every "
                "file with ffprobe and record duration_measured_s. Rows that fail to yield "
                "audio >= 120 s are marked audio_status=missing/short and stay out of the "
                "usable count. gold_how=machine_only: no human transcription, no gold claim.",
        "target_clips_with_audio": 100,
        "clips_with_audio_before": 32,
        "measured_before": {
            "local_wav_files": 34,
            "measured_hours": 16.76,
            "under_120s": ["EN-12 (117 s)", "EN-18 (92 s)"],
            "basis": "ffprobe over experiments/data/*.wav and experiments/data/audio/*.wav",
        },
        "candidates_enqueued": len(rows),
        "by_lang": {k: sum(1 for r in rows if r["lang"] == k) for k in ("tr", "en")},
        "declared_hours_enqueued": round(sum(r["duration_declared_s"] for r in rows) / 3600, 2),
        "rows": rows,
    }
    if args.dry_run:
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=1))
        print("dry run: yazilmadi")
        return
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"kuyruk: {len(rows)} aday | dil {out['by_lang']} | ilan edilen toplam "
          f"{out['declared_hours_enqueued']} h")
    print(f"yazildi: {OUT}")


if __name__ == "__main__":
    main()
