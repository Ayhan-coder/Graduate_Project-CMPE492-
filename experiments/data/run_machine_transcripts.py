"""Transcribe every clip that has audio, in full, into machine-only transcripts.

The transcripts already in outputs/transcripts/openai_whisper-large-v3-turbo are
truncated for the extended clips (e.g. 36 words out of a 50-minute recording), so
they cannot back a label sheet. This driver re-runs the same ASR model over the
whole file and writes one JSON per clip under transcripts/machine_only/.

Nothing here is a gold reference: every record carries gold_how="machine_only".

Usage
-----
python experiments/data/run_machine_transcripts.py --smoke      # one short clip
python experiments/data/run_machine_transcripts.py              # all clips with audio
python experiments/data/run_machine_transcripts.py --ids TR-08 EN-05 --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_asr as R  # noqa: E402

MANIFEST = ROOT / "data" / "manifest_32_machine.jsonl"
OUT_DIR = ROOT / "outputs" / "transcripts" / "machine_only"
MODEL_ID = "openai/whisper-large-v3-turbo"
MIN_WPM = 30.0


def decoding_args() -> Namespace:
    return Namespace(
        chunk_s=30.0,
        language_detect=False,
        condition_on_prev=False,
        use_vad=False,
        prompt_glossary=False,
        prompt_text=None,
    )


def is_complete(rec: dict) -> bool:
    dur = rec.get("duration_s") or 0
    words = rec.get("word_count") or 0
    if dur <= 0:
        return words > 0
    return words / (dur / 60) >= MIN_WPM


def main() -> int:
    ap = argparse.ArgumentParser(description="Full-audio machine transcripts for the 32-clip scope.")
    ap.add_argument("--ids", nargs="+", default=None)
    ap.add_argument("--smoke", action="store_true", help="only the shortest clip with audio")
    ap.add_argument("--force", action="store_true", help="re-transcribe even if a complete transcript exists")
    ap.add_argument("--order", choices=["duration", "manifest"], default="duration",
                    help="duration: shortest first, so partial runs still cover the most clips")
    ap.add_argument("--manifest", type=Path, default=MANIFEST,
                    help="machine manifest jsonl to read clips from (default: the 32-clip scope)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["audio_status"] == "present"]
    if args.ids:
        keep = set(args.ids)
        rows = [r for r in rows if r["id"] in keep]
    if args.smoke:
        rows = sorted(rows, key=lambda r: r["duration_measured_s"] or 0)[:1]
    elif args.order == "duration":
        rows = sorted(rows, key=lambda r: r["duration_measured_s"] or 0)
    if not rows:
        print("nothing selected")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dec = decoding_args()
    model = {"id": MODEL_ID, "backend": "whisper", "subset": "all"}

    done = skipped = failed = 0
    t_all = time.perf_counter()
    for i, row in enumerate(rows, 1):
        clip = row["id"]
        out_path = OUT_DIR / f"{clip}.json"
        if out_path.exists() and not args.force:
            prior = json.loads(out_path.read_text(encoding="utf-8"))
            if is_complete(prior):
                skipped += 1
                print(f"[{i}/{len(rows)}] {clip}: complete transcript exists, skipped", flush=True)
                continue

        audio = row["audio"]
        if not Path(audio).is_absolute():
            audio = str((ROOT / audio).resolve())
        dur = row["duration_measured_s"]
        print(f"[{i}/{len(rows)}] {clip}: {dur / 60:.1f} min audio ...", flush=True)
        try:
            t0 = time.perf_counter()
            text = R.transcribe_whisper(MODEL_ID, audio, row["lang"], dec, clip_id=clip)
            elapsed = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001 - one bad clip must not kill the batch
            failed += 1
            print(f"  {clip}: FAILED {type(exc).__name__}: {exc}", flush=True)
            continue

        words = len(text.split())
        rec = {
            "id": clip,
            "lang": row["lang"],
            "model": MODEL_ID,
            "audio_file": row["audio"],
            "duration_s": dur,
            "duration_declared_s": row["duration_declared_s"],
            "latency_s": round(elapsed, 1),
            "rtf": round(elapsed / dur, 4) if dur else None,
            "word_count": words,
            "wpm": round(words / (dur / 60), 1) if dur else None,
            "gold_how": "machine_only",
            "reference_source": "machine_transcript",
            "decoding": {
                "task": "transcribe",
                "num_beams": 1,
                "temperature": 0.0,
                "condition_on_prev_tokens": False,
                "language": row["lang"],
            },
            "full_text": text.strip(),
        }
        out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        done += 1
        print(f"  {clip}: {words} words, {elapsed:.0f}s, rtf {rec['rtf']}", flush=True)

    total = time.perf_counter() - t_all
    print(f"\ndone={done} skipped={skipped} failed={failed} in {total / 60:.1f} min")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
