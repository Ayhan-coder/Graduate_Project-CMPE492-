"""Re-run long-form ASR for the 30+ clip inventory.

The original pipeline called the Hugging Face chunked Whisper pipeline with
``return_timestamps=False``. Whisper cannot stitch chunks without timestamps,
so only the first 30 s chunk survived (36-93 words for 50-minute lectures).
This script forces timestamps, batches chunks, applies a words-per-minute QA
gate and writes one JSON per clip. It is resumable: clips that already passed
QA are skipped unless ``--force`` is given.

Usage:
    python experiments/data/retranscribe_longform.py --dry-run
    python experiments/data/retranscribe_longform.py --ids EN-12 --force
    python experiments/data/retranscribe_longform.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "experiments" / "data" / "manifest_30plus.jsonl"
AUDIO_DIR = REPO / "experiments" / "data" / "audio"
DEFAULT_OUT = REPO / "experiments" / "outputs" / "transcripts_v2"
CACHE_DIR = REPO / "experiments" / "hf_cache"
DEFAULT_MODEL = "openai/whisper-large-v3-turbo"
MIN_WPM = 60.0

LOG = logging.getLogger("retranscribe")


def load_manifest() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with MANIFEST.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                rows[rec["id"]] = rec
    return rows


def resolve_audio(vid: str, field: str | None) -> Path | None:
    candidates = [AUDIO_DIR / f"{vid}.wav"]
    if field:
        candidates.append(REPO / "experiments" / field)
        candidates.append(AUDIO_DIR / f"{Path(field).stem}.wav")
    for path in candidates:
        if path.exists():
            return path
    return None


def wav_duration(path: Path) -> float:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:
        with wave.open(str(path), "rb") as fh:
            return fh.getnframes() / float(fh.getframerate())


def already_passed(out_file: Path) -> bool:
    if not out_file.exists():
        return False
    try:
        data = json.loads(out_file.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(data.get("qa_passed"))


def transcribe_one(pipe, audio: Path, lang: str, model: str) -> dict:
    started = time.time()
    result = pipe(
        str(audio),
        generate_kwargs={
            "language": "turkish" if lang == "tr" else "english",
            "task": "transcribe",
        },
        return_timestamps=True,
        batch_size=8,
    )
    elapsed = round(time.time() - started, 1)
    text = (result.get("text") or "").strip()
    chunks = result.get("chunks") or []
    duration = wav_duration(audio)
    words = len(text.split())
    wpm = round(words / max(0.01, duration / 60.0), 1)
    passed = wpm >= MIN_WPM
    return {
        "id": audio.stem,
        "model": model,
        "language": lang,
        "audio_file": str(audio),
        "duration_s": round(duration, 2),
        "word_count": words,
        "wpm": wpm,
        "qa_passed": passed,
        "qa_status": "QA_PASSED" if passed else "QA_REJECTED_LOW_WPM",
        "latency_s": elapsed,
        "segment_count": len(chunks),
        "full_text": text,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", help="comma separated clip ids, e.g. TR-04,EN-12")
    ap.add_argument("--all", action="store_true", help="every manifest clip that has audio")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="re-run clips that already passed QA")
    ap.add_argument("--dry-run", action="store_true", help="list what would run")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    if args.ids:
        selected = [x.strip() for x in args.ids.split(",") if x.strip()]
    else:
        selected = list(manifest)

    jobs: list[tuple[str, Path, str]] = []
    for vid in selected:
        rec = manifest.get(vid)
        if rec is None:
            LOG.warning("unknown clip id %s", vid)
            continue
        audio = resolve_audio(vid, rec.get("audio"))
        if audio is None:
            LOG.warning("%s: no audio file, skipped", vid)
            continue
        if not args.force and already_passed(out_dir / f"{vid}.json"):
            LOG.info("%s: already passed QA, skipped", vid)
            continue
        jobs.append((vid, audio, rec.get("lang", "tr")))

    if args.limit:
        jobs = jobs[: args.limit]

    total_min = sum(wav_duration(a) for _, a, _ in jobs) / 60.0
    LOG.info("%d clip(s) queued, %.1f minutes of audio", len(jobs), total_min)
    if args.dry_run:
        for vid, audio, lang in jobs:
            LOG.info("  %s | %s | %.1f min", vid, audio.name, wav_duration(audio) / 60.0)
        return 0
    if not jobs:
        return 0

    import torch
    from transformers import pipeline

    LOG.info("loading %s on cuda (fp16)", args.model)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device="cuda",
        torch_dtype=torch.float16,
        chunk_length_s=30,
        stride_length_s=5,
        model_kwargs={"cache_dir": str(CACHE_DIR)},
    )

    done_min = 0.0
    for idx, (vid, audio, lang) in enumerate(jobs, 1):
        LOG.info("[%d/%d] %s (%.1f min) -> %s", idx, len(jobs), vid, wav_duration(audio) / 60.0, audio.name)
        try:
            payload = transcribe_one(pipe, audio, lang, args.model)
        except Exception as exc:  # one bad clip must not kill the queue
            LOG.error("%s: transcription failed: %s", vid, exc)
            continue
        (out_dir / f"{vid}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        done_min += payload["duration_s"] / 60.0
        LOG.info(
            "  %s: %d words, %.1f wpm, %s, audio-time %.1f min in %.1f s",
            vid, payload["word_count"], payload["wpm"], payload["qa_status"],
            payload["duration_s"] / 60.0, payload["latency_s"],
        )
        if idx < len(jobs):
            eta = (total_min - done_min) * (payload["latency_s"] / max(0.01, payload["duration_s"]))
            LOG.info("  running ETA %.0f min", eta / 60.0)

    index = {}
    for path in sorted(out_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        index[data["id"]] = {
            k: data.get(k) for k in ("word_count", "wpm", "qa_passed", "duration_s", "latency_s")
        }
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    passed = sum(1 for v in index.values() if v["qa_passed"])
    LOG.info("index written: %d clips, %d passed QA", len(index), passed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
