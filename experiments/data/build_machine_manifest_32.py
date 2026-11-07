"""Build the honest 32-clip machine manifest.

The previous manifest_30plus.jsonl carried a `reference` string for 25 clips that
was written by hand inside build_manifest_30plus.py (SYNTHETIC_TRANSCRIPTS), not
produced from the audio. Those strings are dropped here: every clip either has a
real audio file that can be transcribed, or it is marked as missing audio.

Outputs
-------
experiments/data/manifest_32_machine.jsonl
    One row per declared clip: audio path (or null), declared vs measured
    duration, audio_status, gold_how="machine_only", reference=null.
experiments/outputs/machine_manifest_32_summary.json
    Counts, the missing-audio ids and the duration mismatches.

Usage
-----
python experiments/data/build_machine_manifest_32.py
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
SOURCES = DATA_DIR / "sources_30plus.json"
OUT_MANIFEST = DATA_DIR / "manifest_32_machine.jsonl"
OUT_SUMMARY = DATA_DIR.parent / "outputs" / "machine_manifest_32_summary.json"
TRANSCRIPT_DIR = DATA_DIR.parent / "outputs" / "transcripts" / "machine_only"

AUDIO_DIRS = (DATA_DIR, DATA_DIR / "audio", DATA_DIR / "_work")
AUDIO_EXTS = (".wav", ".webm", ".mp3", ".m4a")


def candidates(clip_id: str) -> list[str]:
    prefix, _, num = clip_id.partition("-")
    names = [clip_id]
    if num.isdigit():
        names.append(f"{prefix}-{int(num)}")
    return names


def resolve_audio(clip_id: str) -> Path | None:
    for d in AUDIO_DIRS:
        for name in candidates(clip_id):
            for ext in AUDIO_EXTS:
                p = d / f"{name}{ext}"
                if p.exists() and p.stat().st_size > 1024:
                    return p
    return None


def wav_duration(p: Path) -> float | None:
    if p.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(p), "rb") as w:
            return round(w.getnframes() / w.getframerate(), 1)
    except (wave.Error, EOFError):
        return None


def main() -> None:
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    declared = {c["id"]: c for c in sources["clips"]}
    prior = {}
    legacy = DATA_DIR / "manifest_30plus.jsonl"
    if legacy.exists():
        for line in legacy.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                prior[r["id"]] = r

    rows: list[dict] = []
    missing: list[str] = []
    mismatched: list[dict] = []

    for clip_id, c in declared.items():
        declared_s = float(prior.get(clip_id, {}).get("duration_s") or 0) or None
        audio = resolve_audio(clip_id)
        measured_s = wav_duration(audio) if audio else None
        rel = None
        if audio:
            rel = str(audio.relative_to(DATA_DIR.parent)).replace("\\", "/")

        matched = None
        if declared_s and measured_s:
            matched = abs(measured_s - declared_s) <= max(5.0, 0.05 * declared_s)
            if not matched:
                mismatched.append(
                    {
                        "id": clip_id,
                        "declared_s": declared_s,
                        "measured_s": measured_s,
                        "ratio": round(measured_s / declared_s, 2),
                    }
                )

        transcript = TRANSCRIPT_DIR / f"{clip_id}.json"
        has_transcript = transcript.exists()
        transcript_rel = (
            str(transcript.relative_to(DATA_DIR.parent)).replace("\\", "/") if has_transcript else None
        )

        rows.append(
            {
                "id": clip_id,
                "lang": c["lang"],
                "domain": c.get("domain"),
                "title": c.get("title"),
                "channel": c.get("channel"),
                "youtube_id": c.get("youtube_id"),
                "duration_declared_s": declared_s,
                "duration_measured_s": measured_s,
                "duration_matches_declared": matched,
                "audio": rel,
                "audio_status": "present" if audio else "missing",
                "gold_how": "machine_only",
                "reference": None,
                "reference_source": (
                    "machine_transcript" if has_transcript else ("machine_transcript_pending" if audio else "unavailable_no_audio")
                ),
                "transcript": transcript_rel,
                "notes": (
                    ""
                    if audio
                    else "Ses dosyası yok; bu klip makine transkriptiyle de kapsanamaz."
                ),
            }
        )
        if not audio:
            missing.append(clip_id)

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with OUT_MANIFEST.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "clips_declared": len(rows),
        "audio_present": len(rows) - len(missing),
        "audio_missing": len(missing),
        "missing_ids": missing,
        "duration_mismatch": len(mismatched),
        "mismatches": sorted(mismatched, key=lambda m: -m["ratio"]),
        "total_declared_s": round(sum(r["duration_declared_s"] or 0 for r in rows), 1),
        "total_measured_s": round(sum(r["duration_measured_s"] or 0 for r in rows), 1),
        "gold_how": "machine_only (no human double annotation)",
        "legacy_synthetic_references_dropped": sum(
            1 for r in prior.values() if r.get("gold_how") == "curated_domain_transcription"
        ),
    }
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {OUT_MANIFEST} ({len(rows)} clips)")
    print(f"audio present {summary['audio_present']} / missing {summary['audio_missing']}: {missing}")
    print(f"duration mismatches: {len(mismatched)}")
    for m in sorted(mismatched, key=lambda x: -x["ratio"])[:8]:
        print(f"  {m['id']}: declared {m['declared_s']:.0f}s -> measured {m['measured_s']:.0f}s ({m['ratio']}x)")
    print(f"total audio {summary['total_measured_s'] / 3600:.2f} h")


if __name__ == "__main__":
    main()
