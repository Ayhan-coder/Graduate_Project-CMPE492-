"""Download confirmed (or --allow-candidates) lecture videos and make 16 kHz mono wavs.

Requires: ffmpeg on PATH, and yt-dlp (pip install yt-dlp).

Usage:
    python prepare_audio.py --list
    python prepare_audio.py --only EN-3 EN-4
    python prepare_audio.py --allow-candidates --only TR-1 EN-2 EN-3 EN-4
    python prepare_audio.py --from-local TR-1 path/to/file.mp4

Does not invent gold transcripts. If YouTube captions exist they are stored as
reference_source=caption_draft, which is NOT the paper gold.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SOURCES = DATA / "sources.json"


def load_sources() -> dict:
    return json.loads(SOURCES.read_text(encoding="utf-8"))


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def convert_to_wav(src: Path, dest: Path) -> float:
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-vn",
            str(dest),
        ]
    )
    probe = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(dest),
        ],
        text=True,
    ).strip()
    return float(probe)


def caption_text(folder: Path, clip_id: str) -> str:
    preferred = [
        f"{clip_id}.tr-orig.vtt",
        f"{clip_id}.en-orig.vtt",
        f"{clip_id}.tr.vtt",
        f"{clip_id}.en.vtt",
    ]
    path = None
    for name in preferred:
        cand = folder / name
        if cand.exists():
            path = cand
            break
    if path is None:
        hits = sorted(folder.glob(f"{clip_id}*.vtt")) + sorted(folder.glob(f"{clip_id}*.srt"))
        path = hits[0] if hits else None
    if path is None:
        return ""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines = []
    prev = ""
    for line in raw.splitlines():
        line = re.sub(r"<[^>]+>", "", line).strip()
        if not line or line.startswith("WEBVTT") or "-->" in line or line.isdigit():
            continue
        if line.startswith(("NOTE", "Kind:", "Language:", "Style:", "Region:")):
            continue
        if line == prev:
            continue
        lines.append(line)
        prev = line
    return " ".join(lines)


def write_manifest(rows: list[dict]) -> None:
    path = DATA / "manifest.jsonl"
    existing = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                existing[rec["id"]] = rec
    for row in rows:
        existing[row["id"]] = row
    order = ["TR-1", "TR-2", "TR-3", "EN-1", "EN-2", "EN-3", "EN-4"]
    with path.open("w", encoding="utf-8") as fh:
        for cid in order:
            if cid in existing:
                fh.write(json.dumps(existing[cid], ensure_ascii=False) + "\n")
    print("wrote", path)


def download_one(clip: dict) -> dict:
    cand = clip.get("candidate") or {}
    url = cand.get("url")
    if not url:
        raise SystemExit(f"{clip['id']} has no candidate URL")
    raw_dir = DATA / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    stem = raw_dir / clip["id"]
    cmd = [
        "yt-dlp",
        "-f",
        "bestaudio/best",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--ignore-errors",
        "--no-warnings",
        "-o",
        str(stem) + ".%(ext)s",
        url,
    ]
    # Captions are optional. A 429 on subs must not kill the audio download.
    try:
        run(cmd + ["--write-auto-sub", "--write-sub", "--sub-langs", "en.*,tr.*", "--skip-unavailable-fragments"])
    except subprocess.CalledProcessError:
        print("caption fetch failed; retrying audio only")
        run(cmd)
    audio_exts = (".wav", ".webm", ".m4a", ".mp3", ".opus", ".ogg", ".flac")
    src = None
    for ext in audio_exts:
        hit = raw_dir / f"{clip['id']}{ext}"
        if hit.exists():
            src = hit
            break
    if src is None:
        raise SystemExit(f"no audio file for {clip['id']} in {raw_dir}")
    # yt-dlp may already write wav; still force 16 kHz mono
    wav = DATA / f"{clip['id']}.wav"
    duration = convert_to_wav(src, wav)
    cap = caption_text(raw_dir, clip["id"])
    rec = {
        "id": clip["id"],
        "lang": clip["lang"],
        "audio": f"data/{clip['id']}.wav",
        "duration_s": round(duration, 2),
        "source_url": url,
        "confirmed": bool(clip.get("confirmed")),
        "reference_source": "caption_draft" if cap else "missing",
        "reference": cap,
    }
    return rec


def from_local(clip_id: str, src: Path, lang: str) -> dict:
    wav = DATA / f"{clip_id}.wav"
    duration = convert_to_wav(src, wav)
    return {
        "id": clip_id,
        "lang": lang,
        "audio": f"data/{clip_id}.wav",
        "duration_s": round(duration, 2),
        "source_url": str(src),
        "confirmed": True,
        "reference_source": "missing",
        "reference": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--allow-candidates", action="store_true",
                        help="download high/medium confidence matches even if not confirmed")
    parser.add_argument("--from-local", nargs=2, metavar=("ID", "PATH"))
    parser.add_argument("--rebuild-manifest", action="store_true",
                        help="rebuild manifest.jsonl from existing wavs + captions, no download")
    args = parser.parse_args()

    src = load_sources()
    if args.rebuild_manifest:
        rows = []
        for clip in src["clips"]:
            wav = DATA / f"{clip['id']}.wav"
            if not wav.exists():
                continue
            duration = float(
                subprocess.check_output(
                    [
                        "ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", str(wav),
                    ],
                    text=True,
                ).strip()
            )
            cap = caption_text(DATA / "raw", clip["id"])
            cand = clip.get("candidate") or {}
            rows.append({
                "id": clip["id"],
                "lang": clip["lang"],
                "audio": f"data/{clip['id']}.wav",
                "duration_s": round(duration, 2),
                "source_url": cand.get("url", ""),
                "confirmed": bool(clip.get("confirmed")),
                "reference_source": "caption_draft" if cap else "missing",
                "reference": cap,
            })
        write_manifest(rows)
        return
    if args.list:
        for c in src["clips"]:
            cand = c.get("candidate") or {}
            print(f"{c['id']:5} conf={c['confidence']:6} confirmed={c['confirmed']}  {cand.get('url', 'NO URL')}")
        return

    if args.from_local:
        clip_id, path = args.from_local
        lang = "tr" if clip_id.startswith("TR") else "en"
        rec = from_local(clip_id, Path(path), lang)
        write_manifest([rec])
        return

    if not have("ffmpeg"):
        raise SystemExit("ffmpeg not on PATH")
    if not have("yt-dlp"):
        print("yt-dlp missing. Install with:  pip install yt-dlp", file=sys.stderr)
        raise SystemExit(1)

    rows = []
    for clip in src["clips"]:
        if args.only and clip["id"] not in args.only:
            continue
        if not clip.get("confirmed") and not args.allow_candidates:
            print(f"skip {clip['id']}: not confirmed (pass --allow-candidates to use the public match)")
            continue
        if not clip.get("candidate"):
            print(f"skip {clip['id']}: no candidate URL")
            continue
        rows.append(download_one(clip))
    if rows:
        write_manifest(rows)


if __name__ == "__main__":
    main()
