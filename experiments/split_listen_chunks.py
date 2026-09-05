"""Split lecture wavs into short chunks for gold listening. Local work files only."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
WORK = DATA / "_work" / "chunks"

CLIPS = ("TR-1", "TR-2", "TR-3", "EN-1", "EN-2", "EN-3", "EN-4")


def split_one(clip_id: str, chunk_s: int) -> list[Path]:
    src = DATA / f"{clip_id}.wav"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    dest = WORK / clip_id
    dest.mkdir(parents=True, exist_ok=True)
    pattern = dest / "%03d.wav"
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-f", "segment",
            "-segment_time", str(chunk_s),
            "-c", "copy",
            str(pattern),
        ]
    )
    return sorted(dest.glob("*.wav"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*", default=list(CLIPS))
    parser.add_argument("--chunk-s", type=int, default=90)
    args = parser.parse_args()
    for cid in args.ids:
        files = split_one(cid, args.chunk_s)
        print(f"{cid}: {len(files)} chunks in {WORK / cid}")


if __name__ == "__main__":
    main()
