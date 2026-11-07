"""Download and verify the 100-clip expansion worklist (exact video ids only).

Deliberate differences from download_and_transcribe_corpus.py:
  * no title/channel ytsearch fallback - a substituted video would silently change what
    the corpus contains, so a wrong or unavailable id is recorded as audio_status=failed;
  * every downloaded file is measured with ffprobe and the *declared* YouTube duration is
    compared against it (18/32 records in the existing corpus mismatched);
  * resumable: rows already carrying audio_status=present are skipped.

Usage:
    python experiments/data/fetch_clips_100plus.py --dry-run
    python experiments/data/fetch_clips_100plus.py --limit 20
    python experiments/data/fetch_clips_100plus.py            # whole worklist

Result is written back into experiments/data/worklist_100plus.json.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments" / "data"
WORKLIST = DATA / "worklist_100plus.json"
MIN_USABLE_S = 120


def probe_duration(path: Path) -> float | None:
    try:
        res = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=60)
        return float(res.stdout.strip())
    except Exception:
        return None


def fetch(row: dict, force: bool = False, dry: bool = False) -> dict:
    out = Path(ROOT) / row["audio_path"]
    if row["audio_status"] == "present" and out.exists() and not force:
        return {**row, "note": "zaten indirilmis"}
    if dry:
        return {**row, "audio_status": "dry_run"}

    ytdlp = shutil.which("yt-dlp")
    if not ytdlp:
        return {**row, "audio_status": "failed", "error": "yt-dlp yok"}

    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    cmd = [ytdlp, "-f", "bestaudio/best", "-x", "--audio-format", "wav", "--audio-quality", "0",
           "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
           "--no-playlist", "--retries", "2", "--socket-timeout", "30",
           "-o", f"{stem}.%(ext)s", f"https://www.youtube.com/watch?v={row['youtube_id']}"]
    t0 = time.time()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return {**row, "audio_status": "timeout", "error": "900 sn asimi"}
    if res.returncode != 0:
        msg = re.sub(r"\s+", " ", (res.stderr or res.stdout or ""))[-300:]
        return {**row, "audio_status": "failed", "error": msg}

    wav = out if out.exists() else None
    if wav is None:
        cand = [p for p in out.parent.glob(stem.name + ".*") if p.suffix.lower() == ".wav"]
        if cand:
            cand[0].replace(out)
            wav = out
    if wav is None:
        return {**row, "audio_status": "failed", "error": "wav uretilemedi"}

    measured = probe_duration(out)
    if measured is None:
        return {**row, "audio_status": "failed", "error": "ffprobe sure okuyamadi"}
    declared = row["duration_declared_s"] or 0
    status = "short" if measured < MIN_USABLE_S else "present"
    return {
        **row,
        "audio_status": status,
        "duration_measured_s": round(measured, 1),
        "size_bytes": out.stat().st_size,
        "matches_declared": bool(declared) and abs(measured - declared) <= max(30, 0.05 * declared),
        "fetch_seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    doc = json.loads(WORKLIST.read_text(encoding="utf-8"))
    rows = doc["rows"]
    todo = [r for r in rows if r["audio_status"] in ("pending", "failed", "timeout", "short")]
    if args.limit:
        todo = todo[: args.limit]

    done = 0
    for r in todo:
        upd = fetch(r, force=args.force, dry=args.dry_run)
        for k, v in upd.items():
            if k != "note":
                r[k] = v
        if not args.dry_run:
            WORKLIST.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{r['id']}: {r['audio_status']} | ilan {r['duration_declared_s']:.0f}s / "
                  f"olculen {(r['duration_measured_s'] or 0):.0f}s | eslesiyor="
                  f"{r.get('matches_declared')}", flush=True)
            done += 1
        if args.dry_run:
            print(f"{r['id']}: [{upd['audio_status']}] {r['url']} {r['title'][:50]}")

    if args.dry_run:
        print(f"\ndry run: {len(todo)} satir sirada; indirilmedi.")
        return

    present = [r for r in rows if r["audio_status"] == "present"]
    hours = sum(r["duration_measured_s"] or 0 for r in present) / 3600
    mismatch = sum(1 for r in present if r.get("matches_declared") is False)
    print(f"\nguncel usable klip sayisi (yeni): {len(present)} | olculen saat {hours:.2f} | "
          f"declare-uyusmaz {mismatch}")
    print(f"durum: {json.dumps({k: sum(1 for r in rows if r['audio_status'] == k) for k in set(r['audio_status'] for r in rows)})}")
    print(f"işlenen: {done}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--":
        sys.argv.pop(1)
    main()
