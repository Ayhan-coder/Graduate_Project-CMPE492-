"""Metadata-only discovery for the 100-clip candidate corpus expansion.

The existing corpus yields 26 usable clips out of 32 recorded (measured, 30.10.2026), so
oversourcing is required to land on 100 clips with real audio. This script searches
YouTube for per-domain lecture-style videos, filters them by measured duration bounds and
already-used video ids, and writes a candidate pool for review. It downloads nothing.

Usage:
    python experiments/data/discover_100plus.py --per-cell 5
    python experiments/data/discover_100plus.py --domains "Biology & Genetics" --per-cell 8

Output: experiments/data/candidates_100plus.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments" / "data"
INVENTORY = DATA / "data_inventory.json"
OUT = DATA / "candidates_100plus.json"

MIN_S, MAX_S = 300, 2700  # 5-45 min: the 21283 s outlier is a full lecture recording

# domain -> (turkish query, english query)
QUERIES: dict[str, tuple[str, str]] = {
    "Software Engineering & Project Management": (
        "proje yönetimi ders izle kontrol kazanılmış değer analizi",
        "software engineering lecture earned value analysis monitoring control",
    ),
    "Computer Science & Data Science": (
        "veri bilimi ders normallaştırma özellik çıkarımı",
        "computer science lecture data normalization feature extraction",
    ),
    "Artificial Intelligence & Machine Learning": (
        "makine öğrenmesi ders gradient descent regularizasyon",
        "machine learning lecture gradient descent regularization overfitting",
    ),
    "Mechanical & Thermal Engineering": (
        "termodinamik ders entropi ısı transferi",
        "thermodynamics lecture entropy heat transfer conduction",
    ),
    "Electrical Engineering & Circuit Theory": (
        "devre analizi ders thevenin norton",
        "circuit theory lecture thevenin Norton equivalent",
    ),
    "Mathematics (Linear Algebra, Calculus)": (
        "lineer cebir ders özdeğer özvektor",
        "linear algebra lecture eigenvalues eigenvectors",
    ),
    "Physics (Mechanics, Electromagnetism)": (
        "fizik ders elektromanyetik indüksiyon",
        "physics lecture electromagnetic induction faraday",
    ),
    "Economics & Finance": (
        "ekonomi ders arzi talep esneklik",
        "economics lecture supply demand elasticity",
    ),
    "Cognitive Psychology & Neuroscience": (
        "bilişsel psikoloji ders çalışma belleği",
        "cognitive neuroscience lecture working memory attention",
    ),
    "History & Cultural Heritage": (
        "tarih ders kültürel miras arkeoloji",
        "history lecture cultural heritage preservation archaeology",
    ),
    "Biology & Genetics": (
        "biyoloji ders mitoz kalıtım",
        "biology lecture mitosis meiosis genetics",
    ),
    "Tourism & Sustainable Hospitality": (
        "turizm ders sürdürülebilirlik",
        "sustainable tourism hospitality lecture",
    ),
    "Occupational Health and Safety (ISG / OHS)": (
        "iş sağlığı güvenliği ders risk değerlendirme",
        "occupational health safety lecture risk assessment hazard",
    ),
    "Corporate Compliance & Personal Data Protection (KVKK/GDPR)": (
        "kvkk kişisel veri güvenliği eğitim",
        "gdpr data protection compliance training lecture",
    ),
}

YID = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")


def used_ids() -> set[str]:
    ids: set[str] = set()
    if INVENTORY.exists():
        for r in json.loads(INVENTORY.read_text(encoding="utf-8"))["records"]:
            src = r.get("source") or {}
            for val in (src.get("youtube_id"), src.get("url"), r.get("reference")):
                if not val:
                    continue
                m = YID.search(str(val))
                ids.add(m.group(1) if m else str(val))
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=5, help="kept candidates per domain x lang")
    ap.add_argument("--search", type=int, default=15, help="results pulled per query")
    ap.add_argument("--domains", type=str, default=None, help="comma separated subset")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    import yt_dlp

    taken = used_ids()
    already = json.loads(args.out.read_text(encoding="utf-8"))["candidates"] if args.out.exists() else []
    taken |= {c["youtube_id"] for c in already}
    pool: list[dict] = already
    stats: dict[str, dict] = defaultdict(lambda: {"seen": 0, "kept": 0, "rejected_short": 0,
                                                 "rejected_long": 0, "duplicate": 0})

    domains = list(QUERIES)
    if args.domains:
        want = [d.strip() for d in args.domains.split(",")]
        domains = [d for d in domains if d in want]

    opts = {"quiet": True, "extract_flat": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        for domain in domains:
            for lang, query in zip(("tr", "en"), QUERIES[domain]):
                cell = f"{domain}|{lang}"
                kept_here = 0
                res = ydl.extract_info(f"ytsearch{args.search}:{query}", download=False)
                for e in (res or {}).get("entries") or []:
                    if not e or e.get("_type") == "playlist":
                        continue
                    vid = e.get("id")
                    dur = e.get("duration")
                    title = (e.get("title") or "").strip()
                    if not vid or not title:
                        continue
                    stats[cell]["seen"] += 1
                    if vid in taken:
                        stats[cell]["duplicate"] += 1
                        continue
                    if dur is None:
                        continue
                    if dur < MIN_S:
                        stats[cell]["rejected_short"] += 1
                        continue
                    if dur > MAX_S:
                        stats[cell]["rejected_long"] += 1
                        continue
                    if kept_here >= args.per_cell:
                        continue
                    kept_here += 1
                    taken.add(vid)
                    stats[cell]["kept"] += 1
                    pool.append({
                        "youtube_id": vid,
                        "lang": lang,
                        "domain": domain,
                        "title": title,
                        "duration_s": float(dur),
                        "channel": e.get("uploader") or e.get("channel") or "",
                        "url": f"https://www.youtube.com/watch?v={vid}",
                    })
                print(f"{cell}: gorulen={stats[cell]['seen']} saklanan={stats[cell]['kept']}", flush=True)

    args.out.write_text(json.dumps({
        "note": "Metadata-only candidate pool for the 100-clip expansion; measured duration "
                "bounds 300-2700 s, video ids deduplicated against data_inventory.json. "
                "No audio downloaded, no gold claim: anything accepted later stays "
                "gold_how=machine_only.",
        "tool": "yt-dlp " + str(getattr(yt_dlp.version, "__version__", "?")),
        "bounds_s": [MIN_S, MAX_S],
        "total_candidates": len(pool),
        "by_lang": {k: sum(1 for c in pool if c["lang"] == k) for k in ("tr", "en")},
        "per_cell_stats": {k: dict(v) for k, v in sorted(stats.items())},
        "candidates": pool,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    hours = sum(c["duration_s"] for c in pool) / 3600
    print(f"toplam aday: {len(pool)} | tahmini ses: {hours:.1f} h | "
          f"tr {sum(1 for c in pool if c['lang']=='tr')} / en {sum(1 for c in pool if c['lang']=='en')}")
    print(f"yazildi: {args.out}")
    print("ffmpeg:", "var" if shutil.which("ffmpeg") else "YOK")


if __name__ == "__main__":
    main()
