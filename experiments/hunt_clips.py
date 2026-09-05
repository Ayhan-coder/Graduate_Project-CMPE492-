"""One-off YouTube hunt for the two clips that still have no unique URL.

Prints duration-matched hits only. Does not download and does not mark
anything confirmed. Safe to delete after the hunt.
"""

from __future__ import annotations

import yt_dlp

QUERIES = {
    "TR-2": [
        "ytsearch15:Kapsamlı Turizm Eğitimi Turist Rehberleri",
        "ytsearch15:Comprehensive Tourism Training for Tour Guides",
        "ytsearch15:turist rehberleri için kapsamlı turizm eğitimi 12 dakika",
        "ytsearch15:AÖF turizm rehberliği ünite konu anlatımı",
        "ytsearch15:turist rehberliği eğitim ders 12 dakika",
    ],
    "TR-3": [
        "ytsearch15:VERİ BİLİMİ NEDİR VERİ BİLİMİNE GİRİŞ PYTHON",
        "ytsearch15:What Is Data Science Intro to Data Science Python Turkish",
        "ytsearch15:veri bilimi nedir python 14 dakika",
        "ytsearch15:AÖF veri bilimi ünite konu anlatımı",
        "ytsearch10:Tirendaz veri bilimi nedir",
    ],
}

# Paper: TR-2 = 12 min, TR-3 ≈ 14 min
WINDOWS = {
    "TR-2": (680, 780, 600, 900),   # hit 11:20-13:00, near 10-15
    "TR-3": (780, 900, 720, 960),   # hit 13:00-15:00, near 12-16
}


def search(query: str) -> list[dict]:
    opts = {"quiet": True, "skip_download": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(query, download=False)
    return [e for e in (info or {}).get("entries") or [] if e]


def main() -> None:
    for clip_id, queries in QUERIES.items():
        hit_lo, hit_hi, near_lo, near_hi = WINDOWS[clip_id]
        print("=" * 72)
        print(clip_id)
        seen: set[str] = set()
        for q in queries:
            print(f"\nQ: {q}")
            try:
                entries = search(q)
            except Exception as exc:
                print(f"  error: {exc}")
                continue
            for e in entries:
                vid = e.get("id") or ""
                if vid in seen:
                    continue
                seen.add(vid)
                d = int(e.get("duration") or 0)
                title = e.get("title") or ""
                ch = e.get("uploader") or e.get("channel") or ""
                stamp = f"{d // 60}:{d % 60:02d}"
                if hit_lo <= d <= hit_hi:
                    tag = "HIT "
                elif near_lo <= d <= near_hi:
                    tag = "near"
                else:
                    continue
                print(f"  {tag} {vid}  {stamp:>5}  {title}  |  {ch}")


if __name__ == "__main__":
    main()
