"""Stratified human-label sample over the 100+ machine-only candidate sheet.

The expansion scope is a candidate/terminology scale, not a gold scale: every
item keeps gold_how=machine_only and the sample is what a human actually labels
(~180 items, not all ~900). So the sampling design has to be written down — the
labelled subset is the evidence base, and its strata are what any future
precision claim is conditioned on.

Design
------
stratum      = (lang, domain) from the manifest, so all 14 fields are visible
quota        = proportional to stratum size, largest-remainder rounding to n
within stratum= round-robin one item per clip (clip ids sorted, items by item_id)
               until the quota fills, so no single lecture carries a stratum

Deterministic: no RNG, no thresholds, no judge scores. Re-running on the same
candidate CSV gives the same sample.

Outputs
-------
experiments/annotation/labels_todo_sample_100plus.csv
experiments/outputs/sample_100plus_design.json

Usage
-----
python experiments/data/build_sample_100plus.py                # from repo root
python experiments/data/build_sample_100plus.py --n 150
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
ROOT = DATA_DIR.parent
CANDIDATES = ROOT / "annotation" / "labels_todo_100plus_machine.csv"
MANIFESTS = [
    DATA_DIR / "manifest_100plus.jsonl",
    DATA_DIR / "manifest_32_machine.jsonl",
]
OUT_CSV = ROOT / "annotation" / "labels_todo_sample_100plus.csv"
DESIGN_JSON = ROOT / "outputs" / "sample_100plus_design.json"

FIELDS = [
    "item_id", "video_id", "model", "lang", "term", "explanation",
    "context", "R", "D", "comment", "gold_how",
]


def domains() -> dict[str, str]:
    out: dict[str, str] = {}
    for m in MANIFESTS:
        if not m.exists():
            continue
        for line in m.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["id"]] = rec.get("domain") or "unknown"
    return out


def quota_by_stratum(sizes: dict[tuple[str, str], int], n: int) -> dict[tuple[str, str], int]:
    """Proportional allocation with largest-remainder rounding, floor of 1 per stratum."""
    total = sum(sizes.values())
    raw = {k: sizes[k] * n / total for k in sizes}
    give = {k: min(max(1, int(v)), sizes[k]) for k, v in raw.items()}
    # largest-remainder on what is still unassigned, never above the stratum size
    left = n - sum(give.values())
    order = sorted(
        (k for k in sizes if give[k] < sizes[k]),
        key=lambda k: (-(raw[k] - int(raw[k])), -sizes[k], k),
    )
    i = 0
    while left > 0 and order:
        k = order[i % len(order)]
        give[k] += 1
        left -= 1
        i += 1
        order = [x for x in order if give[x] < sizes[x]]
    while left < 0:
        k = max((x for x in sizes if give[x] > 1), key=lambda x: (give[x], sizes[x]))
        give[k] -= 1
        left += 1
    return give


def round_robin(items: list[dict], want: int) -> list[dict]:
    """Take one item per clip in turn so the stratum is not carried by one lecture."""
    by_clip: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_clip[it["video_id"]].append(it)
    for clip in by_clip:
        by_clip[clip].sort(key=lambda r: r["item_id"])
    clips = sorted(by_clip)
    picked: list[dict] = []
    cursor = {c: 0 for c in clips}
    while len(picked) < want:
        progressed = False
        for c in clips:
            if len(picked) >= want:
                break
            if cursor[c] < len(by_clip[c]):
                picked.append(by_clip[c][cursor[c]])
                cursor[c] += 1
                progressed = True
        if not progressed:
            break
    return picked


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT.parent))
    except ValueError:
        return str(p)


def main() -> int:
    ap = argparse.ArgumentParser(description="Stratified sample of the 100+ candidate sheet.")
    ap.add_argument("--candidates", type=Path, default=CANDIDATES)
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--design", type=Path, default=DESIGN_JSON)
    ap.add_argument("--n", type=int, default=180)
    args = ap.parse_args()

    if not args.candidates.exists():
        raise SystemExit(f"{args.candidates} yok; önce aday üretimini koşun.")

    dom = domains()
    with args.candidates.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("term") or "").strip()]
    for r in rows:
        r["domain"] = dom.get(r["video_id"], "unknown")

    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        strata[(r["lang"], r["domain"])].append(r)
    for k in strata:
        strata[k].sort(key=lambda r: (r["video_id"], r["item_id"]))

    sizes = {k: len(v) for k, v in strata.items()}
    n = min(args.n, len(rows))
    quota = quota_by_stratum(sizes, n)

    sample: list[dict] = []
    for k in sorted(strata):
        sample.extend(round_robin(strata[k], quota[k]))
    sample.sort(key=lambda r: (r["video_id"], r["item_id"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sample)

    design = {
        "frame_csv": rel(args.candidates),
        "frame_items": len(rows),
        "frame_clips": len({r["video_id"] for r in rows}),
        "target_n": args.n,
        "sampled_n": len(sample),
        "sampling_fraction": round(len(sample) / len(rows), 4) if rows else None,
        "strata": "lang x domain (14 alan x 2 dil)",
        "allocation": "proportional, largest-remainder, floor 1 per stratum",
        "within_stratum": "round-robin across clips (deterministic, no RNG)",
        "deterministic": True,
        "gold_how": "machine_only",
        "per_stratum": [
            {
                "lang": k[0],
                "domain": k[1],
                "frame_items": sizes[k],
                "quota": quota[k],
                "sampled": sum(1 for r in sample if (r["lang"], r["domain"]) == k),
                "clips_in_stratum": len({r["video_id"] for r in strata[k]}),
                "clips_sampled": len({r["video_id"] for r in sample if (r["lang"], r["domain"]) == k}),
            }
            for k in sorted(strata)
        ],
        "sampled_clips": len({r["video_id"] for r in sample}),
        "by_lang": {
            lang: sum(1 for r in sample if r["lang"] == lang) for lang in ("tr", "en")
        },
        "reading": (
            "the labelled subset, not the frame, is the human evidence base; report R1/D1 "
            "with this design and state that items are machine-only candidates"
        ),
    }
    args.design.parent.mkdir(parents=True, exist_ok=True)
    args.design.write_text(json.dumps(design, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"wrote {args.out} ({len(sample)} of {len(rows)} items, {design['sampled_clips']} clips)")
    print(f"design -> {args.design}")
    print("by_lang:", design["by_lang"])
    short = [p for p in design["per_stratum"] if p["sampled"] < p["quota"]]
    if short:
        print("under-quota strata:", short)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
