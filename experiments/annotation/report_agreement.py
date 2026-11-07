"""Agreement report for the 15-clip sheet.

Three arms, reported separately and never conflated:
  human–human      labels_a1 vs labels_a2          (inter-annotator agreement)
  human–LLM (A1)   labels_a1 vs labels_llmjudge    (judge validation, NOT IAA)
  human–LLM (A2)   labels_a2 vs labels_llmjudge

Usage
-----
python experiments/annotation/report_agreement.py
Writes experiments/outputs/iaa_report.md and iaa_report.json.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from iaa import cohen_kappa  # noqa: E402

ANN = ROOT / "experiments" / "annotation"
OUT_MD = ROOT / "experiments" / "outputs" / "iaa_report.md"
OUT_JSON = ROOT / "experiments" / "outputs" / "iaa_report.json"


def load(path: Path, rcol: str = "R", dcol: str = "D") -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            r, d = str(row.get(rcol, "")).strip(), str(row.get(dcol, "")).strip()
            if r in ("0", "1") and d in ("0", "1"):
                out[row["item_id"]] = {
                    "video_id": row.get("video_id", ""),
                    "term": row.get("term", ""),
                    "R": int(r),
                    "D": int(d),
                }
    return out


def compare(a: dict, b: dict, ids: list[str]) -> dict:
    if not ids:
        return {"n": 0, "R": cohen_kappa([], []), "D": cohen_kappa([], [])}
    return {
        "n": len(ids),
        "R": cohen_kappa([a[i]["R"] for i in ids], [b[i]["R"] for i in ids]),
        "D": cohen_kappa([a[i]["D"] for i in ids], [b[i]["D"] for i in ids]),
    }


def fmt(block: dict) -> str:
    if not block["n"]:
        return "—"
    r, d = block["R"], block["D"]
    kr = "n/a" if r["kappa"] is None else f"{r['kappa']:.3f}"
    kd = "n/a" if d["kappa"] is None else f"{d['kappa']:.3f}"
    return f"n={block['n']} · R {r['agreement']:.3f} (κ={kr}) · D {d['agreement']:.3f} (κ={kd})"


def main() -> None:
    a1 = load(ANN / "labels_a1.csv")
    a2 = load(ANN / "labels_a2.csv")
    jd = load(ANN / "labels_llmjudge.csv", "judge_R", "judge_D")

    ids_a1 = set(a1)
    ids_a2 = set(a2)
    ids_jd = set(jd)

    hh = sorted(ids_a1 & ids_a2)
    hl1 = sorted(ids_a1 & ids_jd)
    hl2 = sorted(ids_a2 & ids_jd)

    block_hh = compare(a1, a2, hh)
    block_h1 = compare(a1, jd, hl1)
    block_h2 = compare(a2, jd, hl2)

    per_video = defaultdict(lambda: {"items": 0, "a1": 0, "a2": 0, "judge": 0})
    for iid in set(a1) | set(a2) | set(jd):
        vid = (a1.get(iid) or a2.get(iid) or jd.get(iid))["video_id"]
        per_video[vid]["items"] += 1
        per_video[vid]["a1"] += iid in ids_a1
        per_video[vid]["a2"] += iid in ids_a2
        per_video[vid]["judge"] += iid in ids_jd

    lines = [
        "# IAA report — 15 doğrulanmış klip",
        "",
        "Ölçütler `experiments/iaa.py` ile Cohen's κ olarak hesaplandı. "
        "İnsan–LLM satırları **inter-annotator agreement değildir**; hakem modelin (Qwen3-8B) insan etiketleriyle uyumudur.",
        "",
        "| Karşılaştırma | Sonuç |",
        "| :--- | :--- |",
        f"| A1 ↔ A2 (insan–insan) | {fmt(block_hh)} |",
        f"| A1 ↔ LLM judge (insan–LLM) | {fmt(block_h1)} |",
        f"| A2 ↔ LLM judge (insan–LLM) | {fmt(block_h2)} |",
        "",
        "## Klip bazında etiket kapsamı",
        "",
        "| Klip | Madde | A1 | A2 | Judge |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for vid in sorted(per_video):
        v = per_video[vid]
        lines.append(f"| {vid} | {v['items']} | {v['a1']} | {v['a2']} | {v['judge']} |")

    md = "\n".join(lines) + "\n"
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md, encoding="utf-8")

    payload = {
        "human_human": block_hh,
        "human_llm_a1": block_h1,
        "human_llm_a2": block_h2,
        "per_video": {k: dict(v) for k, v in sorted(per_video.items())},
        "counts": {"a1": len(ids_a1), "a2": len(ids_a2), "judge": len(ids_jd)},
        "adjudication": "not performed (independent first-pass labels only)",
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
