"""Cohen's kappa and raw agreement for the two-annotator R/D protocol.

CSV columns (see annotation/labels.example.csv):
    item_id, video_id, model, lang, term, explanation, R, D

R and D are 0/1 (or R0/R1, D0/D1). Rows are aligned by item_id.

Usage:
    python iaa.py --a1 annotation/labels_a1.csv --a2 annotation/labels_a2.csv
    python iaa.py --a1 a1.csv --a2 a2.csv --json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _norm_label(value: str, prefix: str) -> int:
    s = str(value).strip().upper()
    if s in {"1", f"{prefix}1", "YES", "Y", "TRUE"}:
        return 1
    if s in {"0", f"{prefix}0", "NO", "N", "FALSE"}:
        return 0
    raise ValueError(f"cannot parse {prefix} label {value!r}")


def load_labels(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"item_id", "R", "D"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise SystemExit(f"{path} must have columns item_id, R, D")
        for rec in reader:
            item_id = rec["item_id"].strip()
            if not item_id or item_id.startswith("#"):
                continue
            rows[item_id] = {
                "R": _norm_label(rec["R"], "R"),
                "D": _norm_label(rec["D"], "D"),
                "video_id": rec.get("video_id", ""),
                "model": rec.get("model", ""),
                "lang": rec.get("lang", ""),
                "term": rec.get("term", ""),
            }
    return rows


def cohen_kappa(y1: list[int], y2: list[int]) -> dict:
    if len(y1) != len(y2):
        raise ValueError("label lists must have the same length")
    n = len(y1)
    if n == 0:
        return {"n": 0, "agreement": None, "kappa": None, "po": None, "pe": None}
    agree = sum(a == b for a, b in zip(y1, y2))
    po = agree / n
    c1, c2 = Counter(y1), Counter(y2)
    labels = set(c1) | set(c2)
    pe = sum((c1[k] / n) * (c2[k] / n) for k in labels)
    kappa = None if pe == 1.0 else (po - pe) / (1.0 - pe)
    return {
        "n": n,
        "agreement": agree / n,
        "po": po,
        "pe": pe,
        "kappa": kappa,
        "annotator1": dict(sorted(c1.items())),
        "annotator2": dict(sorted(c2.items())),
    }


def score_pair(a1: dict[str, dict], a2: dict[str, dict]) -> dict:
    common = sorted(set(a1) & set(a2))
    missing_a2 = sorted(set(a1) - set(a2))
    missing_a1 = sorted(set(a2) - set(a1))
    out = {
        "n_a1": len(a1),
        "n_a2": len(a2),
        "n_common": len(common),
        "missing_in_a2": missing_a2,
        "missing_in_a1": missing_a1,
        "R": cohen_kappa([a1[i]["R"] for i in common], [a2[i]["R"] for i in common]),
        "D": cohen_kappa([a1[i]["D"] for i in common], [a2[i]["D"] for i in common]),
    }
    return out


def _fmt_k(block: dict) -> str:
    if block["n"] == 0:
        return "n=0"
    k = block["kappa"]
    k_s = "undefined" if k is None else f"{k:.3f}"
    return f"n={block['n']}  agreement={block['agreement']:.3f}  kappa={k_s}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1", type=Path, required=True)
    parser.add_argument("--a2", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = score_pair(load_labels(args.a1), load_labels(args.a2))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    print(f"items: a1={report['n_a1']}  a2={report['n_a2']}  common={report['n_common']}")
    if report["missing_in_a2"]:
        print("missing in a2:", ", ".join(report["missing_in_a2"][:12]))
    if report["missing_in_a1"]:
        print("missing in a1:", ", ".join(report["missing_in_a1"][:12]))
    print("R (context consistency):", _fmt_k(report["R"]))
    print("D (explanation accuracy):", _fmt_k(report["D"]))


if __name__ == "__main__":
    main()
