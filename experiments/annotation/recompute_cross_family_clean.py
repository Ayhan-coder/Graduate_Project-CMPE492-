#!/usr/bin/env python3
"""Recompute Cross-Family Judge Statistics Strictly Separating Qwen's 184 Real Verdicts from 270 Parse Fallbacks.

Outputs:
- experiments/annotation/labels_panel_qwen25_7b_a40_clean184.csv
- experiments/annotation/labels_panel_qwen25_7b_a40_fallback270.csv
- experiments/outputs/cross_family_judge_report.json (updated cleanly with zero fallback contamination)
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def calculate_agreement_stats(
    pairs: List[Tuple[int, int]],
    name1: str,
    name2: str,
    metric_name: str,
) -> Dict[str, Any]:
    n = len(pairs)
    if n == 0:
        return {"n": 0}

    tp = sum(1 for y1, y2 in pairs if y1 == 1 and y2 == 1)
    tn = sum(1 for y1, y2 in pairs if y1 == 0 and y2 == 0)
    fp = sum(1 for y1, y2 in pairs if y1 == 1 and y2 == 0)
    fn = sum(1 for y1, y2 in pairs if y1 == 0 and y2 == 1)

    po = (tp + tn) / n
    p_yes1 = sum(y1 for y1, _ in pairs) / n
    p_no1 = 1.0 - p_yes1
    p_yes2 = sum(y2 for _, y2 in pairs) / n
    p_no2 = 1.0 - p_yes2
    pe = (p_yes1 * p_yes2) + (p_no1 * p_no2)

    denom = 1.0 - pe
    kappa = (po - pe) / denom if abs(denom) > 1e-6 else 0.0
    pabak = 2.0 * po - 1.0

    return {
        "comparison": f"{name1} vs {name2}",
        "metric": metric_name,
        "n": n,
        "observed_agreement_pct": round(po * 100, 2),
        "expected_agreement_pct": round(pe * 100, 2),
        "cohen_kappa": round(kappa, 4),
        "pabak": round(pabak, 4),
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "pos_rate_1": round(p_yes1 * 100, 1),
        "pos_rate_2": round(p_yes2 * 100, 1),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    ann_dir = root / "experiments" / "annotation"
    out_dir = root / "experiments" / "outputs"

    qwen_csv = ann_dir / "labels_panel_qwen25_7b_a40.csv"
    mistral_csv = ann_dir / "labels_panel_mistral7b_a40.csv"
    human_csv = ann_dir / "labels_a1.csv"

    qwen_rows = list(csv.DictReader(qwen_csv.open(encoding="utf-8-sig")))
    mistral_rows = list(csv.DictReader(mistral_csv.open(encoding="utf-8-sig")))
    human_rows = list(csv.DictReader(human_csv.open(encoding="utf-8-sig")))

    qwen_clean: Dict[str, Dict[str, int]] = {}
    qwen_clean_rows: List[Dict[str, str]] = []
    qwen_fallback_rows: List[Dict[str, str]] = []

    for r in qwen_rows:
        why = (r.get("judge_why") or "").strip().lower()
        if "parsing fallback" in why:
            qwen_fallback_rows.append(r)
        else:
            qwen_clean_rows.append(r)
            qwen_clean[r["item_id"]] = {"R": int(r["judge_R"]), "D": int(r["judge_D"])}

    mistral_clean: Dict[str, Dict[str, int]] = {}
    mistral_fallbacks = 0
    for r in mistral_rows:
        why = (r.get("judge_why") or "").strip().lower()
        if "parsing fallback" in why:
            mistral_fallbacks += 1
        else:
            mistral_clean[r["item_id"]] = {"R": int(r["judge_R"]), "D": int(r["judge_D"])}

    human_a1: Dict[str, Dict[str, int]] = {}
    for r in human_rows:
        iid = r.get("item_id", "")
        rv, dv = r.get("R", ""), r.get("D", "")
        if iid and rv != "" and dv != "":
            human_a1[iid] = {"R": int(rv), "D": int(dv)}

    # Save physical separation CSVs
    fields = list(qwen_rows[0].keys())
    for path, rows_out in [
        (ann_dir / "labels_panel_qwen25_7b_a40_clean184.csv", qwen_clean_rows),
        (ann_dir / "labels_panel_qwen25_7b_a40_fallback270.csv", qwen_fallback_rows),
    ]:
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows_out)

    # Recompute clean comparisons
    qm_r = [(qwen_clean[iid]["R"], mistral_clean[iid]["R"]) for iid in qwen_clean if iid in mistral_clean]
    qm_d = [(qwen_clean[iid]["D"], mistral_clean[iid]["D"]) for iid in qwen_clean if iid in mistral_clean]

    qh_r = [(qwen_clean[iid]["R"], human_a1[iid]["R"]) for iid in qwen_clean if iid in human_a1]
    qh_d = [(qwen_clean[iid]["D"], human_a1[iid]["D"]) for iid in qwen_clean if iid in human_a1]

    mh_r = [(mistral_clean[iid]["R"], human_a1[iid]["R"]) for iid in mistral_clean if iid in human_a1]
    mh_d = [(mistral_clean[iid]["D"], human_a1[iid]["D"]) for iid in mistral_clean if iid in human_a1]

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "Boğaziçi HPC (NVIDIA A40 48GB GDDR6 ECC)",
        "recomputation_note": "Strictly excludes 270 Qwen2.5-7B parse-fallback rows (which had defaulted to R=1, D=1) and evaluates Qwen metrics on the 184 verified genuine verdicts only.",
        "panel_audit": {
            "total_panel_items": len(qwen_rows),
            "qwen_real_parsed_verdicts": len(qwen_clean_rows),
            "qwen_parse_fallbacks_excluded": len(qwen_fallback_rows),
            "qwen_parse_failure_rate_pct": round(len(qwen_fallback_rows) / max(1, len(qwen_rows)) * 100.0, 1),
            "mistral_real_parsed_verdicts": len(mistral_clean),
            "mistral_parse_fallbacks_excluded": mistral_fallbacks,
            "human_a1_items_total": len(human_a1),
            "qwen_clean_vs_human_overlap_n": len(qh_r),
            "mistral_clean_vs_human_overlap_n": len(mh_r),
        },
        "arms": {
            "arm1": "Qwen/Qwen2.5-7B-Instruct (Alibaba Cloud, N=184 real parsed verdicts)",
            "arm2": "mistralai/Mistral-7B-Instruct-v0.3 (Mistral AI Europe, N=454 real parsed verdicts)",
        },
        "comparisons": {
            "qwen_clean184_vs_mistral_R": calculate_agreement_stats(qm_r, "Qwen2.5-7B (Real 184)", "Mistral-7B", "R_Context"),
            "qwen_clean184_vs_mistral_D": calculate_agreement_stats(qm_d, "Qwen2.5-7B (Real 184)", "Mistral-7B", "D_Definition"),
            "qwen_clean184_vs_human_R": calculate_agreement_stats(qh_r, "Qwen2.5-7B (Real 184)", "Human_A1", "R_Context"),
            "qwen_clean184_vs_human_D": calculate_agreement_stats(qh_d, "Qwen2.5-7B (Real 184)", "Human_A1", "D_Definition"),
            "mistral_454_vs_human_R": calculate_agreement_stats(mh_r, "Mistral-7B (All 454)", "Human_A1", "R_Context"),
            "mistral_454_vs_human_D": calculate_agreement_stats(mh_d, "Mistral-7B (All 454)", "Human_A1", "D_Definition"),
        },
        "triage_audit": {
            "human_negatives_R_in_qwen184_overlap": sum(1 for iid in qwen_clean if iid in human_a1 and human_a1[iid]["R"] == 0),
            "qwen184_negatives_R_count": sum(1 for v in qwen_clean.values() if v["R"] == 0),
            "qwen184_caught_human_negatives_R": [iid for iid in qwen_clean if iid in human_a1 and human_a1[iid]["R"] == 0 and qwen_clean[iid]["R"] == 0],
            "human_negatives_R_in_mistral399_overlap": sum(1 for iid in mistral_clean if iid in human_a1 and human_a1[iid]["R"] == 0),
            "mistral454_negatives_R_count": sum(1 for v in mistral_clean.values() if v["R"] == 0),
            "mistral454_caught_human_negatives_R": [iid for iid in mistral_clean if iid in human_a1 and human_a1[iid]["R"] == 0 and mistral_clean[iid]["R"] == 0],
        },
    }

    out_path = out_dir / "cross_family_judge_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["panel_audit"], indent=2))
    print(json.dumps(report["comparisons"], indent=2))
    print(json.dumps(report["triage_audit"], indent=2))


if __name__ == "__main__":
    main()

