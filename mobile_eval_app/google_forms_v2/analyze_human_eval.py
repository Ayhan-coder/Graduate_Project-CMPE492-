# -*- coding: utf-8 -*-
"""
EduTerm CMPE 492 — Tidy-Data Converter & Statistical Analyzer (V2)

Converts Google Forms CSV exports (Form V1 and Form V2) into a tidy CSV table
matching `human_eval_codebook.csv`, assigns post-hoc random anonymous participant IDs,
unblinds A/B labels using `human_eval_manifest.json`, and runs statistical tests
per model_pair_id (never pooling distinct pairs into one composite score).

Note: Requires real participant CSV input. Does not fabricate thesis results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PRIOR_ORDER = {
    "Hiç bilgim yok": 1,
    "Başlangıç": 2,
    "Orta": 3,
    "İleri": 4,
    "Uzman": 5,
}

DOMAIN_PRIOR_COL = {
    "ISG": "İş Sağlığı ve Güvenliği (İSG)",
    "KVKK": "KVKK ve Veri Gizliliği",
    "CS": "Bilgisayar Bilimleri",
    "Finance": "Finans",
}


def parse_likert(val: str) -> Optional[float]:
    v = (val or "").strip()
    if not v or v == "Değerlendiremiyorum":
        return None
    m = re.match(r"^([1-9])\b", v)
    return float(m.group(1)) if m else None


def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def summarize_numeric(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0, "median": None, "iqr": None, "mean": None, "sd": None}
    sv = sorted(vals)
    q1 = percentile(sv, 0.25)
    q3 = percentile(sv, 0.75)
    return {
        "n": len(vals),
        "median": round(statistics.median(sv), 3),
        "iqr": round(q3 - q1, 3),
        "mean": round(statistics.mean(sv), 3),
        "sd": round(statistics.stdev(sv), 3) if len(sv) > 1 else 0.0,
    }


def bootstrap_ci_mean(vals: List[float], n_boot: int = 2000, seed: int = 42) -> List[Optional[float]]:
    if not vals:
        return [None, None]
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(rng.choice(vals) for _ in range(n)) / n for _ in range(n_boot))
    return [round(means[int(0.025 * n_boot)], 3), round(means[int(0.975 * n_boot)], 3)]


def wilcoxon_paired_test(x: List[float], y: List[float]) -> Dict[str, Any]:
    """Paired Wilcoxon signed-rank test (normal approximation with tie handling) + rank-biserial r."""
    diffs = [a - b for a, b in zip(x, y) if a != b]
    n = len(diffs)
    if n == 0:
        return {"n_nonzero": 0, "W_pos": 0.0, "p_value": 1.0, "rank_biserial_r": 0.0}
    abs_diffs = sorted((abs(d), idx, d > 0) for idx, d in enumerate(diffs))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_diffs[j + 1][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    w_pos = sum(r for (_, _, is_pos), r in zip(abs_diffs, ranks) if is_pos)
    w_neg = sum(r for (_, _, is_pos), r in zip(abs_diffs, ranks) if not is_pos)
    total_rank = n * (n + 1) / 2.0
    r_rb = (w_pos - w_neg) / total_rank if total_rank > 0 else 0.0
    mean_w = total_rank / 2.0
    sd_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sd_w == 0:
        p_val = 1.0
    else:
        z = (min(w_pos, w_neg) - mean_w) / sd_w
        p_val = min(1.0, 2.0 * 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    return {
        "n_nonzero": n,
        "W_pos": round(w_pos, 2),
        "W_neg": round(w_neg, 2),
        "p_value": round(p_val, 6),
        "rank_biserial_r": round(r_rb, 4),
    }


def holm_bonferroni(p_values: List[float]) -> List[float]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda t: t[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = round(running_max, 6)
    return adjusted


def spearman_corr(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 3 or len(x) != len(y):
        return None
    def rank_vec(v: List[float]) -> List[float]:
        indexed = sorted(enumerate(v), key=lambda t: t[1])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and indexed[j + 1][1] == indexed[i][1]:
                j += 1
            avg_r = (i + 1 + j + 1) / 2.0
            for k in range(i, j + 1):
                r[indexed[k][0]] = avg_r
            i = j + 1
        return r
    rx, ry = rank_vec(x), rank_vec(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(num / den, 4) if den > 0 else 0.0


def build_scenario_lookup(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for mod in manifest.get("modules", {}).values():
        for sc in mod.get("scenarios", []):
            lookup[sc["scenario_id"]] = sc
    return lookup


def convert_to_tidy(
    csv_files: List[Tuple[str, Path]], manifest: Dict[str, Any], seed: int = 42
) -> List[Dict[str, Any]]:
    sc_lookup = build_scenario_lookup(manifest)
    rng = random.Random(seed)
    all_rows: List[Tuple[str, Dict[str, str]]] = []
    for form_ver, path in csv_files:
        with path.open("r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                consent_col = next((c for c in row if "kabul ediyorum" in c.lower()), None)
                if consent_col and row[consent_col].strip() != "Kabul ediyorum.":
                    continue
                all_rows.append((form_ver, row))

    pids = [f"P-{i:03d}" for i in range(1, len(all_rows) + 1)]
    rng.shuffle(pids)

    tidy: List[Dict[str, Any]] = []
    for pid, (form_ver, row) in zip(pids, all_rows):
        ts = row.get("Timestamp") or row.get("Zaman damgası") or ""
        flip = (form_ver == "V2")
        for col, val in row.items():
            m = re.match(r"^\[([A-Z]+_\d+)\s*\|\s*(Öğretmen|Yanıt)\s+(A|B)\]\s*(.+)$", col.strip())
            if m:
                sc_id, _, disp_pos, dim_label = m.groups()
                sc = sc_lookup.get(sc_id)
                if not sc:
                    continue
                true_pos = ("B" if disp_pos == "A" else "A") if flip else disp_pos
                model_ckpt = sc[f"v1_{true_pos}_model"]
                dom = sc["domain"].split(" ")[0]
                prior_col = next((c for c in row if DOMAIN_PRIOR_COL.get(dom, "NONE") in c), None)
                prior_val = row.get(prior_col, "NA") if prior_col else "NA"
                tidy.append({
                    "participant_id": pid,
                    "form_version": form_ver,
                    "scenario_id": sc_id,
                    "domain": dom,
                    "model_pair_id": sc["model_pair_id"],
                    "display_position": disp_pos,
                    "model_checkpoint": model_ckpt,
                    "response_source": sc["response_source"],
                    "rating_dimension": dim_label.strip(),
                    "rating_value": parse_likert(val),
                    "mental_effort": None,
                    "preference": "",
                    "comprehension_correct": None,
                    "prior_knowledge": prior_val,
                    "completion_timestamp": ts,
                })
                continue

            m_pref = re.match(r"^\[([A-Z]+_\d+)\]\s*(Hangi|Genel)", col.strip())
            if m_pref:
                sc_id = m_pref.group(1)
                sc = sc_lookup.get(sc_id)
                if not sc:
                    continue
                v = (val or "").strip()
                if v in ("Öğretmen A", "Yanıt A"):
                    true_pos = "B" if flip else "A"
                    pref = sc[f"v1_{true_pos}_model"]
                elif v in ("Öğretmen B", "Yanıt B"):
                    true_pos = "A" if flip else "B"
                    pref = sc[f"v1_{true_pos}_model"]
                elif v == "İkisi de eşit":
                    pref = "tie"
                elif v == "Hiçbiri":
                    pref = "neither"
                else:
                    pref = "undecided"
                tidy.append({
                    "participant_id": pid,
                    "form_version": form_ver,
                    "scenario_id": sc_id,
                    "domain": sc["domain"].split(" ")[0],
                    "model_pair_id": sc["model_pair_id"],
                    "display_position": "PAIR",
                    "model_checkpoint": "PAIR",
                    "response_source": sc["response_source"],
                    "rating_dimension": "preference",
                    "rating_value": None,
                    "mental_effort": None,
                    "preference": pref,
                    "comprehension_correct": None,
                    "prior_knowledge": "NA",
                    "completion_timestamp": ts,
                })
    return tidy


def analyze_tidy(tidy: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Participant-clustered paired analysis per model_pair_id."""
    pairs = sorted(set(r["model_pair_id"] for r in tidy))
    results_by_pair: Dict[str, Any] = {}
    raw_pvals: List[Tuple[str, float]] = []

    for pair_id in pairs:
        p_rows = [r for r in tidy if r["model_pair_id"] == pair_id]
        models = sorted(set(r["model_checkpoint"] for r in p_rows if r["model_checkpoint"] != "PAIR"))
        rating_summary: Dict[str, Any] = {}
        by_part_model: Dict[Tuple[str, str], List[float]] = {}

        for r in p_rows:
            if r["rating_dimension"] != "preference" and r["rating_value"] is not None:
                by_part_model.setdefault((r["participant_id"], r["model_checkpoint"]), []).append(float(r["rating_value"]))

        for m in models:
            vals = [r["rating_value"] for r in p_rows if r["model_checkpoint"] == m and r["rating_value"] is not None]
            unratable = sum(1 for r in p_rows if r["model_checkpoint"] == m and r["rating_dimension"] != "preference" and r["rating_value"] is None)
            part_means = [statistics.mean(v) for (pid, mod), v in by_part_model.items() if mod == m]
            rating_summary[m] = {
                "item_level": summarize_numeric(vals),
                "participant_clustered_mean_95ci": bootstrap_ci_mean(part_means),
                "unratable_count": unratable,
            }

        wilcoxon_res = None
        if len(models) == 2:
            m1, m2 = models
            pids = sorted(set(pid for (pid, _) in by_part_model))
            x_vec, y_vec = [], []
            for pid in pids:
                if (pid, m1) in by_part_model and (pid, m2) in by_part_model:
                    x_vec.append(statistics.mean(by_part_model[(pid, m1)]))
                    y_vec.append(statistics.mean(by_part_model[(pid, m2)]))
            wilcoxon_res = wilcoxon_paired_test(x_vec, y_vec)
            raw_pvals.append((pair_id, wilcoxon_res["p_value"]))

        pref_rows = [r for r in p_rows if r["rating_dimension"] == "preference"]
        pref_counts: Dict[str, int] = {}
        for r in pref_rows:
            pref_counts[r["preference"]] = pref_counts.get(r["preference"], 0) + 1
        n_pref = len(pref_rows)
        pref_rates = {k: round(v / max(n_pref, 1), 4) for k, v in pref_counts.items()}

        results_by_pair[pair_id] = {
            "models": models,
            "ratings": rating_summary,
            "participant_clustered_wilcoxon": wilcoxon_res,
            "preferences": {"n": n_pref, "counts": pref_counts, "rates": pref_rates},
        }

    if raw_pvals:
        adj = holm_bonferroni([p for _, p in raw_pvals])
        for (pid, _), ap in zip(raw_pvals, adj):
            results_by_pair[pid]["participant_clustered_wilcoxon"]["holm_adjusted_p"] = ap

    return {
        "total_participants": len(set(r["participant_id"] for r in tidy)),
        "total_tidy_rows": len(tidy),
        "results_by_model_pair": results_by_pair,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--v1-csv", default=None)
    parser.add_argument("--v2-csv", default=None)
    parser.add_argument("--out-tidy", default="tidy_human_eval.csv")
    parser.add_argument("--out-report", default="human_eval_analysis_report.json")
    args = parser.parse_args()

    inputs: List[Tuple[str, Path]] = []
    if args.v1_csv:
        inputs.append(("V1", Path(args.v1_csv)))
    if args.v2_csv:
        inputs.append(("V2", Path(args.v2_csv)))
    if not inputs:
        raise SystemExit("Provide at least --v1-csv or --v2-csv.")

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    tidy = convert_to_tidy(inputs, manifest)
    if tidy:
        with Path(args.out_tidy).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(tidy[0].keys()))
            writer.writeheader()
            writer.writerows(tidy)
    report = analyze_tidy(tidy)
    Path(args.out_report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
