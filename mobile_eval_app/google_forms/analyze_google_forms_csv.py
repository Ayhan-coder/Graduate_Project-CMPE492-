# -*- coding: utf-8 -*-
"""
Google Forms CSV Analyzer for EduTerm CMPE 492 Human Evaluation.

Usage:
    python analyze_google_forms_csv.py path/to/responses.csv
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

# Matches the blinding map in create_eduterm_google_form.gs
AB_UNBLIND = {
    "AB_01": {"Öğretmen A": "task_separated_hybrid", "Öğretmen B": "onpolicy_dpo_v2"},
    "AB_02": {"Öğretmen A": "onpolicy_dpo_v2", "Öğretmen B": "task_separated_hybrid"},
    "AB_03": {"Öğretmen A": "task_separated_hybrid", "Öğretmen B": "onpolicy_dpo_v2"},
}

QUIZ_GOLD = {
    "Kontrol Sorusu 1": "Evet, zarar verme potansiyeli taşıdığı için tehlikedir",
    "Kontrol Sorusu 2": "Hayır, sözleşmenin ifası kapsamında hukuki sebep vardır",
    "Kontrol Sorusu 3": "Örtüşen alt problemlerin sonuçlarını bellekte saklaması",
}


def exact_binomial_p(wins_a: int, wins_b: int) -> float:
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = min(wins_a, wins_b)
    p = sum(math.comb(n, i) * (0.5 ** n) for i in range(0, k + 1))
    return round(min(1.0, 2.0 * p), 6)


def analyze(csv_path: Path) -> Dict:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8-sig")))
    clarity_scores: List[float] = []
    turkish_scores: List[float] = []
    quiz_correct = 0
    quiz_total = 0
    ab_counts = {"onpolicy_dpo_v2": 0, "task_separated_hybrid": 0, "tie": 0}
    socratic_ratings: List[float] = []

    for row in rows:
        for col, val in row.items():
            v = (val or "").strip()
            if not v:
                continue
            if ".1]" in col:
                clarity_scores.append(float(v))
            elif ".2]" in col:
                turkish_scores.append(float(v))
            elif "Sokratik yönlendirmesi" in col:
                socratic_ratings.append(float(v))
            else:
                for q_prefix, gold in QUIZ_GOLD.items():
                    if q_prefix in col:
                        quiz_total += 1
                        if v == gold:
                            quiz_correct += 1
                for ab_id, mapping in AB_UNBLIND.items():
                    if f"[{ab_id}]" in col and "Hangi öğretmen" in col:
                        if v in mapping:
                            ab_counts[mapping[v]] += 1
                        else:
                            ab_counts["tie"] += 1

    dpo_w = ab_counts["onpolicy_dpo_v2"]
    hyb_w = ab_counts["task_separated_hybrid"]
    return {
        "total_participants": len(rows),
        "cefr_mean_clarity_1_to_5": round(sum(clarity_scores) / max(len(clarity_scores), 1), 3),
        "cefr_mean_turkish_1_to_5": round(sum(turkish_scores) / max(len(turkish_scores), 1), 3),
        "comprehension_quiz_accuracy": round(quiz_correct / max(quiz_total, 1), 4),
        "ab_preference_unblinded": ab_counts,
        "ab_exact_sign_test_p": exact_binomial_p(dpo_w, hyb_w),
        "mean_preferred_socratic_1_to_5": round(sum(socratic_ratings) / max(len(socratic_ratings), 1), 3),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_google_forms_csv.py <google_forms_responses.csv>")
        sys.exit(1)
    res = analyze(Path(sys.argv[1]))
    print(json.dumps(res, ensure_ascii=False, indent=2))
