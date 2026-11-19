# -*- coding: utf-8 -*-
"""
Google Forms CSV Analyzer for EduTerm CMPE 492 Thesis-Grade Human Evaluation.

Matches `create_eduterm_google_form.gs` (4 CEFR grid items + 5 blinded A/B pairs).

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
    "AB_04": {"Öğretmen A": "onpolicy_dpo_v2", "Öğretmen B": "prompt_dag_rag_base_14b"},
    "AB_05": {"Öğretmen A": "base_14b_leak", "Öğretmen B": "dpo_v4_guardrail_sysguard"},
}

QUIZ_GOLD = {
    "Kontrol Sorusu 1": "Evet, zarar verme potansiyeli taşıdığı için tehlikedir",
    "Kontrol Sorusu 2": "Hayır, sözleşmenin ifası kapsamında hukuki sebep vardır",
    "Kontrol Sorusu 3": "Örtüşen alt problemlerin sonuçlarını bellekte saklaması",
    "Kontrol Sorusu 4": "Reel olarak azalır (negatif reel getiri)",
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
    cogload_scores: List[float] = []
    quiz_correct = 0
    quiz_total = 0

    ab_counts: Dict[str, int] = {}
    socratic_ratings: List[float] = []
    prereq_ratings: List[float] = []
    style_ratings: List[float] = []
    adoption_scores: List[float] = []
    guardrail_utility_scores: List[float] = []

    for row in rows:
        for col, val in row.items():
            v = (val or "").strip()
            if not v:
                continue
            if "[K" in col and "Anlaşılırlık" in col:
                clarity_scores.append(float(v))
            elif "[K" in col and "Türkçe akıcılığı" in col:
                turkish_scores.append(float(v))
            elif "[K" in col and "Öğrenme kolaylığı" in col:
                cogload_scores.append(float(v))
            elif "Sokratik yönlendirme" in col and "[AB_" in col:
                socratic_ratings.append(float(v))
            elif "Ön koşul" in col and "[AB_" in col:
                prereq_ratings.append(float(v))
            elif "Türkçe akıcılığı ve üslup" in col and "[AB_" in col:
                style_ratings.append(float(v))
            elif "derslerinizde kullanmak ister miydiniz" in col:
                adoption_scores.append(float(v))
            elif "doğrudan cevap vermek yerine" in col:
                guardrail_utility_scores.append(float(v))
            else:
                for q_prefix, gold in QUIZ_GOLD.items():
                    if q_prefix in col:
                        quiz_total += 1
                        if v == gold:
                            quiz_correct += 1
                for ab_id, mapping in AB_UNBLIND.items():
                    if f"[{ab_id}]" in col and "Genel Tercih" in col:
                        key = mapping.get(v, f"{ab_id}_tie")
                        ab_counts[key] = ab_counts.get(key, 0) + 1

    dpo_v2_vs_hyb_dpo = sum(
        1 for row in rows
        for ab_id in ("AB_01", "AB_02", "AB_03")
        for col, val in row.items()
        if f"[{ab_id}]" in col and "Genel Tercih" in col and AB_UNBLIND[ab_id].get((val or "").strip()) == "onpolicy_dpo_v2"
    )
    dpo_v2_vs_hyb_hyb = sum(
        1 for row in rows
        for ab_id in ("AB_01", "AB_02", "AB_03")
        for col, val in row.items()
        if f"[{ab_id}]" in col and "Genel Tercih" in col and AB_UNBLIND[ab_id].get((val or "").strip()) == "task_separated_hybrid"
    )

    mean = lambda lst: round(sum(lst) / max(len(lst), 1), 3)
    return {
        "total_participants": len(rows),
        "cefr_explanations": {
            "mean_clarity_1_to_5": mean(clarity_scores),
            "mean_turkish_1_to_5": mean(turkish_scores),
            "mean_cognitive_ease_1_to_5": mean(cogload_scores),
            "comprehension_quiz_accuracy": round(quiz_correct / max(quiz_total, 1), 4),
        },
        "ab_preference_unblinded_all": ab_counts,
        "ab_01_03_dpo_v2_vs_hybrid": {
            "onpolicy_dpo_v2_wins": dpo_v2_vs_hyb_dpo,
            "task_separated_hybrid_wins": dpo_v2_vs_hyb_hyb,
            "exact_sign_test_p": exact_binomial_p(dpo_v2_vs_hyb_dpo, dpo_v2_vs_hyb_hyb),
        },
        "preferred_tutor_rubric_means_1_to_5": {
            "socratic": mean(socratic_ratings),
            "prerequisite_grounding": mean(prereq_ratings),
            "turkish_fluency": mean(style_ratings),
        },
        "overall_system_feedback": {
            "adoption_intent_1_to_5": mean(adoption_scores),
            "anti_cheat_guardrail_utility_1_to_5": mean(guardrail_utility_scores),
        },
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_google_forms_csv.py <google_forms_responses.csv>")
        sys.exit(1)
    res = analyze(Path(sys.argv[1]))
    print(json.dumps(res, ensure_ascii=False, indent=2))
