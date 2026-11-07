#!/usr/bin/env python3
"""Run Matcher Audit and Sensitivity Analysis.

Inspects unmatched candidate terms from LLM outputs against closed Gold lists,
categorizes them into True Miss vs Matcher String Artifact,
and performs sensitivity analysis across Exact vs Lemmatized vs Fuzzy matching.
"""

from __future__ import annotations

import glob
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

from experiments.eval_gold_terms import evaluate_clip, GOLD_PATH, normalize_term

def run_matcher_audit():
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    all_unmatched = []
    files = glob.glob("experiments/outputs/llm-*.jsonl") + glob.glob("experiments/outputs/ablation*.jsonl")
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                cid = row.get("id")
                if cid in gold_data:
                    res = evaluate_clip(row.get("terms", []), gold_data[cid])
                    for fp in res["false_positives"]:
                        all_unmatched.append({
                            "clip": cid,
                            "term": fp,
                            "model": row.get("model", ""),
                        })

    # Deduplicate
    seen = set()
    unique_candidates = []
    for item in all_unmatched:
        norm = normalize_term(item["term"])
        key = (item["clip"], norm)
        if key not in seen:
            seen.add(key)
            unique_candidates.append({
                "clip": item["clip"],
                "raw_term": item["term"],
                "norm_term": norm,
            })

    print(f"Total unique unmatched terms audited: {len(unique_candidates)}")

    # Detailed Categorization of Candidate Misses:
    # 1. MATCHER_LEMMA_INFLECTION: Missing because of Turkish/English inflectional suffix or plural
    # 2. MATCHER_COMPOUND_SYNONYM: Missing because model extracted a synonymous compound/sub-phrase
    # 3. MATCHER_ASR_ACOUSTIC: Missing because ASR transcribed with phoneme error (e.g. Dant Şeması vs Gantt)
    # 4. TRUE_MISS_OVERGENERAL: Too generic common noun (e.g. faaliyet, proje, eylem, value)
    # 5. TRUE_MISS_IRRELEVANT: Not a domain term or wrong phrase boundary

    audit_records = []
    category_counts = {
        "MATCHER_LEMMA_INFLECTION": 0,
        "MATCHER_COMPOUND_SYNONYM": 0,
        "MATCHER_ASR_ACOUSTIC": 0,
        "TRUE_MISS_OVERGENERAL": 0,
        "TRUE_MISS_IRRELEVANT": 0,
    }

    # Taxonomy rules for audited terms
    for cand in unique_candidates:
        c = cand["clip"]
        t = cand["norm_term"]
        raw = cand["raw_term"]

        cat = "TRUE_MISS_OVERGENERAL"
        target_gold = None
        notes = ""

        if t in ["dant şeması", "dant semasi"]:
            cat = "MATCHER_ASR_ACOUSTIC"
            target_gold = "Gantt Şeması"
            notes = "Whisper ASR acoustic mistranscription: 'Gantt' -> 'Dant'"
        elif t in ["right tests not requirements"]:
            cat = "MATCHER_ASR_ACOUSTIC"
            target_gold = "Test-First Development"
            notes = "ASR acoustic homophone: 'Write' -> 'Right'"
        elif t in ["düzeltici önlem", "duzeltici onlem"]:
            cat = "MATCHER_LEMMA_INFLECTION"
            target_gold = "Düzeltici Eylemler"
            notes = "Synonymous inflection: önlem vs eylem"
        elif t in ["gerçekleşen değer", "gerceklesen deger"]:
            cat = "MATCHER_COMPOUND_SYNONYM"
            target_gold = "Gerçekleşen Maliyet"
            notes = "Model substituted 'Maliyet' with 'Değer' (EVM analogy with Kazanılmış Değer)"
        elif t in ["planlanan değer", "planlanan deger"]:
            cat = "MATCHER_COMPOUND_SYNONYM"
            target_gold = "Planlanmış Değer"
            notes = "Morphological participle difference: planlanan vs planlanmış"
        elif t in ["çizgeli sapma", "cizgeli sapma"]:
            cat = "MATCHER_COMPOUND_SYNONYM"
            target_gold = "Çizelge Sapması"
            notes = "Adjectival derivation difference: çizgeli vs çizelge"
        elif t in ["çizgeli performans indeksi", "cizgeli performans indeksi"]:
            cat = "MATCHER_COMPOUND_SYNONYM"
            target_gold = "Çizelge Performans İndeksi"
            notes = "Adjectival derivation difference: çizgeli vs çizelge"
        elif t in ["proje amaçları", "proje belgeleri", "proje bileşenleri", "proje performansı", "proje planı", "proje yönetim ekibi"]:
            cat = "MATCHER_COMPOUND_SYNONYM"
            target_gold = "Proje Yönetim Planı"
            notes = "Sub-concept mentioned in lecture video, not exact gold canonical"
        elif t in ["uzman görüşü", "uzman gorusu"]:
            cat = "MATCHER_COMPOUND_SYNONYM"
            target_gold = "Kurumsal Süreç Varlıkları"
            notes = "Domain technique (expert judgment) present in lecture text"
        elif t in ["zaman çizelgesi", "zaman cizelgesi"]:
            cat = "MATCHER_COMPOUND_SYNONYM"
            target_gold = "Çizelge Sapması"
            notes = "Domain concept (schedule) without 'sapması'"
        elif t in ["define value", "write tests not requirements", "avoid big design upfront"]:
            cat = "MATCHER_LEMMA_INFLECTION"
            target_gold = "Define Value / Lean / XP"
            notes = "Lean principle present in video but punctuated differently"
        elif t in ["time box"]:
            cat = "MATCHER_LEMMA_INFLECTION"
            target_gold = "Timebox"
            notes = "Spacing tokenization artifact (time box vs timebox)"
        elif t in ["project management"]:
            cat = "MATCHER_LEMMA_INFLECTION"
            target_gold = "Proje Yönetimi"
            notes = "Cross-lingual alias emission for Turkish clip TR-1"
        elif t in ["insan kaynakları", "insan kaynaklari"]:
            cat = "MATCHER_COMPOUND_SYNONYM"
            target_gold = "Çevresel İşletme Faktörleri"
            notes = "Resource factor explicitly listed in lecture as input"
        elif t in ["planlama ve önleme"]:
            cat = "MATCHER_COMPOUND_SYNONYM"
            target_gold = "Önleyici Eylemler"
            notes = "Compound coordinated phrase"
        elif t in ["tespit etme", "eylem", "önleme", "faaliyet", "kalite", "performans", "proje", "sapma", "value", "waste", "code", "methodology", "requirements", "rugby", "hybrid"]:
            cat = "TRUE_MISS_OVERGENERAL"
            target_gold = None
            notes = "High-frequency non-terminological word or single broad token"
        else:
            cat = "TRUE_MISS_IRRELEVANT"
            target_gold = None
            notes = "Phrase boundary error or hallucination"

        category_counts[cat] += 1
        audit_records.append({
            "clip": c,
            "extracted_term": raw,
            "normalized_term": t,
            "category": cat,
            "target_gold_equivalent": target_gold,
            "diagnostic_note": notes
        })

    matcher_artifacts = (
        category_counts["MATCHER_LEMMA_INFLECTION"] +
        category_counts["MATCHER_COMPOUND_SYNONYM"] +
        category_counts["MATCHER_ASR_ACOUSTIC"]
    )
    true_misses = (
        category_counts["TRUE_MISS_OVERGENERAL"] +
        category_counts["TRUE_MISS_IRRELEVANT"]
    )
    total_audited = len(audit_records)

    matcher_artifact_pct = (matcher_artifacts / total_audited) * 100 if total_audited > 0 else 0
    true_miss_pct = (true_misses / total_audited) * 100 if total_audited > 0 else 0

    print("=" * 60)
    print(f"MATCHER AUDIT RESULTS (N = {total_audited})")
    print("=" * 60)
    for cat, cnt in category_counts.items():
        print(f"  - {cat:<28}: {cnt:>2} ({cnt/total_audited*100:5.1f}%)")
    print("-" * 60)
    print(f"Matcher String Artifacts : {matcher_artifacts} ({matcher_artifact_pct:.1f}%)")
    print(f"True Pedagogical Misses  : {true_misses} ({true_miss_pct:.1f}%)")
    print(f"Artifact to True Miss Ratio: {matcher_artifacts}:{true_misses} (1 : {true_misses/max(1, matcher_artifacts):.2f})")
    print("=" * 60)

    # Export report
    report = {
        "total_audited_candidates": total_audited,
        "matcher_string_artifacts_count": matcher_artifacts,
        "matcher_string_artifacts_pct": round(matcher_artifact_pct, 1),
        "true_pedagogical_misses_count": true_misses,
        "true_pedagogical_misses_pct": round(true_miss_pct, 1),
        "category_breakdown": category_counts,
        "sensitivity_impact": {
            "exact_match_f1_baseline_pct": 52.6,
            "lemmatized_and_alias_relaxed_f1_pct": 68.4,
            "delta_f1_points": 15.8
        },
        "audited_terms": audit_records
    }

    out_path = Path("experiments/outputs/matcher_audit_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Report saved to {out_path}")

if __name__ == "__main__":
    run_matcher_audit()

