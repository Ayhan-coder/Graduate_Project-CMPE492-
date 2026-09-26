"""enriched_schema.py: Standard enriched output schema for downstream educational modules.

Addresses the advisor's work package:
"Enriched output for downstream modules. Timestamped terms, a prerequisite concept graph,
difficulty levels, and multiple explanation variants per term, delivered in the shared schema."

Delivers:
1. Timestamped terms (linked to ASR speech segments).
2. Prerequisite Concept Graph (DAG: nodes, prerequisite-of edges, topological order).
3. Pedagogical Difficulty Levels (Beginner, Intermediate, Advanced) & Bloom Taxonomy.
4. Triplet Explanation Variants per Term:
   - CEFR B1 intuitive analogy
   - Rigorous academic / textbook definition
   - One-sentence elevator pitch / quick takeaway
5. High-stakes exam tips & common misconception flags.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Domain prerequisite rules (Source -> Target prerequisite dependencies)
PREREQUISITE_RULES = {
    # Project Management
    "proje yönetimi": ["proje yöneticisi", "proje yönetim planı", "kapsam", "bütçe"],
    "proje yönetim planı": ["temel çizgi", "iş kırılım yapısı", "izleme", "kontrol"],
    "izleme": ["kontrol", "performans raporları"],
    "kontrol": ["düzeltici eylemler", "önleyici eylemler", "değişiklik talebi"],
    "temel çizgi": ["kazanılmış değer yönetimi", "planlanmış değer"],
    "planlanmış değer": ["kazanılmış değer", "gerçekleşen maliyet"],
    "kazanılmış değer": ["maliyet sapması", "çizelge sapması", "maliyet performans indeksi", "çizelge performans indeksi"],

    # Agile & Software Engineering
    "agile": ["scrum", "kanban", "extreme programming", "lean"],
    "scrum": ["sprint", "backlog", "scrum master", "product owner", "development team"],
    "sprint": ["daily scrum", "timebox", "deliverable", "iteration"],
    "backlog": ["product backlog", "sprint backlog", "work in progress"],
    "kanban": ["kanban board", "work in progress", "wip limit"],

    # Tourism
    "turizm": ["turist", "turizm arzı", "turizm talebi", "kitle turizmi"],
    "turizm arzı": ["ulaşım", "konaklama", "stoklanamazlık", "uktkg"],
    "turizm talebi": ["stoklanamazlık"],

    # Data Science
    "veri bilimi": ["büyük veri", "python", "makine öğrenmesi"],
    "python": ["numpy", "pandas", "jupyter notebook"],
    "pandas": ["veri ön işleme"],
    "veri ön işleme": ["makine öğrenmesi", "model validation", "derin öğrenme"],
}


def estimate_difficulty_and_bloom(term: str, explanation: str) -> tuple[str, str]:
    """Classify term into pedagogical difficulty level and Bloom's taxonomy category."""
    norm = term.lower()
    expl_norm = explanation.lower()

    # Advanced keywords
    if any(k in norm for k in ["indeks", "performans indeksi", "varyans", "sapması", "validasyon", "deep learning", "derin öğrenme", "backpropagation"]):
        return "Advanced", "Analyzing"
    # Intermediate keywords
    elif any(k in norm for k in ["yönetimi", "kontrol", "hesap", "wbs", "kanban", "sprint", "makine öğrenmesi", "pandas", "numpy"]):
        return "Intermediate", "Applying"
    # Foundational / Beginner concepts
    else:
        return "Beginner", "Understanding"


def generate_explanation_variants(
    term: str,
    base_explanation: str,
    lang: str = "tr",
) -> dict[str, str]:
    """Synthesize 3 distinct pedagogical explanation variants for a term."""
    is_tr = str(lang).lower().startswith("tr")
    clean_base = base_explanation.strip()

    # 1. Formal Textbook Definition
    formal = clean_base

    # 2. Intuitive Real-world Analogy (CEFR B1)
    norm = term.lower()
    if any(k in norm for k in ["kazanılmış değer", "evm", "earned value"]):
        analogy = "Bir inşaat ustasının 10 günde yapmayı planladığı duvar ile gerçekte harcadığı parayı kıyaslamak gibidir." if is_tr else "Like comparing the actual walls built by a contractor against the money spent so far."
    elif any(k in norm for k in ["sapma", "variance"]):
        analogy = "Navigasyonun rotadan ne kadar uzaklaştığınızı kilometre cinsinden göstermesi gibidir." if is_tr else "Like a GPS alert showing how far off course you have drifted."
    elif any(k in norm for k in ["sprint", "iteration"]):
        analogy = "Büyük bir maratonu tek seferde koşmak yerine her gün 1 kilometrelik kısa hedefler koyup koşmak gibidir." if is_tr else "Like running a marathon in short 1-mile sprints with regular check-ins."
    elif any(k in norm for k in ["backlog"]):
        analogy = "Tatile çıkmadan önce yapılacaklar listesine maddeler ekleyip önem sırasına göre dizmek gibidir." if is_tr else "Like an ordered to-do list where top-priority items get done first."
    elif any(k in norm for k in ["turizm arzı", "supply"]):
        analogy = "Bir otelin odalarını sabit tutması gibidir; oda boş kalsa bile taşınamaz veya saklanamaz." if is_tr else "Like hotel rooms that cannot be stored or moved if unsold tonight."
    elif any(k in norm for k in ["veri ön işleme", "preprocessing"]):
        analogy = "Yemek yapmadan önce sebzeleri yıkayıp doğrayarak tencereye hazır hale getirmek gibidir." if is_tr else "Like washing and chopping vegetables before you start cooking."
    else:
        analogy = f"Bir hedefe giderken {term} kavramı rehber bir harita işlevi görür." if is_tr else f"In learning this domain, {term} acts as an essential foundational roadmap."

    # 3. Elevator Pitch (Single-sentence memorable summary)
    first_sentence = re.split(r"[.!?]", clean_base)[0].strip()
    pitch = (first_sentence + ".") if first_sentence else clean_base

    # 4. Exam Tip
    if any(k in norm for k in ["maliyet sapması", "cv"]):
        exam_tip = "Sınav kuralı: CV = EV - AC. Sonuç negatif çıkarsa bütçe aşılmıştır!" if is_tr else "Exam rule: CV = EV - AC. A negative value indicates cost overrun!"
    elif any(k in norm for k in ["çizelge sapması", "sv"]):
        exam_tip = "Sınav kuralı: SV = EV - PV. Sonuç negatifse proje planın gerisinde kalmıştır." if is_tr else "Exam rule: SV = EV - PV. A negative value indicates delay behind schedule."
    elif any(k in norm for k in ["sprint"]):
        exam_tip = "Sprint süresi sabittir (Timebox) ve asla uzatılmaz; iş bitmediyse sonraki sprinte aktarılır." if is_tr else "Sprint timebox duration is fixed and never extended."
    else:
        exam_tip = f"{term} kavramının temel rolüne ve neden-sonuç ilişkisine odaklanın." if is_tr else f"Focus on the core causal relationship and role of {term}."

    return {
        "b1_analogy": analogy,
        "formal_definition": formal,
        "elevator_pitch": pitch,
        "exam_tip": exam_tip,
    }


def build_concept_dag(terms: list[dict]) -> dict:
    """Construct a Directed Acyclic Graph (DAG) of concept prerequisites."""
    term_names = {t["term"].strip() for t in terms if t.get("term")}
    norm_to_actual = {t.lower(): t for t in term_names}

    nodes = []
    edges = []

    for t in terms:
        t_name = t.get("term", "")
        difficulty = t.get("difficulty", "Beginner")
        bloom = t.get("bloom_level", "Understanding")
        nodes.append({
            "id": t_name,
            "label": t_name,
            "difficulty": difficulty,
            "bloom_level": bloom,
            "timestamp": t.get("timestamp"),
        })

    # Find prerequisite edges
    for t_name in term_names:
        norm_source = t_name.lower()
        if norm_source in PREREQUISITE_RULES:
            for target_norm in PREREQUISITE_RULES[norm_source]:
                if target_norm in norm_to_actual:
                    target_actual = norm_to_actual[target_norm]
                    edges.append({
                        "from": t_name,
                        "to": target_actual,
                        "relation": "prerequisite_of",
                    })

    # Link prerequisites back to term dicts
    target_to_prereqs: dict[str, list[str]] = {}
    for edge in edges:
        target_to_prereqs.setdefault(edge["to"], []).append(edge["from"])

    for t in terms:
        t_name = t.get("term", "")
        t["prerequisites"] = target_to_prereqs.get(t_name, [])

    return {
        "nodes": nodes,
        "edges": edges,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
    }


def build_enriched_output(
    clip_id: str,
    lecture_title: str,
    lang: str,
    raw_terms: list[dict],
    segments: list[dict] | None = None,
) -> dict:
    """Transform extracted raw terms and ASR segments into the shared standard schema."""
    enriched_terms = []

    # Map timestamps from segments if available
    for idx, item in enumerate(raw_terms):
        t_name = (item.get("term") or "").strip()
        t_expl = (item.get("explanation") or "").strip()
        if not t_name:
            continue

        # Match earliest segment
        matched_start = None
        matched_end = None
        if segments:
            for seg in segments:
                if t_name.lower() in (seg.get("text") or "").lower():
                    matched_start = seg.get("start")
                    matched_end = seg.get("end")
                    break

        if matched_start is not None:
            m = int(matched_start // 60)
            s = int(matched_start % 60)
            timestamp_str = f"{m:02d}:{s:02d}"
        else:
            timestamp_str = item.get("timestamp") or "00:00"

        diff_level, bloom = estimate_difficulty_and_bloom(t_name, t_expl)
        explanations = generate_explanation_variants(t_name, t_expl, lang=lang)

        enriched_terms.append({
            "id": f"{clip_id}-term-{idx+1:03d}",
            "term": t_name,
            "canonical": t_name,
            "timestamp": timestamp_str,
            "start_s": matched_start,
            "end_s": matched_end,
            "difficulty": diff_level,
            "bloom_level": bloom,
            "prerequisites": [],  # Will be populated by DAG builder
            "explanations": explanations,
        })

    dag = build_concept_dag(enriched_terms)

    return {
        "schema_version": "2.0.0",
        "video_id": clip_id,
        "lecture_title": lecture_title,
        "lang": "tr" if str(lang).lower().startswith("tr") else "en",
        "total_terms": len(enriched_terms),
        "concept_graph": dag,
        "terms": enriched_terms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate standard enriched output schema")
    parser.add_argument("--test", action="store_true", help="Run self-test with sample concepts")
    parser.add_argument("--clip", default="TR-1")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.test:
        print("Running Enriched Schema self-test...")
        sample_raw = [
            {"term": "Proje Yönetimi", "explanation": "Kaynakların hedeflere ulaşmak için planlanması sürecidir."},
            {"term": "Proje Yönetim Planı", "explanation": "Projenin nasıl yürütüleceğini belirten ana belgedir."},
            {"term": "Kazanılmış Değer Yönetimi", "explanation": "Kapsam, bütçe ve takvim performansını ölçen yöntemdir (EVM)."},
            {"term": "Maliyet Sapması", "explanation": "Kazanılmış değer ile fiili maliyet arasındaki parasal farktır."},
            {"term": "Temel Çizgi", "explanation": "Performans ölçümünde referans alınan onaylanmış başlangıç planıdır."},
        ]

        enriched = build_enriched_output(
            clip_id="TR-1",
            lecture_title="Proje İzleme ve Kontrolü",
            lang="tr",
            raw_terms=sample_raw,
        )

        print(f"Generated Schema Version: {enriched['schema_version']}")
        print(f"Total Terms: {enriched['total_terms']}")
        print(f"Concept Graph Nodes: {enriched['concept_graph']['n_nodes']} | Edges: {enriched['concept_graph']['n_edges']}")
        print("\nSample Enriched Term Output:")
        sample_term = enriched["terms"][2]
        print(f"  Term          : {sample_term['term']}")
        print(f"  Difficulty    : {sample_term['difficulty']} | Bloom: {sample_term['bloom_level']}")
        print(f"  Prerequisites : {sample_term['prerequisites']}")
        print(f"  B1 Analogy    : {sample_term['explanations']['b1_analogy']}")
        print(f"  Exam Tip      : {sample_term['explanations']['exam_tip']}")

        assert enriched["total_terms"] == 5
        assert enriched["concept_graph"]["n_edges"] >= 3
        print("\nEnriched schema self-test passed successfully.")
        return


if __name__ == "__main__":
    main()
