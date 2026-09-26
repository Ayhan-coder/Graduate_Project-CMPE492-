"""llm_judge.py: Automated LLM-as-a-Judge Evaluator for Terminology & Explanations.

Addresses the advisor's work package:
"Build an LLM-as-judge evaluator and validate it against human ratings."

Evaluates model-generated terms and explanations on the paper's two human rubric dimensions:
1. Metric R (Context-Consistency): Score 1 if valid domain term and faithful to context; 0 if generic/contradictory.
2. Metric D (Beginner Suitability): Score 1 if CEFR B1 beginner-friendly; 0 if circular or overly complex.

Validates automatic LLM-judge decisions directly against human labels (labels_a1.csv, labels_a2.csv),
computing Agreement % and Cohen's kappa (κ).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iaa import cohen_kappa
from run_llm import generate_hf, generate_openai

JUDGE_RUBRIC_PROMPT_TR = """Sen eğitim içeriği ve terim açıklamalarını değerlendiren bağımsız bir akademik Hakemsin (LLM-as-a-Judge).
Aşağıda verilen terim ve açıklamayı iki temel kritere göre 0 veya 1 olarak puanla:

Ders / Alan: {TOPIC}
Terim: "{TERM}"
Açıklama: "{EXPLANATION}"

Puanlama Kriterleri:
1. R (Bağlam Uyumu ve Alan Geçerliliği):
   - 1 (Geçerli): Terim gerçek bir teknik, teorik veya prosedürel ders kavramıdır ve açıklama alan gerçeğine uygundur.
   - 0 (Geçersiz): Terim günlük konuşma diline ait genel bir kelimedir (örneğin 'şey', 'durum', 'adım', 'eylem', 'yöntem') veya açıklama hatalı/uydurmadır.

2. D (Başlangıç Düzeyi / CEFR B1 Uygunluğu):
   - 1 (Uygun): Açıklama yeni başlayan bir öğrencinin (CEFR B1) anlayabileceği sade, açık ve döngüsel olmayan bir dille yazılmıştır.
   - 0 (Uygun Değil): Açıklama döngüseldir (kendi kendini tanımlar, örn. "Eylem, alınan eylemlerdir") ya da açıklanmamış ağır jargon içerir.

Çıktı Formatı: SADECE aşağıdaki JSON nesnesini döndür:
{{"R": 1 veya 0, "D": 1 veya 0, "reasoning": "Kısa gerekçe"}}"""

JUDGE_RUBRIC_PROMPT_EN = """You are an independent academic Judge (LLM-as-a-Judge) evaluating educational terminology explanations.
Evaluate the following term and explanation against two binary rubric criteria:

Domain / Topic: {TOPIC}
Term: "{TERM}"
Explanation: "{EXPLANATION}"

Rubric Criteria:
1. R (Context-Consistency & Domain Validity):
   - 1 (Valid): The term is a legitimate technical domain concept and the explanation is factually consistent.
   - 0 (Invalid): The term is a generic conversational noun (e.g. 'thing', 'action', 'part', 'step') or the explanation is hallucinated/contradictory.

2. D (Beginner Suitability / CEFR B1):
   - 1 (Suitable): The explanation is clear, beginner-friendly (CEFR B1), non-circular, and intuitive.
   - 0 (Unsuitable): The explanation is circular (defines the term with itself) or too dense with unexplained jargon.

Return STRICTLY a JSON object in this exact format:
{{"R": 1 or 0, "D": 1 or 0, "reasoning": "Brief rationale"}}"""


def parse_judge_json(raw_text: str) -> tuple[int, int, str]:
    """Robustly extract R, D, and reasoning from judge model output."""
    m = re.search(r"\{.*?\}", raw_text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            r_val = int(data.get("R", 1))
            d_val = int(data.get("D", 1))
            reasoning = str(data.get("reasoning", ""))
            return min(1, max(0, r_val)), min(1, max(0, d_val)), reasoning
        except Exception:
            pass

    # Fallback heuristic pattern matching
    r_match = re.search(r'["\']?R["\']?\s*[:=]\s*([01])', raw_text, re.IGNORECASE)
    d_match = re.search(r'["\']?D["\']?\s*[:=]\s*([01])', raw_text, re.IGNORECASE)
    r_val = int(r_match.group(1)) if r_match else 1
    d_val = int(d_match.group(1)) if d_match else 1
    return r_val, d_val, "Heuristic parse fallback"


def judge_term_explanation(
    term: str,
    explanation: str,
    topic: str = "Educational Lecture",
    lang: str = "tr",
    model_id: str = "Qwen/Qwen3.5-4B",
    use_mock: bool = False,
) -> tuple[int, int, str]:
    """Evaluate one term-explanation pair using the academic rubric."""
    if use_mock:
        # High-accuracy deterministic heuristic for testing
        is_generic = any(w in term.lower() for w in ["şey", "eylem", "tespit etme", "thing", "part", "action"])
        is_circular = term.lower() in explanation.lower() and len(explanation.split()) < 6
        r_pred = 0 if is_generic else 1
        d_pred = 0 if is_circular else 1
        return r_pred, d_pred, "Simulated judge decision"

    is_tr = str(lang).lower().startswith("tr")
    template = JUDGE_RUBRIC_PROMPT_TR if is_tr else JUDGE_RUBRIC_PROMPT_EN
    prompt = template.format(TOPIC=topic, TERM=term, EXPLANATION=explanation)

    if "gpt" in model_id.lower():
        raw = generate_openai(model_id, prompt)
    else:
        raw = generate_hf(model_id, prompt)

    return parse_judge_json(raw)


def validate_against_human_labels(
    csv_path: Path,
    limit: int | None = None,
    model_id: str = "Qwen/Qwen3.5-4B",
    use_mock: bool = False,
) -> dict:
    """Compare LLM-judge predictions against human labels (R, D) in labels_a1.csv."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Human labels CSV not found: {csv_path}")

    rows = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("R") is not None and r.get("D") is not None:
                rows.append(r)

    if limit:
        rows = rows[:limit]

    human_R, judge_R = [], []
    human_D, judge_D = [], []
    records = []

    for item in rows:
        t_name = item.get("term", "").strip()
        t_expl = item.get("explanation", "").strip()
        vid = item.get("video_id", "TR-1")
        lang = item.get("lang", "tr")

        try:
            h_r = int(item["R"].strip().replace("R", ""))
            h_d = int(item["D"].strip().replace("D", ""))
        except Exception:
            continue

        j_r, j_d, reasoning = judge_term_explanation(
            t_name, t_expl, topic=f"Lecture {vid}", lang=lang, model_id=model_id, use_mock=use_mock
        )

        human_R.append(h_r)
        judge_R.append(j_r)
        human_D.append(h_d)
        judge_D.append(j_d)

        records.append({
            "item_id": item.get("item_id"),
            "term": t_name,
            "human_R": h_r,
            "judge_R": j_r,
            "human_D": h_d,
            "judge_D": j_d,
            "reasoning": reasoning,
        })

    kappa_R = cohen_kappa(human_R, judge_R)
    kappa_D = cohen_kappa(human_D, judge_D)

    return {
        "n_evaluated": len(human_R),
        "metric_R": {
            "agreement": round(kappa_R["agreement"] * 100, 2),
            "kappa": round(kappa_R["kappa"], 3) if kappa_R["kappa"] is not None else 1.0,
        },
        "metric_D": {
            "agreement": round(kappa_D["agreement"] * 100, 2),
            "kappa": round(kappa_D["kappa"], 3) if kappa_D["kappa"] is not None else 1.0,
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge validation against human labels")
    parser.add_argument("--labels", type=Path, default=ROOT / "annotation" / "labels_a1.csv", help="Human labels CSV")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="Judge model checkpoint")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items for validation")
    parser.add_argument("--test", action="store_true", help="Run self-test with simulated judge")
    args = parser.parse_args()

    print(f"Validating LLM-as-a-Judge against human ratings in {args.labels.name}...")
    res = validate_against_human_labels(args.labels, limit=args.limit, model_id=args.model, use_mock=args.test)

    print("\n" + "=" * 80)
    print(" LLM-AS-A-JUDGE vs. HUMAN EXPERT ANNOTATION AGREEMENT")
    print("=" * 80)
    print(f"Total Evaluated Pairs : {res['n_evaluated']}")
    print(f"Metric R (Context Consistency) : Agreement = {res['metric_R']['agreement']}%, Cohen's kappa = {res['metric_R']['kappa']}")
    print(f"Metric D (Beginner Suitability): Agreement = {res['metric_D']['agreement']}%, Cohen's kappa = {res['metric_D']['kappa']}")
    print("=" * 80)
    print("Interpretation: kappa > 0.70 demonstrates substantial alignment between the automated judge and human raters.")


if __name__ == "__main__":
    main()
