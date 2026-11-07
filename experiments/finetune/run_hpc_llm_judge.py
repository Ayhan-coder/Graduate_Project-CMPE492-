"""Empirical LLM-as-a-Judge vs. Human Agreement Benchmark on Boğaziçi HPC A40.

Addresses the core thesis methodological limitation:
"2026 genişleme hücrelerinde kullanılan yerel LLM hakeminin insan etiketleriyle
doğrulaması yapılmamıştır (hakem-insan uyumu diskte artefaktı olan bir deney değildir)."

This script:
1. Loads Qwen2.5-7B-Instruct on NVIDIA A40 (bfloat16).
2. Evaluates the full human-annotated corpus (labels_a1.csv: 233 items) and
   the double-labeled benchmark subset (labels_a2.csv: 115 items).
3. Prompts the 7B model using the identical academic rubric:
   - R (Context-Consistency & Domain Validity: 0 or 1)
   - D (Beginner Suitability & CEFR B1 Plain Language: 0 or 1)
4. Saves item-by-item judge predictions to disk: labels_7b_judge.csv
5. Computes pairwise Inter-Rater Agreement:
   - Judge (7B) vs. Human Annotator 1 (233 items & 115 items)
   - Judge (7B) vs. Human Annotator 2 (115 items)
   - Human Annotator 1 vs. Human Annotator 2 (Human baseline: κ = 0.782)
   - Calculates observed agreement (Po), expected agreement (Pe),
     Cohen's Kappa (κ), and PABAK (Prevalence-Adjusted Bias-Adjusted Kappa).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("LLMJudgeBenchmark")

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


def parse_judge_json(raw_text: str) -> Tuple[int, int, str]:
    """Robustly extract R, D, and reasoning from judge output."""
    m = re.search(r"\{.*?\}", raw_text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            r_val = int(data.get("R", 1))
            d_val = int(data.get("D", 1))
            reasoning = str(data.get("reasoning", "")).strip()
            return min(1, max(0, r_val)), min(1, max(0, d_val)), reasoning
        except Exception:
            pass

    # Heuristic fallback
    r_match = re.search(r'["\']?R["\']?\s*[:=]\s*([01])', raw_text, re.IGNORECASE)
    d_match = re.search(r'["\']?D["\']?\s*[:=]\s*([01])', raw_text, re.IGNORECASE)
    r_val = int(r_match.group(1)) if r_match else 1
    d_val = int(d_match.group(1)) if d_match else 1
    return r_val, d_val, "Heuristic parse fallback"


def compute_metrics(y1: List[int], y2: List[int]) -> Dict[str, Any]:
    """Compute Po, Pe, Cohen's Kappa, and PABAK."""
    if len(y1) != len(y2) or len(y1) == 0:
        return {"n": len(y1), "agreement_pct": 0.0, "po": 0.0, "pe": 0.0, "kappa": 0.0, "pabak": 0.0}
    n = len(y1)
    agree = sum(a == b for a, b in zip(y1, y2))
    po = agree / n
    
    c1, c2 = Counter(y1), Counter(y2)
    labels = set(c1) | set(c2)
    pe = sum((c1[k] / n) * (c2[k] / n) for k in labels)
    
    kappa = 1.0 if pe == 1.0 and po == 1.0 else ((po - pe) / (1.0 - pe) if pe < 1.0 else 0.0)
    # PABAK = 2 * Po - 1
    pabak = (2.0 * po) - 1.0

    # Confusion matrix
    tp = sum(1 for a, b in zip(y1, y2) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y1, y2) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y1, y2) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y1, y2) if a == 1 and b == 0)

    return {
        "n": n,
        "agreement_pct": round(po * 100.0, 2),
        "po": round(po, 4),
        "pe": round(pe, 4),
        "kappa": round(kappa, 3),
        "pabak": round(pabak, 3),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    }


def main():
    parser = argparse.ArgumentParser(description="LLM Judge Validation Benchmark")
    parser.add_argument("--model-path", type=str, default="/users/ali.gunder/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--a1-csv", type=str, default="annotation/labels_a1.csv")
    parser.add_argument("--a2-csv", type=str, default="annotation/labels_a2.csv")
    parser.add_argument("--output-csv", type=str, default="annotation/labels_7b_judge.csv")
    parser.add_argument("--output-json", type=str, default="experiments/outputs/llm_judge_human_agreement_report.json")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("=" * 80)
    logger.info("STARTING LLM-AS-A-JUDGE EMPIRICAL HUMAN AGREEMENT BENCHMARK")
    logger.info("=" * 80)

    # 1. Load human labels
    a1_rows = {}
    with open(args.a1_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            item_id = r["item_id"].strip()
            if item_id and not item_id.startswith("#"):
                a1_rows[item_id] = {
                    "item_id": item_id,
                    "video_id": r.get("video_id", ""),
                    "model": r.get("model", ""),
                    "lang": r.get("lang", "tr"),
                    "term": r.get("term", "").strip(),
                    "explanation": r.get("explanation", "").strip(),
                    "R": int(r["R"].strip().replace("R", "")),
                    "D": int(r["D"].strip().replace("D", ""))
                }

    a2_rows = {}
    if os.path.exists(args.a2_csv):
        with open(args.a2_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                item_id = r["item_id"].strip()
                if item_id and not item_id.startswith("#"):
                    a2_rows[item_id] = {
                        "R": int(r["R"].strip().replace("R", "")),
                        "D": int(r["D"].strip().replace("D", ""))
                    }

    logger.info(f"Loaded {len(a1_rows)} items from A1, {len(a2_rows)} items from A2.")
    common_ids = sorted([i for i in a1_rows if i in a2_rows])
    logger.info(f"Common double-labeled subset: {len(common_ids)} items.")

    # 2. Load Model
    logger.info(f"Loading Judge Model from {args.model_path} in bfloat16 on A40 GPU...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    logger.info("Judge Model Loaded successfully.")

    # 3. Perform Item-by-Item Evaluation
    judge_predictions = {}
    judge_records_for_csv = []
    t_start = time.perf_counter()

    for idx, (item_id, item) in enumerate(a1_rows.items(), 1):
        is_tr = item["lang"].lower().startswith("tr")
        template = JUDGE_RUBRIC_PROMPT_TR if is_tr else JUDGE_RUBRIC_PROMPT_EN
        user_prompt = template.format(
            TOPIC=f"Lecture {item['video_id']}",
            TERM=item["term"],
            EXPLANATION=item["explanation"]
        )

        full_prompt = (
            f"<|im_start|>system\nYou are an objective academic evaluator.<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=96,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        j_r, j_d, reasoning = parse_judge_json(gen_text)

        judge_predictions[item_id] = {
            "judge_R": j_r,
            "judge_D": j_d,
            "reasoning": reasoning
        }

        judge_records_for_csv.append({
            "item_id": item_id,
            "video_id": item["video_id"],
            "model": item["model"],
            "lang": item["lang"],
            "term": item["term"],
            "explanation": item["explanation"],
            "a1_R": item["R"],
            "a1_D": item["D"],
            "a2_R": a2_rows[item_id]["R"] if item_id in a2_rows else "",
            "a2_D": a2_rows[item_id]["D"] if item_id in a2_rows else "",
            "judge_7b_R": j_r,
            "judge_7b_D": j_d,
            "judge_reasoning": reasoning
        })

        if idx % 25 == 0 or idx == len(a1_rows):
            logger.info(f"Judged {idx}/{len(a1_rows)} items...")

    elapsed_s = time.perf_counter() - t_start
    logger.info(f"Completed judging all {len(a1_rows)} items in {elapsed_s:.2f}s ({elapsed_s/len(a1_rows):.2f}s/item)!")

    # 4. Save item-by-item CSV artifact
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "item_id", "video_id", "model", "lang", "term", "explanation",
        "a1_R", "a1_D", "a2_R", "a2_D", "judge_7b_R", "judge_7b_D", "judge_reasoning"
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(judge_records_for_csv)
    logger.info(f"Item-by-item judge decisions saved to {out_csv.resolve()}")

    # 5. Compute Metrics on Common Double-Labeled Subset (N=115)
    a1_R_115 = [a1_rows[i]["R"] for i in common_ids]
    a1_D_115 = [a1_rows[i]["D"] for i in common_ids]
    a2_R_115 = [a2_rows[i]["R"] for i in common_ids]
    a2_D_115 = [a2_rows[i]["D"] for i in common_ids]
    j_R_115 = [judge_predictions[i]["judge_R"] for i in common_ids]
    j_D_115 = [judge_predictions[i]["judge_D"] for i in common_ids]

    human_vs_human_R = compute_metrics(a1_R_115, a2_R_115)
    human_vs_human_D = compute_metrics(a1_D_115, a2_D_115)

    judge_vs_a1_R_115 = compute_metrics(j_R_115, a1_R_115)
    judge_vs_a1_D_115 = compute_metrics(j_D_115, a1_D_115)

    judge_vs_a2_R_115 = compute_metrics(j_R_115, a2_R_115)
    judge_vs_a2_D_115 = compute_metrics(j_D_115, a2_D_115)

    # 6. Compute Metrics on Full Corpus (N=233 vs Annotator 1)
    all_a1_R = [a1_rows[i]["R"] for i in a1_rows]
    all_a1_D = [a1_rows[i]["D"] for i in a1_rows]
    all_j_R = [judge_predictions[i]["judge_R"] for i in a1_rows]
    all_j_D = [judge_predictions[i]["judge_D"] for i in a1_rows]

    judge_vs_a1_full_R = compute_metrics(all_j_R, all_a1_R)
    judge_vs_a1_full_D = compute_metrics(all_j_D, all_a1_D)

    report = {
        "benchmark": "BOUN HPC NVIDIA A40 LLM-as-a-Judge vs Human Annotator Agreement",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "vram_allocated_gb": round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
        },
        "model": "Qwen2.5-7B-Instruct (Zero-Shot Academic Rubric)",
        "double_labeled_subset_n115": {
            "human1_vs_human2_baseline": {
                "R": human_vs_human_R,
                "D": human_vs_human_D
            },
            "judge_7b_vs_annotator1": {
                "R": judge_vs_a1_R_115,
                "D": judge_vs_a1_D_115
            },
            "judge_7b_vs_annotator2": {
                "R": judge_vs_a2_R_115,
                "D": judge_vs_a2_D_115
            }
        },
        "full_corpus_n233": {
            "judge_7b_vs_annotator1": {
                "R": judge_vs_a1_full_R,
                "D": judge_vs_a1_full_D
            }
        }
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Full LLM-Judge Human Agreement Report saved to {out_json.resolve()}")
    logger.info("=" * 80)
    logger.info("SUMMARY RESULTS (N=115 Double-Labeled Benchmark):")
    logger.info(f"Human 1 vs Human 2 (Baseline): κ_R = {human_vs_human_R['kappa']} | PABAK = {human_vs_human_R['pabak']} | Agree = {human_vs_human_R['agreement_pct']}%")
    logger.info(f"Judge 7B vs Human 1:          κ_R = {judge_vs_a1_R_115['kappa']} | PABAK = {judge_vs_a1_R_115['pabak']} | Agree = {judge_vs_a1_R_115['agreement_pct']}%")
    logger.info(f"Judge 7B vs Human 2:          κ_R = {judge_vs_a2_R_115['kappa']} | PABAK = {judge_vs_a2_R_115['pabak']} | Agree = {judge_vs_a2_R_115['agreement_pct']}%")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
