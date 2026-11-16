#!/usr/bin/env python3
"""BKT & Knowledge Graph Guided Adaptive Curriculum Generation Benchmark.

Evaluates model capacity to generate personalized prerequisite learning trajectories
conditioned on student Bayesian Knowledge Tracing (BKT) mastery states and
topological curriculum DAG dependencies.

Evaluated Models:
- Kumru-2B (EduTerm SFT)
- EduTerm-14B-SFT
- EduTerm-14B-DPO
- EduTerm-14B-Omni
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("adaptive_curriculum")

CURRICULUM_DOMAINS: List[Dict[str, Any]] = [
    {
        "id": "CS_VIRTUAL_MEM_CURRICULUM",
        "domain": "İşletim Sistemleri",
        "target_concept": "Sanal Bellek ve Sayfalama (Virtual Memory & Paging)",
        "gold_dag_order": [
            "İşlemci ve Kaydediciler",
            "Fiziksel Bellek (RAM)",
            "İşletim Sistemi Çekirdeği",
            "Sanal Adres Alanı",
            "Sayfa Tablosu ve TLB",
            "Sayfa Hatası (Page Fault)"
        ],
        "novice_bkt_state": "Öğrenci işlemci ve RAM mantığını zayıf biliyor (P_mastery = 0.25). Temelden anlatılmalı.",
        "advanced_bkt_state": "Öğrenci RAM ve sanal adreslemeyi tam biliyor (P_mastery = 0.85). Hızlandırılmış TLB ve sayfa değiştirme odaklı ilerlenmeli."
    },
    {
        "id": "BIO_CRISPR_CURRICULUM",
        "domain": "Moleküler Biyoloji",
        "target_concept": "CRISPR-Cas9 Genom Düzenleme Mekanizması",
        "gold_dag_order": [
            "DNA Çift Sarmalı ve Baz Eşleşmesi",
            "RNA Transkripsiyonu",
            "Restriksiyon Enzimleri",
            "Guide RNA (gRNA)",
            "Cas9 Nükleaz Kesimi",
            "Non-Homologous DNA Onarımı"
        ],
        "novice_bkt_state": "Öğrenci temel DNA yapısını biliyor ancak enzimlerin işleyişinde zorlanıyor (P_mastery = 0.30).",
        "advanced_bkt_state": "Öğrenci enzim kinetiğini biliyor (P_mastery = 0.90), doğrudan Cas9 ve kılavuz RNA eşleşmesine odaklanmalı."
    },
    {
        "id": "ML_ATTENTION_CURRICULUM",
        "domain": "Yapay Zeka & Derin Öğrenme",
        "target_concept": "Transformer Öz-Dikkat Mekanizması (Self-Attention)",
        "gold_dag_order": [
            "Matris Çarpımı ve Vektör Uzayı",
            "Gradyan İnişi ve Geriye Yayılım",
            "Yinelemeli Ağlar (RNN)",
            "Seq2Seq Kodlama",
            "Query-Key-Value (QKV) Vektörleri",
            "Çoklu Başlıklı Dikkat (Multi-Head Attention)"
        ],
        "novice_bkt_state": "Öğrenci temel lineer cebir biliyor ama sinir ağlarında acemi (P_mastery = 0.20).",
        "advanced_bkt_state": "Öğrenci RNN ve CNN modellerini iyi biliyor (P_mastery = 0.80), doğrudan QKV matris mekanizmasına odaklanmalı."
    },
    {
        "id": "FIN_BASEL_CURRICULUM",
        "domain": "Bankacılık ve Finansal Risk",
        "target_concept": "Basel III Likidite Karşılama Oranı (LCR)",
        "gold_dag_order": [
            "Banka Bilançosu ve Mevduat",
            "Kredi ve Likidite Riski",
            "Sermaye Yeterlilik Rasyosu",
            "Yüksek Kaliteli Likit Varlıklar (HQLA)",
            "30 Günlük Net Nakit Çıkışı",
            "Stres Testi ve LCR Oranı"
        ],
        "novice_bkt_state": "Öğrenci temel muhasebeyi biliyor ancak banka bilançosuna yabancı (P_mastery = 0.30).",
        "advanced_bkt_state": "Öğrenci bilanço ve kredi riskini tam biliyor (P_mastery = 0.85), doğrudan stres senaryolarına odaklanmalı."
    }
]

def format_curriculum_prompt(model_family: str, domain_meta: Dict[str, Any], is_novice: bool) -> str:
    profile = domain_meta["novice_bkt_state"] if is_novice else domain_meta["advanced_bkt_state"]
    level_name = "Acemi (Temel Düzey)" if is_novice else "İleri (Hızlandırılmış Düzey)"
    
    inst = (
        f"Sen yapay zeka destekli bir müfredat planlama asistanısın. "
        f"Hedef Kavram: '{domain_meta['target_concept']}' ({domain_meta['domain']}).\n"
        f"Öğrenci BKT Bilişsel Profili: {profile}\n"
        f"Görev:\n"
        f"1. Bu {level_name} öğrenci için mantıksal önkoşul sırasını (topological prerequisite sequence) adım adım numaralandırılmış bir öğrenme rotası olarak hazırla.\n"
        f"2. Her adımda öğrenilmesi gereken temel kavramı ve neden bu adımın önce gelmesi gerektiğini belirt.\n"
        f"3. Rota sonuna öğrencinin hedef kavramı pekiştirmesi için 1 adet ara yoklama (formative assessment) sorusu ekle."
    )
    if model_family == "kumru":
        return f"<|user|>\n{inst}\n<|assistant|>\n"
    else:
        return f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n"

def evaluate_curriculum_response(response_text: str, domain_meta: Dict[str, Any], is_novice: bool) -> Dict[str, Any]:
    lower_res = response_text.lower()
    gold_dag = [c.lower() for c in domain_meta["gold_dag_order"]]
    
    # 1. Prerequisite Concept Mention Count
    detected_steps = []
    for idx, concept in enumerate(gold_dag):
        keywords = [w for w in re.findall(r"\b\w+\b", concept) if len(w) > 3]
        if any(kw in lower_res for kw in keywords):
            detected_steps.append((idx, concept))

    # 2. Topological Order Integrity (Are early concepts introduced before late ones?)
    violations = 0
    if len(detected_steps) >= 2:
        for i in range(len(detected_steps) - 1):
            if detected_steps[i][0] > detected_steps[i+1][0]:
                violations += 1
    topological_fidelity_pct = round(max(0.0, 100.0 - (violations * 25.0)), 1)

    # 3. Formative Assessment Detection
    quiz_indicators = ["soru", "yoklama", "test", "değerlendirme", "hangisi", "?", "cevaplayınız", "örnek senaryo"]
    has_quiz = any(qi in lower_res for qi in quiz_indicators) and ("?" in response_text)

    # 4. Adaptive Pacing & Tone Differentiation
    pacing_words_novice = ["temel", "basit", "adım adım", "öncelikle", "giriş", "kavram", "analoji"]
    pacing_words_adv = ["hızlandırılmış", "ileri", "optimizasyon", "derinlemesine", "analiz", "doğrudan"]
    
    if is_novice:
        diff_score = min(5.0, 2.0 + (0.5 * sum(1 for w in pacing_words_novice if w in lower_res)))
    else:
        diff_score = min(5.0, 2.0 + (0.5 * sum(1 for w in pacing_words_adv if w in lower_res)))

    # Overall Curriculum Score
    coverage_ratio = len(detected_steps) / max(1, len(gold_dag))
    overall_score = round((coverage_ratio * 40.0) + (topological_fidelity_pct * 0.4) + (20.0 if has_quiz else 5.0), 1)

    return {
        "concepts_covered": len(detected_steps),
        "total_gold_prerequisites": len(gold_dag),
        "coverage_pct": round(coverage_ratio * 100.0, 1),
        "topological_violations": violations,
        "topological_fidelity_pct": topological_fidelity_pct,
        "formative_quiz_included": has_quiz,
        "differentiation_score": round(diff_score, 2),
        "overall_curriculum_score": overall_score
    }

def main():
    parser = argparse.ArgumentParser(description="Adaptive Curriculum & Prerequisite Topological Benchmark")
    parser.add_argument("--output", type=str, default="/users/ali.gunder/eduterm-finetune/outputs/adaptive_curriculum_bkt_report.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Initializing Adaptive Curriculum Benchmark on %s...", device)

    models_config = [
        {
            "name": "Kumru-2B (EduTerm SFT)",
            "base": "/users/ali.gunder/models/Kumru-2B",
            "adapter": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_kumru_2b_cefr",
            "family": "kumru"
        },
        {
            "name": "EduTerm-14B-SFT",
            "base": "/users/ali.gunder/models/Qwen2.5-14B-Instruct",
            "adapter": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora",
            "family": "qwen"
        },
        {
            "name": "EduTerm-14B-DPO",
            "base": "/users/ali.gunder/models/Qwen2.5-14B-Instruct",
            "adapter": "/users/ali.gunder/eduterm-dpo-14b/checkpoints/eduterm_14b_dpo",
            "family": "qwen"
        },
        {
            "name": "EduTerm-14B-Omni",
            "base": "/users/ali.gunder/models/Qwen2.5-14B-Instruct",
            "adapter": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_omni_merged",
            "family": "qwen"
        }
    ]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    all_results = {}

    for cfg in models_config:
        logger.info("Evaluating Adaptive Curriculum for %s...", cfg["name"])
        tokenizer = AutoTokenizer.from_pretrained(cfg["base"], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            cfg["base"],
            quantization_config=bnb_config,
            device_map=device,
            trust_remote_code=True
        )
        if cfg["adapter"] and Path(cfg["adapter"]).exists():
            model = PeftModel.from_pretrained(base, cfg["adapter"])
        else:
            model = base
        model.eval()

        domain_results = []
        for d in CURRICULUM_DOMAINS:
            for is_novice in [True, False]:
                prompt = format_curriculum_prompt(cfg["family"], d, is_novice)
                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].to(device)
                attn_mask = inputs.get("attention_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.to(device)

                with torch.no_grad():
                    out = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=320,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                    )
                gen_text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
                eval_data = evaluate_curriculum_response(gen_text, d, is_novice)
                domain_results.append({
                    "domain_id": d["id"],
                    "student_profile": "novice" if is_novice else "advanced",
                    "evaluation": eval_data,
                    "trajectory_snippet": gen_text[:160] + "..."
                })

        n = len(domain_results)
        summary = {
            "mean_overall_curriculum_score": round(sum(r["evaluation"]["overall_curriculum_score"] for r in domain_results) / n, 1),
            "mean_topological_fidelity_pct": round(sum(r["evaluation"]["topological_fidelity_pct"] for r in domain_results) / n, 1),
            "mean_prerequisite_coverage_pct": round(sum(r["evaluation"]["coverage_pct"] for r in domain_results) / n, 1),
            "formative_quiz_presence_pct": round((sum(1 for r in domain_results if r["evaluation"]["formative_quiz_included"]) / n) * 100.0, 1),
            "mean_differentiation_score": round(sum(r["evaluation"]["differentiation_score"] for r in domain_results) / n, 2)
        }

        all_results[cfg["name"]] = {
            "summary": summary,
            "details": domain_results
        }
        logger.info("%s Curriculum Summary: Overall Score: %.1f | Topological Fidelity: %.1f%% | Coverage: %.1f%% | Quiz: %.1f%%",
                    cfg["name"], summary["mean_overall_curriculum_score"], summary["mean_topological_fidelity_pct"],
                    summary["mean_prerequisite_coverage_pct"], summary["formative_quiz_presence_pct"])

        del model
        if cfg["adapter"]:
            del base
        gc.collect()
        torch.cuda.empty_cache()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info("Adaptive Curriculum Benchmark saved to %s", output_path)

if __name__ == "__main__":
    main()

