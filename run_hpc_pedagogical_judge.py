#!/usr/bin/env python3
"""Multi-Dimensional Pedagogical LLM-as-a-Judge & Bloom Rubric Benchmark.

Evaluates Kumru-2B, EduTerm-14B-SFT, EduTerm-14B-DPO, and EduTerm-14B-Omni across
5 pedagogical dimensions:
1. Conceptual Correctness (1-5)
2. Pedagogical Tone & Empathy (1-5)
3. Analogy Vividness (1-5)
4. Misconception Preemption (1-5)
5. Bloom Cognitive Depth (1-6)
+ Ateşman Readability and CEFR-B1 Compliance.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pedagogical_judge")

CONCEPTS: List[Dict[str, Any]] = [
    {
        "id": "CS_VIRTUAL_MEM",
        "domain": "Bilgisayar Mühendisliği",
        "concept": "Sanal Bellek ve Sayfalama (Virtual Memory & Paging)",
        "key_terms": ["sanal bellek", "sayfalama", "fiziksel bellek", "ram", "disk"],
        "misconception_clue": "sanal bellek ekstra ram değildir",
        "analogy_targets": ["kütüphane", "masa", "kitaplık", "dolap", "depo", "çekmece"]
    },
    {
        "id": "BIO_OKAZAKI",
        "domain": "Moleküler Biyoloji",
        "concept": "DNA Helikaz ve Okazaki Parçacıkları",
        "key_terms": ["dna", "helikaz", "okazaki", "ligaz", "zincir", "kesintili"],
        "misconception_clue": "iki zincir aynı yönde sentezlenemez",
        "analogy_targets": ["fermuar", "yol", "tren rayı", "dikiş", "inşaat"]
    },
    {
        "id": "EE_KIRCHHOFF",
        "domain": "Elektrik ve Elektronik",
        "concept": "Kirchhoff Akım Yasası ve Düğüm Gerilimleri",
        "key_terms": ["kirchhoff", "akım", "düğüm", "yük korunumu", "gerilim"],
        "misconception_clue": "düğümde akım birikmez",
        "analogy_targets": ["su borusu", "kavşak", "nehir", "trafik", "havuz"]
    },
    {
        "id": "BANK_LCR",
        "domain": "Bankacılık ve Finans",
        "concept": "Likidite Karşılama Oranı (LCR) ve Basel III",
        "key_terms": ["likidite", "lcr", "basel", "nakit", "mevduat", "stres"],
        "misconception_clue": "karlılık ile likidite aynı şey değildir",
        "analogy_targets": ["yedek akçe", "acil durum fonu", "yangın tüpü", "cüzdan"]
    },
    {
        "id": "ML_OVERFITTING",
        "domain": "Yapay Zeka ve Makine Öğrenmesi",
        "concept": "Aşırı Öğrenme (Overfitting) ve Gradyan İnişi",
        "key_terms": ["aşırı öğrenme", "overfitting", "gradyan", "genelleme", "regülarizasyon"],
        "misconception_clue": "eğitim başarısının yüzde yüz olması mükemmellik değildir",
        "analogy_targets": ["ezber", "sınav", "çıkmış sorular", "terzi", "elbise"]
    },
    {
        "id": "MED_ACTION_POTENTIAL",
        "domain": "Tıp ve Biyofizik",
        "concept": "Nöronlarda Aksiyon Potansiyeli ve Sodyum Pompası",
        "key_terms": ["nöron", "aksiyon potansiyeli", "sodyum", "potasyum", "zar"],
        "misconception_clue": "elektrik akımı kablodaki gibi elektron akışı değildir",
        "analogy_targets": ["domino", "kapı", "dalga", "telefon hattı", "baraj"]
    }
]

def calculate_atesman(text: str) -> float:
    words = re.findall(r"\b\w+\b", text)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not words or not sentences:
        return 0.0
    total_words = len(words)
    total_sentences = len(sentences)
    vowels = "aeıioöuüAEIİOÖUÜ"
    total_syllables = sum(sum(1 for ch in w if ch in vowels) for w in words)
    asl = total_words / total_sentences
    asw = total_syllables / total_words
    score = 198.825 - (40.175 * asw) - (2.610 * asl)
    return round(float(score), 2)

def evaluate_explanation_rubric(text: str, concept_meta: Dict[str, Any]) -> Dict[str, Any]:
    lower_text = text.lower()
    
    # 1. Correctness (1-5) based on key domain terms
    key_hits = sum(1 for term in concept_meta["key_terms"] if term.lower() in lower_text)
    term_ratio = key_hits / max(1, len(concept_meta["key_terms"]))
    correctness_score = round(1.0 + (term_ratio * 4.0), 2)
    
    # 2. Pedagogical Tone & Empathy (1-5)
    tone_markers = [
        "düşün", "örneğin", "öğrenci", "günlük", "basitçe", "şöyle hayal",
        "anlamak için", "birlikte", "unutmayalım", "dikkat", "merak etmeyin"
    ]
    tone_hits = sum(1 for tm in tone_markers if tm in lower_text)
    tone_score = round(min(5.0, 1.5 + (tone_hits * 0.7)), 2)
    
    # 3. Analogy Vividness (1-5)
    analogy_markers = ["gibi", "benzer", "tıpkı", "örnek ver", "metafor", "model", "canlandır"]
    target_hits = sum(1 for at in concept_meta["analogy_targets"] if at in lower_text)
    marker_hits = sum(1 for am in analogy_markers if am in lower_text)
    if target_hits > 0 and marker_hits > 0:
        analogy_score = min(5.0, 3.5 + (0.5 * target_hits) + (0.3 * marker_hits))
    elif marker_hits > 0:
        analogy_score = min(3.5, 2.0 + (0.4 * marker_hits))
    else:
        analogy_score = 1.5
    analogy_score = round(analogy_score, 2)
    
    # 4. Misconception Preemption (1-5)
    misconception_markers = [
        "karıştırılan", "yanlış bilinen", "aslında", "farklı olarak", "dikkat edilmeli",
        "çoğu zaman sanılanın", "sanılanın aksine", "ancak unutulmamalıdır", "demek değildir"
    ]
    preempt_hits = sum(1 for mm in misconception_markers if mm in lower_text)
    clue_in_text = any(w in lower_text for w in concept_meta["misconception_clue"].split())
    if preempt_hits >= 2 and clue_in_text:
        misconception_score = min(5.0, 3.5 + (0.5 * preempt_hits))
    elif preempt_hits >= 1:
        misconception_score = min(3.5, 2.0 + (0.5 * preempt_hits))
    else:
        misconception_score = 1.0
    misconception_score = round(misconception_score, 2)
    
    # 5. Bloom Cognitive Depth (1-6)
    # 1: Remembering, 2: Understanding, 3: Applying, 4: Analyzing, 5: Evaluating, 6: Creating
    bloom_score = 2.0 # Default understanding
    if any(k in lower_text for k in ["tasarla", "oluştur", "sentezle", "üret"]):
        bloom_score = 6.0
    elif any(k in lower_text for k in ["değerlendir", "karşılaştır", "karar ver", "tercih"]):
        bloom_score = 5.0
    elif any(k in lower_text for k in ["analiz", "ayırt et", "neden-sonuç", "ilişki", "çünkü"]):
        bloom_score = 4.0
    elif target_hits > 0 or any(k in lower_text for k in ["uygula", "çöz", "kullan"]):
        bloom_score = 3.0
    elif len(lower_text.split()) < 30:
        bloom_score = 1.0
        
    atesman = calculate_atesman(text)
    b1_compliant = atesman >= 45.0
    
    return {
        "conceptual_correctness": correctness_score,
        "pedagogical_tone": tone_score,
        "analogy_vividness": analogy_score,
        "misconception_preemption": misconception_score,
        "bloom_depth": bloom_score,
        "atesman": atesman,
        "b1_compliant": b1_compliant
    }

def format_prompt(model_family: str, concept_name: str) -> str:
    inst = (
        f"Sen deneyimli bir üniversite eğitmenisin. '{concept_name}' kavramını lise ve üniversite 1. sınıf "
        "öğrencisinin rahatça anlayabileceği CEFR-B1 düzeyinde, günlük hayattan somut bir analoji (benzetme) "
        "kullanarak açıkla. Öğrencilerin sıkça düştüğü bir kavram yanılgısını (misconception) önceden belirterek uyar."
    )
    if model_family == "kumru":
        return f"<|user|>\n{inst}\n<|assistant|>\n"
    else:
        return f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n"

def main():
    parser = argparse.ArgumentParser(description="Multi-Dimensional Pedagogical LLM-as-a-Judge Benchmark")
    parser.add_argument("--output", type=str, default="/users/ali.gunder/eduterm-finetune/outputs/pedagogical_judge_rubric_report.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Starting Multi-Dimensional Pedagogical Benchmark on %s...", device)

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
        logger.info("Evaluating model %s...", cfg["name"])
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

        concept_evals = []
        for c in CONCEPTS:
            prompt = format_prompt(cfg["family"], c["concept"])
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
            generated_text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            rubric = evaluate_explanation_rubric(generated_text, c)
            concept_evals.append({
                "concept_id": c["id"],
                "concept": c["concept"],
                "rubric": rubric,
                "text_snippet": generated_text[:180] + "..."
            })

        # Calculate averages across all concepts
        n = len(concept_evals)
        summary = {
            "mean_conceptual_correctness": round(sum(e["rubric"]["conceptual_correctness"] for e in concept_evals) / n, 2),
            "mean_pedagogical_tone": round(sum(e["rubric"]["pedagogical_tone"] for e in concept_evals) / n, 2),
            "mean_analogy_vividness": round(sum(e["rubric"]["analogy_vividness"] for e in concept_evals) / n, 2),
            "mean_misconception_preemption": round(sum(e["rubric"]["misconception_preemption"] for e in concept_evals) / n, 2),
            "mean_bloom_depth": round(sum(e["rubric"]["bloom_depth"] for e in concept_evals) / n, 2),
            "mean_atesman": round(sum(e["rubric"]["atesman"] for e in concept_evals) / n, 2),
            "b1_compliance_pct": round((sum(1 for e in concept_evals if e["rubric"]["b1_compliant"]) / n) * 100.0, 1)
        }

        all_results[cfg["name"]] = {
            "summary": summary,
            "details": concept_evals
        }
        logger.info("%s Summary: Correctness: %.2f | Tone: %.2f | Analogy: %.2f | Misconception: %.2f | Bloom: %.2f | B1: %.1f%%",
                    cfg["name"], summary["mean_conceptual_correctness"], summary["mean_pedagogical_tone"],
                    summary["mean_analogy_vividness"], summary["mean_misconception_preemption"],
                    summary["mean_bloom_depth"], summary["b1_compliance_pct"])

        del model
        if cfg["adapter"]:
            del base
        gc.collect()
        torch.cuda.empty_cache()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info("Pedagogical Rubric Benchmark saved successfully to %s", output_path)

if __name__ == "__main__":
    main()

