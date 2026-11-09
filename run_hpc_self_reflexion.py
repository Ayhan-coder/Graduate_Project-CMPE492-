#!/usr/bin/env python3
"""
Autonomous Pedagogical Self-Reflexion & Agentic Refinement Benchmark
Evaluates EduTerm-14B and EduTerm-14B-Omni across a 2-stage Reflexion loop:
  Stage 1: Direct Generation (Initial technical explanation)
  Stage 2: Pedagogical Self-Critique (Detecting jargon, complex syntax, missing analogies)
  Stage 3: Refinement (Rewriting with CEFR-B1 target constraints)

Measures:
  - Delta Atesman Readability (S_initial -> S_refined)
  - Average Sentence Length (ASL) reduction
  - Technical Term Retention vs. Jargon Elimination
  - CEFR-B1 Compliance Rate
"""

import os
import sys
import json
import logging
import time
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

BASE_MODEL_PATH = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"

MODELS_CONFIG = [
    {
        "id": "EduTerm-14B-CEFR",
        "adapter_path": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora"
    },
    {
        "id": "EduTerm-14B-Omni",
        "adapter_path": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_omni_merged"
    }
]

BENCHMARK_CONCEPTS = [
    {
        "id": "CS_VIRTUAL_MEM",
        "domain": "Bilgisayar Mühendisliği",
        "concept": "Sanal Bellek ve Sayfalama (Virtual Memory & Paging)",
        "prompt": "İşletim sistemlerindeki sanal bellek ve sayfalama (paging) mekanizmasını açıkla.",
        "core_terms": ["bellek", "sayfa", "ram", "disk", "adres"],
        "heavy_jargon_to_simplify": ["page fault", "tlb", "swapping", "parçalanma"]
    },
    {
        "id": "BIO_OKAZAKI",
        "domain": "Moleküler Biyoloji",
        "concept": "DNA Replikasyonu ve Okazaki Parçacıkları",
        "prompt": "DNA ikileşmesinde kesintili zincir ve Okazaki parçacıklarının oluşum mantığını açıkla.",
        "core_terms": ["dna", "zincir", "parça", "enzim", "iplik"],
        "heavy_jargon_to_simplify": ["polimeraz", "ligaz", "primer", "replikasyon çatalı"]
    },
    {
        "id": "EE_KIRCHHOFF",
        "domain": "Elektrik-Elektronik",
        "concept": "Düğüm Gerilimleri ve Kirchhoff Akım Yasası",
        "prompt": "Elektrik devrelerinde Kirchhoff akım yasası (KCL) ve yük korunumunu açıkla.",
        "core_terms": ["akım", "düğüm", "yük", "korun", "devre"],
        "heavy_jargon_to_simplify": ["iletkenlik matrisi", "süper düğüm", "referans potansiyeli"]
    },
    {
        "id": "FIN_RESERVE_LCR",
        "domain": "Bankacılık & Finans",
        "concept": "Zorunlu Karşılık ve Likidite Karşılama Oranı (LKR)",
        "prompt": "Bankacılıkta munzam karşılık oranı ile Basel III Likidite Karşılama Oranının (LKR) farkını ve önemini açıkla.",
        "core_terms": ["zorunlu karşılık", "likidite", "merkez bankası", "kredi", "risk"],
        "heavy_jargon_to_simplify": ["munzam", "stres testi", "yüksek kaliteli likit varlık", "hqlu"]
    },
    {
        "id": "ML_REGULARIZATION",
        "domain": "Yapay Zeka & Veri Bilimi",
        "concept": "Aşırı Öğrenme (Overfitting) ve Düzenlileştirme",
        "prompt": "Makine öğrenmesinde modelin aşırı öğrenmesini (overfitting) ve L2 düzenlileştirmenin (weight decay) bunu nasıl engellediğini açıkla.",
        "core_terms": ["model", "aşırı öğrenme", "ezber", "genelleme", "hata"],
        "heavy_jargon_to_simplify": ["ceza terimi", "kayıp fonksiyonu", "gradyan inişi", "hiperparametre"]
    },
    {
        "id": "PM_EVM",
        "domain": "Proje Yönetimi",
        "concept": "Kazanılmış Değer Analizi (Earned Value Management)",
        "prompt": "Proje yönetiminde Kazanılmış Değer Analizini (EVM) ve CPI ile SPI göstergelerini açıkla.",
        "core_terms": ["maliyet", "bütçe", "zaman", "ilerleme", "proje"],
        "heavy_jargon_to_simplify": ["kümülatif sapma", "tamamlanma tahmini", "pv", "ev", "ac"]
    }
]

def compute_readability_metrics(text: str) -> dict:
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
    words = re.findall(r'\b[a-zA-ZçğıöşüÇĞİÖŞÜ]+\b', text)
    if not sentences or not words:
        return {"atesman": 0.0, "asl": 0.0, "word_count": 0, "b1_compliant": False}
    
    num_sentences = max(1, len(sentences))
    num_words = max(1, len(words))
    num_syllables = sum(len(re.findall(r'[aeıioöuüAEIİOÖUÜ]', w)) for w in words)
    asl = num_words / num_sentences
    asw = num_syllables / num_words
    score = 198.825 - (40.175 * asw) - (2.610 * asl)
    atesman = round(float(max(0.0, min(100.0, score))), 2)
    b1_comp = (40.0 <= atesman <= 75.0)
    
    return {
        "atesman": atesman,
        "asl": round(asl, 2),
        "word_count": num_words,
        "b1_compliant": b1_comp
    }

def evaluate_term_retention(text: str, core_terms: list) -> float:
    text_lower = text.lower()
    hits = sum(1 for t in core_terms if t.lower() in text_lower)
    return round((hits / max(1, len(core_terms))) * 100.0, 1)

def run_reflexion_for_model(model, tokenizer, model_id: str) -> dict:
    logging.info(f"=== Running Agentic Self-Reflexion on {model_id} ===")
    results = []

    for item in BENCHMARK_CONCEPTS:
        logging.info(f"\nEvaluating Concept: [{item['concept']}] ({item['domain']})")

        # Stage 1: Direct Generation (Round 1)
        messages_r1 = [
            {"role": "system", "content": "Sen alanında uzman teknik bir eğitim asistanısın. Verilen teknik kavramı öğrencilere anlaşılır şekilde açıkla."},
            {"role": "user", "content": item["prompt"]}
        ]
        prompt_r1 = tokenizer.apply_chat_template(messages_r1, tokenize=False, add_generation_prompt=True)
        inputs_r1 = tokenizer(prompt_r1, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_r1 = model.generate(
                **inputs_r1,
                max_new_tokens=180,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_r1 = tokenizer.decode(out_r1[0][inputs_r1["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        m_r1 = compute_readability_metrics(gen_r1)
        ret_r1 = evaluate_term_retention(gen_r1, item["core_terms"])

        # Stage 2: Reflexion & Critique
        critique_prompt = (
            f"Aşağıdaki teknik açıklamayı CEFR-B1 pedagojik dil standartlarına göre denetle:\n"
            f"\"\"\"{gen_r1}\"\"\"\n\n"
            f"Kriterler:\n"
            f"1. Lise/hazırlık öğrencisi için gereksiz ağır akademik jargon veya yabancı terim var mı?\n"
            f"2. Cümleler çok uzun veya karmaşık mı?\n"
            f"3. Konuyu somutlaştıran günlük bir yaşam benzetmesi (analoji) eksik mi?\n"
            f"Eksikleri maddeler halinde kısaca eleştir."
        )
        messages_critique = [
            {"role": "system", "content": "Sen titiz bir pedagojik denetçisin. Açıklamaları CEFR-B1 standartlarına göre nesnel biçimde eleştirirsin."},
            {"role": "user", "content": critique_prompt}
        ]
        prompt_c = tokenizer.apply_chat_template(messages_critique, tokenize=False, add_generation_prompt=True)
        inputs_c = tokenizer(prompt_c, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_c = model.generate(
                **inputs_c,
                max_new_tokens=140,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        critique_text = tokenizer.decode(out_c[0][inputs_c["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # Stage 3: Self-Correction & Refinement (Round 2)
        refine_prompt = (
            f"Teknik Konu: {item['concept']}\n"
            f"İlk Açıklaman: \"\"\"{gen_r1}\"\"\"\n"
            f"Öz-Eleştirin:\n{critique_text}\n\n"
            f"Talimat: Yaptığın bu öz-eleştiriyi dikkate alarak açıklamayı baştan yaz. Açıklamayı CEFR-B1 düzeyinde (kısa ve net cümleler, sade Türkçe, açıklayıcı bir benzetme) hazırla, ancak temel teknik doğruluğu asla bozma."
        )
        messages_r2 = [
            {"role": "system", "content": "Sen pedagojik açıdan kusursuz, sade ve CEFR-B1 düzeyinde Türkçe eğitim veren bir asistansın."},
            {"role": "user", "content": refine_prompt}
        ]
        prompt_r2 = tokenizer.apply_chat_template(messages_r2, tokenize=False, add_generation_prompt=True)
        inputs_r2 = tokenizer(prompt_r2, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_r2 = model.generate(
                **inputs_r2,
                max_new_tokens=180,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_r2 = tokenizer.decode(out_r2[0][inputs_r2["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        m_r2 = compute_readability_metrics(gen_r2)
        ret_r2 = evaluate_term_retention(gen_r2, item["core_terms"])

        delta_atesman = round(m_r2["atesman"] - m_r1["atesman"], 2)
        asl_reduction = round(m_r1["asl"] - m_r2["asl"], 2)

        results.append({
            "id": item["id"],
            "domain": item["domain"],
            "concept": item["concept"],
            "round1_direct": {
                "text_snippet": gen_r1[:200] + "...",
                "atesman": m_r1["atesman"],
                "asl": m_r1["asl"],
                "b1_compliant": m_r1["b1_compliant"],
                "term_retention_pct": ret_r1
            },
            "critique_snippet": critique_text[:200] + "...",
            "round2_refined": {
                "text_snippet": gen_r2[:200] + "...",
                "atesman": m_r2["atesman"],
                "asl": m_r2["asl"],
                "b1_compliant": m_r2["b1_compliant"],
                "term_retention_pct": ret_r2
            },
            "delta_atesman": delta_atesman,
            "asl_reduction": asl_reduction
        })
        logging.info(f"[{item['id']}] Atesman: {m_r1['atesman']} -> {m_r2['atesman']} (Delta: +{delta_atesman}) | ASL: {m_r1['asl']} -> {m_r2['asl']} | Term Ret: {ret_r2}%")

    mean_atesman_r1 = round(sum(r["round1_direct"]["atesman"] for r in results) / len(results), 2)
    mean_atesman_r2 = round(sum(r["round2_refined"]["atesman"] for r in results) / len(results), 2)
    mean_asl_r1 = round(sum(r["round1_direct"]["asl"] for r in results) / len(results), 2)
    mean_asl_r2 = round(sum(r["round2_refined"]["asl"] for r in results) / len(results), 2)
    mean_b1_r1 = round((sum(1 for r in results if r["round1_direct"]["b1_compliant"]) / len(results)) * 100.0, 1)
    mean_b1_r2 = round((sum(1 for r in results if r["round2_refined"]["b1_compliant"]) / len(results)) * 100.0, 1)
    mean_term_ret_r1 = round(sum(r["round1_direct"]["term_retention_pct"] for r in results) / len(results), 2)
    mean_term_ret_r2 = round(sum(r["round2_refined"]["term_retention_pct"] for r in results) / len(results), 2)

    return {
        "model_id": model_id,
        "round1_direct_summary": {
            "mean_atesman": mean_atesman_r1,
            "mean_asl": mean_asl_r1,
            "b1_compliance_pct": mean_b1_r1,
            "mean_term_retention_pct": mean_term_ret_r1
        },
        "round2_refined_summary": {
            "mean_atesman": mean_atesman_r2,
            "mean_asl": mean_asl_r2,
            "b1_compliance_pct": mean_b1_r2,
            "mean_term_retention_pct": mean_term_ret_r2
        },
        "reflexion_gains": {
            "atesman_gain": round(mean_atesman_r2 - mean_atesman_r1, 2),
            "asl_reduction": round(mean_asl_r1 - mean_asl_r2, 2),
            "b1_compliance_gain_pct": round(mean_b1_r2 - mean_b1_r1, 1),
            "term_retention_delta_pct": round(mean_term_ret_r2 - mean_term_ret_r1, 2)
        },
        "evaluations": results
    }

def main():
    output_dir = "/users/ali.gunder/eduterm-finetune/outputs"
    report_path = os.path.join(output_dir, "self_reflexion_benchmark_report.json")
    os.makedirs(output_dir, exist_ok=True)

    logging.info("Loading Tokenizer and Base Qwen2.5-14B in NF4 Quantization...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    all_reports = {}

    for cfg in MODELS_CONFIG:
        logging.info(f"Loading adapter: {cfg['adapter_path']}")
        model = PeftModel.from_pretrained(base_model, cfg["adapter_path"])
        model.eval()
        
        rep = run_reflexion_for_model(model, tokenizer, cfg["id"])
        all_reports[cfg["id"]] = rep

        del model
        torch.cuda.empty_cache()

    final_payload = {
        "experiment": "BOUN HPC NVIDIA A40 Agentic Pedagogical Self-Reflexion Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_evaluated": list(all_reports.keys()),
        "results": all_reports
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2)

    logging.info(f"Self-Reflexion Benchmark completed! Report saved to {report_path}")

if __name__ == "__main__":
    main()

