#!/usr/bin/env python3
"""
run_hpc_job1_arena_winrate.py
Job 1: Head-to-Head Win-Rate Arena: Base vs. SFT vs. DPO (EduTerm-14B)
Evaluates clarity, analogy grounding, and B1 compliance across standard benchmarks.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Job1-ArenaWinRate")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL_PATH = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
SFT_ADAPTER_PATH = "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora"
DPO_ADAPTER_PATH = "/users/ali.gunder/eduterm-dpo-14b/checkpoints/eduterm_14b_dpo"
OUTPUT_REPORT_PATH = "/users/ali.gunder/eduterm-dpo-14b/outputs/arena_winrate_report.json"

BENCHMARKS = [
    {
        "id": "EE",
        "domain": "Elektrik ve Elektronik",
        "text": "Düğüm gerilimleri analizinde bağımsız düğüm noktaları toprağa göre referans alınarak KCL akım denklemleri kurulur.",
        "terms": ["Düğüm Gerilimleri Analizi", "KCL Akım Denklemleri"]
    },
    {
        "id": "BIO",
        "domain": "Biyomühendislik & Biyoloji",
        "text": "DNA replikasyonu sırasında helikaz enzimi çift zinciri açar ve okazaki parçaları ligaz enzimi ile birleştirilir.",
        "terms": ["DNA Replikasyonu", "Helikaz Enzimi", "Okazaki Parçaları", "Ligaz Enzimi"]
    },
    {
        "id": "CS",
        "domain": "Bilgisayar Mühendisliği",
        "text": "İkili arama ağaçlarında silme ve arama işlemleri dengeli ağaç yapısında logaritmik O(log n) karmaşıklık sunar.",
        "terms": ["İkili Arama Ağacı", "Logaritmik Karmaşıklık"]
    },
    {
        "id": "TOUR",
        "domain": "Turizm & Seyahat",
        "text": "Dinamik paketleme motorunda uçak bileti ve otel kontenjanı birleştirilirken no-show riski için erken rezervasyon sigortası uygulanır.",
        "terms": ["Dinamik Paketleme", "No-Show Riski", "Erken Rezervasyon Sigortası"]
    },
    {
        "id": "ISG",
        "domain": "İş Sağlığı ve Güvenliği",
        "text": "Şantiyede meydana gelen ramak kala olayları kayıt altına alınmalı ve acil durum eylem planı yenilenmelidir.",
        "terms": ["Ramak Kala Olayı", "Acil Durum Eylem Planı"]
    },
    {
        "id": "KVKK",
        "domain": "Kişisel Verilerin Korunması",
        "text": "Veri sorumlusu aydınlatma yükümlülüğünü yerine getirdikten sonra ilgili kişiden açık rıza temin etmelidir.",
        "terms": ["Veri Sorumlusu", "Aydınlatma Yükümlülüğü", "Açık Rıza"]
    },
    {
        "id": "SEC",
        "domain": "Siber Güvenlik",
        "text": "Sıfırıncı gün açığı (zero-day) tespit edildiğinde derhal yama uygulanmalı ve çok faktörlü kimlik doğrulama zorunlu kılınmalıdır.",
        "terms": ["Sıfırıncı Gün Açığı", "Çok Faktörlü Kimlik Doğrulama"]
    },
    {
        "id": "BNK",
        "domain": "Bankacılık & Finans",
        "text": "Bankaların likidite karşılama oranı Basel III standartlarına göre belirlenir ve TCMB zorunlu karşılık oranına tabidir.",
        "terms": ["Likidite Karşılama Oranı", "Basel III", "Zorunlu Karşılık"]
    }
]

def calculate_atesman(text: str) -> float:
    if not text or not text.strip():
        return 60.0
    vowels = "aeıioöuü"
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    if not words:
        return 60.0
    syllables = sum(sum(1 for c in w if c in vowels) for w in words)
    spw = syllables / max(1, len(words))
    wps = len(words) / max(1, len(sentences))
    score = 198.825 - (40.175 * spw) - (2.610 * wps)
    return max(0.0, min(100.0, round(score, 1)))

def has_analogy(text: str) -> bool:
    analogy_keywords = ["analoji", "benzetme", "gibi", "benzer", "düşünelim", "örneğin", "fırın", "restoran", "kamera", "kitap"]
    text_lower = text.lower()
    return any(k in text_lower for k in analogy_keywords)

def term_retention_score(text: str, terms: list) -> float:
    text_lower = text.lower()
    found = sum(1 for t in terms if t.lower() in text_lower or any(word.lower() in text_lower for word in t.split()))
    return round((found / max(1, len(terms))) * 100.0, 1)

def run():
    start_time = time.time()
    logger.info("Initializing Job 1: Head-to-Head Win-Rate Arena (Base vs SFT vs DPO)...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    logger.info("Loading Base Model in 4-bit...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="cuda:0",
        trust_remote_code=True
    )

    models_dict = {}

    # 1. Base Model Evaluation
    logger.info("Generating with Base Model...")
    base_results = []
    for b in BENCHMARKS:
        prompt = (
            f"<|im_start|>system\nSen eğitim konuşmalarını ve teknik terimleri başlangıç seviyesindeki öğrenciler için CEFR B1 düzeyinde sade, analojili ve anlaşılır biçimde açıklayan uzman bir pedagojik asistansın.<|im_end|>\n"
            f"<|im_start|>user\nAşağıdaki ders notunu CEFR B1 seviyesindeki bir üniversite öğrencisi için sadeleştir, teknik terimleri koru ve günlük hayattan bir analojiyle açıkla:\n{b['text']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = base_model.generate(**inputs, max_new_tokens=150, do_sample=False)
        gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        at = calculate_atesman(gen)
        an = has_analogy(gen)
        tr = term_retention_score(gen, b["terms"])
        base_results.append({"domain": b["domain"], "text": gen, "atesman": at, "analogy": an, "retention": tr, "b1": 50 <= at <= 75})
    models_dict["Base"] = base_results

    # 2. SFT Model Evaluation
    logger.info("Attaching SFT LoRA Adapter...")
    sft_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    sft_results = []
    for b in BENCHMARKS:
        prompt = (
            f"<|im_start|>system\nSen eğitim konuşmalarını ve teknik terimleri başlangıç seviyesindeki öğrenciler için CEFR B1 düzeyinde sade, analojili ve anlaşılır biçimde açıklayan uzman bir pedagojik asistansın.<|im_end|>\n"
            f"<|im_start|>user\nAşağıdaki ders notunu CEFR B1 seviyesindeki bir üniversite öğrencisi için sadeleştir, teknik terimleri koru ve günlük hayattan bir analojiyle açıkla:\n{b['text']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = sft_model.generate(**inputs, max_new_tokens=150, do_sample=False)
        gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        at = calculate_atesman(gen)
        an = has_analogy(gen)
        tr = term_retention_score(gen, b["terms"])
        sft_results.append({"domain": b["domain"], "text": gen, "atesman": at, "analogy": an, "retention": tr, "b1": 50 <= at <= 75})
    models_dict["SFT"] = sft_results

    # 3. DPO Model Evaluation
    logger.info("Switching to DPO LoRA Adapter...")
    dpo_model = PeftModel.from_pretrained(base_model, DPO_ADAPTER_PATH)
    dpo_results = []
    for b in BENCHMARKS:
        prompt = (
            f"<|im_start|>system\nSen eğitim konuşmalarını ve teknik terimleri başlangıç seviyesindeki öğrenciler için CEFR B1 düzeyinde sade, analojili ve anlaşılır biçimde açıklayan uzman bir pedagojik asistansın.<|im_end|>\n"
            f"<|im_start|>user\nAşağıdaki ders notunu CEFR B1 seviyesindeki bir üniversite öğrencisi için sadeleştir, teknik terimleri koru ve günlük hayattan bir analojiyle açıkla:\n{b['text']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = dpo_model.generate(**inputs, max_new_tokens=150, do_sample=False)
        gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        at = calculate_atesman(gen)
        an = has_analogy(gen)
        tr = term_retention_score(gen, b["terms"])
        dpo_results.append({"domain": b["domain"], "text": gen, "atesman": at, "analogy": an, "retention": tr, "b1": 50 <= at <= 75})
    models_dict["DPO"] = dpo_results

    # Calculate Metrics and Pairwise Win Rates
    def score_item(item):
        score = 0.0
        if item["b1"]: score += 3.0
        elif 40 <= item["atesman"] <= 85: score += 1.5
        if item["analogy"]: score += 2.0
        score += (item["retention"] / 100.0) * 3.0
        return score

    win_matrix = {"DPO_vs_Base": {"win": 0, "loss": 0, "tie": 0}, "DPO_vs_SFT": {"win": 0, "loss": 0, "tie": 0}, "SFT_vs_Base": {"win": 0, "loss": 0, "tie": 0}}

    for i in range(len(BENCHMARKS)):
        s_base = score_item(base_results[i])
        s_sft = score_item(sft_results[i])
        s_dpo = score_item(dpo_results[i])

        if s_dpo > s_base: win_matrix["DPO_vs_Base"]["win"] += 1
        elif s_dpo < s_base: win_matrix["DPO_vs_Base"]["loss"] += 1
        else: win_matrix["DPO_vs_Base"]["tie"] += 1

        if s_dpo > s_sft: win_matrix["DPO_vs_SFT"]["win"] += 1
        elif s_dpo < s_sft: win_matrix["DPO_vs_SFT"]["loss"] += 1
        else: win_matrix["DPO_vs_SFT"]["tie"] += 1

        if s_sft > s_base: win_matrix["SFT_vs_Base"]["win"] += 1
        elif s_sft < s_base: win_matrix["SFT_vs_Base"]["loss"] += 1
        else: win_matrix["SFT_vs_Base"]["tie"] += 1

    summary = {
        "duration_sec": round(time.time() - start_time, 2),
        "mean_atesman": {
            "Base": round(sum(r["atesman"] for r in base_results)/len(base_results), 1),
            "SFT": round(sum(r["atesman"] for r in sft_results)/len(sft_results), 1),
            "DPO": round(sum(r["atesman"] for r in dpo_results)/len(dpo_results), 1)
        },
        "b1_compliance_pct": {
            "Base": round(sum(1 for r in base_results if r["b1"])/len(base_results)*100, 1),
            "SFT": round(sum(1 for r in sft_results if r["b1"])/len(sft_results)*100, 1),
            "DPO": round(sum(1 for r in dpo_results if r["b1"])/len(dpo_results)*100, 1)
        },
        "analogy_grounding_pct": {
            "Base": round(sum(1 for r in base_results if r["analogy"])/len(base_results)*100, 1),
            "SFT": round(sum(1 for r in sft_results if r["analogy"])/len(sft_results)*100, 1),
            "DPO": round(sum(1 for r in dpo_results if r["analogy"])/len(dpo_results)*100, 1)
        },
        "term_retention_pct": {
            "Base": round(sum(r["retention"] for r in base_results)/len(base_results), 1),
            "SFT": round(sum(r["retention"] for r in sft_results)/len(sft_results), 1),
            "DPO": round(sum(r["retention"] for r in dpo_results)/len(dpo_results), 1)
        },
        "pairwise_win_matrix": win_matrix,
        "dpo_win_rate_vs_sft_pct": round((win_matrix["DPO_vs_SFT"]["win"] / len(BENCHMARKS)) * 100, 1),
        "dpo_win_rate_vs_base_pct": round((win_matrix["DPO_vs_Base"]["win"] / len(BENCHMARKS)) * 100, 1)
    }

    os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Job 1 Completed Successfully! Win rate vs SFT: %.1f%%", summary["dpo_win_rate_vs_sft_pct"])

if __name__ == "__main__":
    run()

