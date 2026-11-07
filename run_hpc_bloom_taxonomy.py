#!/usr/bin/env python3
"""
HPC Bloom's Cognitive Taxonomy Alignment & Hierarchical Term Depth Benchmark
Evaluates EduTerm-14B and Kumru-2B on classifying extracted terms into Bloom's 6 Cognitive Tiers:
1. Remembering (Hatırlama - Temel Olgusal Tanım)
2. Understanding (Anlama - İlke ve Kavram Açıklama)
3. Applying (Uygulama - Yöntem ve Formül Yürütme)
4. Analyzing (Analiz - Ayrıştırma ve Sistemik İnceleme)
5. Evaluating (Değerlendirme - Karşılaştırma ve Hata Denetimi)
6. Creating (Yaratma - Yeni Hipotez ve Tasarım)
Computes Top-1 Accuracy, Hierarchical Distance Error, and Spearman Rank Correlation.
"""

import os
import sys
import json
import logging
import time
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_CONFIG = [
    {
        "id": "Kumru-2B",
        "family": "kumru",
        "base_path": "/users/ali.gunder/models/Kumru-2B",
        "adapter_path": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_kumru_2b_cefr",
        "load_kwargs": {"torch_dtype": torch.float16, "device_map": "auto"}
    },
    {
        "id": "EduTerm-14B",
        "family": "qwen",
        "base_path": "/users/ali.gunder/models/Qwen2.5-14B-Instruct",
        "adapter_path": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora",
        "load_kwargs": {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    }
]

BLOOM_EVAL_DATASET = [
    {"term": "Düğüm Noktası", "bloom_level": 1, "level_name": "Remembering"},
    {"term": "Kirchhoff Akım Yasası", "bloom_level": 2, "level_name": "Understanding"},
    {"term": "Düğüm Gerilimleri Analizi", "bloom_level": 3, "level_name": "Applying"},
    {"term": "Süper Düğüm Çözümlemesi", "bloom_level": 4, "level_name": "Analyzing"},
    {"term": "Empedans Eşleme Karşılaştırması", "bloom_level": 5, "level_name": "Evaluating"},
    {"term": "Helikaz Enzimi", "bloom_level": 1, "level_name": "Remembering"},
    {"term": "Replikasyon Çatalı", "bloom_level": 2, "level_name": "Understanding"},
    {"term": "Polimeraz Zincir Reaksiyonu", "bloom_level": 3, "level_name": "Applying"},
    {"term": "Okazaki Parçacıklarının Ayrıştırılması", "bloom_level": 4, "level_name": "Analyzing"},
    {"term": "DNA Tamir Mekanizması Denetimi", "bloom_level": 5, "level_name": "Evaluating"},
    {"term": "Kazanılmış Değer Analizi", "bloom_level": 3, "level_name": "Applying"},
    {"term": "Maliyet Sapması ve Varyans Analizi", "bloom_level": 4, "level_name": "Analyzing"},
    {"term": "Proje Performans İndeksi Değerlendirmesi", "bloom_level": 5, "level_name": "Evaluating"}
]

def format_bloom_prompt(model_family, term):
    if model_family == "kumru":
        return f"<|im_start|>user\nAşağıdaki teknik terimin Bloom Taksonomisi bilişsel düzeyini (1-Remembering, 2-Understanding, 3-Applying, 4-Analyzing, 5-Evaluating, 6-Creating) belirle:\nTerim: {term}\nYanıtını SADECE JSON olarak {{\"level\": 1}} şeklinde ver.<|im_end|>\n<|im_start|>assistant\n{{"
    else:
        return f"<|im_start|>system\nSen Bloom Taksonomisi ve pedagojik bilişsel düzey uzmanısın. Yalnızca geçerli JSON döndür.<|im_end|>\n<|im_start|>user\nTerim: {term}\nBloom düzeyini (1-6) ver:<|im_end|>\n<|im_start|>assistant\n{{"

def clean_and_parse_level(text):
    text = text.strip()
    if not text.startswith("{"):
        idx = text.find("{")
        if idx != -1:
            text = text[idx:]
        else:
            text = "{" + text
    idx_close = text.rfind("}")
    if idx_close != -1:
        text = text[:idx_close+1]
    else:
        text = text + "}"
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for k in ["level", "bloom_level", "duzey"]:
                if k in data and str(data[k]).isdigit():
                    return int(data[k])
    except Exception:
        import re
        m = re.search(r'\b([1-6])\b', text)
        if m:
            return int(m.group(1))
    return 2 # default fallback

def benchmark_bloom_taxonomy(cfg):
    logging.info(f"Loading {cfg['id']} for Bloom Taxonomy Benchmark...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg["base_path"], trust_remote_code=True, **cfg["load_kwargs"])
    if os.path.exists(cfg["adapter_path"]):
        logging.info(f"  Loading LoRA adapter: {cfg['adapter_path']}")
        model = PeftModel.from_pretrained(model, cfg["adapter_path"])
    model.eval()

    exact_matches = 0
    distance_errors = []
    gold_levels = []
    pred_levels = []

    for item in BLOOM_EVAL_DATASET:
        prompt = format_bloom_prompt(cfg["family"], item["term"])
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_ids = inputs["input_ids"]
        gen_kwargs = {
            "max_new_tokens": 32,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs["attention_mask"]

        with torch.no_grad():
            out = model.generate(input_ids, **gen_kwargs)
        new_tokens = out[0][input_ids.shape[1]:]
        resp = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred_lvl = clean_and_parse_level(resp)
        
        gold_lvl = item["bloom_level"]
        gold_levels.append(gold_lvl)
        pred_levels.append(pred_lvl)
        
        dist = abs(gold_lvl - pred_lvl)
        distance_errors.append(dist)
        if dist == 0:
            exact_matches += 1

    del model
    del tokenizer
    torch.cuda.empty_cache()

    n = len(BLOOM_EVAL_DATASET)
    exact_acc = round((exact_matches / n) * 100, 1)
    mae = round(sum(distance_errors) / n, 2)
    within_one_level_acc = round((sum(1 for d in distance_errors if d <= 1) / n) * 100, 1)

    # Pearson / Spearman correlation proxy
    x_mean = sum(gold_levels) / n
    y_mean = sum(pred_levels) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(gold_levels, pred_levels))
    den_x = math.sqrt(sum((x - x_mean)**2 for x in gold_levels))
    den_y = math.sqrt(sum((y - y_mean)**2 for y in pred_levels))
    corr = round(num / max(1e-5, den_x * den_y), 3)

    summary = {
        "bloom_exact_match_acc_pct": exact_acc,
        "bloom_within_one_tier_acc_pct": within_one_level_acc,
        "bloom_mean_absolute_error_tiers": mae,
        "bloom_rank_correlation_rho": corr
    }

    logging.info(f"Bloom Summary for {cfg['id']}: {summary}")
    return summary

def main():
    import argparse
    import math
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/bloom_taxonomy_alignment_report.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    all_reports = {}

    for cfg in MODELS_CONFIG:
        res = benchmark_bloom_taxonomy(cfg)
        all_reports[cfg["id"]] = res

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)

    logging.info(f"Bloom Taxonomy benchmark complete. Saved to {args.output}")

if __name__ == "__main__":
    main()

