#!/usr/bin/env python3
"""
HPC Code-Switching & Spoken Cross-Lingual Terminology Extraction Benchmark
Evaluates EduTerm-14B, Trendyol-8B, and Kumru-2B across 3 language modes:
1. Pure Turkish (Tam Türkçe Terimler)
2. Pure English (Tam İngilizce Terimler)
3. Spoken Code-Switching (Türkçe Sentaks İçinde İngilizce Terim Kullanımı)
"""

import os
import sys
import json
import logging
import time
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
        "id": "Trendyol-8B",
        "family": "trendyol",
        "base_path": "/users/ali.gunder/models/Trendyol-LLM-8B-T1",
        "adapter_path": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_trendyol_8b_cefr",
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

CODE_SWITCH_SUITES = [
    {
        "id": "CSW-ML-1",
        "domain": "Yapay Zeka ve Makine Öğrenmesi",
        "modes": {
            "pure_tr": {
                "text": "Bugün yapay sinir ağlarında gradyan inişi ve geriye yayılım algoritmasını işleyeceğiz. Aşırı öğrenmeyi önlemek için düzenlileştirme ve öğrenme oranı ayarlaması kritik önem taşır.",
                "gold": ["gradyan inişi", "geriye yayılım", "aşırı öğrenme", "düzenlileştirme", "öğrenme oranı"]
            },
            "pure_en": {
                "text": "Today we cover gradient descent and backpropagation in deep neural networks. Regularization and learning rate scheduling are essential to prevent overfitting.",
                "gold": ["gradient descent", "backpropagation", "deep neural networks", "regularization", "learning rate", "overfitting"]
            },
            "code_switched": {
                "text": "Arkadaşlar bugünkü dersimizde deep learning modellerinde gradient descent ve backpropagation algoritmalarını işleyeceğiz. Özellikle overfitting olmasın diye regularization tekniklerini ve learning rate scheduling yöntemini devreye alıyoruz.",
                "gold": ["deep learning", "gradient descent", "backpropagation", "overfitting", "regularization", "learning rate scheduling"]
            }
        }
    },
    {
        "id": "CSW-SE-1",
        "domain": "Yazılım Mühendisliği ve Çevik Yönetim",
        "modes": {
            "pure_tr": {
                "text": "Çevik yazılım geliştirmede ürün iş listesindeki kullanıcı hikayeleri koşu planlama toplantısında seçilir ve kalan iş grafiği ile ilerleme izlenir.",
                "gold": ["çevik yazılım geliştirme", "ürün iş listesi", "kullanıcı hikayeleri", "koşu planlama", "kalan iş grafiği"]
            },
            "pure_en": {
                "text": "In agile software engineering, user stories from the product backlog are selected during sprint planning and tracked via burn-down charts.",
                "gold": ["agile software engineering", "user stories", "product backlog", "sprint planning", "burn-down charts"]
            },
            "code_switched": {
                "text": "Agile takımlarda sprint planning sırasında product backlog içerisindeki user story'ler sprint backlog'a çekilir ve velocity burn-down chart üzerinden kontrol edilir.",
                "gold": ["agile", "sprint planning", "product backlog", "user story", "sprint backlog", "velocity", "burn-down chart"]
            }
        }
    },
    {
        "id": "CSW-BIO-1",
        "domain": "Biyomühendislik",
        "modes": {
            "pure_tr": {
                "text": "Hücresel döngüde DNA ikileşmesi sırasında helikaz enzimi çift sarmalı açar ve kesintili zincirde Okazaki parçacıkları ligaz ile birleştirilir.",
                "gold": ["dna ikileşmesi", "helikaz enzimi", "çift sarmal", "kesintili zincir", "okazaki parçacıkları", "ligaz"]
            },
            "pure_en": {
                "text": "During DNA replication, helicase unwinds the double helix, and Okazaki fragments on the lagging strand are joined by DNA ligase.",
                "gold": ["dna replication", "helicase", "double helix", "okazaki fragments", "lagging strand", "dna ligase"]
            },
            "code_switched": {
                "text": "Derste gördüğümüz üzere DNA replication sürecinde helicase enzimi çift sarmalı ayırır ve lagging strand üzerinde oluşan okazaki fragment'lar DNA ligase vasıtasıyla bağlanır.",
                "gold": ["dna replication", "helicase", "lagging strand", "okazaki fragment", "dna ligase"]
            }
        }
    }
]

def format_prompt(model_family, lecture_text):
    if model_family == "kumru":
        return f"<|im_start|>user\nAşağıdaki ders anlatımında geçen teknik ve bilimsel terimleri (Türkçe veya İngilizce ayrımı yapmaksızın) JSON listesi olarak çıkar:\nMetin: {lecture_text}\nYanıtını SADECE [\"terim1\", \"terim2\"] formatında ver.<|im_end|>\n<|im_start|>assistant\n["
    elif model_family == "trendyol":
        return f"<|im_start|>system\nSen uzman bir teknik terim çıkarım asistanısın.<|im_end|>\n<|im_start|>user\nMetindeki teknik terimleri JSON listesi olarak ver:\n{lecture_text}<|im_end|>\n<|im_start|>assistant\n["
    else: # qwen
        return f"<|im_start|>system\nSen pedagojik terim analiz uzmanısın. Yalnızca geçerli bir JSON listesi döndür.<|im_end|>\n<|im_start|>user\nMetin: {lecture_text}\nTerimleri çıkar:<|im_end|>\n<|im_start|>assistant\n["

def clean_and_parse_json(text):
    text = text.strip()
    if not text.startswith("["):
        idx = text.find("[")
        if idx != -1:
            text = text[idx:]
        else:
            text = "[" + text
    idx_close = text.rfind("]")
    if idx_close != -1:
        text = text[:idx_close+1]
    else:
        text = text + "]"
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        import re
        matches = re.findall(r'"([^"]+)"', text)
        if matches:
            return [m.strip() for m in matches if m.strip()]
    return []

def normalize_term(t):
    return t.lower().replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c").strip()

def evaluate_metrics(extracted, gold):
    norm_gold = [normalize_term(g) for g in gold]
    norm_ext = [normalize_term(e) for e in extracted]
    tp = 0
    for e in norm_ext:
        if any(e == g or e in g or g in e for g in norm_gold):
            tp += 1
    prec = tp / max(1, len(norm_ext))
    rec = tp / max(1, len(norm_gold))
    f1 = (2 * prec * rec) / max(1e-5, (prec + rec))
    return prec, rec, f1

def benchmark_model_codeswitch(cfg):
    logging.info(f"Loading {cfg['id']} for Code-Switching Benchmark...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg["base_path"], trust_remote_code=True, **cfg["load_kwargs"])
    if os.path.exists(cfg["adapter_path"]):
        logging.info(f"  Loading LoRA adapter: {cfg['adapter_path']}")
        model = PeftModel.from_pretrained(model, cfg["adapter_path"])
    model.eval()

    mode_scores = {
        "pure_tr": {"prec": [], "rec": [], "f1": []},
        "pure_en": {"prec": [], "rec": [], "f1": []},
        "code_switched": {"prec": [], "rec": [], "f1": []}
    }

    start_time = time.time()

    for suite in CODE_SWITCH_SUITES:
        for mode, data in suite["modes"].items():
            prompt = format_prompt(cfg["family"], data["text"])
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            input_ids = inputs["input_ids"]
            gen_kwargs = {
                "max_new_tokens": 128,
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
            extracted = clean_and_parse_json(resp)
            p, r, f1 = evaluate_metrics(extracted, data["gold"])
            mode_scores[mode]["prec"].append(p)
            mode_scores[mode]["rec"].append(r)
            mode_scores[mode]["f1"].append(f1)

    del model
    del tokenizer
    torch.cuda.empty_cache()

    duration = time.time() - start_time
    summary = {"duration_sec": round(duration, 1)}

    for mode in ["pure_tr", "pure_en", "code_switched"]:
        avg_p = sum(mode_scores[mode]["prec"]) / len(mode_scores[mode]["prec"])
        avg_r = sum(mode_scores[mode]["rec"]) / len(mode_scores[mode]["rec"])
        avg_f1 = sum(mode_scores[mode]["f1"]) / len(mode_scores[mode]["f1"])
        summary[mode] = {
            "precision_pct": round(avg_p * 100, 1),
            "recall_pct": round(avg_r * 100, 1),
            "f1_pct": round(avg_f1 * 100, 1)
        }

    # Code-Switching Resilience Ratio (CS-F1 / max(TR-F1, EN-F1))
    max_monolingual = max(summary["pure_tr"]["f1_pct"], summary["pure_en"]["f1_pct"])
    resilience = (summary["code_switched"]["f1_pct"] / max(1e-5, max_monolingual)) * 100
    summary["code_switching_resilience_pct"] = round(min(150.0, resilience), 1)

    logging.info(f"Code-Switching Summary for {cfg['id']}: {summary}")
    return summary

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/code_switching_benchmark_report.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    all_reports = {}

    for cfg in MODELS_CONFIG:
        res = benchmark_model_codeswitch(cfg)
        all_reports[cfg["id"]] = res

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)

    logging.info(f"Code-Switching benchmark complete. Saved to {args.output}")

if __name__ == "__main__":
    main()

