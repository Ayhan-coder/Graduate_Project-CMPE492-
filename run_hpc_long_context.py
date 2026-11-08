#!/usr/bin/env python3
"""
HPC Long-Context Needle-in-a-Haystack & Position Saturation Benchmark
Evaluates EduTerm-14B and Kumru-2B across context window expansions:
1K, 2K, 4K, 8K tokens.
Tests key domain term retrieval when placed at:
- Head (10% of lecture)
- Middle (50% of lecture - Lost in the Middle test)
- Tail (90% of lecture)
"""

import os
import sys
import json
import logging
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_CONFIG = [
    {
        "id": "Kumru-2B",
        "family": "kumru",
        "base_path": "/users/ali.gunder/models/Kumru-2B",
        "adapter_path": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_kumru_2b_cefr",
        "max_ctx": 4096,
        "load_kwargs": {"torch_dtype": torch.float16, "device_map": "auto"}
    },
    {
        "id": "EduTerm-14B",
        "family": "qwen",
        "base_path": "/users/ali.gunder/models/Qwen2.5-14B-Instruct",
        "adapter_path": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora",
        "max_ctx": 8192,
        "use_nf4": True
    },
    {
        "id": "EduTerm-14B-Omni",
        "family": "qwen",
        "base_path": "/users/ali.gunder/models/Qwen2.5-14B-Instruct",
        "adapter_path": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_omni_merged",
        "max_ctx": 8192,
        "use_nf4": True
    }
]

BACKGROUND_PARAGRAPHS = [
    "Üniversitemiz mühendislik fakültesinde bu dönem yürütülen temel bilim ve teknoloji seminerlerinde genel akademik ilkeler ele alınmaktadır. Öğrencilerin ders takibi, araştırma metodolojileri ve laboratuvar güvenlik kuralları düzenli olarak değerlendirilir. Çalışma gruplarının haftalık raporlama standartları belirlenmiştir.",
    "Proje planlama süreçlerinde zaman yönetimi ve kaynak tahsisi en kritik başarı faktörleri arasında yer alır. Ekiplerin periyodik toplantılarla görev dağılımını güncellemesi, olası risk faktörlerinin erkenden tespit edilerek bertaraf edilmesini sağlar.",
    "Akademik araştırmalarda literatür taramasının kapsamlı yapılması, özgün hipotezlerin geliştirilmesi açısından temel teşkil eder. Veri toplama aşamasında etik kurallara uyulması ve sonuçların tarafsız biçimde analiz edilmesi bilimsel güvenilirliğin ön şartıdır.",
    "Teknolojik dönüşümün eğitim sistemlerine entegrasyonu, uzaktan eğitim platformlarının ve dijital kütüphanelerin etkin kullanımını zorunlu kılmaktadır. Öğrencilerin bilgiye hızlı erişimi öğrenme verimliliğini doğrudan artırmaktadır."
]

NEEDLE_TERMS = [
    {"needle": "Kazanılmış Değer Analizi", "domain": "Proje Yönetimi"},
    {"needle": "Okazaki Parçacıkları", "domain": "Moleküler Biyoloji"},
    {"needle": "Kirchhoff Akım Yasası", "domain": "Elektrik-Elektronik"},
    {"needle": "Likidite Karşılama Oranı", "domain": "Bankacılık & Finans"},
    {"needle": "Sendikasyon Kredisi", "domain": "Bankacılık & Finans"}
]

def generate_haystack(target_tokens, needle_info, position_pct, tokenizer):
    needle_sentence = f" Bugünkü uzmanlık konumuzda özellikle {needle_info['needle']} kavramının matematiksel ve pratik temelleri üzerinde duracağız. "
    haystack = []
    current_tokens = 0
    bg_idx = 0
    
    while current_tokens < target_tokens:
        p = BACKGROUND_PARAGRAPHS[bg_idx % len(BACKGROUND_PARAGRAPHS)]
        haystack.append(p)
        current_tokens += len(tokenizer.tokenize(p))
        bg_idx += 1
        
    insert_idx = int(len(haystack) * (position_pct / 100.0))
    haystack.insert(insert_idx, needle_sentence)
    full_text = " ".join(haystack)
    return full_text

def format_prompt(model_family, lecture_text):
    if model_family == "kumru":
        return f"<|im_start|>user\nAşağıdaki uzun ders metninde geçen temel teknik uzmanlık terimini JSON listesi olarak çıkar:\nMetin: {lecture_text}\nYanıtını SADECE [\"terim\"] formatında ver.<|im_end|>\n<|im_start|>assistant\n["
    else:
        return f"<|im_start|>system\nSen pedagojik terim analiz uzmanısın. Metindeki uzmanlık terimini JSON listesi olarak ver.<|im_end|>\n<|im_start|>user\nMetin: {lecture_text}\nTerimleri çıkar:<|im_end|>\n<|im_start|>assistant\n["

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

def benchmark_long_context(cfg):
    logging.info(f"Loading {cfg['id']} for Long-Context Haystack Benchmark...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(cfg.get("load_kwargs", {}))
    if cfg.get("use_nf4", False):
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(cfg["base_path"], trust_remote_code=True, **load_kwargs)
    if os.path.exists(cfg["adapter_path"]):
        logging.info(f"  Loading LoRA adapter: {cfg['adapter_path']}")
        model = PeftModel.from_pretrained(model, cfg["adapter_path"])
    model.eval()

    context_lengths = [1024, 2048, 4096] if cfg["max_ctx"] <= 4096 else [1024, 2048, 4096, 8192]
    positions = [10, 50, 90] # Head, Middle, Tail

    results = {}

    for length in context_lengths:
        len_key = f"{length}_tokens"
        results[len_key] = {"head_10pct": [], "middle_50pct": [], "tail_90pct": []}
        pos_keys = {10: "head_10pct", 50: "middle_50pct", 90: "tail_90pct"}
        
        for pos in positions:
            pos_key = pos_keys[pos]
            hits = 0
            for item in NEEDLE_TERMS:
                text = generate_haystack(length, item, pos, tokenizer)
                prompt = format_prompt(cfg["family"], text)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg["max_ctx"]).to(model.device)
                input_ids = inputs["input_ids"]
                gen_kwargs = {
                    "max_new_tokens": 64,
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
                
                norm_needle = normalize_term(item["needle"])
                if any(norm_needle in normalize_term(e) or normalize_term(e) in norm_needle for e in extracted):
                    hits += 1
            accuracy_pct = round((hits / len(NEEDLE_TERMS)) * 100, 1)
            results[len_key][pos_key] = accuracy_pct

    del model
    del tokenizer
    torch.cuda.empty_cache()

    # Calculate Middle Retention Index (Middle accuracy / max(Head, Tail))
    total_mid = sum(results[f"{l}_tokens"]["middle_50pct"] for l in context_lengths)
    total_extremes = sum((results[f"{l}_tokens"]["head_10pct"] + results[f"{l}_tokens"]["tail_90pct"]) / 2.0 for l in context_lengths)
    middle_retention = round((total_mid / max(1e-5, total_extremes)) * 100, 1)
    results["middle_context_retention_index"] = middle_retention

    logging.info(f"Long-Context Summary for {cfg['id']}: {results}")
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/long_context_haystack_report.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    all_reports = {}

    for cfg in MODELS_CONFIG:
        res = benchmark_long_context(cfg)
        all_reports[cfg["id"]] = res

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)

    logging.info(f"Long-Context Haystack benchmark complete. Saved to {args.output}")

if __name__ == "__main__":
    main()

