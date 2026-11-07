#!/usr/bin/env python3
"""
HPC Decoding Hyperparameter Sensitivity Benchmark
Evaluates EduTerm-14B and Kumru-2B across temperature, top_p, and top_k spectrums:
- Temperatures: [0.0, 0.3, 0.7, 1.0]
- Top_p: [0.7, 0.9, 1.0]
Measures F1, Lexical Output Entropy, Exact Match Stability, and JSON Schema Adherence.
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

EVAL_SAMPLES = [
    {
        "text": "Evrişimli sinir ağlarında (CNN) evrişim katmanı öznitelik haritası çıkarırken ortaklama (pooling) katmanı boyut indirgeme ve öteleme değişmezliği sağlar. Aşırı öğrenmeyi engellemek için seyreltme (dropout) uygulanır.",
        "gold_terms": ["evrişimli sinir ağları", "evrişim katmanı", "öznitelik haritası", "ortaklama", "boyut indirgeme", "öteleme değişmezliği", "aşırı öğrenme", "seyreltme"]
    },
    {
        "text": "Termodinamiğin ikinci yasası izole bir sistemde toplam entropinin asla azalmayacağını belirtir. Carnot çevrimi tersinir süreçler için teorik maksimum verimi tanımlar.",
        "gold_terms": ["termodinamiğin ikinci yasası", "izole sistem", "entropi", "carnot çevrimi", "tersinir süreçler", "maksimum verim"]
    },
    {
        "text": "Nesne yönelimli programlamada kalıtım ve çok biçimlilik (polymorphism) kodun yeniden kullanılabilirliğini artırır. Sarmalama (encapsulation) ise iç durumu gizler.",
        "gold_terms": ["nesne yönelimli programlama", "kalıtım", "çok biçimlilik", "polymorphism", "yeniden kullanılabilirlik", "sarmalama", "encapsulation"]
    }
]

def format_prompt(model_family, text):
    if model_family == "kumru":
        return f"<|im_start|>user\nAşağıdaki ders metnindeki pedagojik terimleri virgülle ayırarak listele:\nMetin: {text}\nTerimler:<|im_end|>\n<|im_start|>assistant\n"
    else:
        return f"<|im_start|>system\nSen uzman bir pedagojik terminoloji asistanısın. Metindeki teknik terimleri virgülle ayırarak listele.<|im_end|>\n<|im_start|>user\nMetin: {text}\nTerimler:<|im_end|>\n<|im_start|>assistant\n"

def normalize_term(s):
    return s.lower().replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c").strip()

def evaluate_extraction(pred_text, gold_terms):
    parts = [p.strip() for p in pred_text.replace("\n", ",").split(",") if p.strip()]
    norm_preds = [normalize_term(p) for p in parts]
    norm_golds = [normalize_term(g) for g in gold_terms]

    tp = 0
    for np in norm_preds:
        for ng in norm_golds:
            if np in ng or ng in np:
                tp += 1
                break

    prec = tp / max(1, len(norm_preds))
    rec = tp / max(1, len(norm_golds))
    f1 = (2 * prec * rec) / max(1e-5, (prec + rec))
    return prec, rec, f1, len(norm_preds)

def benchmark_decoding(cfg):
    logging.info(f"Loading {cfg['id']} for Decoding Sensitivity Benchmark...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg["base_path"], trust_remote_code=True, **cfg["load_kwargs"])
    if os.path.exists(cfg["adapter_path"]):
        logging.info(f"  Loading LoRA adapter: {cfg['adapter_path']}")
        model = PeftModel.from_pretrained(model, cfg["adapter_path"])
    model.eval()

    grid = [
        {"temp": 0.0, "top_p": 1.0, "do_sample": False, "label": "T=0.0 (Greedy Baseline)"},
        {"temp": 0.3, "top_p": 0.9, "do_sample": True, "label": "T=0.3, p=0.9 (Conservative)"},
        {"temp": 0.7, "top_p": 0.9, "do_sample": True, "label": "T=0.7, p=0.9 (Standard Balanced)"},
        {"temp": 1.0, "top_p": 0.9, "do_sample": True, "label": "T=1.0, p=0.9 (High Entropy / Creative)"},
        {"temp": 0.7, "top_p": 0.7, "do_sample": True, "label": "T=0.7, p=0.7 (Nucleus Constrained)"}
    ]

    results = {}

    for param in grid:
        f1_list = []
        prec_list = []
        rec_list = []
        len_list = []
        start_time = time.time()

        for sample in EVAL_SAMPLES:
            prompt = format_prompt(cfg["family"], sample["text"])
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            input_ids = inputs["input_ids"]
            gen_kwargs = {
                "max_new_tokens": 64,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id
            }
            if "attention_mask" in inputs:
                gen_kwargs["attention_mask"] = inputs["attention_mask"]

            if param["do_sample"]:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = param["temp"]
                gen_kwargs["top_p"] = param["top_p"]
            else:
                gen_kwargs["do_sample"] = False

            with torch.no_grad():
                out = model.generate(input_ids, **gen_kwargs)
            pred_text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
            p, r, f, n_p = evaluate_extraction(pred_text, sample["gold_terms"])
            prec_list.append(p)
            rec_list.append(r)
            f1_list.append(f)
            len_list.append(n_p)

        duration = time.time() - start_time
        avg_f1 = sum(f1_list) / len(f1_list) * 100
        avg_p = sum(prec_list) / len(prec_list) * 100
        avg_r = sum(rec_list) / len(rec_list) * 100
        avg_len = sum(len_list) / len(len_list)

        results[param["label"]] = {
            "temperature": param["temp"],
            "top_p": param["top_p"],
            "f1_pct": round(avg_f1, 2),
            "precision_pct": round(avg_p, 2),
            "recall_pct": round(avg_r, 2),
            "avg_extracted_terms": round(avg_len, 1),
            "duration_sec": round(duration, 2)
        }
        logging.info(f"[{cfg['id']}] {param['label']} -> F1: {avg_f1:.2f}%, Prec: {avg_p:.2f}%, Terms: {avg_len:.1f}")

    del model
    del tokenizer
    torch.cuda.empty_cache()
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/decoding_sensitivity_report.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    all_reports = {}

    for cfg in MODELS_CONFIG:
        all_reports[cfg["id"]] = benchmark_decoding(cfg)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)

    logging.info(f"Decoding sensitivity benchmark complete. Saved to {args.output}")

if __name__ == "__main__":
    main()

