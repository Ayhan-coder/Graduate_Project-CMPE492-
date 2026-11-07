#!/usr/bin/env python3
"""
HPC Cross-Disciplinary Out-of-Distribution (OOD) Transfer Benchmark
Evaluates EduTerm-14B and Kumru-2B across 5 extreme unseen academic domains:
1. Forensic Toxicology (Adli Toksikoloji)
2. Quantum Cryptography (Kuantum Kriptografi)
3. Maritime Law (Deniz Hukuku)
4. Nuclear Reactor Physics (Nükleer Reaktör Fiziği)
5. Financial Econometrics (Finansal Ekonometri)
Computes In-Distribution vs Out-of-Distribution F1 Retention and Generalization Drop.
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
        "id": "EduTerm-14B",
        "family": "qwen",
        "base_path": "/users/ali.gunder/models/Qwen2.5-14B-Instruct",
        "adapter_path": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora",
        "load_kwargs": {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    }
]

OOD_DOMAINS_DATASET = [
    {
        "domain": "Forensic Toxicology",
        "text": "Adli toksikolojide kromatografik kütle spektrometrisi biyolojik örneklerdeki ksenobiyotik metabolitleri ve zehirli bileşikleri nitel olarak teşhis eder.",
        "gold_terms": ["adli toksikoloji", "kromatografik kütle spektrometrisi", "ksenobiyotik metabolitler", "zehirli bileşikler"]
    },
    {
        "domain": "Quantum Cryptography",
        "text": "Kuantum anahtar dağıtımında BB84 protokolü fotonların polarizasyon durumlarını kullanarak kopyalanamazlık teoremi temelinde şifreleme anahtarı üretir.",
        "gold_terms": ["kuantum anahtar dağıtımı", "bb84 protokolü", "polarizasyon durumları", "kopyalanamazlık teoremi", "şifreleme anahtarı"]
    },
    {
        "domain": "Maritime Law",
        "text": "Deniz ticaret hukukunda müşterek avarya gemi ve yükün ortak selametini sağlamak amacıyla kaptanın kasıtlı fedakarlığı sonucu doğan zararların paylaştırılmasıdır.",
        "gold_terms": ["deniz ticaret hukuku", "müşterek avarya", "ortak selamet", "kasıtlı fedakarlık"]
    },
    {
        "domain": "Nuclear Reactor Physics",
        "text": "Nükleer fisyon reaktörlerinde nötron moderatörü hızlı nötronları termal enerji düzeyine yavaşlatırken kontrol çubukları reaktivite katsayısını dengeler.",
        "gold_terms": ["nükleer fisyon", "nötron moderatörü", "hızlı nötronlar", "termal enerji düzeyi", "kontrol çubukları", "reaktivite katsayısı"]
    },
    {
        "domain": "Financial Econometrics",
        "text": "Zaman serisi ekonometrisinde otoregresif koşullu değişen varyans modeli finansal getiri serilerindeki dalgalanma kümelenmelerini tahminler.",
        "gold_terms": ["zaman serisi ekonometrisi", "otoregresif koşullu değişen varyans", "finansal getiri serileri", "dalgalanma kümelenmeleri"]
    }
]

def format_prompt(model_family, text):
    if model_family == "kumru":
        return f"<|im_start|>user\nAşağıdaki ders metnindeki teknik terimleri virgülle ayırarak listele:\nMetin: {text}\nTerimler:<|im_end|>\n<|im_start|>assistant\n"
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
    return prec, rec, f1

def benchmark_ood_transfer(cfg):
    logging.info(f"Loading {cfg['id']} for OOD Cross-Disciplinary Benchmark...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg["base_path"], trust_remote_code=True, **cfg["load_kwargs"])
    if os.path.exists(cfg["adapter_path"]):
        logging.info(f"  Loading LoRA adapter: {cfg['adapter_path']}")
        model = PeftModel.from_pretrained(model, cfg["adapter_path"])
    model.eval()

    domain_results = {}
    f1_list, prec_list, rec_list = [], [], []

    start_time = time.time()

    for item in OOD_DOMAINS_DATASET:
        prompt = format_prompt(cfg["family"], item["text"])
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
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
        pred_text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        p, r, f1 = evaluate_extraction(pred_text, item["gold_terms"])
        
        domain_results[item["domain"]] = {
            "precision_pct": round(p * 100, 2),
            "recall_pct": round(r * 100, 2),
            "f1_pct": round(f1 * 100, 2)
        }
        prec_list.append(p)
        rec_list.append(r)
        f1_list.append(f1)
        logging.info(f"[{cfg['id']}] {item['domain']} -> F1: {f1*100:.2f}%, Prec: {p*100:.2f}%")

    duration = time.time() - start_time
    macro_f1 = (sum(f1_list) / len(f1_list)) * 100
    macro_prec = (sum(prec_list) / len(prec_list)) * 100
    macro_rec = (sum(rec_list) / len(rec_list)) * 100

    # In-Distribution reference baseline (from CEFR engineering benchmarks)
    in_dist_baseline_f1 = 43.8 if cfg["id"] == "Kumru-2B" else 46.7
    ood_retention_index = (macro_f1 / in_dist_baseline_f1) * 100.0

    summary = {
        "model_id": cfg["id"],
        "macro_f1_pct": round(macro_f1, 2),
        "macro_precision_pct": round(macro_prec, 2),
        "macro_recall_pct": round(macro_rec, 2),
        "in_distribution_reference_f1": in_dist_baseline_f1,
        "ood_retention_index": round(ood_retention_index, 2),
        "domain_breakdown": domain_results,
        "duration_sec": round(duration, 2)
    }

    del model
    del tokenizer
    torch.cuda.empty_cache()
    return summary

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/ood_cross_disciplinary_report.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    all_reports = {}

    for cfg in MODELS_CONFIG:
        all_reports[cfg["id"]] = benchmark_ood_transfer(cfg)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)

    logging.info(f"OOD Transfer benchmark complete. Saved to {args.output}")

if __name__ == "__main__":
    main()

