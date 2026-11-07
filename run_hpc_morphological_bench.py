#!/usr/bin/env python3
"""
HPC Morphological & Suffix Robustness Benchmark
Evaluates EduTerm-14B, Trendyol-8B, and Kumru-2B across 4 agglutinative morphological variations:
1. Root / Bare Form (Yalın Hâl)
2. Case Suffixes (Dative, Locative, Ablative: -e, -de, -den)
3. Genitive & Possessive (Tamlayan / İyelik: -nin, -inin, -ı)
4. Copula / Predicative (Ek-fiil: -dir, -tur)
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

# 4 morphological test suites with base gold terms
MORPH_SUITES = [
    {
        "id": "MORPH-SUITE-1",
        "base_gold": ["polimeraz zincir reaksiyonu", "termal döngü", "jel elektroforezi"],
        "variations": {
            "bare": "Moleküler analizde polimeraz zincir reaksiyonu ve termal döngü adımları tamamlandıktan sonra jel elektroforezi uygulanır.",
            "case_suffixes": "Moleküler analizde polimeraz zincir reaksiyonuna ve termal döngüye geçildikten sonra jel elektroforezinden veriler toplanır.",
            "possessive_genitive": "Moleküler analizde polimeraz zincir reaksiyonunun ve termal döngünün çıktısı jel elektroforezinin kalitesiyle doğrulanır.",
            "copula_predicative": "Kullanılan yöntem polimeraz zincir reaksiyonudur; izlenen protokol ise termal döngüdür ve son aşama jel elektroforezidir."
        }
    },
    {
        "id": "MORPH-SUITE-2",
        "base_gold": ["düğüm gerilimleri analizi", "eşdeğer devre", "süper düğüm"],
        "variations": {
            "bare": "Devre teorisinde düğüm gerilimleri analizi ile eşdeğer devre modeli kurulur ve süper düğüm hesaplanır.",
            "case_suffixes": "Devre teorisinde düğüm gerilimleri analizine başvurularak eşdeğer devrede çözüm aranır ve süper düğüme odaklanılır.",
            "possessive_genitive": "Devre teorisinde düğüm gerilimleri analizinin prensibi ve eşdeğer devrenin parametreleri süper düğümün yapısını belirler.",
            "copula_predicative": "Bugün uyguladığımız teknik düğüm gerilimleri analizidir, elde edilen sonuç eşdeğer devredir ve kritik bileşen süper düğümdür."
        }
    },
    {
        "id": "MORPH-SUITE-3",
        "base_gold": ["ikili arama ağacı", "dengeli ağaç", "logaritmik karmaşıklık"],
        "variations": {
            "bare": "Veri yapılarında ikili arama ağacı dengeli ağaç prensibiyle tasarlandığında logaritmik karmaşıklık elde edilir.",
            "case_suffixes": "Veri yapılarında ikili arama ağacına yeni eleman eklendiğinde dengeli ağaçta logaritmik karmaşıklığa ulaşılır.",
            "possessive_genitive": "Veri yapılarında ikili arama ağacının yüksekliği dengeli ağacın yapısıyla logaritmik karmaşıklığın sınırlarında kalır.",
            "copula_predicative": "Buradaki veri yapısı ikili arama ağacıdır, uygulanan model dengeli ağaçtır ve elde edilen başarım logaritmik karmaşıklıktır."
        }
    }
]

def format_prompt(model_family, lecture_text):
    if model_family == "kumru":
        return f"<|im_start|>user\nAşağıdaki Türkçe metinden temel teknik terimleri YALIN KÖK HALLERİYLE çıkar:\nMetin: {lecture_text}\nYanıtını SADECE JSON listesi olarak [\"terim1\", \"terim2\"] ver.<|im_end|>\n<|im_start|>assistant\n["
    elif model_family == "trendyol":
        return f"<|im_start|>system\nSen uzman bir Türkçe dilbilim ve terim çıkarım asistanısın.<|im_end|>\n<|im_start|>user\nMetindeki teknik terimleri yalın halde JSON listesi olarak ver:\n{lecture_text}<|im_end|>\n<|im_start|>assistant\n["
    else: # qwen
        return f"<|im_start|>system\nSen pedagojik terim analiz uzmanısın. Terimleri yalın kök formlarında JSON listesi olarak ver.<|im_end|>\n<|im_start|>user\nMetin: {lecture_text}\nTerimleri çıkar:<|im_end|>\n<|im_start|>assistant\n["

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

def evaluate_morph_match(extracted, gold):
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

def benchmark_model_morphology(cfg):
    logging.info(f"Loading {cfg['id']} for Morphological Robustness Benchmark...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg["base_path"], trust_remote_code=True, **cfg["load_kwargs"])
    if os.path.exists(cfg["adapter_path"]):
        logging.info(f"  Loading LoRA adapter: {cfg['adapter_path']}")
        model = PeftModel.from_pretrained(model, cfg["adapter_path"])
    model.eval()

    variation_scores = {
        "bare": {"f1": []},
        "case_suffixes": {"f1": []},
        "possessive_genitive": {"f1": []},
        "copula_predicative": {"f1": []}
    }

    start_time = time.time()

    for suite in MORPH_SUITES:
        gold = suite["base_gold"]
        for var_name, var_text in suite["variations"].items():
            prompt = format_prompt(cfg["family"], var_text)
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            input_ids = inputs["input_ids"]
            gen_kwargs = {
                "max_new_tokens": 100,
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
            _, _, f1 = evaluate_morph_match(extracted, gold)
            variation_scores[var_name]["f1"].append(f1)

    del model
    del tokenizer
    torch.cuda.empty_cache()

    duration = time.time() - start_time
    summary = {"duration_sec": round(duration, 1)}
    
    bare_f1 = sum(variation_scores["bare"]["f1"]) / len(variation_scores["bare"]["f1"])
    summary["bare_root_f1_pct"] = round(bare_f1 * 100, 1)
    
    for var_name in ["case_suffixes", "possessive_genitive", "copula_predicative"]:
        avg_f1 = sum(variation_scores[var_name]["f1"]) / len(variation_scores[var_name]["f1"])
        retention = (avg_f1 / max(1e-5, bare_f1)) * 100 if bare_f1 > 0 else 0.0
        summary[f"{var_name}_f1_pct"] = round(avg_f1 * 100, 1)
        summary[f"{var_name}_retention_pct"] = round(min(150.0, retention), 1)

    # Average Suffix Robustness Index (0 - 100)
    retentions = [summary[f"{v}_retention_pct"] for v in ["case_suffixes", "possessive_genitive", "copula_predicative"]]
    summary["overall_suffix_robustness_index"] = round(sum(retentions) / len(retentions), 1)

    logging.info(f"Morphological Summary for {cfg['id']}: {summary}")
    return summary

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/morphological_robustness_report.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    all_reports = {}

    for cfg in MODELS_CONFIG:
        res = benchmark_model_morphology(cfg)
        all_reports[cfg["id"]] = res

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)

    logging.info(f"Morphological robustness benchmark complete. Saved to {args.output}")

if __name__ == "__main__":
    main()

