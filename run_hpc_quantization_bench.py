#!/usr/bin/env python3
"""
HPC Quantization Precision & Compression Degradation Benchmark
Evaluates EduTerm-14B and Kumru-2B across 3 hardware precision regimes on NVIDIA A40:
1. Native Precision (FP16 / BF16)
2. INT8 Quantization (bitsandbytes LLM.int8())
3. 4-Bit NF4 Quantization (bitsandbytes QLoRA NF4)
Measures VRAM footprint, Latency, and Terminology Extraction F1 retention.
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

EVAL_SNIPPETS = [
    {
        "text": "Moleküler analizde polimeraz zincir reaksiyonu ve termal döngü adımları jel elektroforezi ile doğrulanır.",
        "gold": ["polimeraz zincir reaksiyonu", "termal döngü", "jel elektroforezi"]
    },
    {
        "text": "Elektrik devrelerinde düğüm gerilimleri analizi Kirchhoff akım yasası denklemleri ile kurulur ve süper düğüm hesaplanır.",
        "gold": ["düğüm gerilimleri analizi", "kirchhoff akım yasası", "süper düğüm"]
    },
    {
        "text": "Veri yapılarında dengeli ikili arama ağacı kullanılarak logaritmik arama karmaşıklığı elde edilir.",
        "gold": ["dengeli ikili arama ağacı", "logaritmik arama karmaşıklığı"]
    }
]

MODELS_CONFIG = [
    {
        "id": "Kumru-2B",
        "family": "kumru",
        "base_path": "/users/ali.gunder/models/Kumru-2B",
        "adapter_path": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_kumru_2b_cefr",
    },
    {
        "id": "EduTerm-14B",
        "family": "qwen",
        "base_path": "/users/ali.gunder/models/Qwen2.5-14B-Instruct",
        "adapter_path": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora",
    }
]

def format_prompt(model_family, lecture_text):
    if model_family == "kumru":
        return f"<|im_start|>user\nAşağıdaki ders metnindeki terimleri JSON listesi olarak çıkar:\nMetin: {lecture_text}\nYanıtını SADECE [\"terim1\", \"terim2\"] formatında ver.<|im_end|>\n<|im_start|>assistant\n["
    else:
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

def benchmark_precision_tier(cfg, precision_name, load_kwargs):
    logging.info(f"Loading {cfg['id']} in {precision_name}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(cfg["base_path"], trust_remote_code=True, **load_kwargs)
    if os.path.exists(cfg["adapter_path"]):
        model = PeftModel.from_pretrained(model, cfg["adapter_path"])
    model.eval()

    vram_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
    latencies = []
    f1_list = []

    for item in EVAL_SNIPPETS:
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
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(input_ids, **gen_kwargs)
        latencies.append(time.time() - t0)
        new_tokens = out[0][input_ids.shape[1]:]
        resp = tokenizer.decode(new_tokens, skip_special_tokens=True)
        extracted = clean_and_parse_json(resp)
        _, _, f1 = evaluate_metrics(extracted, item["gold"])
        f1_list.append(f1)

    del model
    del tokenizer
    torch.cuda.empty_cache()

    avg_f1 = sum(f1_list) / len(f1_list)
    avg_lat_ms = (sum(latencies) / len(latencies)) * 1000

    return {
        "vram_gb": vram_gb,
        "avg_latency_ms": round(avg_lat_ms, 1),
        "f1_score_pct": round(avg_f1 * 100, 1)
    }

def benchmark_model_quantization(cfg):
    results = {}
    
    # 1. Native FP16/BF16
    dtype = torch.bfloat16 if cfg["family"] == "qwen" else torch.float16
    fp_res = benchmark_precision_tier(cfg, "Native_FP16", {"torch_dtype": dtype, "device_map": "auto"})
    results["Native_FP16"] = fp_res

    # 2. INT8
    int8_cfg = BitsAndBytesConfig(load_in_8bit=True)
    int8_res = benchmark_precision_tier(cfg, "INT8_Quantized", {"quantization_config": int8_cfg, "device_map": "auto"})
    results["INT8_Quantized"] = int8_res

    # 3. 4-bit NF4
    nf4_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    nf4_res = benchmark_precision_tier(cfg, "NF4_4Bit_Quantized", {"quantization_config": nf4_cfg, "device_map": "auto"})
    results["NF4_4Bit_Quantized"] = nf4_res

    # Compute F1 Retention
    native_f1 = fp_res["f1_score_pct"]
    for tier in ["INT8_Quantized", "NF4_4Bit_Quantized"]:
        retention = (results[tier]["f1_score_pct"] / max(1e-5, native_f1)) * 100
        results[tier]["f1_retention_pct"] = round(min(120.0, retention), 1)

    logging.info(f"Quantization Summary for {cfg['id']}: {results}")
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/quantization_compression_report.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    all_reports = {}

    for cfg in MODELS_CONFIG:
        res = benchmark_model_quantization(cfg)
        all_reports[cfg["id"]] = res

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)

    logging.info(f"Quantization benchmark complete. Saved to {args.output}")

if __name__ == "__main__":
    main()

