#!/usr/bin/env python3
"""Hardware & Inference Latency Profiling Benchmark on NVIDIA A40.

Measures precise hardware metrics across model scales and quantization levels:
1. Kumru-2B (FP16)
2. EduTerm-14B (NF4 / 4-bit)
3. EduTerm-14B (INT8 / 8-bit)
4. EduTerm-14B (BF16 / Unquantized)

Metrics collected with PyTorch CUDA Events:
- TTFT (Time-To-First-Token in ms)
- Throughput (tokens/second)
- Total Latency (ms)
- Peak VRAM (GB)
- Model Weights Memory (GB)
- KV-Cache Allocation (MB)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("inference_profiling")

PROMPT = (
    "Aşağıdaki kavramı üniversite birinci sınıf öğrencisine açıkla: "
    "Sanal bellek ve sayfalama mekanizması modern işletim sistemlerinde nasıl çalışır?"
)

def benchmark_configuration(
    model_name: str,
    model_path: str,
    quant_mode: str,
    device: str
) -> Dict[str, Any]:
    logger.info("Setting up profiling for %s (quant=%s)...", model_name, quant_mode)
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    torch_dtype = torch.bfloat16
    if quant_mode == "nf4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    elif quant_mode == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quant_mode == "fp16":
        torch_dtype = torch.float16
    elif quant_mode == "bf16":
        torch_dtype = torch.bfloat16

    load_start = time.time()
    if bnb_config:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True
        )
    load_time = round(time.time() - load_start, 2)
    model.eval()

    weights_vram_gb = round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
    logger.info("%s loaded in %.2fs. Weights VRAM: %.2f GB", model_name, load_time, weights_vram_gb)

    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    # Warmup passes
    for _ in range(2):
        with torch.no_grad():
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
    torch.cuda.synchronize()

    # Benchmark loop (10 trials)
    num_trials = 10
    target_tokens = 128
    ttft_list = []
    latency_list = []
    throughput_list = []

    for trial in range(num_trials):
        torch.cuda.reset_peak_memory_stats()
        
        # 1. Measure TTFT (First token latency)
        t_start = torch.cuda.Event(enable_timing=True)
        t_first = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)

        t_start.record()
        with torch.no_grad():
            first_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        t_first.record()

        # 2. Measure Full Generation Latency
        with torch.no_grad():
            full_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=target_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        t_end.record()
        torch.cuda.synchronize()

        ttft_ms = t_start.elapsed_time(t_first)
        total_ms = t_start.elapsed_time(t_end)
        gen_tokens = full_out.shape[1] - input_ids.shape[1]
        throughput = (gen_tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0

        ttft_list.append(ttft_ms)
        latency_list.append(total_ms)
        throughput_list.append(throughput)

    peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
    kv_cache_mb = round(max(0.0, (peak_vram_gb - weights_vram_gb) * 1024), 1)

    mean_ttft = round(sum(ttft_list) / len(ttft_list), 1)
    mean_lat = round(sum(latency_list) / len(latency_list), 1)
    mean_tps = round(sum(throughput_list) / len(throughput_list), 1)

    logger.info("%s (%s): TTFT=%.1f ms | Speed=%.1f tok/s | Peak VRAM=%.2f GB",
                model_name, quant_mode, mean_ttft, mean_tps, peak_vram_gb)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model_name": model_name,
        "quant_mode": quant_mode,
        "weights_vram_gb": weights_vram_gb,
        "peak_vram_gb": peak_vram_gb,
        "kv_cache_mb": kv_cache_mb,
        "mean_ttft_ms": mean_ttft,
        "mean_total_latency_ms": mean_lat,
        "throughput_tokens_per_sec": mean_tps,
        "latency_per_token_ms": round(mean_lat / target_tokens, 2)
    }

def main():
    parser = argparse.ArgumentParser(description="Inference Latency & VRAM Profiling Benchmark")
    parser.add_argument("--output", type=str, default="/users/ali.gunder/eduterm-finetune/outputs/inference_latency_profiling_report.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Initializing Real-World Inference Profiling on %s...", device)

    configs = [
        {
            "name": "Kumru-2B",
            "path": "/users/ali.gunder/models/Kumru-2B",
            "quant": "fp16"
        },
        {
            "name": "EduTerm-14B (NF4)",
            "path": "/users/ali.gunder/models/Qwen2.5-14B-Instruct",
            "quant": "nf4"
        },
        {
            "name": "EduTerm-14B (INT8)",
            "path": "/users/ali.gunder/models/Qwen2.5-14B-Instruct",
            "quant": "int8"
        },
        {
            "name": "EduTerm-14B (BF16 Unquantized)",
            "path": "/users/ali.gunder/models/Qwen2.5-14B-Instruct",
            "quant": "bf16"
        }
    ]

    results = {}
    for cfg in configs:
        key = f"{cfg['name']}_{cfg['quant']}"
        res = benchmark_configuration(cfg["name"], cfg["path"], cfg["quant"], device)
        results[key] = res

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Profiling results saved successfully to %s", output_path)

if __name__ == "__main__":
    main()

