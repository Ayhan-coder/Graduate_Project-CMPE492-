#!/usr/bin/env python3
"""
HPC Cognitive Load & Eye-Tracking Surprisal Proxy Benchmark
Models reading difficulty and cognitive fixation durations (Rayner 1998, Kliegl 2006)
by calculating token-level negative log-likelihood (surprisal) across raw lecture transcripts
versus CEFR-B1 simplified pedagogical explanations.
"""

import os
import sys
import json
import logging
import time
import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LECTURE_PAIRS = [
    {
        "id": "Pair-1 (Deep Learning)",
        "raw_text": "Evrişimli sinir ağlarında konvolüsyonel öznitelik haritası tensör boyutu küçültülürken aşırı uyum riskini azaltmak gayesiyle stokastik seyreltme regülarizasyonu tatbik edilmektedir.",
        "simplified_b1": "Yapay zeka modellerinde resimleri işlerken ezberlemeyi önlemek için bazı bağlantılar rastgele kapatılır."
    },
    {
        "id": "Pair-2 (Thermodynamics)",
        "raw_text": "Termodinamiğin ikinci postülatı mucibince izole termal sistemlerin entropi değişimi adyabatik süreçlerde asgari sıfır olup tersinmezlik mikroskobik düzensizliği tırmandırır.",
        "simplified_b1": "Isı kanunlarına göre kapalı bir ortamda düzensizlik zamanla kendiliğinden artar ve süreçler geri döndürülemez."
    },
    {
        "id": "Pair-3 (Biochemistry)",
        "raw_text": "Replikatif enzim polimeraz trifosfat hidrolizinden türetilen serbest entalpi gradyanını kullanarak 5 üssü ucundan 3 üssü istikametine fosfodiester esterleşmesini katalizler.",
        "simplified_b1": "Hücre bölünürken enzimler enerji harcayarak DNA zincirini adım adım kopyalar ve birleştirir."
    }
]

def calculate_token_surprisal(model, tokenizer, text):
    """
    Computes per-token negative log-likelihood (surprisal in bits) and cognitive reading metrics.
    """
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits # [1, seq_len, vocab_size]
        
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_surprisals = (token_losses / math.log(2)).cpu().tolist() # convert to bits
    
    avg_surprisal = sum(token_surprisals) / max(1, len(token_surprisals))
    perplexity = math.exp(sum(token_losses.cpu().tolist()) / max(1, len(token_losses)))
    
    # Cognitive eye-tracking proxy formulas (Kliegl et al. 2006 / Rayner 1998):
    # Mean Fixation Duration (ms) ≈ Baseline (220ms) + 12.5 * Surprisal (bits)
    # Saccade Regression Probability ≈ 0.05 + 0.02 * max(0, Surprisal - 4.0)
    simulated_fixation_ms = 220.0 + (12.5 * avg_surprisal)
    simulated_regression_prob = min(0.40, 0.05 + (0.02 * max(0.0, avg_surprisal - 4.0)))
    
    return {
        "num_tokens": input_ids.shape[1],
        "perplexity": round(perplexity, 2),
        "mean_surprisal_bits": round(avg_surprisal, 3),
        "simulated_fixation_duration_ms": round(simulated_fixation_ms, 1),
        "simulated_saccade_regression_prob": round(simulated_regression_prob, 3)
    }

def benchmark_cognitive_load():
    base_path = "/users/ali.gunder/models/Kumru-2B"
    logging.info(f"Loading {base_path} for Cognitive Load Surprisal Profiling...")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()

    pairs_results = []
    total_raw_fixation = []
    total_b1_fixation = []
    total_raw_surprisal = []
    total_b1_surprisal = []

    start_time = time.time()

    for pair in LECTURE_PAIRS:
        raw_metrics = calculate_token_surprisal(model, tokenizer, pair["raw_text"])
        b1_metrics = calculate_token_surprisal(model, tokenizer, pair["simplified_b1"])
        
        reduction_fixation_ms = raw_metrics["simulated_fixation_duration_ms"] - b1_metrics["simulated_fixation_duration_ms"]
        relative_reduction_pct = (reduction_fixation_ms / raw_metrics["simulated_fixation_duration_ms"]) * 100.0
        
        pairs_results.append({
            "id": pair["id"],
            "raw_academic_text": raw_metrics,
            "simplified_b1_text": b1_metrics,
            "cognitive_fixation_reduction_ms": round(reduction_fixation_ms, 1),
            "cognitive_load_reduction_pct": round(relative_reduction_pct, 2)
        })
        
        total_raw_fixation.append(raw_metrics["simulated_fixation_duration_ms"])
        total_b1_fixation.append(b1_metrics["simulated_fixation_duration_ms"])
        total_raw_surprisal.append(raw_metrics["mean_surprisal_bits"])
        total_b1_surprisal.append(b1_metrics["mean_surprisal_bits"])

    duration = time.time() - start_time
    avg_raw_fix = sum(total_raw_fixation) / len(total_raw_fixation)
    avg_b1_fix = sum(total_b1_fixation) / len(total_b1_fixation)
    mean_load_reduction_pct = ((avg_raw_fix - avg_b1_fix) / avg_raw_fix) * 100.0

    summary = {
        "model": "Kumru-2B Surprisal Density Profiler",
        "duration_sec": round(duration, 2),
        "mean_raw_academic_surprisal_bits": round(sum(total_raw_surprisal) / len(total_raw_surprisal), 2),
        "mean_simplified_b1_surprisal_bits": round(sum(total_b1_surprisal) / len(total_b1_surprisal), 2),
        "mean_raw_fixation_duration_ms": round(avg_raw_fix, 1),
        "mean_simplified_b1_fixation_duration_ms": round(avg_b1_fix, 1),
        "mean_simulated_cognitive_load_reduction_pct": round(mean_load_reduction_pct, 2),
        "pairs_detail": pairs_results
    }
    
    logging.info(f"Cognitive Load Summary: Mean Fixation Drop {avg_raw_fix:.1f}ms -> {avg_b1_fix:.1f}ms (-{mean_load_reduction_pct:.2f}%)")

    del model
    del tokenizer
    torch.cuda.empty_cache()
    return summary

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/cognitive_load_surprisal_report.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    res = benchmark_cognitive_load()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    logging.info(f"Cognitive load benchmark complete. Saved to {args.output}")

if __name__ == "__main__":
    main()

