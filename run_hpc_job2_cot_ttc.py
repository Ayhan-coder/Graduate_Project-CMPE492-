#!/usr/bin/env python3
"""
run_hpc_job2_cot_ttc.py
Job 2: Test-Time Compute (TTC) & Chain-of-Thought (CoT) Reasoning Scaling
Evaluates reasoning depth, majority voting consistency, and term accuracy across sample scales N=1, 2, 4, 8.
"""

import os
import sys
import json
import logging
import time
import re
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Job2-TTCReasoning")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
ADAPTER_PATH = "/users/ali.gunder/eduterm-dpo-14b/checkpoints/eduterm_14b_dpo"
OUTPUT_REPORT_PATH = "/users/ali.gunder/eduterm-dpo-14b/outputs/ttc_reasoning_scaling_report.json"

COMPLEX_SCENARIOS = [
    {
        "id": "TTC-1",
        "domain": "Dağıtık Sistemler & Konsensüs",
        "prompt": "Bizans Hata Toleransı (BFT) protokollerinde f sayıda hain düğüm varken sistemin tutarlı çalışabilmesi için neden toplam 3f + 1 düğüme ihtiyaç duyulduğunu adım adım akıl yürüterek açıkla ve temel teknik terimleri listeleyerek pedagojik olarak özetle.",
        "gold_terms": ["Bizans Hata Toleransı", "Hain Düğüm", "Konsensüs"]
    },
    {
        "id": "TTC-2",
        "domain": "Kuantum Hesaplama",
        "prompt": "Shor algoritmasının klasik RSA açık anahtarlı şifrelemeyi kırmada kuantum fourier dönüşümü ve periyot bulma adımlarını adım adım analiz ederek pedagojik düzeyde açıkla.",
        "gold_terms": ["Shor Algoritması", "Kuantum Fourier Dönüşümü", "Periyot Bulma", "RSA"]
    },
    {
        "id": "TTC-3",
        "domain": "Derin Pekiştirmeli Öğrenme",
        "prompt": "Policy Gradient yöntemlerinde actor-critic mimarisinin varyansı azaltırken yanlılığı (bias) nasıl kontrol ettiğini adım adım mantıksal silsileyle açıkla.",
        "gold_terms": ["Actor-Critic", "Varyans Azaltma", "Policy Gradient", "Yanlılık"]
    }
]

def extract_terms_simple(text: str) -> list:
    candidates = re.findall(r"[\"']([^\"']+)[\"']|\*\*([^*]+)\*\*|([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)+)", text)
    extracted = set()
    for c in candidates:
        for item in c:
            if item and len(item.strip()) > 3 and not item.lower().startswith("adım"):
                extracted.add(item.strip().lower())
    return list(extracted)

def run():
    start_time = time.time()
    logger.info("Initializing Job 2: Test-Time Compute (TTC) & CoT Scaling on 14B...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="cuda:0",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    n_samples_list = [1, 2, 4, 8]
    ttc_results = {}

    for n in n_samples_list:
        logger.info(f"Evaluating Test-Time Compute with N={n} rollouts...")
        n_start = time.time()
        scenario_metrics = []

        for sc in COMPLEX_SCENARIOS:
            chatml = (
                f"<|im_start|>system\nSen karmaşık teknik problemleri önce adım adım düşünen, mantıksal nedensellik zinciri kuran ve ardından en net pedagojik terim çıkarımını yapan uzman bir asistansın. Çıkarım yapmadan önce <think> ... </think> blokları arasında akıl yürüt.<|im_end|>\n"
                f"<|im_start|>user\n{sc['prompt']}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n"
            )
            inputs = tokenizer(chatml, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=320,
                    do_sample=(n > 1),
                    temperature=0.7 if n > 1 else 0.0,
                    top_p=0.9 if n > 1 else 1.0,
                    num_return_sequences=n
                )

            rollout_texts = []
            rollout_terms = []
            reasoning_token_counts = []

            for seq in outputs:
                full_gen = tokenizer.decode(seq[inputs.input_ids.shape[1]:], skip_special_tokens=False)
                # Parse think block
                think_match = re.search(r"^(.*?)(?:</think>|$)", full_gen, re.DOTALL)
                think_text = think_match.group(1).strip() if think_match else ""
                reasoning_tokens = len(tokenizer.encode(think_text))
                reasoning_token_counts.append(reasoning_tokens)

                ext_terms = extract_terms_simple(full_gen)
                rollout_terms.append(ext_terms)
                rollout_texts.append(full_gen)

            # Majority voting across rollouts
            all_term_mentions = [t for r in rollout_terms for t in r]
            term_counts = Counter(all_term_mentions)
            consensus_terms = [t for t, count in term_counts.items() if count >= (n / 2.0)]

            # Check precision/recall against gold terms
            gold_lower = [g.lower() for g in sc["gold_terms"]]
            hits = sum(1 for g in gold_lower if any(g in ct or ct in g for ct in consensus_terms))
            recall = hits / max(1, len(gold_lower))
            precision = hits / max(1, len(consensus_terms)) if consensus_terms else 0.0
            f1 = (2 * precision * recall) / max(1e-6, (precision + recall))

            scenario_metrics.append({
                "id": sc["id"],
                "mean_reasoning_tokens": round(sum(reasoning_token_counts) / len(reasoning_token_counts), 1),
                "consensus_terms_count": len(consensus_terms),
                "recall_pct": round(recall * 100, 1),
                "precision_pct": round(precision * 100, 1),
                "f1_pct": round(f1 * 100, 1)
            })

        avg_f1 = sum(m["f1_pct"] for m in scenario_metrics) / len(scenario_metrics)
        avg_tokens = sum(m["mean_reasoning_tokens"] for m in scenario_metrics) / len(scenario_metrics)
        latency = round(time.time() - n_start, 2)

        ttc_results[f"N_{n}"] = {
            "num_rollouts": n,
            "latency_sec": latency,
            "mean_reasoning_tokens": round(avg_tokens, 1),
            "macro_f1_pct": round(avg_f1, 1),
            "scenarios": scenario_metrics
        }
        logger.info(f"N={n} finished: Macro F1 = {avg_f1:.1f}%, Reasoning Tokens = {avg_tokens:.1f}, Latency = {latency}s")

    report = {
        "duration_sec": round(time.time() - start_time, 2),
        "ttc_scaling_summary": ttc_results,
        "f1_gain_n1_to_n8_pct": round(ttc_results["N_8"]["macro_f1_pct"] - ttc_results["N_1"]["macro_f1_pct"], 1)
    }

    os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Job 2 Completed Successfully! F1 Gain N=1 -> N=8: +%.1f%%", report["f1_gain_n1_to_n8_pct"])

if __name__ == "__main__":
    run()

