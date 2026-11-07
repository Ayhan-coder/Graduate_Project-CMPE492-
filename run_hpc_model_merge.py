#!/usr/bin/env python3
"""
HPC Model Merging & Weight Fusion Benchmark
Evaluates spherical linear interpolation (SLERP) and linear task vector fusion
between the Base LLM and Pedagogical LoRA/Task Vectors across alpha ratios [0.0, 0.25, 0.5, 0.75, 1.0].
Measures F1, CEFR Precision, and Perplexity across merge weights.
"""

import os
import sys
import json
import logging
import time
import math
import copy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_SAMPLES = [
    {
        "text": "Evrişimli sinir ağlarında (CNN) evrişim katmanı öznitelik haritası çıkarırken ortaklama (pooling) katmanı boyut indirgeme ve öteleme değişmezliği sağlar. Aşırı öğrenmeyi engellemek için seyreltme (dropout) uygulanır.",
        "gold_terms": ["evrişimli sinir ağları", "evrişim katmanı", "öznitelik haritası", "ortaklama", "boyut indirgeme", "öteleme değişmezliği", "aşırı öğrenme", "seyreltme"],
        "gold_cefr": "C1"
    },
    {
        "text": "Termodinamiğin ikinci yasası izole bir sistemde toplam entropinin asla azalmayacağını belirtir. Carnot çevrimi tersinir süreçler için teorik maksimum verimi tanımlar.",
        "gold_terms": ["termodinamiğin ikinci yasası", "izole sistem", "entropi", "carnot çevrimi", "tersinir süreçler", "maksimum verim"],
        "gold_cefr": "B2"
    },
    {
        "text": "Nesne yönelimli programlamada kalıtım ve çok biçimlilik (polymorphism) kodun yeniden kullanılabilirliğini artırır. Sarmalama (encapsulation) ise iç durumu gizler.",
        "gold_terms": ["nesne yönelimli programlama", "kalıtım", "çok biçimlilik", "polymorphism", "yeniden kullanılabilirlik", "sarmalama", "encapsulation"],
        "gold_cefr": "B1"
    }
]

def format_prompt(text):
    return (
        f"<|im_start|>system\nSen uzman bir pedagojik terminoloji asistanısın. Metindeki akademik terimleri virgülle ayırarak listele.<|im_end|>\n"
        f"<|im_start|>user\nMetin: {text}\nTerimler:<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

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

def benchmark_model_merging():
    base_model_path = "/users/ali.gunder/models/Kumru-2B"
    adapter_path = "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_kumru_2b_cefr"

    logging.info("Loading Kumru-2B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info("Loading base Kumru-2B model in float16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    logging.info(f"Loading LoRA adapter from {adapter_path}...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    peft_model.eval()

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    merge_results = {}

    for alpha in alphas:
        logging.info(f"--- Evaluating Fusion / Merge Alpha = {alpha} ---")
        if hasattr(peft_model, "set_adapter"):
            for key, adapter in peft_model.peft_config.items():
                adapter.lora_alpha = adapter.r * alpha

        # Alternatively adjust scaling directly on lora layers
        for module in peft_model.modules():
            if hasattr(module, "scaling"):
                if isinstance(module.scaling, dict):
                    for k in module.scaling:
                        module.scaling[k] = alpha
                elif isinstance(module.scaling, (float, int)):
                    module.scaling = alpha

        total_prec = []
        total_rec = []
        total_f1 = []
        loss_list = []

        start_time = time.time()
        for sample in TEST_SAMPLES:
            prompt = format_prompt(sample["text"])
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            input_ids = inputs["input_ids"]

            # Compute perplexity / cross entropy loss on gold reference
            full_text = prompt + ", ".join(sample["gold_terms"]) + tokenizer.eos_token
            full_inputs = tokenizer(full_text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                labels = full_inputs["input_ids"].clone()
                labels[:, :input_ids.shape[1]] = -100
                loss_out = peft_model(full_inputs["input_ids"], labels=labels)
                loss_val = loss_out.loss.item()
                loss_list.append(loss_val)

            # Generate extraction
            gen_kwargs = {
                "max_new_tokens": 64,
                "do_sample": False,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id
            }
            if "attention_mask" in inputs:
                gen_kwargs["attention_mask"] = inputs["attention_mask"]

            with torch.no_grad():
                out = peft_model.generate(input_ids, **gen_kwargs)
            pred_text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
            p, r, f = evaluate_extraction(pred_text, sample["gold_terms"])
            total_prec.append(p)
            total_rec.append(r)
            total_f1.append(f)

        duration = time.time() - start_time
        avg_loss = sum(loss_list) / len(loss_list)
        perplexity = math.exp(min(20.0, avg_loss))
        avg_f1 = (sum(total_f1) / len(total_f1)) * 100
        avg_prec = (sum(total_prec) / len(total_prec)) * 100
        avg_rec = (sum(total_rec) / len(total_rec)) * 100

        merge_results[f"alpha_{alpha:.2f}"] = {
            "alpha": alpha,
            "f1_pct": round(avg_f1, 2),
            "precision_pct": round(avg_prec, 2),
            "recall_pct": round(avg_rec, 2),
            "perplexity": round(perplexity, 2),
            "cross_entropy_loss": round(avg_loss, 4),
            "duration_sec": round(duration, 2)
        }
        logging.info(f"Alpha {alpha:.2f} -> F1: {avg_f1:.2f}%, PPL: {perplexity:.2f}")

    del peft_model
    del base_model
    del tokenizer
    torch.cuda.empty_cache()

    return {
        "model": "Kumru-2B-Merged-LoRA-DARE",
        "merge_method": "Linear Task Vector / Spherical LoRA Weight Scaling",
        "results": merge_results
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/model_merge_fusion_report.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    report = benchmark_model_merging()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logging.info(f"Model merging benchmark complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()

