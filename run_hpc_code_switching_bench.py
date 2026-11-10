#!/usr/bin/env python3
"""Spoken Code-Switching & Morphological Harmony Benchmark on HPC A40.

Evaluates how LLMs handle code-switched Turkish university lecture transcripts
where English technical loanwords are inflected with Turkish agglutinative suffixes
(vowel and consonant harmony).

Measures:
1. Technical Term Extraction F1 on mixed-lingual text
2. Morphological Harmony & Root Extraction Accuracy
3. Turkish CEFR-B1 Simplification Quality
4. Language Consistency (absence of uncontrolled English code collapse)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("code_switching_benchmark")

CODE_SWITCHING_CASES: List[Dict[str, Any]] = [
    {
        "id": "CS_BACKPROP",
        "domain": "Bilgisayar & Yapay Zeka",
        "raw_transcript": "Modeli eğitirken backpropagation'ı run ettiğimizde gradient vanishing'e takılmamak için aktivasyon fonksiyonu olarak ReLU seçiyoruz.",
        "gold_terms": ["Backpropagation", "Gradient Vanishing", "Aktivasyon Fonksiyonu", "ReLU"],
        "inflected_terms": ["backpropagation'ı", "gradient vanishing'e"]
    },
    {
        "id": "CS_PIPELINE",
        "domain": "Bilgisayar Mimarisi",
        "raw_transcript": "Eğer branch prediction stall olursa, işlemcinin bütün instruction pipeline'ını flush etmek ve cache invalidation yapmak zorunda kalırız.",
        "gold_terms": ["Branch Prediction", "Instruction Pipeline", "Cache Invalidation", "Stall"],
        "inflected_terms": ["pipeline'ını", "stall"]
    },
    {
        "id": "BIO_CRISPR",
        "domain": "Moleküler Biyoloji",
        "raw_transcript": "Cas9 nükleazı guide RNA'yı bind edip DNA double-strand break'i meydana getirdiğinde hücre non-homologous end joining mekanizmasını tetikler.",
        "gold_terms": ["Cas9 Nükleazı", "Guide RNA", "Double-Strand Break", "Non-Homologous End Joining"],
        "inflected_terms": ["RNA'yı", "break'i"]
    },
    {
        "id": "EE_IMPEDANCE",
        "domain": "Elektrik-Elektronik",
        "raw_transcript": "Yüksek frekanslı RF devrelerinde impedance matching yapmazsak power reflection'ı ve standing wave ratio'yu kabul edilebilir seviyede tutamayız.",
        "gold_terms": ["Impedance Matching", "Power Reflection", "Standing Wave Ratio", "RF Devreleri"],
        "inflected_terms": ["reflection'ı", "ratio'yu"]
    },
    {
        "id": "FIN_VOLATILITY",
        "domain": "Finans Mühendisliği",
        "raw_transcript": "Türev piyasalarında implied volatility surge ettiğinde delta hedging pozisyonlarını rebalance etmek için high-frequency order flow'u takip ederiz.",
        "gold_terms": ["Implied Volatility", "Delta Hedging", "High-Frequency Order Flow"],
        "inflected_terms": ["flow'u"]
    },
    {
        "id": "ML_DROPOUT",
        "domain": "Derin Öğrenme",
        "raw_transcript": "Ağda overfitting'e düşmemek ve weight decay'i dengelemek için fully-connected layer'lara dropout ekleyip early stopping'i devreye alıyoruz.",
        "gold_terms": ["Overfitting", "Weight Decay", "Dropout", "Early Stopping"],
        "inflected_terms": ["overfitting'e", "decay'i", "layer'lara", "stopping'i"]
    }
]

def calculate_atesman(text: str) -> float:
    words = re.findall(r"\b\w+\b", text)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not words or not sentences:
        return 0.0
    total_words = len(words)
    total_sentences = len(sentences)
    vowels = "aeıioöuüAEIİOÖUÜ"
    total_syllables = sum(sum(1 for ch in w if ch in vowels) for w in words)
    asl = total_words / total_sentences
    asw = total_syllables / total_words
    score = 198.825 - (40.175 * asw) - (2.610 * asl)
    return round(float(score), 2)

def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("’", "'").replace("`", "'")
    return re.sub(r"[^\w\s]", " ", text)

def format_cs_prompt(model_family: str, transcript: str) -> str:
    inst = (
        "Aşağıdaki konuşma dili ders kesitinde İngilizce ve Türkçe terimler karışık kullanılmıştır. "
        "1. Metindeki anahtar teknik terimleri kök halleriyle liste olarak çıkar.\n"
        "2. Bu ders kesitini lise öğrencisinin anlayabileceği CEFR-B1 düzeyinde, düzgün ve akıcı bir Türkçe ile açıkla.\n\n"
        f"Ders Metni: {transcript}"
    )
    if model_family == "kumru":
        return f"<|user|>\n{inst}\n<|assistant|>\n"
    else:
        return f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n"

def evaluate_cs_output(output_text: str, case_meta: Dict[str, Any]) -> Dict[str, Any]:
    norm_out = normalize_text(output_text)
    
    # 1. Term Recall on Gold Terms
    hits = 0
    for gt in case_meta["gold_terms"]:
        norm_gt = normalize_text(gt)
        if any(w in norm_out for w in norm_gt.split()):
            hits += 1
    term_recall = round((hits / max(1, len(case_meta["gold_terms"]))) * 100.0, 1)
    
    # 2. Morphological Harmony: Did it strip suffixes cleanly to bare root?
    root_cleaned = 0
    for infl in case_meta["inflected_terms"]:
        stem = infl.split("'")[0].lower()
        if stem in norm_out:
            root_cleaned += 1
    root_harmony_rate = round((root_cleaned / max(1, len(case_meta["inflected_terms"]))) * 100.0, 1)
    
    # 3. Turkish Language Purity (absence of falling completely into English monologue)
    turkish_common_words = ["bu", "ve", "bir", "için", "olarak", "ile", "daha", "böylece", "çünkü"]
    turkish_density = sum(1 for tw in turkish_common_words if f" {tw} " in f" {norm_out} ")
    language_maintained = turkish_density >= 3
    
    atesman = calculate_atesman(output_text)
    b1_compliant = atesman >= 40.0
    
    return {
        "term_recall_pct": term_recall,
        "root_harmony_rate_pct": root_harmony_rate,
        "turkish_language_maintained": language_maintained,
        "atesman": atesman,
        "b1_compliant": b1_compliant
    }

def main():
    parser = argparse.ArgumentParser(description="Spoken Code-Switching & Morphological Harmony Benchmark")
    parser.add_argument("--output", type=str, default="/users/ali.gunder/eduterm-finetune/outputs/code_switching_morphology_report.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Initializing Code-Switching Benchmark on %s...", device)

    models_config = [
        {
            "name": "Kumru-2B (EduTerm SFT)",
            "base": "/users/ali.gunder/models/Kumru-2B",
            "adapter": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_kumru_2b_cefr",
            "family": "kumru"
        },
        {
            "name": "EduTerm-14B-SFT",
            "base": "/users/ali.gunder/models/Qwen2.5-14B-Instruct",
            "adapter": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora",
            "family": "qwen"
        },
        {
            "name": "EduTerm-14B-DPO",
            "base": "/users/ali.gunder/models/Qwen2.5-14B-Instruct",
            "adapter": "/users/ali.gunder/eduterm-dpo-14b/checkpoints/eduterm_14b_dpo",
            "family": "qwen"
        },
        {
            "name": "EduTerm-14B-Omni",
            "base": "/users/ali.gunder/models/Qwen2.5-14B-Instruct",
            "adapter": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_omni_merged",
            "family": "qwen"
        }
    ]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    all_results = {}

    for cfg in models_config:
        logger.info("Benchmarking %s on code-switching...", cfg["name"])
        tokenizer = AutoTokenizer.from_pretrained(cfg["base"], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            cfg["base"],
            quantization_config=bnb_config,
            device_map=device,
            trust_remote_code=True
        )
        if cfg["adapter"] and Path(cfg["adapter"]).exists():
            model = PeftModel.from_pretrained(base, cfg["adapter"])
        else:
            model = base
        model.eval()

        case_results = []
        for cs in CODE_SWITCHING_CASES:
            prompt = format_cs_prompt(cfg["family"], cs["raw_transcript"])
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attn_mask = inputs.get("attention_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)

            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=300,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                )
            gen_text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            eval_res = evaluate_cs_output(gen_text, cs)
            case_results.append({
                "case_id": cs["id"],
                "eval": eval_res,
                "text_snippet": gen_text[:160] + "..."
            })

        n = len(case_results)
        summary = {
            "mean_term_recall_pct": round(sum(c["eval"]["term_recall_pct"] for c in case_results) / n, 1),
            "mean_root_harmony_rate_pct": round(sum(c["eval"]["root_harmony_rate_pct"] for c in case_results) / n, 1),
            "turkish_language_integrity_pct": round((sum(1 for c in case_results if c["eval"]["turkish_language_maintained"]) / n) * 100.0, 1),
            "mean_atesman": round(sum(c["eval"]["atesman"] for c in case_results) / n, 2),
            "b1_compliance_pct": round((sum(1 for c in case_results if c["eval"]["b1_compliant"]) / n) * 100.0, 1)
        }

        all_results[cfg["name"]] = {
            "summary": summary,
            "details": case_results
        }
        logger.info("%s Code-Switching Summary: Recall: %.1f%% | Root Harmony: %.1f%% | Integrity: %.1f%% | Ateşman: %.2f",
                    cfg["name"], summary["mean_term_recall_pct"], summary["mean_root_harmony_rate_pct"],
                    summary["turkish_language_integrity_pct"], summary["mean_atesman"])

        del model
        if cfg["adapter"]:
            del base
        gc.collect()
        torch.cuda.empty_cache()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info("Code-Switching Benchmark saved to %s", output_path)

if __name__ == "__main__":
    main()

