#!/usr/bin/env python3
"""Cross-Lingual Zero-Shot Multilingual Pedagogical Transfer Benchmark.

Evaluates how well Turkish-fine-tuned EduTerm models transfer core pedagogical
concepts into English, Azerbaijani, and German under zero-shot prompting.

Evaluated Models:
- Kumru-2B (EduTerm SFT)
- EduTerm-14B-SFT
- EduTerm-14B-DPO
- EduTerm-14B-Omni
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
logger = logging.getLogger("cross_lingual_transfer")

MULTILINGUAL_CONCEPTS: List[Dict[str, Any]] = [
    {
        "id": "CS_VIRTUAL_MEM",
        "domain": "Computer Science",
        "tr_concept": "Sanal Bellek ve Sayfalama",
        "targets": {
            "en": {
                "lang_name": "English",
                "gold_terms": ["virtual memory", "paging", "page fault", "ram", "disk"],
                "stop_words": ["the", "is", "and", "in", "of", "to", "memory"]
            },
            "az": {
                "lang_name": "Azərbaycan dili",
                "gold_terms": ["virtual yaddaş", "səhifələmə", "fiziki yaddaş", "əməliyyat sistemi"],
                "stop_words": ["və", "bir", "üçün", "ilə", "kimi", "yaddaş"]
            },
            "de": {
                "lang_name": "Deutsch",
                "gold_terms": ["virtueller speicher", "paging", "arbeitsspeicher", "seitenfehler"],
                "stop_words": ["und", "der", "die", "das", "ein", "im", "speicher"]
            }
        }
    },
    {
        "id": "BIO_OKAZAKI",
        "domain": "Molecular Biology",
        "tr_concept": "DNA Helikaz ve Okazaki Parçacıkları",
        "targets": {
            "en": {
                "lang_name": "English",
                "gold_terms": ["okazaki fragments", "dna helicase", "dna ligase", "replication"],
                "stop_words": ["the", "dna", "and", "during", "strand", "enzyme"]
            },
            "az": {
                "lang_name": "Azərbaycan dili",
                "gold_terms": ["okazaki fraqmentləri", "dnt helikaza", "liqaza", "replikasiya"],
                "stop_words": ["və", "dnt", "üçün", "ilə", "zamanı"]
            },
            "de": {
                "lang_name": "Deutsch",
                "gold_terms": ["okazaki-fragmente", "dna-helikase", "ligase", "replikation"],
                "stop_words": ["die", "der", "und", "dna", "strang", "enzym"]
            }
        }
    },
    {
        "id": "BANK_LCR",
        "domain": "Finance & Banking",
        "tr_concept": "Likidite Karşılama Oranı (LCR) ve Basel III",
        "targets": {
            "en": {
                "lang_name": "English",
                "gold_terms": ["liquidity coverage ratio", "lcr", "basel iii", "cash outflow"],
                "stop_words": ["the", "ratio", "bank", "and", "liquidity", "stress"]
            },
            "az": {
                "lang_name": "Azərbaycan dili",
                "gold_terms": ["likvidlik əhatə dairəsi", "lcr", "basel", "nağd pul"],
                "stop_words": ["və", "bank", "üçün", "ilə", "nisbəti"]
            },
            "de": {
                "lang_name": "Deutsch",
                "gold_terms": ["liquiditätsdeckungsquote", "lcr", "basel iii", "abflüsse"],
                "stop_words": ["die", "und", "bank", "quote", "liquidität"]
            }
        }
    },
    {
        "id": "ML_OVERFITTING",
        "domain": "Machine Learning",
        "tr_concept": "Aşırı Öğrenme ve Gradyan İnişi",
        "targets": {
            "en": {
                "lang_name": "English",
                "gold_terms": ["overfitting", "gradient descent", "generalization", "regularization"],
                "stop_words": ["the", "model", "and", "training", "gradient", "data"]
            },
            "az": {
                "lang_name": "Azərbaycan dili",
                "gold_terms": ["həddindən artıq uyğunlaşma", "qradiyent enişi", "ümumiləşdirmə"],
                "stop_words": ["və", "model", "üçün", "təlim", "ilə"]
            },
            "de": {
                "lang_name": "Deutsch",
                "gold_terms": ["überanpassung", "gradientenverfahren", "generalisierung"],
                "stop_words": ["das", "die", "modell", "und", "gradient", "daten"]
            }
        }
    }
]

def format_multilingual_prompt(model_family: str, tr_concept: str, target_lang_code: str, target_meta: Dict[str, Any]) -> str:
    lang_name = target_meta["lang_name"]
    inst = (
        f"Explain the technical concept '{tr_concept}' in fluent {lang_name} for a high-school student (CEFR-B1 level). "
        f"Use an intuitive real-world analogy and preserve accurate academic terminology in {lang_name}. "
        f"Output MUST be entirely in {lang_name}."
    )
    if model_family == "kumru":
        return f"<|user|>\n{inst}\n<|assistant|>\n"
    else:
        return f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n"

def evaluate_multilingual_output(response_text: str, target_meta: Dict[str, Any]) -> Dict[str, Any]:
    lower_res = response_text.lower()
    
    # 1. Term Retention
    gold_terms = [gt.lower() for gt in target_meta["gold_terms"]]
    hits = sum(1 for gt in gold_terms if gt in lower_res)
    term_retention_pct = round((hits / max(1, len(gold_terms))) * 100.0, 1)

    # 2. Target Language Integrity (Are stop words of target language present?)
    stop_words = target_meta["stop_words"]
    stop_hits = sum(1 for sw in stop_words if f" {sw} " in f" {lower_res} ")
    lang_fidelity_pct = round(min(100.0, (stop_hits / max(1, len(stop_words) * 0.5)) * 100.0), 1)

    # 3. Analogy presence
    analogy_markers = ["like", "imagine", "analog", "gibi", "təsəvvür", "wie", "vergleichbar", "metapher"]
    has_analogy = any(am in lower_res for am in analogy_markers)

    # Composite Transfer Score
    transfer_score = round((term_retention_pct * 0.5) + (lang_fidelity_pct * 0.3) + (20.0 if has_analogy else 0.0), 1)

    return {
        "term_retention_pct": term_retention_pct,
        "language_fidelity_pct": lang_fidelity_pct,
        "has_analogy": has_analogy,
        "transfer_score": transfer_score
    }

def main():
    parser = argparse.ArgumentParser(description="Cross-Lingual Multilingual Pedagogical Transfer Benchmark")
    parser.add_argument("--output", type=str, default="/users/ali.gunder/eduterm-finetune/outputs/cross_lingual_transfer_report.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Initializing Cross-Lingual Transfer Benchmark on %s...", device)

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
        logger.info("Benchmarking Cross-Lingual Transfer for %s...", cfg["name"])
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

        transfer_evals = []
        for concept in MULTILINGUAL_CONCEPTS:
            for lang_code, target_meta in concept["targets"].items():
                prompt = format_multilingual_prompt(cfg["family"], concept["tr_concept"], lang_code, target_meta)
                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].to(device)
                attn_mask = inputs.get("attention_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.to(device)

                with torch.no_grad():
                    out = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                    )
                gen_text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
                eval_res = evaluate_multilingual_output(gen_text, target_meta)
                transfer_evals.append({
                    "concept_id": concept["id"],
                    "target_language": lang_code,
                    "evaluation": eval_res,
                    "output_snippet": gen_text[:140] + "..."
                })

        # Summaries per language and overall
        by_lang = {}
        for l_code in ["en", "az", "de"]:
            subset = [e for e in transfer_evals if e["target_language"] == l_code]
            by_lang[l_code] = {
                "term_retention_pct": round(sum(s["evaluation"]["term_retention_pct"] for s in subset) / max(1, len(subset)), 1),
                "language_fidelity_pct": round(sum(s["evaluation"]["language_fidelity_pct"] for s in subset) / max(1, len(subset)), 1),
                "mean_transfer_score": round(sum(s["evaluation"]["transfer_score"] for s in subset) / max(1, len(subset)), 1)
            }

        total_n = len(transfer_evals)
        summary = {
            "mean_overall_transfer_score": round(sum(e["evaluation"]["transfer_score"] for e in transfer_evals) / total_n, 1),
            "mean_overall_term_retention_pct": round(sum(e["evaluation"]["term_retention_pct"] for e in transfer_evals) / total_n, 1),
            "mean_overall_language_fidelity_pct": round(sum(e["evaluation"]["language_fidelity_pct"] for e in transfer_evals) / total_n, 1),
            "by_language": by_lang
        }

        all_results[cfg["name"]] = {
            "summary": summary,
            "details": transfer_evals
        }
        logger.info("%s Cross-Lingual: Overall: %.1f | EN Term: %.1f%% | AZ Term: %.1f%% | DE Term: %.1f%%",
                    cfg["name"], summary["mean_overall_transfer_score"],
                    by_lang["en"]["term_retention_pct"], by_lang["az"]["term_retention_pct"],
                    by_lang["de"]["term_retention_pct"])

        del model
        if cfg["adapter"]:
            del base
        gc.collect()
        torch.cuda.empty_cache()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info("Cross-Lingual Benchmark saved successfully to %s", output_path)

if __name__ == "__main__":
    main()

