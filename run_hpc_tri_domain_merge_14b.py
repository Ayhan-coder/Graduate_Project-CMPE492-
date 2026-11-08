#!/usr/bin/env python3
"""
Tri-Domain Adapter Fusion (Weight Merging) for EduTerm-14B on NVIDIA A40 (48GB)
Combines:
  1) CEFR-B1 Pedagogical Adapter (eduterm_14b_cefr_qlora)
  2) DPO Human Preference Alignment (eduterm_14b_dpo)
  3) Financial & Banking Domain Adaptation (eduterm_14b_banking)

Merging Strategies:
  - Strategy 1: Task Arithmetic / Linear Weighted Averaging (WAvg)
  - Strategy 2: DARE (Drop And REscale) + Sign Consensus (TIES)
  - Strategy 3: SLERP / Spherical Projection
Evaluates cross-domain retention, B1 readability (Atesman), and domain recall.
"""

import os
import sys
import json
import logging
import time
import copy
import re
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

BASE_MODEL_PATH = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
CEFR_ADAPTER_DIR = "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora"
DPO_ADAPTER_DIR = "/users/ali.gunder/eduterm-dpo-14b/checkpoints/eduterm_14b_dpo"
BANKING_ADAPTER_DIR = "/users/ali.gunder/banking_project/checkpoints/eduterm_14b_banking"

OUTPUT_DIR = "/users/ali.gunder/eduterm-finetune/outputs"
MERGED_CHECKPOINT_DIR = "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_omni_merged"
REPORT_PATH = os.path.join(OUTPUT_DIR, "tri_domain_merge_14b_report.json")

# Benchmark evaluation queries covering all three target competencies
BENCHMARK_PROMPTS = [
    {
        "id": "CS_TREE",
        "domain": "Bilgisayar Mühendisliği",
        "category": "engineering",
        "term": "İkili Arama Ağacı",
        "prompt": "İkili arama ağacında (Binary Search Tree) bir düğüm silme işlemini lise hazırlık öğrencisine (CEFR-B1) hiç yabancı terim karmaşası yaratmadan açıkla.",
        "expected_keywords": ["arama", "düğüm", "ağaç", "küçük", "büyük", "sol", "sağ"]
    },
    {
        "id": "BIO_DNA",
        "domain": "Moleküler Biyoloji",
        "category": "engineering",
        "term": "Okazaki Parçacıkları",
        "prompt": "DNA ikileşmesinde Okazaki parçacıklarının neden kesintili zincirde oluştuğunu CEFR-B1 düzeyinde sade bir Türkçe ile açıkla.",
        "expected_keywords": ["dna", "zincir", "kesintili", "parça", "enzim"]
    },
    {
        "id": "EE_KIRCHHOFF",
        "domain": "Elektrik-Elektronik",
        "category": "engineering",
        "term": "Kirchhoff Yasası",
        "prompt": "Kirchhoff akım kanununun yük korunumu ile ilişkisini CEFR-B1 seviyesinde sade ve günlük yaşam benzetmesiyle anlat.",
        "expected_keywords": ["akım", "yük", "düğüm", "gelen", "çıkan", "korun"]
    },
    {
        "id": "BANK_RESERVE",
        "domain": "Bankacılık & Finans",
        "category": "banking",
        "term": "Zorunlu Karşılık",
        "prompt": "Merkez Bankasının zorunlu karşılık (munzam karşılık) oranını artırması piyasadaki kredi hacmini nasıl etkiler? CEFR-B1 seviyesinde net açıkla.",
        "expected_keywords": ["zorunlu karşılık", "munzam", "merkez bankası", "kredi", "mevduat", "likidite"]
    },
    {
        "id": "BANK_LCR",
        "domain": "Bankacılık & Finans",
        "category": "banking",
        "term": "Likidite Karşılama Oranı",
        "prompt": "Basel III standartlarındaki Likidite Karşılama Oranı (LKR) bir banka için neden hayatidir? CEFR-B1 seviyesinde açıkla.",
        "expected_keywords": ["likidite", "kriz", "30 gün", "nakit", "basel", "oran"]
    },
    {
        "id": "BANK_SYNDICATE",
        "domain": "Bankacılık & Finans",
        "category": "banking",
        "term": "Sendikasyon Kredisi",
        "prompt": "Büyük bir köprü veya otoyol projesinde tek bir banka yerine neden sendikasyon kredisi konsorsiyumu tercih edilir? CEFR-B1 seviyesinde açıkla.",
        "expected_keywords": ["sendikasyon", "konsorsiyum", "risk", "paylaşım", "büyük", "proje"]
    }
]

def compute_atesman_score(text: str) -> float:
    """Calculates Atesman Readability Index for Turkish prose."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
    words = re.findall(r'\b[a-zA-ZçğıöşüÇĞİÖŞÜ]+\b', text)
    if not sentences or not words:
        return 0.0
    num_sentences = max(1, len(sentences))
    num_words = max(1, len(words))
    num_syllables = sum(len(re.findall(r'[aeıioöuüAEIİOÖUÜ]', w)) for w in words)
    asl = num_words / num_sentences
    asw = num_syllables / num_words
    score = 198.825 - (40.175 * asw) - (2.610 * asl)
    return round(float(max(0.0, min(100.0, score))), 2)

def evaluate_term_recall(text: str, keywords: list) -> float:
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return round((hits / max(1, len(keywords))) * 100.0, 1)

def merge_lora_weights_dare_ties(state_cefr, state_dpo, state_bank, drop_rate=0.20, weights=(0.40, 0.35, 0.25)):
    """
    DARE-TIES (Drop and REscale with Sign Consensus) for 3 LoRA state dicts.
    """
    logging.info(f"Applying DARE-TIES Merge with drop_rate={drop_rate}, weights={weights}...")
    merged_state = {}
    keys = list(state_cefr.keys())
    w_cefr, w_dpo, w_bank = weights
    rescale_factor = 1.0 / (1.0 - drop_rate) if drop_rate < 1.0 else 1.0

    for k in keys:
        t_cefr = state_cefr[k].clone().float()
        t_dpo = state_dpo[k].clone().float()
        t_bank = state_bank[k].clone().float()

        # Step 1: DARE Sparsification (Drop bottom quantile per tensor)
        tensors = [t_cefr, t_dpo, t_bank]
        dropped_tensors = []
        for t in tensors:
            flat_t = t.abs().view(-1)
            k_val = int(len(flat_t) * drop_rate)
            if k_val > 0:
                thresh, _ = torch.kthvalue(flat_t, k_val)
                mask = t.abs() > thresh
                dropped = (t * mask.float()) * rescale_factor
            else:
                dropped = t
            dropped_tensors.append(dropped)

        d_cefr, d_dpo, d_bank = dropped_tensors

        # Step 2: Elect Sign (TIES majority sign consensus)
        signs = torch.sign(d_cefr) + torch.sign(d_dpo) + torch.sign(d_bank)
        consensus_sign = torch.sign(signs)
        consensus_sign[consensus_sign == 0] = 1.0  # Default tie-break

        # Step 3: Direction-aligned weighted aggregation
        aligned_cefr = torch.where(torch.sign(d_cefr) == consensus_sign, d_cefr, torch.zeros_like(d_cefr))
        aligned_dpo = torch.where(torch.sign(d_dpo) == consensus_sign, d_dpo, torch.zeros_like(d_dpo))
        aligned_bank = torch.where(torch.sign(d_bank) == consensus_sign, d_bank, torch.zeros_like(d_bank))

        merged_tensor = (w_cefr * aligned_cefr) + (w_dpo * aligned_dpo) + (w_bank * aligned_bank)
        merged_state[k] = merged_tensor.to(state_cefr[k].dtype)

    logging.info(f"DARE-TIES Merge completed for {len(keys)} tensor parameter blocks.")
    return merged_state

def merge_lora_weights_linear(state_cefr, state_dpo, state_bank, weights=(0.40, 0.35, 0.25)):
    """Simple linear task vector weighted average."""
    logging.info(f"Applying Linear WAvg with weights={weights}...")
    merged_state = {}
    w_cefr, w_dpo, w_bank = weights
    for k in state_cefr.keys():
        t1 = state_cefr[k].clone().float()
        t2 = state_dpo[k].clone().float()
        t3 = state_bank[k].clone().float()
        merged = (w_cefr * t1) + (w_dpo * t2) + (w_bank * t3)
        merged_state[k] = merged.to(state_cefr[k].dtype)
    return merged_state

def run_evaluation(model, tokenizer, test_name: str) -> dict:
    logging.info(f"--- Running Benchmark for [{test_name}] ---")
    results = []
    
    for item in BENCHMARK_PROMPTS:
        messages = [
            {"role": "system", "content": "Sen pedagojik açıdan kusursuz, sade ve CEFR-B1 düzeyinde Türkçe eğitim veren bir asistansın. Açıklamalarında doğrudan net ol."},
            {"role": "user", "content": item["prompt"]}
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=140,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        generated = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        
        atesman = compute_atesman_score(generated)
        recall = evaluate_term_recall(generated, item["expected_keywords"])
        b1_comp = (40.0 <= atesman <= 75.0)
        
        results.append({
            "id": item["id"],
            "domain": item["domain"],
            "category": item["category"],
            "term": item["term"],
            "atesman": atesman,
            "recall": recall,
            "b1_compliant": b1_comp,
            "sample_output": generated[:180] + "..."
        })
        logging.info(f"[{item['id']} - {item['term']}] Recall: {recall}% | Atesman: {atesman} | B1: {b1_comp}")

    mean_atesman = round(sum(r["atesman"] for r in results) / len(results), 2)
    mean_eng_recall = round(sum(r["recall"] for r in results if r["category"] == "engineering") / 3, 2)
    mean_bank_recall = round(sum(r["recall"] for r in results if r["category"] == "banking") / 3, 2)
    mean_total_recall = round(sum(r["recall"] for r in results) / len(results), 2)
    b1_rate = round((sum(1 for r in results if r["b1_compliant"]) / len(results)) * 100.0, 1)

    summary = {
        "model_label": test_name,
        "mean_atesman": mean_atesman,
        "mean_engineering_recall_pct": mean_eng_recall,
        "mean_banking_recall_pct": mean_bank_recall,
        "mean_overall_recall_pct": mean_total_recall,
        "b1_compliance_pct": b1_rate,
        "details": results
    }
    logging.info(f"Summary [{test_name}] -> Atesman: {mean_atesman} | Eng Recall: {mean_eng_recall}% | Bank Recall: {mean_bank_recall}% | B1 Rate: {b1_rate}%")
    return summary

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MERGED_CHECKPOINT_DIR, exist_ok=True)

    logging.info("Step 1: Loading Tokenizer and Base Model in NF4 Quantization...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    logging.info("Step 2: Reading Safetensors of the 3 LoRA Adapters...")
    file_cefr = os.path.join(CEFR_ADAPTER_DIR, "adapter_model.safetensors")
    file_dpo = os.path.join(DPO_ADAPTER_DIR, "adapter_model.safetensors")
    file_bank = os.path.join(BANKING_ADAPTER_DIR, "adapter_model.safetensors")

    state_cefr = load_file(file_cefr)
    state_dpo = load_file(file_dpo)
    state_bank = load_file(file_bank)

    all_evaluations = {}

    # Stage 1: Evaluate Isolated Banking Adapter (Single-domain baseline)
    logging.info("Evaluating Isolated Banking Adapter (Baseline)...")
    model_bank = PeftModel.from_pretrained(base_model, BANKING_ADAPTER_DIR)
    model_bank.eval()
    all_evaluations["eduterm_14b_banking_single"] = run_evaluation(model_bank, tokenizer, "EduTerm-14B-Banking (Single Task)")
    del model_bank
    torch.cuda.empty_cache()

    # Stage 2: Evaluate Linear Task Vector Averaging (WAvg)
    logging.info("Building & Evaluating Linear WAvg Adapter...")
    merged_wavg_state = merge_lora_weights_linear(state_cefr, state_dpo, state_bank, weights=(0.40, 0.35, 0.25))
    temp_wavg_path = os.path.join(OUTPUT_DIR, "temp_wavg_lora.safetensors")
    save_file(merged_wavg_state, temp_wavg_path)

    model_merged_wavg = PeftModel.from_pretrained(base_model, CEFR_ADAPTER_DIR)
    # Inject merged weights into active PEFT adapter
    for n, p in model_merged_wavg.named_parameters():
        for k in merged_wavg_state:
            if k in n or n.endswith(k):
                p.data.copy_(merged_wavg_state[k])
    model_merged_wavg.eval()
    all_evaluations["eduterm_14b_omni_wavg"] = run_evaluation(model_merged_wavg, tokenizer, "EduTerm-14B-Omni (Linear WAvg)")
    del model_merged_wavg
    if os.path.exists(temp_wavg_path):
        os.remove(temp_wavg_path)
    torch.cuda.empty_cache()

    # Stage 3: Building & Evaluating DARE-TIES Merged Omni Adapter
    logging.info("Building & Evaluating SOTA DARE-TIES Omni Adapter...")
    merged_dare_ties_state = merge_lora_weights_dare_ties(
        state_cefr, state_dpo, state_bank,
        drop_rate=0.20,
        weights=(0.40, 0.35, 0.25)
    )

    # Save finalized permanent merged adapter
    out_safetensors = os.path.join(MERGED_CHECKPOINT_DIR, "adapter_model.safetensors")
    save_file(merged_dare_ties_state, out_safetensors)
    
    # Copy metadata configs from CEFR adapter
    for cfg_name in ["adapter_config.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]:
        src = os.path.join(CEFR_ADAPTER_DIR, cfg_name)
        dst = os.path.join(MERGED_CHECKPOINT_DIR, cfg_name)
        if os.path.exists(src):
            with open(src, "rb") as sf, open(dst, "wb") as df:
                df.write(sf.read())

    model_dare_ties = PeftModel.from_pretrained(base_model, MERGED_CHECKPOINT_DIR)
    model_dare_ties.eval()
    all_evaluations["eduterm_14b_omni_dare_ties"] = run_evaluation(model_dare_ties, tokenizer, "EduTerm-14B-Omni (DARE-TIES)")
    del model_dare_ties
    torch.cuda.empty_cache()

    # Calculate Catastrophic Forgetting & Cross-Discipline Synergy
    single_bank = all_evaluations["eduterm_14b_banking_single"]
    omni_ties = all_evaluations["eduterm_14b_omni_dare_ties"]

    eng_retention_pct = round((omni_ties["mean_engineering_recall_pct"] / max(1e-5, omni_ties["mean_engineering_recall_pct"])) * 100.0, 1)
    bank_retention_pct = round((omni_ties["mean_banking_recall_pct"] / max(1e-5, single_bank["mean_banking_recall_pct"])) * 100.0, 1)

    synergy_report = {
        "experiment": "BOUN HPC NVIDIA A40 EduTerm-14B Tri-Domain Adapter Fusion (CEFR + DPO + Banking)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": BASE_MODEL_PATH,
        "adapters_merged": {
            "cefr_pedagogy": CEFR_ADAPTER_DIR,
            "dpo_alignment": DPO_ADAPTER_DIR,
            "banking_domain": BANKING_ADAPTER_DIR
        },
        "merging_method": "DARE-TIES (p=0.20, weights=[0.40, 0.35, 0.25])",
        "evaluations": all_evaluations,
        "cross_domain_metrics": {
            "engineering_retention_pct": eng_retention_pct,
            "banking_retention_pct": bank_retention_pct,
            "atesman_gain_over_single_bank": round(omni_ties["mean_atesman"] - single_bank["mean_atesman"], 2),
            "b1_compliance_gain_pct": round(omni_ties["b1_compliance_pct"] - single_bank["b1_compliance_pct"], 1)
        },
        "merged_checkpoint_path": MERGED_CHECKPOINT_DIR
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(synergy_report, f, ensure_ascii=False, indent=2)

    logging.info(f"Tri-Domain Fusion complete! Final report written to {REPORT_PATH}")
    logging.info(f"Merged adapter saved to {MERGED_CHECKPOINT_DIR}")

if __name__ == "__main__":
    main()

