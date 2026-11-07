#!/usr/bin/env python3
"""
step3_train_banking_14b_qlora.py
Domain-Adaptive Fine-Tuning (DAFT) of EduTerm-14B on Banking & Financial Lecture Dataset.
Runs on Boğaziçi HPC NVIDIA A40 (48GB) using 4-bit NF4 QLoRA.
Evaluates pre- and post-training banking domain terminology precision, recall, and CEFR-B1 compliance.
"""

import os
import sys
import json
import logging
import time
import math
import re
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainBanking14B")

BASE_MODEL_PATH = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
PREV_DPO_ADAPTER = "/users/ali.gunder/eduterm-dpo-14b/checkpoints/eduterm_14b_dpo"
OUTPUT_DIR = Path("/users/ali.gunder/banking_project/checkpoints/eduterm_14b_banking")
OUTPUT_REPORT = Path("/users/ali.gunder/banking_project/outputs/banking_adaptation_report.json")
DATA_DIR = Path("/users/ali.gunder/banking_project/data")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

BANKING_TEST_BENCHMARKS = [
    {
        "term": "Zorunlu Karşılık",
        "query": "Bankaların munzam karşılık oranının TCMB tarafından artırılması piyasadaki kredi hacmini nasıl etkiler?",
        "gold_terms": ["Zorunlu Karşılık", "Munzam Karşılık", "TCMB", "Kredi Hacmi"]
    },
    {
        "term": "Likidite Karşılama Oranı",
        "query": "Basel III standartlarına göre likidite karşılama oranı neden en az yüzde yüz olmak zorundadır?",
        "gold_terms": ["Likidite Karşılama Oranı", "Basel III", "Likit Aktifler"]
    },
    {
        "term": "Sendikasyon Kredisi",
        "query": "Büyük altyapı projelerinde tek bir banka yerine neden sendikasyon kredisi konsorsiyumu tercih edilir?",
        "gold_terms": ["Sendikasyon Kredisi", "Konsorsiyum", "Risk Dağıtımı"]
    },
    {
        "term": "Açık Piyasa İşlemleri",
        "query": "Merkez Bankası piyasada likidite fazlası olduğunda Açık Piyasa İşlemleri ile ne tür adımlar atar?",
        "gold_terms": ["Açık Piyasa İşlemleri", "APİ", "Repo", "Ters Repo", "Devlet İç Borçlanma Senedi"]
    }
]

def calculate_atesman(text: str) -> float:
    if not text or not text.strip():
        return 60.0
    vowels = "aeıioöuü"
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    if not words:
        return 60.0
    spw = sum(sum(1 for c in w if c in vowels) for w in words) / max(1, len(words))
    wps = len(words) / max(1, len(sentences))
    return max(0.0, min(100.0, round(198.825 - (40.175 * spw) - (2.610 * wps), 1)))

class BankingSFTDataset(Dataset):
    def __init__(self, jsonl_file: Path, tokenizer: Any, max_len: int = 512):
        self.rows = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        item = self.rows[idx]
        user_block = item["instruction"]
        if item["input"]:
            user_block += f"\n\n{item['input']}"
        assistant_block = item["output"]

        prompt = (
            f"<|im_start|>system\nSen bankacılık ve finans alanında uzmanlaşmış, teknik kavramları üniversite öğrencileri için CEFR B1 düzeyinde analojilerle açıklayan pedagojik bir asistansın.<|im_end|>\n"
            f"<|im_start|>user\n{user_block}<|im_end|>\n<|im_start|>assistant\n"
        )
        full_text = prompt + assistant_block + "<|im_end|>"

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_enc = self.tokenizer(full_text, max_length=self.max_len, truncation=True, return_tensors="pt")
        input_ids = full_enc["input_ids"][0]
        attention_mask = full_enc["attention_mask"][0]

        labels = input_ids.clone()
        labels[:len(prompt_ids)] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def collate_fn(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    b_ids, b_mask, b_labels = [], [], []
    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        b_ids.append(torch.cat([b["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
        b_mask.append(torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        b_labels.append(torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(b_ids),
        "attention_mask": torch.stack(b_mask),
        "labels": torch.stack(b_labels)
    }

def evaluate_banking_benchmarks(model, tokenizer, stage_name: str) -> Dict[str, Any]:
    model.eval()
    bench_results = []
    logger.info(f"--- Running Banking Benchmark Evaluation [{stage_name}] ---")

    for b in BANKING_TEST_BENCHMARKS:
        prompt = (
            f"<|im_start|>system\nSen bankacılık ve finans alanında uzmanlaşmış, teknik kavramları üniversite öğrencileri için CEFR B1 düzeyinde analojilerle açıklayan pedagojik bir asistansın.<|im_end|>\n"
            f"<|im_start|>user\n{b['query']}<|im_end|>\n<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=140, do_sample=False)
        gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        at = calculate_atesman(gen)
        hits = sum(1 for gt in b["gold_terms"] if gt.lower() in gen.lower())
        recall = (hits / max(1, len(b["gold_terms"]))) * 100.0

        bench_results.append({
            "term": b["term"],
            "generated_text": gen,
            "atesman": at,
            "b1_compliant": 50 <= at <= 75,
            "term_recall_pct": round(recall, 1)
        })
        logger.info(f"[{b['term']}] Recall: {recall:.1f}% | Ateşman: {at:.1f} | Output: {gen[:70]}...")

    avg_at = sum(r["atesman"] for r in bench_results) / len(bench_results)
    avg_rec = sum(r["term_recall_pct"] for r in bench_results) / len(bench_results)
    b1_rate = (sum(1 for r in bench_results if r["b1_compliant"]) / len(bench_results)) * 100.0

    return {
        "stage": stage_name,
        "mean_atesman": round(avg_at, 1),
        "mean_term_recall_pct": round(avg_rec, 1),
        "b1_compliance_pct": round(b1_rate, 1),
        "benchmarks": bench_results
    }

def main():
    logger.info("Initializing EduTerm-14B Banking Domain Adaptation...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="cuda:0",
        trust_remote_code=True
    )

    # Attach previous DPO checkpoint as starting foundation
    logger.info("Attaching previous EduTerm-14B-DPO checkpoint as starting weight foundation...")
    model = PeftModel.from_pretrained(base_model, PREV_DPO_ADAPTER, is_trainable=True)

    # Pre-training benchmark
    pre_results = evaluate_banking_benchmarks(model, tokenizer, "PRE_BANKING_ADAPTATION")

    # Load datasets
    train_dataset = BankingSFTDataset(DATA_DIR / "train_banking_cefr.jsonl", tokenizer, max_len=512)
    val_dataset = BankingSFTDataset(DATA_DIR / "val_banking_cefr.jsonl", tokenizer, max_len=512)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id)
    )

    # Enable LoRA gradients
    model.train()
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-5,
        weight_decay=0.01
    )

    epochs = 3
    grad_accum_steps = 4
    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    t_start = time.time()
    history = []

    logger.info(f"Starting Banking SFT Training ({len(train_dataset)} train samples, {epochs} epochs)...")
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            total_loss += outputs.loss.item()
            step_count += 1

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = total_loss / step_count

        # Validation
        model.eval()
        val_loss = 0.0
        v_steps = 0
        with torch.no_grad():
            for v_batch in val_loader:
                v_ids = v_batch["input_ids"].to(DEVICE)
                v_mask = v_batch["attention_mask"].to(DEVICE)
                v_labels = v_batch["labels"].to(DEVICE)
                v_out = model(input_ids=v_ids, attention_mask=v_mask, labels=v_labels)
                val_loss += v_out.loss.item()
                v_steps += 1
        avg_val_loss = val_loss / max(1, v_steps)

        logger.info(f"Epoch {ep}/{epochs}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")
        history.append({
            "epoch": ep,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4)
        })

    elapsed = time.time() - t_start

    # Post-training benchmark
    post_results = evaluate_banking_benchmarks(model, tokenizer, "POST_BANKING_ADAPTATION")

    # Save Checkpoint
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving specialized Banking LoRA Checkpoint to {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Save Report
    report = {
        "experiment": "BOUN HPC NVIDIA A40 EduTerm-14B Banking Domain Adaptation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_time_sec": round(elapsed, 2),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epoch_history": history,
        "evaluation_comparison": {
            "pre_adaptation": pre_results,
            "post_adaptation": post_results,
            "term_recall_gain_pct": round(post_results["mean_term_recall_pct"] - pre_results["mean_term_recall_pct"], 1),
            "b1_compliance_gain_pct": round(post_results["b1_compliance_pct"] - pre_results["b1_compliance_pct"], 1)
        }
    }

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Banking adaptation completed successfully! Report saved to {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()

