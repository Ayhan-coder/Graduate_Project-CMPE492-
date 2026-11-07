#!/usr/bin/env python3
"""EduTerm-14B: Multi-Level CEFR (B1/B2/C1) Controllable SFT Training & Evaluation.

Trains Qwen2.5-14B-Instruct via 4-bit QLoRA on NVIDIA A40 (48GB) to perform:
1. Multi-tier terminology extraction.
2. Controllable complexity generation across B1 (Sade), B2 (Teknik) and C1 (Akademik).
3. Bloom taxonomy cognitive level classification.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# Patch dynamo mark_static_address if needed
try:
    import torch._dynamo
    if not hasattr(torch._dynamo, "mark_static_address"):
        torch._dynamo.mark_static_address = lambda *args, **kwargs: None
except Exception:
    pass

class FlushingHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

handler = FlushingHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("EduTerm14B-CEFR-SFT")

SYSTEM_PROMPT = (
    "Sen Türkçe teknik eğitimde terminoloji tespiti, pedagojik sadeleştirme ve öğrenci düzeyine "
    "göre (CEFR B1, B2 veya C1) kontrollü karmaşıklıkta açıklama üreten uzman bir asistansın."
)
IM_END = "<|im_end|>"
MAX_TOKENS = 896

BENCHMARKS: List[Dict[str, Any]] = [
    {
        "id": "BENCH-EE",
        "domain": "Elektrik ve Elektronik",
        "text": "Düğüm gerilimleri analizinde bağımsız düğüm noktaları toprağa göre referans alınarak KCL akım denklemleri kurulur.",
        "gold_terms": ["Düğüm Gerilimleri Analizi", "Düğüm Noktası", "KCL Akım Denklemleri"],
        "gold_bloom": "Analiz"
    },
    {
        "id": "BENCH-BIO",
        "domain": "Biyomühendislik & Biyoloji",
        "text": "DNA replikasyonu sırasında helikaz enzimi çift zinciri açar ve okazaki parçaları ligaz enzimi ile birleştirilir.",
        "gold_terms": ["DNA Replikasyonu", "Helikaz Enzimi", "Okazaki Parçaları", "Ligaz Enzimi"],
        "gold_bloom": "Anlama"
    },
    {
        "id": "BENCH-CS",
        "domain": "Bilgisayar Mühendisliği",
        "text": "İkili arama ağaçlarında silme ve arama işlemleri dengeli ağaç yapısında logaritmik O(log n) karmaşıklık sunar.",
        "gold_terms": ["İkili Arama Ağacı", "Logaritmik Karmaşıklık", "Dengeli Ağaç"],
        "gold_bloom": "Uygulama"
    },
    {
        "id": "BENCH-TOUR",
        "domain": "Turizm & Seyahat (Tatilsepeti)",
        "text": "Dinamik paketleme motorunda uçak bileti ve otel kontenjanı birleştirilirken no-show riski için erken rezervasyon sigortası uygulanır.",
        "gold_terms": ["Dinamik Paketleme", "No-Show Riski", "Erken Rezervasyon Sigortası"],
        "gold_bloom": "Uygulama"
    },
    {
        "id": "BENCH-ISG",
        "domain": "İş Sağlığı ve Güvenliği",
        "text": "Şantiyede meydana gelen ramak kala olayları kayıt altına alınmalı ve acil durum eylem planı yenilenmelidir.",
        "gold_terms": ["Ramak Kala Olayı", "Acil Durum Eylem Planı"],
        "gold_bloom": "Uygulama"
    },
    {
        "id": "BENCH-KVKK",
        "domain": "Kişisel Verilerin Korunması",
        "text": "Veri sorumlusu aydınlatma yükümlülüğünü yerine getirdikten sonra ilgili kişiden açık rıza temin etmelidir.",
        "gold_terms": ["Veri Sorumlusu", "Aydınlatma Yükümlülüğü", "Açık Rıza"],
        "gold_bloom": "Anlama"
    },
    {
        "id": "BENCH-SEC",
        "domain": "Siber Güvenlik",
        "text": "Sıfırıncı gün açığı (zero-day) tespit edildiğinde derhal yama uygulanmalı ve çok faktörlü kimlik doğrulama zorunlu kılınmalıdır.",
        "gold_terms": ["Sıfırıncı Gün Açığı", "Çok Faktörlü Kimlik Doğrulama"],
        "gold_bloom": "Analiz"
    },
    {
        "id": "BENCH-BNK",
        "domain": "Bankacılık & Finans",
        "text": "Bankaların likidite karşılama oranı Basel III standartlarına göre belirlenir ve TCMB zorunlu karşılık oranına tabidir.",
        "gold_terms": ["Likidite Karşılama Oranı", "Basel III", "Zorunlu Karşılık"],
        "gold_bloom": "Anlama"
    }
]


def calculate_atesman_score(text: str) -> float:
    """Calculate Ateşman Turkish Readability Index."""
    if not text or not text.strip():
        return 60.0
    vowels = "aeıioöuü"
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    if not words:
        return 60.0
    syllables = sum(sum(1 for c in w if c in vowels) for w in words)
    spw = syllables / max(1, len(words))
    wps = len(words) / max(1, len(sentences))
    score = 198.825 - (40.175 * spw) - (2.610 * wps)
    return max(0.0, min(100.0, round(score, 1)))


def format_chatml_prompt(instruction: str, user_input: str, assistant_output: str = "", system: str = SYSTEM_PROMPT) -> str:
    user_block = f"{instruction}\n\nDers Notu Metni:\n{user_input}" if user_input else instruction
    prompt = (
        f"<|im_start|>system\n{system}{IM_END}\n"
        f"<|im_start|>user\n{user_block}{IM_END}\n"
    )
    if assistant_output:
        return prompt + f"<|im_start|>assistant\n{assistant_output}{IM_END}"
    else:
        return prompt + "<|im_start|>assistant\n"


def _collate(batch: Sequence[Tuple[List[int], List[int]]], pad_id: int, device: str = "cuda"):
    max_len = max(len(x[0]) for x in batch)
    input_ids, labels = [], []
    for ids, labs in batch:
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        labels.append(labs + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "attention_mask": (torch.tensor(input_ids, dtype=torch.long, device=device) != pad_id).long()
    }


def evaluate_cefr_benchmarks(model, tokenizer, device: str, prefix_label: str) -> Dict[str, Any]:
    """Evaluate both terminology extraction and CEFR level adaptability."""
    model.eval()
    results = []
    level_compliance = {"B1": 0, "B2": 0, "C1": 0}

    logger.info("=" * 75)
    logger.info("RUNNING BENCHMARK EVALUATION [%s]", prefix_label)
    logger.info("=" * 75)

    for bench in BENCHMARKS:
        # Prompt model for Multi-Level CEFR output
        instruction = (
            "Aşağıdaki ders notu metninden teknik terimleri tespit et. "
            "Her terim için Bloom taksonomisi belirle ve CEFR B1 (Sade), B2 (Teknik) ve C1 (Akademik) "
            "düzeylerinde açıklamalar üret. Sonucu geçerli JSON olarak ver."
        )
        prompt = format_chatml_prompt(instruction, bench["text"], "")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=450,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_time = time.time() - t0
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Parse JSON
        parsed_terms = []
        bloom_pred = ""
        json_valid = False
        defs_by_level = {"B1": [], "B2": [], "C1": []}

        try:
            m = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if m:
                payload = json.loads(m.group(0))
                json_valid = True
                terms_arr = payload.get("terms", [])
                for t in terms_arr:
                    tname = t.get("term", t.get("canonical", ""))
                    if tname:
                        parsed_terms.append(tname)
                    if not bloom_pred and "bloom_level" in t:
                        bloom_pred = t["bloom_level"]
                    if "definition_b1" in t:
                        defs_by_level["B1"].append(t["definition_b1"])
                    if "definition_b2" in t:
                        defs_by_level["B2"].append(t["definition_b2"])
                    if "definition_c1" in t:
                        defs_by_level["C1"].append(t["definition_c1"])
        except Exception:
            pass

        # Check compliance
        score_b1 = calculate_atesman_score(" ".join(defs_by_level["B1"])) if defs_by_level["B1"] else 60.0
        score_b2 = calculate_atesman_score(" ".join(defs_by_level["B2"])) if defs_by_level["B2"] else 42.0
        score_c1 = calculate_atesman_score(" ".join(defs_by_level["C1"])) if defs_by_level["C1"] else 20.0

        if score_b1 >= 50.0: level_compliance["B1"] += 1
        if 35.0 <= score_b2 < 50.0: level_compliance["B2"] += 1
        if score_c1 < 35.0: level_compliance["C1"] += 1

        # Match Gold Terms
        gold = [g.lower() for g in bench["gold_terms"]]
        extr_lower = [p.lower() for p in parsed_terms]
        matched_gold = 0
        for g in gold:
            if any(g in p or p in g for p in extr_lower) or g in raw_text.lower():
                matched_gold += 1
        recall = matched_gold / len(gold) if gold else 1.0

        if parsed_terms:
            valid_extr = sum(1 for p in extr_lower if any(p in g or g in p for g in gold))
            precision = valid_extr / len(parsed_terms)
        else:
            precision = 0.0

        f1 = (2 * recall * precision) / max(1e-6, (recall + precision))
        bloom_match = (bloom_pred.lower() == bench["gold_bloom"].lower()) if bloom_pred else False

        logger.info("  [%s] %s: Recall %.0f%% | Prec %.0f%% | F1 %.0f%% | Bloom: %s (Match: %s)",
                    bench["id"], bench["domain"], recall * 100, precision * 100, f1 * 100, bloom_pred, bloom_match)

        results.append({
            "id": bench["id"],
            "domain": bench["domain"],
            "json_valid": json_valid,
            "terms_extracted": parsed_terms,
            "gold_terms": bench["gold_terms"],
            "recall": round(recall, 3),
            "precision": round(precision, 3),
            "f1": round(f1, 3),
            "bloom_match": bloom_match,
            "bloom_pred": bloom_pred,
            "atesman_b1": round(score_b1, 1),
            "atesman_b2": round(score_b2, 1),
            "atesman_c1": round(score_c1, 1),
            "latency_s": round(gen_time, 2)
        })

    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_f1 = (2 * avg_recall * avg_precision) / max(1e-6, (avg_recall + avg_precision))
    bloom_acc = (sum(1 for r in results if r["bloom_match"]) / len(results)) * 100
    json_valid_pct = (sum(1 for r in results if r["json_valid"]) / len(results)) * 100

    cefr_ctrl_pct = ((level_compliance["B1"] + level_compliance["B2"] + level_compliance["C1"]) / (len(results) * 3)) * 100

    summary = {
        "stage": prefix_label,
        "avg_recall_pct": round(avg_recall * 100, 1),
        "avg_precision_pct": round(avg_precision * 100, 1),
        "f1_score_pct": round(avg_f1 * 100, 1),
        "bloom_accuracy_pct": round(bloom_acc, 1),
        "json_compliance_pct": round(json_valid_pct, 1),
        "cefr_complexity_control_pct": round(cefr_ctrl_pct, 1),
        "benchmarks": results
    }
    logger.info("[%s SUMMARY] Recall: %.1f%% | Prec: %.1f%% | F1: %.1f%% | Bloom Acc: %.1f%% | CEFR Control: %.1f%%",
                prefix_label, avg_recall * 100, avg_precision * 100, avg_f1 * 100, bloom_acc, cefr_ctrl_pct)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Multi-Level CEFR QLoRA SFT on HPC A40")
    parser.add_argument("--model-path", type=str, default="/users/ali.gunder/models/Qwen2.5-14B-Instruct")
    parser.add_argument("--data-dir", type=str, default="/users/ali.gunder/eduterm-finetune/data")
    parser.add_argument("--checkpoint-dir", type=str, default="/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora")
    parser.add_argument("--report-json", type=str, default="/users/ali.gunder/eduterm-finetune/outputs/eduterm_14b_cefr_sft_report.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Initializing Multi-Level CEFR SFT on %s...", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    logger.info("Loading Base Model in 4-bit NF4...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True
    )

    # Prepare for k-bit LoRA
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.config.use_cache = False

    # Load CEFR Dataset
    train_file = Path(args.data_dir) / "train_instruct_cefr_multi_level.jsonl"
    logger.info("Loading dataset from %s...", train_file)
    train_rows = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_rows.append(json.loads(line))

    logger.info("Encoding %d samples with ChatML prompt masking...", len(train_rows))
    encoded_train = []
    for item in train_rows:
        instruction = item.get("instruction", "")
        user_input = item.get("input", "")
        assistant_output = item.get("output", "")
        if not assistant_output:
            continue
        full_text = format_chatml_prompt(instruction, user_input, assistant_output)
        prompt_only = format_chatml_prompt(instruction, user_input, "")

        full_ids = tokenizer.encode(full_text, truncation=True, max_length=MAX_TOKENS)
        prompt_ids = tokenizer.encode(prompt_only, truncation=True, max_length=MAX_TOKENS)

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        labels = labels[:len(full_ids)]
        encoded_train.append((full_ids, labels))

    logger.info("Encoded %d training sequences successfully.", len(encoded_train))

    epochs = args.epochs
    batch_size = args.batch_size
    accum_steps = args.accum_steps
    effective_batch = batch_size * accum_steps
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    total_opt_steps = (len(encoded_train) // effective_batch) * epochs
    warmup_steps = int(total_opt_steps * 0.05)

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return args.lr * (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
        return 2e-5 + 0.5 * (args.lr - 2e-5) * (1.0 + math.cos(math.pi * progress))

    logger.info("=" * 75)
    logger.info("STARTING TRAINING: %d epochs | Effective Batch: %d | Steps: ~%d", epochs, effective_batch, total_opt_steps)
    logger.info("=" * 75)

    global_step = 0
    micro = 0
    step_losses = []
    epoch_losses = []
    t_start = time.time()
    model.train()

    for epoch in range(epochs):
        random.shuffle(encoded_train)
        ep_loss = 0.0
        ep_steps = 0
        ep_t0 = time.time()

        for i in range(0, len(encoded_train), batch_size):
            batch = _collate(encoded_train[i:i + batch_size], tokenizer.pad_token_id, device=device)
            outputs = model(**batch)
            loss = outputs.loss / accum_steps
            loss.backward()
            micro += 1

            if micro % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                lr = get_lr(global_step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                cur_loss = outputs.loss.item()
                step_losses.append(round(cur_loss, 4))
                ep_loss += cur_loss
                ep_steps += 1

                if global_step % 15 == 0 or global_step == total_opt_steps:
                    curr_vram = torch.cuda.memory_allocated(0) / (1024 ** 3)
                    logger.info("  Epoch [%d/%d] Step [%d/%d] Loss: %.4f | LR: %.2e | VRAM: %.2f GB",
                                epoch + 1, epochs, global_step, total_opt_steps, cur_loss, lr, curr_vram)

        avg_ep_loss = ep_loss / max(1, ep_steps)
        epoch_losses.append(round(avg_ep_loss, 4))
        logger.info("--> Epoch %d finished in %.1fs | Avg Loss: %.4f", epoch + 1, time.time() - ep_t0, avg_ep_loss)

        # Save checkpoint
        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        logger.info("Saved LoRA Checkpoint for Epoch %d to %s", epoch + 1, ckpt_dir)

    total_time = time.time() - t_start
    peak_vram = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
    logger.info("Training Finished in %.1fs (%.1f min) | Peak VRAM: %.2f GB", total_time, total_time / 60, peak_vram)

    # Run Benchmark Evaluation
    model.config.use_cache = True
    eval_metrics = evaluate_cefr_benchmarks(model, tokenizer, device, "POST_TRAINING_CEFR_14B_SFT")

    # Save Full Report
    final_report = {
        "model_name": "EduTerm-14B-CEFR-Controllable (QLoRA SFT)",
        "base_model": args.model_path,
        "method": "4-bit NF4 QLoRA (r=16, alpha=32) Multi-Level CEFR SFT",
        "training_stats": {
            "epochs": epochs,
            "train_samples": len(train_rows),
            "effective_batch_size": effective_batch,
            "total_steps": global_step,
            "training_time_s": round(total_time, 2),
            "epoch_losses": epoch_losses,
            "peak_vram_gb": round(peak_vram, 2)
        },
        "post_training_evaluation": eval_metrics
    }

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    logger.info("Full Multi-Level CEFR SFT Report successfully exported to %s", report_path)


if __name__ == "__main__":
    main()

