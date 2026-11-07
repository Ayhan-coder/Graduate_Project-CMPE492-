#!/usr/bin/env python3
"""EduTerm-Trendyol-8B: Multi-Level CEFR (B1/B2/C1) SFT Training & Evaluation on HPC.

Fine-tunes the state-of-the-art Turkish reasoning model Trendyol/Trendyol-LLM-8B-T1 on NVIDIA A40 (48GB)
using the 4,666 multi-tier CEFR dataset to benchmark against Kumru-2B and Qwen2.5-14B.
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
logger = logging.getLogger("EduTerm-Trendyol-8B-SFT")

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
        "text": "Bankaların likidite karşılama oranı Basel III standartlarına göre belirlir ve TCMB zorunlu karşılık oranına tabidir.",
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

def format_trendyol_prompt(instruction: str, user_input: str, assistant_output: str = "", system: str = SYSTEM_PROMPT) -> str:
    """Trendyol-LLM-8B-T1 uses ChatML: <|im_start|>system...<|im_end|><|im_start|>user...<|im_end|><|im_start|>assistant...<|im_end|>"""
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

def evaluate_trendyol_benchmarks(model, tokenizer, device: str, prefix_label: str) -> Dict[str, Any]:
    """Evaluate both terminology extraction and CEFR level adaptability."""
    model.eval()
    results = []
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    eos_id = tokenizer.eos_token_id
    stop_ids = [tid for tid in [im_end_id, eos_id] if tid is not None]

    logger.info("=" * 75)
    logger.info("RUNNING BENCHMARK EVALUATION [%s]", prefix_label)
    logger.info("=" * 75)

    for bench in BENCHMARKS:
        instruction = (
            "Aşağıdaki ders notu metninden teknik terimleri tespit et. "
            "Her terim için Bloom taksonomisi belirle ve CEFR B1 (Sade), B2 (Teknik) ve C1 (Akademik) "
            "düzeylerinde açıklamalar üret. Sonucu geçerli JSON olarak ver."
        )
        prompt = format_trendyol_prompt(instruction, bench["text"], "")
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=450,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=stop_ids
            )
        elapsed = time.time() - t0

        raw_gen = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Parse JSON
        parsed_json = None
        json_match = re.search(r"```json\s*(.*?)\s*```", raw_gen, re.DOTALL)
        candidate = json_match.group(1) if json_match else raw_gen
        try:
            parsed_json = json.loads(candidate)
        except Exception:
            try:
                first_brace = candidate.find("{")
                last_brace = candidate.rfind("}")
                if first_brace != -1 and last_brace != -1:
                    parsed_json = json.loads(candidate[first_brace:last_brace + 1])
            except Exception:
                parsed_json = None

        extracted_terms = []
        cefr_readabilities = {"B1": [], "B2": [], "C1": []}
        predicted_bloom = "Bilinmiyor"

        if isinstance(parsed_json, dict):
            terms_obj = parsed_json.get("terimler") or parsed_json.get("terms") or []
            if isinstance(terms_obj, list):
                for t in terms_obj:
                    if isinstance(t, dict):
                        t_name = t.get("terim") or t.get("term") or ""
                        if t_name:
                            extracted_terms.append(t_name)
                        b_val = t.get("bloom_seviyesi") or t.get("bloom_level") or t.get("bloom") or ""
                        if b_val and predicted_bloom == "Bilinmiyor":
                            predicted_bloom = b_val
                        for lvl in ["B1", "B2", "C1"]:
                            expl = t.get(lvl) or t.get(f"{lvl.lower()}_aciklama") or ""
                            if expl:
                                cefr_readabilities[lvl].append(calculate_atesman_score(expl))
            elif isinstance(terms_obj, dict):
                for k in terms_obj.keys():
                    extracted_terms.append(k)

        # Compute metric overlaps
        def normalize(s: str) -> str:
            return re.sub(r"[^\w\s]", "", s.lower()).strip()

        norm_gold = [normalize(t) for t in bench["gold_terms"]]
        norm_pred = [normalize(t) for t in extracted_terms if t]
        tp = sum(1 for p in norm_pred if any(p in g or g in p for g in norm_gold))
        precision = tp / max(1, len(norm_pred)) if norm_pred else 0.0
        recall = tp / max(1, len(norm_gold)) if norm_gold else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        avg_b1 = sum(cefr_readabilities["B1"]) / max(1, len(cefr_readabilities["B1"])) if cefr_readabilities["B1"] else 0.0
        avg_b2 = sum(cefr_readabilities["B2"]) / max(1, len(cefr_readabilities["B2"])) if cefr_readabilities["B2"] else 0.0
        avg_c1 = sum(cefr_readabilities["C1"]) / max(1, len(cefr_readabilities["C1"])) if cefr_readabilities["C1"] else 0.0

        is_steering_valid = (avg_b1 >= avg_b2 >= avg_c1) if (avg_b1 > 0 and avg_b2 > 0 and avg_c1 > 0) else False

        results.append({
            "id": bench["id"],
            "domain": bench["domain"],
            "elapsed_sec": round(elapsed, 2),
            "json_valid": parsed_json is not None,
            "gold_terms": bench["gold_terms"],
            "extracted_terms": extracted_terms,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "predicted_bloom": predicted_bloom,
            "bloom_correct": normalize(predicted_bloom) in normalize(bench["gold_bloom"]),
            "readability_atesman": {
                "B1": round(avg_b1, 1),
                "B2": round(avg_b2, 1),
                "C1": round(avg_c1, 1)
            },
            "cefr_steering_compliant": is_steering_valid,
            "raw_output": raw_gen[:250] + "..." if len(raw_gen) > 250 else raw_gen
        })

    avg_f1 = sum(r["f1"] for r in results) / len(results)
    avg_p = sum(r["precision"] for r in results) / len(results)
    avg_r = sum(r["recall"] for r in results) / len(results)
    json_acc = sum(1 for r in results if r["json_valid"]) / len(results)
    bloom_acc = sum(1 for r in results if r["bloom_correct"]) / len(results)
    steering_acc = sum(1 for r in results if r["cefr_steering_compliant"]) / len(results)

    logger.info("Summary [%s]: F1=%.2f%%, Precision=%.2f%%, Recall=%.2f%%, JSON=%.1f%%, CEFR Steering=%.1f%%",
                prefix_label, avg_f1 * 100, avg_p * 100, avg_r * 100, json_acc * 100, steering_acc * 100)

    return {
        "label": prefix_label,
        "metrics": {
            "macro_precision": round(avg_p, 4),
            "macro_recall": round(avg_r, 4),
            "macro_f1": round(avg_f1, 4),
            "json_schema_validity": round(json_acc, 4),
            "bloom_accuracy": round(bloom_acc, 4),
            "cefr_steering_rate": round(steering_acc, 4)
        },
        "benchmarks": results
    }

def main():
    parser = argparse.ArgumentParser(description="Multi-Level CEFR QLoRA SFT for Trendyol-LLM-8B-T1 on HPC A40")
    parser.add_argument("--model-path", type=str, default="/users/ali.gunder/models/Trendyol-LLM-8B-T1")
    parser.add_argument("--data-dir", type=str, default="/users/ali.gunder/eduterm-finetune/data")
    parser.add_argument("--checkpoint-dir", type=str, default="/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_trendyol_8b_cefr")
    parser.add_argument("--report-json", type=str, default="/users/ali.gunder/eduterm-finetune/outputs/eduterm_trendyol_8b_cefr_report.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Initializing Trendyol-LLM-8B-T1 Multi-Level CEFR SFT on %s...", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    logger.info("Loading Trendyol-LLM-8B-T1 Base Model in 4-bit NF4...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True
    )

    # Initial Zero-Shot Baseline Evaluation
    logger.info("Running Baseline Zero-Shot Benchmark on un-finetuned Trendyol-LLM-8B-T1...")
    baseline_eval = evaluate_trendyol_benchmarks(model, tokenizer, device, "trendyol-8b-zero-shot")

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
        full_text = format_trendyol_prompt(instruction, user_input, assistant_output)
        prompt_only = format_trendyol_prompt(instruction, user_input, "")

        full_ids = tokenizer.encode(full_text, truncation=True, max_length=MAX_TOKENS)
        prompt_ids = tokenizer.encode(prompt_only, truncation=True, max_length=MAX_TOKENS)

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        labels = labels[:len(full_ids)]
        encoded_train.append((full_ids, labels))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    total_steps = (len(encoded_train) // (args.batch_size * args.accum_steps)) * args.epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    logger.info("Starting SFT Training: %d epochs, total planned steps=%d...", args.epochs, total_steps)
    model.train()
    training_start = time.time()
    step_count = 0
    epoch_losses = []

    for epoch in range(1, args.epochs + 1):
        random.seed(42 + epoch)
        random.shuffle(encoded_train)
        running_loss = 0.0
        epoch_steps = 0
        epoch_start = time.time()

        for i in range(0, len(encoded_train), args.batch_size):
            batch_slice = encoded_train[i : i + args.batch_size]
            batch = _collate(batch_slice, pad_id=pad_id, device=device)

            outputs = model(**batch)
            loss = outputs.loss / args.accum_steps
            loss.backward()

            running_loss += outputs.loss.item()
            epoch_steps += 1

            if epoch_steps % args.accum_steps == 0 or (i + args.batch_size >= len(encoded_train)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                step_count += 1

                if step_count % 50 == 0:
                    current_loss = running_loss / epoch_steps
                    lr_curr = lr_scheduler.get_last_lr()[0]
                    vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    logger.info(
                        "Epoch %d/%d | Step %d/%d | Loss: %.4f | LR: %.2e | Peak VRAM: %.2f GB",
                        epoch, args.epochs, step_count, total_steps, current_loss, lr_curr, vram_gb
                    )

        avg_loss = running_loss / max(1, epoch_steps)
        epoch_duration = time.time() - epoch_start
        epoch_losses.append({
            "epoch": epoch,
            "avg_loss": round(avg_loss, 4),
            "duration_sec": round(epoch_duration, 1)
        })
        logger.info(">>> Epoch %d Completed in %.1fs. Average Loss: %.4f", epoch, epoch_duration, avg_loss)

    total_train_sec = time.time() - training_start
    logger.info("Training Finished in %.1f seconds (~%.2f hours).", total_train_sec, total_train_sec / 3600)

    # Save Adapter
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger.info("Saving LoRA adapter to %s...", args.checkpoint_dir)
    model.save_pretrained(args.checkpoint_dir)
    tokenizer.save_pretrained(args.checkpoint_dir)

    # Post-training Benchmark Evaluation
    post_eval = evaluate_trendyol_benchmarks(model, tokenizer, device, "eduterm-trendyol-8b-cefr")

    # Final summary report
    summary = {
        "model_name": "Trendyol/Trendyol-LLM-8B-T1",
        "checkpoint_dir": args.checkpoint_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "effective_batch_size": args.batch_size * args.accum_steps,
        "learning_rate": args.lr,
        "total_train_sec": round(total_train_sec, 2),
        "total_train_hours": round(total_train_sec / 3600, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2),
        "epoch_losses": epoch_losses,
        "baseline_zero_shot": baseline_eval,
        "post_sft_evaluation": post_eval,
        "dataset_samples": len(encoded_train)
    }

    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Trendyol-8B SFT Report successfully written to %s", args.report_json)

if __name__ == "__main__":
    main()

