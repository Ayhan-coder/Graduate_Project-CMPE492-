#!/usr/bin/env python3
"""
train_dpo_qwen.py: Direct Preference Optimization (DPO) on Boğaziçi HPC NVIDIA A40 (48GB).

Aligns EduTerm-7B-TR with pedagogical CEFR B1 readability standards and eliminates academic over-complication.
Mathematically optimizes the implicit reward margin between:
- Chosen (y_w): Lucid, B1-compliant definitions with intuitive real-world analogies (Ateşman 67.5).
- Rejected (y_l): Overly dense, inaccessible academic / bureaucratic phrasing (Ateşman 11.9).

Loss Formulation (Rafailov et al., 2023):
    L_DPO = -E [ log \sigma ( \beta ( log \pi_\theta(y_w|x) - log \pi_ref(y_w|x) )
                            - \beta ( log \pi_\theta(y_l|x) - log \pi_ref(y_l|x) ) ) ]
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("DPOTraining")


class DPODataset(Dataset):
    """Dataset container for (prompt, chosen, rejected) triplets."""

    def __init__(self, data_file: Path, tokenizer: Any, max_len: int = 768):
        self.samples = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        prompt = item["prompt"]
        chosen = item["chosen"] + "<|im_end|>"
        rejected = item["rejected"] + "<|im_end|>"

        # Tokenize prompt alone to know where response starts
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_tokens)

        # Full sequences: prompt + chosen and prompt + rejected
        full_chosen = prompt + chosen
        full_rejected = prompt + rejected

        chosen_enc = self.tokenizer(
            full_chosen,
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt"
        )
        rejected_enc = self.tokenizer(
            full_rejected,
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "term": item.get("term", ""),
            "domain": item.get("domain", ""),
            "prompt_len": prompt_len,
            "chosen_input_ids": chosen_enc.input_ids[0],
            "chosen_attention_mask": chosen_enc.attention_mask[0],
            "rejected_input_ids": rejected_enc.input_ids[0],
            "rejected_attention_mask": rejected_enc.attention_mask[0],
            "chosen_atesman": item.get("chosen_atesman", 65.0),
            "rejected_atesman": item.get("rejected_atesman", 15.0)
        }


def pad_collate_fn(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    """Pad batch of variable length chosen/rejected tensors."""
    max_chosen_len = max(len(b["chosen_input_ids"]) for b in batch)
    max_rejected_len = max(len(b["rejected_input_ids"]) for b in batch)

    chosen_ids = []
    chosen_mask = []
    rejected_ids = []
    rejected_mask = []
    prompt_lens = []

    for b in batch:
        c_len = len(b["chosen_input_ids"])
        c_pad = max_chosen_len - c_len
        c_id = torch.cat([b["chosen_input_ids"], torch.full((c_pad,), pad_token_id, dtype=torch.long)])
        c_m = torch.cat([b["chosen_attention_mask"], torch.zeros(c_pad, dtype=torch.long)])

        r_len = len(b["rejected_input_ids"])
        r_pad = max_rejected_len - r_len
        r_id = torch.cat([b["rejected_input_ids"], torch.full((r_pad,), pad_token_id, dtype=torch.long)])
        r_m = torch.cat([b["rejected_attention_mask"], torch.zeros(r_pad, dtype=torch.long)])

        chosen_ids.append(c_id)
        chosen_mask.append(c_m)
        rejected_ids.append(r_id)
        rejected_mask.append(r_m)
        prompt_lens.append(b["prompt_len"])

    return {
        "chosen_input_ids": torch.stack(chosen_ids),
        "chosen_attention_mask": torch.stack(chosen_mask),
        "rejected_input_ids": torch.stack(rejected_ids),
        "rejected_attention_mask": torch.stack(rejected_mask),
        "prompt_lens": prompt_lens
    }


def compute_sequence_logps(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: List[int]
) -> torch.Tensor:
    """Compute sum of log probabilities strictly over the completion tokens (excluding prompt and padding)."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # Shift logits and labels by 1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_logps = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(2)).squeeze(2)

    seq_logps = []
    batch_size = input_ids.shape[0]
    for i in range(batch_size):
        # Completion starts after prompt_len - 1 in shifted coordinates
        start_idx = max(0, prompt_lens[i] - 1)
        valid_mask = shift_mask[i].bool()
        # Mask out prompt tokens
        valid_mask[:start_idx] = False
        seq_logps.append((per_token_logps[i] * valid_mask).sum())

    return torch.stack(seq_logps)


def precompute_reference_logps(
    ref_model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda:0"
) -> Tuple[List[float], List[float]]:
    """Precompute log \pi_ref(y_w|x) and log \pi_ref(y_l|x) for all samples."""
    ref_model.eval()
    chosen_ref_logps = []
    rejected_ref_logps = []

    logger.info("Precomputing Reference Model Log-Probabilities...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Reference LogPs"):
            c_ids = batch["chosen_input_ids"].to(device)
            c_mask = batch["chosen_attention_mask"].to(device)
            r_ids = batch["rejected_input_ids"].to(device)
            r_mask = batch["rejected_attention_mask"].to(device)
            p_lens = batch["prompt_lens"]

            c_logps = compute_sequence_logps(ref_model, c_ids, c_mask, p_lens)
            r_logps = compute_sequence_logps(ref_model, r_ids, r_mask, p_lens)

            chosen_ref_logps.extend(c_logps.cpu().tolist())
            rejected_ref_logps.extend(r_logps.cpu().tolist())

    return chosen_ref_logps, rejected_ref_logps


def run_dpo_training(
    base_dir: str,
    model_path: str,
    sft_lora_path: str,
    output_checkpoint_dir: str,
    beta: float = 0.1,
    learning_rate: float = 5e-6,
    num_epochs: int = 3,
    batch_size: int = 4,
    grad_accum_steps: int = 2
):
    base_path = Path(base_dir)
    data_dir = base_path / "data"
    train_file = data_dir / "train_dpo.jsonl"
    val_file = data_dir / "val_dpo.jsonl"
    ckpt_dir = Path(output_checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("BOUN HPC NVIDIA A40 PEDAGOGICAL DPO ALIGNMENT TRAINING")
    logger.info("=" * 80)
    logger.info(f"Base Model:       {model_path}")
    logger.info(f"SFT Reference:    {sft_lora_path}")
    logger.info(f"Output Checkpoint:{output_checkpoint_dir}")
    logger.info(f"DPO Beta:         {beta}")
    logger.info(f"Learning Rate:    {learning_rate}")
    logger.info(f"Epochs:           {num_epochs}")
    logger.info(f"Effective Batch:  {batch_size * grad_accum_steps}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0
    logger.info(f"GPU Hardware: {gpu_name} ({total_vram_gb:.2f} GB VRAM)")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info("Loading Tokenizer and Base Model in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 1. Load SFT Model to Precompute Reference LogPs
    logger.info("Instantiating Reference Policy with SFT adapter...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True
    )

    from peft import PeftModel, LoraConfig, get_peft_model
    if sft_lora_path and Path(sft_lora_path).exists():
        model = PeftModel.from_pretrained(model, sft_lora_path)
        logger.info(f"Loaded SFT adapter from {sft_lora_path} for reference policy.")

    train_dataset = DPODataset(train_file, tokenizer, max_len=768)
    val_dataset = DPODataset(val_file, tokenizer, max_len=768)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order consistent with precomputed logps
        collate_fn=lambda b: pad_collate_fn(b, tokenizer.pad_token_id)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: pad_collate_fn(b, tokenizer.pad_token_id)
    )

    # Precompute Reference LogPs
    ref_chosen_train, ref_rejected_train = precompute_reference_logps(model, train_loader, device=device)
    ref_chosen_val, ref_rejected_val = precompute_reference_logps(model, val_loader, device=device)

    # Clear cache and prepare for active training
    gc.collect()
    torch.cuda.empty_cache()

    # Enable gradients on active model adapter
    model.train()
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable Parameters for DPO: {trainable_params:,} ({100*trainable_params/all_params:.2f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    total_steps = (len(train_loader) // grad_accum_steps) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    training_history = []
    global_step = 0
    t_start = time.time()

    logger.info("\nStarting DPO Optimization...")
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_margin = 0.0
        batch_count = 0
        optimizer.zero_grad()

        sample_offset = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

        for b_idx, batch in enumerate(pbar):
            c_ids = batch["chosen_input_ids"].to(device)
            c_mask = batch["chosen_attention_mask"].to(device)
            r_ids = batch["rejected_input_ids"].to(device)
            r_mask = batch["rejected_attention_mask"].to(device)
            p_lens = batch["prompt_lens"]
            b_size = len(p_lens)

            # Retrieve precomputed reference logps for this slice
            ref_cw = torch.tensor(ref_chosen_train[sample_offset : sample_offset + b_size], device=device)
            ref_cl = torch.tensor(ref_rejected_train[sample_offset : sample_offset + b_size], device=device)
            sample_offset += b_size

            # Compute policy logps with gradients
            pi_cw = compute_sequence_logps(model, c_ids, c_mask, p_lens)
            pi_cl = compute_sequence_logps(model, r_ids, r_mask, p_lens)

            # Implicit rewards: r(x,y) = \beta * (log \pi(y|x) - log \pi_ref(y|x))
            r_w = beta * (pi_cw - ref_cw)
            r_l = beta * (pi_cl - ref_cl)
            margin = r_w - r_l

            # DPO Loss: -log \sigma(r_w - r_l)
            loss = -F.logsigmoid(margin).mean()
            loss = loss / grad_accum_steps
            loss.backward()

            accuracy = (margin > 0).float().mean().item()
            epoch_loss += loss.item() * grad_accum_steps
            epoch_acc += accuracy
            epoch_margin += margin.mean().item()
            batch_count += 1

            if (b_idx + 1) % grad_accum_steps == 0 or (b_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item()*grad_accum_steps:.4f}",
                "acc": f"{accuracy*100:.1f}%",
                "margin": f"{margin.mean().item():.3f}"
            })

        avg_train_loss = epoch_loss / batch_count
        avg_train_acc = (epoch_acc / batch_count) * 100
        avg_train_margin = epoch_margin / batch_count

        # Validation Pass
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_margin = 0.0
        v_offset = 0
        v_batches = 0

        with torch.no_grad():
            for v_batch in val_loader:
                vc_ids = v_batch["chosen_input_ids"].to(device)
                vc_mask = v_batch["chosen_attention_mask"].to(device)
                vr_ids = v_batch["rejected_input_ids"].to(device)
                vr_mask = v_batch["rejected_attention_mask"].to(device)
                vp_lens = v_batch["prompt_lens"]
                vb_size = len(vp_lens)

                v_ref_cw = torch.tensor(ref_chosen_val[v_offset : v_offset + vb_size], device=device)
                v_ref_cl = torch.tensor(ref_rejected_val[v_offset : v_offset + vb_size], device=device)
                v_offset += vb_size

                v_pi_cw = compute_sequence_logps(model, vc_ids, vc_mask, vp_lens)
                v_pi_cl = compute_sequence_logps(model, vr_ids, vr_mask, vp_lens)

                v_rw = beta * (v_pi_cw - v_ref_cw)
                v_rl = beta * (v_pi_cl - v_ref_cl)
                v_m = v_rw - v_rl
                v_l = -F.logsigmoid(v_m).mean()

                val_loss += v_l.item()
                val_acc += (v_m > 0).float().mean().item()
                val_margin += v_m.mean().item()
                v_batches += 1

        avg_val_loss = val_loss / v_batches
        avg_val_acc = (val_acc / v_batches) * 100
        avg_val_margin = val_margin / v_batches

        logger.info(f"Epoch {epoch} Summary: Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.1f}% | Margin: {avg_train_margin:.3f}")
        logger.info(f"                 Val Loss:   {avg_val_loss:.4f} | Val Acc:   {avg_val_acc:.1f}% | Val Margin:   {avg_val_margin:.3f}")

        training_history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "train_acc": round(avg_train_acc, 2),
            "train_margin": round(avg_train_margin, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_acc": round(avg_val_acc, 2),
            "val_margin": round(avg_val_margin, 4)
        })

    elapsed_time = time.time() - t_start
    peak_vram_gb = torch.cuda.max_memory_allocated(0) / (1024**3)

    # Save Aligned DPO Checkpoint
    logger.info(f"\nSaving Aligned DPO Checkpoint to {ckpt_dir}...")
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    # Save Experiment Report
    report = {
        "experiment": "BOUN HPC NVIDIA A40 Pedagogical DPO Alignment",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "gpu": gpu_name,
            "vram_total_gb": total_vram_gb,
            "peak_vram_allocated_gb": round(peak_vram_gb, 2)
        },
        "hyperparameters": {
            "beta": beta,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "effective_batch_size": batch_size * grad_accum_steps
        },
        "dataset_statistics": {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "average_chosen_atesman": 67.53,
            "average_rejected_atesman": 11.97,
            "readability_delta_points": 55.56
        },
        "results": {
            "elapsed_seconds": round(elapsed_time, 2),
            "final_train_loss": training_history[-1]["train_loss"],
            "final_train_accuracy_pct": training_history[-1]["train_acc"],
            "final_train_reward_margin": training_history[-1]["train_margin"],
            "final_val_loss": training_history[-1]["val_loss"],
            "final_val_accuracy_pct": training_history[-1]["val_acc"],
            "final_val_reward_margin": training_history[-1]["val_margin"],
            "preference_optimization_gain_pct": round(training_history[-1]["val_acc"] - 50.0, 2)
        },
        "epoch_history": training_history
    }

    report_path = base_path / "outputs" / "dpo_experiment_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("=" * 80)
    logger.info(f"DPO TRAINING FINISHED SUCCESSFULLY in {elapsed_time/60:.2f} minutes!")
    logger.info(f"Final Validation Preference Accuracy: {training_history[-1]['val_acc']:.1f}%")
    logger.info(f"Report saved to {report_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pedagogical DPO on A40")
    parser.add_argument("--base-dir", type=str, default="/users/ali.gunder/eduterm-dpo", help="Base directory")
    parser.add_argument("--model-path", type=str, default="/users/ali.gunder/models/Qwen2.5-7B-Instruct", help="Base model")
    parser.add_argument("--sft-lora-path", type=str, default="/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_7b_tr_sft", help="SFT adapter")
    parser.add_argument("--output-dir", type=str, default="/users/ali.gunder/eduterm-dpo/checkpoints/eduterm_7b_tr_dpo", help="Save checkpoint")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs")
    args = parser.parse_args()

    run_dpo_training(
        base_dir=args.base_dir,
        model_path=args.model_path,
        sft_lora_path=args.sft_lora_path,
        output_checkpoint_dir=args.output_dir,
        beta=args.beta,
        learning_rate=args.lr,
        num_epochs=args.epochs
    )
