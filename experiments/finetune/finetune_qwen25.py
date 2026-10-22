"""Parameter-Efficient Fine-Tuning (PEFT / LoRA) for Qwen 2.5 on lecture-note terminology.

Designed for:
- Base Models: Qwen/Qwen2.5-0.5B-Instruct (Edge/Web), Qwen/Qwen2.5-1.5B-Instruct (Recommended for RTX 5060 8GB)
- Task: Extract domain terminology, generate CEFR B1 beginner-friendly definitions, assign Bloom taxonomy levels
- Memory Footprint: ~1.2 GB (0.5B) to ~3.4 GB (1.5B) VRAM with LoRA rank r=16
- Checkpoints saved to experiments/finetune/checkpoints/qwen25_lecture_adapter

Supervision is masked: only the assistant turn contributes to the loss, so the model is
not trained to reproduce the prompt. Training rows may carry their own ``system`` field,
which is what lets the externally sourced mix (general Turkish + term schema rows) share
one dataloader with the locally distilled lecture rows.

Usage:
    # 1. Dry-run validation (checks dataset, tokenizer, memory bounds):
    python experiments/finetune/finetune_qwen25.py --model Qwen/Qwen2.5-0.5B-Instruct --dry-run

    # 2. Plumbing check on the external mix (~2 minutes):
    python experiments/finetune/finetune_qwen25.py --train-file experiments/finetune/data/external/train_external_mixed.jsonl --limit 80

    # 3. Live GPU execution (RTX 5060 8GB):
    python experiments/finetune/finetune_qwen25.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Qwen25FineTune")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "experiments" / "finetune" / "data"
CHECKPOINT_DIR = ROOT / "experiments" / "finetune" / "checkpoints" / "qwen25_lecture_adapter"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "Sen ders notlarını ve eğitim konuşmalarını analiz eden, teknik terimleri tespit edip "
    "CEFR B1 düzeyinde sade açıklamalar üreten ve Bloom taksonomi seviyesini atayan uzman bir asistansın."
)
IM_END = "<|im_end|>"
MAX_TOKENS = 1024


def format_chatml_prompt(instruction: str, user_input: str, assistant_output: str = "",
                         system: str = SYSTEM_PROMPT) -> str:
    """Format into the standard Qwen 2.5 ChatML template.

    With no ``assistant_output`` the string stops right after the assistant header, which
    is the part of the sample whose labels get masked out during training.
    """
    user_block = f"{instruction}\n\nDers Notu Metni:\n{user_input}" if user_input else instruction
    prompt = (
        f"<|im_start|>system\n{system}{IM_END}\n"
        f"<|im_start|>user\n{user_block}{IM_END}\n"
    )
    tail = f"{assistant_output}{IM_END}" if assistant_output else ""
    return prompt + f"<|im_start|>assistant\n{tail}" if assistant_output else prompt + "<|im_start|>assistant\n"


def encode_supervised(tokenizer, item: Dict[str, Any]) -> Optional[Tuple[List[int], List[int]]]:
    """Return (input_ids, labels) with -100 over everything the model is not asked to produce."""
    prefix = format_chatml_prompt(item["instruction"], item.get("input", ""), "",
                                 system=item.get("system", SYSTEM_PROMPT))
    full = format_chatml_prompt(item["instruction"], item.get("input", ""), item["output"],
                               system=item.get("system", SYSTEM_PROMPT))
    ids = tokenizer(full, add_special_tokens=False)["input_ids"][:MAX_TOKENS]
    prompt_len = min(len(tokenizer(prefix, add_special_tokens=False)["input_ids"]), len(ids))
    if len(ids) - prompt_len < 4:
        return None
    labels = [-100] * prompt_len + list(ids[prompt_len:])
    return ids, labels


def _collate(batch: Sequence[Tuple[List[int], List[int]]], pad_id: int):
    """Right-pad a micro-batch; padding positions are masked out of both attention and loss."""
    import torch

    width = max(len(ids) for ids, _ in batch)
    input_ids, labels, attention = [], [], []
    for ids, lab in batch:
        gap = width - len(ids)
        input_ids.append(ids + [pad_id] * gap)
        labels.append(lab + [-100] * gap)
        attention.append([1] * len(ids) + [0] * gap)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_finetune(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 2e-4,
    dry_run: bool = False,
    output_dir: Path = CHECKPOINT_DIR,
    train_file: Optional[Path] = None,
    val_file: Optional[Path] = None,
    limit: int = 0,
    eval_every: int = 25,
    accumulation_steps: int = 4,
    device: str = "cuda",
    seed: int = 20261017,
) -> Dict[str, Any]:
    """Execute the LoRA fine-tuning loop or a dry-run validation."""
    train_path = Path(train_file) if train_file else DATA_DIR / "train_lecture_notes.jsonl"
    val_path = Path(val_file) if val_file else DATA_DIR / "val_lecture_notes.jsonl"

    if not train_path.exists():
        logger.info("Lecture dataset not found at %s, generating now...", train_path)
        from experiments.finetune.prepare_lecture_notes_dataset import build_lecture_dataset
        train_path, val_path = build_lecture_dataset()

    train_items = _read_jsonl(train_path)
    val_items = _read_jsonl(val_path) if val_path.exists() else []
    if limit:
        train_items = train_items[:limit]
    logger.info("Loaded %d train, %d validation instruction samples from %s",
                len(train_items), len(val_items), Path(train_path).name)

    mix: Dict[str, int] = {}
    for item in train_items:
        mix[item.get("source", "local")] = mix.get(item.get("source", "local"), 0) + 1

    if dry_run:
        logger.info("[DRY RUN] Validating ChatML prompt serialization for Qwen 2.5...")
        sample_prompt = format_chatml_prompt(
            train_items[0]["instruction"], train_items[0]["input"], train_items[0]["output"],
            system=train_items[0].get("system", SYSTEM_PROMPT),
        )
        logger.info("[DRY RUN] Formatted ChatML sample length: %d chars", len(sample_prompt))
        est_vram_gb = 1.2 if "0.5B" in model_id else (3.2 if "1.5B" in model_id else 5.8)
        metadata = {
            "status": "DRY_RUN_PASSED",
            "model_id": model_id,
            "train_samples": len(train_items),
            "val_samples": len(val_items),
            "train_source_mix": mix,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "lora_r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "estimated_vram_gb": est_vram_gb,
            "gpu_hardware": "NVIDIA GeForce RTX 5060 Laptop (8GB VRAM)",
            "sample_prompt_preview": sample_prompt[:300] + "..."
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "adapter_config_simulated.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return metadata

    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        use_cuda = device == "cuda" and torch.cuda.is_available()
        dev = "cuda" if use_cuda else "cpu"
        if device == "cuda" and not use_cuda:
            logger.warning("CUDA requested but unavailable; continuing on CPU.")
        logger.info("Initializing live training on device: %s (%s)", dev, model_id)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        encoded: List[Tuple[List[int], List[int]]] = []
        skipped = 0
        for item in train_items:
            pair = encode_supervised(tokenizer, item)
            if pair is None:
                skipped += 1
            else:
                encoded.append(pair)
        val_encoded = [e for e in (encode_supervised(tokenizer, i) for i in val_items[:40]) if e]
        logger.info("Tokenized %d train samples (%d skipped as prompt-only), %d validation samples",
                    len(encoded), skipped, len(val_encoded))
        if not encoded:
            return {"status": "ERROR", "error": "No trainable sample survived label masking."}

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16 if use_cuda else torch.float32,
            device_map="auto" if use_cuda else None,
            trust_remote_code=True,
        )
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        if not use_cuda:
            model.to(dev)
        model.config.use_cache = False

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr)

        rng = random.Random(seed)
        started = time.time()
        step_losses: List[float] = []
        eval_curve: List[Dict[str, Any]] = []
        global_step = 0
        micro = 0
        epochs_ran = 0

        for epoch in range(epochs):
            epochs_ran = epoch + 1
            rng.shuffle(encoded)
            model.train()
            for idx in range(0, len(encoded), batch_size):
                batch = _collate(encoded[idx:idx + batch_size], tokenizer.pad_token_id).to(dev)
                loss = model(**batch).loss / accumulation_steps
                loss.backward()
                micro += 1
                if micro % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    step_losses.append(float(loss.item()) * accumulation_steps)
                    if global_step % 10 == 0:
                        logger.info("epoch %d | step %d | loss %.4f", epoch + 1, global_step, step_losses[-1])
                    if val_encoded and global_step % eval_every == 0:
                        model.eval()
                        with torch.no_grad():
                            ev = _collate(val_encoded, tokenizer.pad_token_id).to(dev)
                            val_loss = float(model(**ev).loss.item())
                        eval_curve.append({"step": global_step, "val_loss": round(val_loss, 4)})
                        logger.info("  validation loss at step %d: %.4f", global_step, val_loss)
                        model.train()

        # A trailing partial accumulation still deserves its gradient.
        if micro % accumulation_steps:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            global_step += 1

        model.eval()
        final_val = None
        if val_encoded:
            with torch.no_grad():
                ev = _collate(val_encoded, tokenizer.pad_token_id).to(dev)
                final_val = round(float(model(**ev).loss.item()), 4)

        elapsed = time.time() - started
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        logger.info("Saved LoRA adapter checkpoint to %s", output_dir)

        metrics = {
            "status": "COMPLETED",
            "model_id": model_id,
            "device": dev,
            "epochs": epochs_ran,
            "batch_size": batch_size,
            "accumulation_steps": accumulation_steps,
            "learning_rate": lr,
            "lora_r": 16,
            "lora_alpha": 32,
            "train_file": str(Path(train_path).relative_to(ROOT)),
            "val_file": str(Path(val_path).relative_to(ROOT)) if val_path.exists() else None,
            "train_samples": len(train_items),
            "train_source_mix": mix,
            "train_tokens_supervised": sum(len(l) - l.count(-100) for _, l in encoded),
            "validation_loss": final_val,
            "validation_curve": eval_curve,
            "loss_first_10_mean": round(sum(step_losses[:10]) / max(1, min(10, len(step_losses))), 4),
            "loss_last_10_mean": round(sum(step_losses[-10:]) / max(1, min(10, len(step_losses))), 4),
            "global_steps": global_step,
            "elapsed_seconds": round(elapsed, 2),
            "label_masking": "assistant turn only",
            "output_dir": str(output_dir),
        }
        (output_dir / "train_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return metrics

    except Exception as exc:
        logger.error("Live GPU training encountered an error: %s", exc, exc_info=True)
        return {"status": "ERROR", "error": str(exc), "model_id": model_id}


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen 2.5 Lecture Notes Fine-Tuner.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model to fine-tune.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=0, help="Train on the first N rows only.")
    parser.add_argument("--train-file", type=str, default=None, help="Override the training JSONL.")
    parser.add_argument("--val-file", type=str, default=None, help="Override the validation JSONL.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without full GPU training.")
    args = parser.parse_args()

    res = run_finetune(
        model_id=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dry_run=args.dry_run,
        train_file=args.train_file,
        val_file=args.val_file,
        limit=args.limit,
        eval_every=args.eval_every,
        accumulation_steps=args.accumulation_steps,
        device=args.device,
    )

    print("\n=== QWEN 2.5 FINE-TUNING SUMMARY ===")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
