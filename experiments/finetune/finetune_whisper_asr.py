"""LoRA adaptation of the Turkish Whisper ASR model (Katman 1).

Trains on the 16 kHz wav corpus produced by
``experiments/finetune/prepare_external_datasets.py`` -- never on
``experiments/data/TR-*.wav`` / ``EN-*.wav``, which are the held-out clips the
paper's WER numbers are measured on.

Budget note for the RTX 5060 Laptop (8 GB): whisper-large-v3-turbo is ~809 M
parameters, so fp16 weights alone are ~1.6 GB. The loop therefore keeps one
utterance per step, accumulates gradients, and only touches the attention
projections with LoRA. ``--smoke`` runs a handful of steps to prove the plumbing
before committing to hours of training.

Usage:
    # plumbing check, ~1 minute
    python -m experiments.finetune.finetune_whisper_asr --smoke

    # SRM adaptation (Whisper Small, the report's edge recognition model)
    python -m experiments.finetune.finetune_whisper_asr --model openai/whisper-small --epochs 3 \
        --adapter-dir experiments/finetune/checkpoints/srm_whisper_small_tr

    # measured before/after on the held-out clips
    python -m experiments.finetune.finetune_whisper_asr --model openai/whisper-small \
        --adapter-dir experiments/finetune/checkpoints/srm_whisper_small_tr --eval-only
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WhisperAdapter")

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "experiments" / "finetune" / "data" / "external"
MANIFEST = OUT_DIR / "asr_manifest.jsonl"
CHECKPOINT_DIR = ROOT / "experiments" / "finetune" / "checkpoints" / "whisper_tr_adapter"
METRICS_FILE = CHECKPOINT_DIR / "train_metrics.json"

BASE_MODEL = "selimc/whisper-large-v3-turbo-turkish"
MAX_DURATION_S = 30.0
MAX_TARGET_TOKENS = 96


def load_manifest(path: Path = MANIFEST) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `python -m experiments.finetune.prepare_external_datasets` first."
        )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    train = [r for r in rows if r.get("split") == "train"]
    val = [r for r in rows if r.get("split") != "train"]
    return train, val


def _clip_text(text: str) -> str:
    return " ".join(text.split())


def build_batch(device: str, train_rows: List[Dict[str, Any]], index: int,
                feature_extractor, tokenizer, dtype=None) -> Optional[Dict[str, Any]]:
    """Load one utterance and its label tensors; None when the clip is unusable."""
    import soundfile as sf

    row = train_rows[index]
    wav = OUT_DIR / row["audio_path"]
    if not wav.exists():
        return None
    data, sr = sf.read(str(wav), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if len(data) / sr > MAX_DURATION_S:
        data = data[: int(MAX_DURATION_S * sr)]

    features = feature_extractor(
        data, sampling_rate=int(sr), return_tensors="pt", truncation=True
    )["input_features"][0]
    if dtype is not None:
        # The encoder weights are fp16 on GPU; float32 features trip conv1.
        features = features.to(dtype)
    labels = tokenizer(
        _clip_text(row["target_text"]),
        return_tensors="pt", max_length=MAX_TARGET_TOKENS, truncation=True,
    )["input_ids"][0]
    if labels.numel() < 2:
        return None
    return {
        "input_features": features.to(device),
        "labels": labels.to(device),
        "duration_s": round(len(data) / sr, 3),
    }


def evaluate_wer(generate_fn, val_rows: List[Dict[str, Any]], limit: int,
                 device: str) -> Dict[str, Any]:
    """Transcribe a held-out sample; empty when no validation clips are available."""
    from jiwer import wer

    if not val_rows or limit <= 0:
        return {"count": 0}
    refs: List[str] = []
    hyps: List[str] = []
    for row in val_rows[:limit]:
        hyp = generate_fn(row)
        if hyp is None:
            continue
        refs.append(_clip_text(row["target_text"]))
        hyps.append(_clip_text(hyp))
    if not refs:
        return {"count": 0}
    return {"count": len(refs), "wer": round(wer(refs, hyps), 4)}
def _core(model):
    """Return the module that actually runs Whisper's forward and generate.

    transformers 5 + peft 0.21 collide on the outer ``PeftModel`` wrapper, which injects
    ``input_ids`` a second time, and on ``LoraModel``, whose attribute proxy lands on the
    bare ``WhisperModel`` with no ``generate``. The wrapped HF task model keeps the LoRA
    layers in place and behaves like a normal WhisperForConditionalGeneration.
    """
    return model.model if hasattr(model, "peft_config") else model


def _transcribe_row(model, feature_extractor, tokenizer, row: Dict[str, Any],
                    device: str, max_tokens: int = MAX_TARGET_TOKENS) -> Optional[str]:
    """Greedy-decode one manifest clip with whatever weights the model currently holds."""
    import soundfile as sf
    import torch

    core = _core(model)
    wav = OUT_DIR / row["audio_path"]
    if not wav.exists():
        return None
    data, sr = sf.read(str(wav), dtype="float32")
    compute_dtype = next(model.parameters()).dtype
    feats = feature_extractor(data, sampling_rate=int(sr), return_tensors="pt")["input_features"].to(
        device, dtype=compute_dtype)
    was_training = core.training
    core.eval()
    with torch.no_grad():
        ids = core.generate(feats, max_new_tokens=max_tokens, num_beams=1)
    text = tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
    if was_training:
        core.train()
    return text


def evaluate_corpus(base_model: str, adapter_dir: Optional[Path], device: str = "cuda",
                    limit: int = 40) -> Dict[str, Any]:
    """WER/CER of the same held-out clips, once unadapted and once with the adapter loaded.

    Both numbers come from one process so the comparison shares a tokenizer, a dtype and
    a decode setting -- a before/after pair a reviewer can re-run with ``--eval-only``.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    from jiwer import cer, wer

    _, val_rows = load_manifest()
    val_rows = val_rows[:limit]
    if not val_rows:
        return {"error": "no validation clips in asr_manifest.jsonl"}

    use_cuda = device == "cuda" and torch.cuda.is_available()
    dev = "cuda" if use_cuda else "cpu"
    processor = AutoProcessor.from_pretrained(base_model)
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        base_model, dtype=torch.float16 if use_cuda else torch.float32)
    model.generation_config.language = "tr"
    model.generation_config.task = "transcribe"
    model.to(dev)

    report: Dict[str, Any] = {"base_model": base_model, "device": dev,
                              "clips": len(val_rows),
                              "reference_hours": round(sum(float(r.get("duration_s", 0)) for r in val_rows) / 3600.0, 4)}
    for mode in (["unadapted"] if adapter_dir is None else ["unadapted", "adapted"]):
        if mode == "adapted":
            model = PeftModel.from_pretrained(model, str(adapter_dir))
            model.to(dev)
        refs: List[str] = []
        hyps: List[str] = []
        for row in val_rows:
            hyp = _transcribe_row(model, feature_extractor, tokenizer, row, dev)
            if hyp is None:
                continue
            refs.append(_clip_text(row["target_text"]))
            hyps.append(_clip_text(hyp))
        report[mode] = {
            "count": len(refs),
            "wer": round(wer(refs, hyps), 4) if refs else None,
            "cer": round(cer(refs, hyps), 4) if refs else None,
        }
        logger.info("%s: WER %.2f%% CER %.2f%% over %d clips", mode,
                    100 * (report[mode]["wer"] or 0), 100 * (report[mode]["cer"] or 0), report[mode]["count"])

    if adapter_dir is not None and report.get("unadapted", {}).get("wer") is not None:
        report["wer_delta_abs"] = round(report["unadapted"]["wer"] - report["adapted"]["wer"], 4)

    out = Path(adapter_dir) if adapter_dir else CHECKPOINT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_wer.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run(
    base_model: str = BASE_MODEL,
    epochs: int = 1,
    lr: float = 2e-4,
    accumulation_steps: int = 8,
    max_steps: int = 0,
    eval_after: int = 30,
    device: str = "cuda",
    output_dir: Path = CHECKPOINT_DIR,
    smoke: bool = False,
) -> Dict[str, Any]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    if smoke:
        epochs, max_steps, accumulation_steps, eval_after = 1, 12, 2, 8

    train_rows, val_rows = load_manifest()
    random.Random(20261017).shuffle(train_rows)
    logger.info("Corpus: %d train / %d validation clips", len(train_rows), len(val_rows))

    use_cuda = device == "cuda" and torch.cuda.is_available()
    dev = "cuda" if use_cuda else "cpu"
    if device == "cuda" and not use_cuda:
        logger.warning("CUDA requested but unavailable; continuing on CPU (slow).")

    dtype = torch.float16 if use_cuda else torch.float32
    processor = AutoProcessor.from_pretrained(base_model)
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    model = AutoModelForSpeechSeq2Seq.from_pretrained(base_model, dtype=dtype)
    model.generation_config.language = "tr"
    model.generation_config.task = "transcribe"
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(dev)
    model.config.use_cache = False
    core = _core(model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    started = time.time()
    eval_clips = 4 if smoke else 12

    def transcribe(row: Dict[str, Any]) -> Optional[str]:
        return _transcribe_row(model, feature_extractor, tokenizer, row, dev)

    step_losses: List[float] = []
    optimizer.zero_grad(set_to_none=True)
    micro = 0
    global_step = 0
    history: List[Dict[str, Any]] = []
    total = len(train_rows) * epochs
    stop = False

    for epoch in range(epochs):
        for i in range(len(train_rows)):
            if stop:
                break
            batch = build_batch(dev, train_rows, i, feature_extractor, tokenizer, dtype)
            if batch is None:
                continue
            try:
                out = core(input_features=batch["input_features"].unsqueeze(0),
                           labels=batch["labels"].unsqueeze(0))
            except TypeError as exc:
                logger.warning("Unsupported forward signature: %s", str(exc)[:200])
                return {"status": "ERROR", "error": str(exc)[:400]}
            loss = out.loss / accumulation_steps
            loss.backward()
            micro += 1
            if micro % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                step_losses.append(float(loss.item()) * accumulation_steps)
                if global_step % 10 == 0:
                    logger.info("step %d loss %.4f", global_step, step_losses[-1])
                if max_steps and global_step >= max_steps:
                    stop = True
                elif global_step and global_step % eval_after == 0:
                    eta_min = (time.time() - started) / global_step * (total / accumulation_steps - global_step) / 60
                    probe = evaluate_wer(transcribe, val_rows, eval_clips, dev)
                    logger.info("step %d | checkpoint written, ETA %.1f min | held-out WER probe: %s",
                                global_step, eta_min, json.dumps(probe))
                    save_adapter(model, processor, output_dir)
                    history.append({
                        "step": global_step,
                        "mean_loss": round(sum(step_losses) / len(step_losses), 4),
                        "wer_probe": probe,
                    })
                    write_metrics(history, step_losses, global_step, started, train_rows, val_rows,
                                  base_model, output_dir, dev, dtype)

    elapsed = time.time() - started
    save_adapter(model, processor, output_dir)
    final_wer = evaluate_wer(transcribe, val_rows, eval_clips, dev)
    history.append({"step": global_step, "final": True, "wer_probe": final_wer})
    write_metrics(history, step_losses, global_step, started, train_rows, val_rows,
                  base_model, output_dir, dev, dtype)
    return {
        "status": "COMPLETED",
        "base_model": base_model,
        "device": dev,
        "global_steps": global_step,
        "train_clips": len(train_rows),
        "validation_clips": len(val_rows),
        "held_out_wer_probe": final_wer,
        "mean_loss": round(sum(step_losses) / max(1, len(step_losses)), 4),
        "first_10_mean_loss": round(sum(step_losses[:10]) / max(1, min(10, len(step_losses))), 4),
        "last_10_mean_loss": round(sum(step_losses[-10:]) / max(1, min(10, len(step_losses))), 4),
        "elapsed_s": round(elapsed, 1),
        "adapter_dir": str(output_dir),
    }


def save_adapter(model, processor, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))


def write_metrics(history: List[Dict[str, Any]], losses: List[float], steps: int,
                  started: float, train_rows: List[Dict[str, Any]],
                  val_rows: List[Dict[str, Any]], base_model: str,
                  output_dir: Path, dev: str, dtype) -> None:
    payload = {
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_model": base_model,
        "device": dev,
        "dtype": str(dtype).replace("torch.", ""),
        "train_clips": len(train_rows),
        "validation_clips": len(val_rows),
        "train_hours": round(sum(float(r.get("duration_s", 0)) for r in train_rows) / 3600.0, 3),
        "global_steps": steps,
        "loss_curve": history,
        "losses": [round(l, 5) for l in losses[-200:]],
        "elapsed_s": round(time.time() - started, 1),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="LoRA adaptation for Katman 1 Turkish ASR.")
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--accumulation-steps", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--smoke", action="store_true", help="Short plumbing check.")
    p.add_argument("--adapter-dir", default=str(CHECKPOINT_DIR))
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training and report unadapted vs adapted WER/CER on the held-out clips.")
    p.add_argument("--eval-clips", type=int, default=40)
    args = p.parse_args()

    if args.eval_only:
        result = evaluate_corpus(args.model, Path(args.adapter_dir), args.device, args.eval_clips)
    else:
        result = run(
            base_model=args.model,
            epochs=args.epochs,
            lr=args.lr,
            accumulation_steps=args.accumulation_steps,
            max_steps=args.max_steps,
            device=args.device,
            output_dir=Path(args.adapter_dir),
            smoke=args.smoke,
        )
    print("\n=== KATMAN 1 ASR ADAPTATION SUMMARY ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
