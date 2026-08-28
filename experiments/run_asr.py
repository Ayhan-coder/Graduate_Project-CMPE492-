"""Transcribe a 16 kHz wav with a selected paper ASR model and score WER/CER.

Manifest (JSONL), one object per clip:
    {"id": "TR-1", "lang": "tr", "audio": "data/TR-1.wav", "duration_s": 720, "reference": "gold text"}

Usage:
    python run_asr.py --manifest data/manifest.jsonl --model openai/whisper-large-v3-turbo
    python run_asr.py --manifest data/manifest.jsonl --all-selected
    python run_asr.py --manifest data/manifest.jsonl --all-pilot
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from fetch_models import load_registry

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
REGISTRY = load_registry()


def load_manifest(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def all_asr_entries() -> list[dict]:
    return list(REGISTRY.get("asr_selected", [])) + list(REGISTRY.get("asr_pilot", []))


def asr_entry(model_id: str) -> dict:
    for m in all_asr_entries():
        if m["id"] == model_id:
            return m
    raise SystemExit(f"{model_id} is not in models.json asr_selected or asr_pilot")


def _device():
    import torch

    if torch.cuda.is_available():
        return 0, torch.float16
    return "cpu", torch.float32


def transcribe_whisper(model_id: str, audio_path: str, lang: str | None, args) -> str:
    import torch
    from transformers import pipeline

    device, dtype = _device()
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=dtype,
        device=device,
        return_timestamps=True,
    )
    generate_kwargs = {"task": "transcribe"}
    if not args.language_detect and lang in {"tr", "en"}:
        generate_kwargs["language"] = lang
    generate_kwargs["condition_on_previous_text"] = bool(args.condition_on_prev)
    out = pipe(audio_path, chunk_length_s=args.chunk_s, generate_kwargs=generate_kwargs)
    return out["text"].strip()


def transcribe_hf_ctc(model_id: str, audio_path: str) -> str:
    """Wav2Vec2-style CTC / seq2seq checkpoints via transformers."""
    from transformers import pipeline

    device, _ = _device()
    pipe = pipeline("automatic-speech-recognition", model=model_id, device=device)
    out = pipe(audio_path)
    if isinstance(out, dict):
        return (out.get("text") or "").strip()
    return str(out).strip()


def transcribe_speechbrain(model_id: str, audio_path: str) -> str:
    from speechbrain.inference.ASR import EncoderDecoderASR

    asr = EncoderDecoderASR.from_hparams(source=model_id, savedir=str(ROOT / "hf_cache" / "speechbrain"))
    return asr.transcribe_file(audio_path).strip()


def transcribe_qwen3(model_id: str, audio_path: str, lang: str | None, args=None) -> str:
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForMultimodalLM as _Model
    except ImportError:
        from transformers import AutoModel as _Model

    processor = AutoProcessor.from_pretrained(model_id)
    model = _Model.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")
    lang_prompt = {"tr": "Turkish", "en": "English"}.get(lang or "", "")
    prompt = f"Transcribe the speech into {lang_prompt}." if lang_prompt else "Transcribe the speech."
    inputs = processor(audio=audio_path, text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=2048)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def decoding_config(args) -> dict:
    try:
        import torch
        cuda = torch.cuda.is_available()
        torch_ver = torch.__version__
        gpu = torch.cuda.get_device_name(0) if cuda else None
    except Exception:
        cuda, torch_ver, gpu = False, None, None
    return {
        "task": "transcribe",
        "language": "forced clip.lang" if not args.language_detect else "auto-detect",
        "chunk_length_s": args.chunk_s,
        "return_timestamps": True,
        "condition_on_previous_text": bool(args.condition_on_prev),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch_ver,
        "cuda": cuda,
        "gpu": gpu,
    }


def run_one(model: dict, row: dict, args) -> dict:
    from metrics import pair_metrics

    if model.get("subset") == "tr" and row.get("lang") != "tr":
        return {"id": row["id"], "model": model["id"], "skipped": True, "reason": "TR-only model"}

    audio = str(Path(row["audio"]))
    if not Path(audio).is_absolute():
        audio = str((ROOT / audio).resolve())
    lang = row.get("lang")
    t0 = time.perf_counter()
    backend = model["backend"]
    if backend == "whisper":
        hyp = transcribe_whisper(model["id"], audio, lang, args)
    elif backend == "qwen3_asr":
        hyp = transcribe_qwen3(model["id"], audio, lang, args)
    elif backend == "hf_ctc":
        hyp = transcribe_hf_ctc(model["id"], audio)
    elif backend == "speechbrain":
        hyp = transcribe_speechbrain(model["id"], audio)
    else:
        raise SystemExit(f"unknown backend {backend}")
    elapsed = time.perf_counter() - t0

    rec = {
        "id": row["id"],
        "lang": lang,
        "model": model["id"],
        "revision": model.get("revision"),
        "hypothesis": hyp,
        "latency_s": elapsed,
        "audio_duration_s": row.get("duration_s"),
        "decoding": decoding_config(args),
    }
    if rec["audio_duration_s"]:
        rec["rtf"] = elapsed / float(rec["audio_duration_s"])
    if row.get("reference"):
        rec.update(pair_metrics(row["reference"], hyp))
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", help="one ASR repo id from asr_selected or asr_pilot")
    parser.add_argument("--all-selected", action="store_true", help="2026 refresh set")
    parser.add_argument("--all-pilot", action="store_true", help="original four pilot systems")
    parser.add_argument("--chunk-s", type=float, default=30.0)
    parser.add_argument("--language-detect", action="store_true",
                        help="do not force language=tr|en (Whisper auto-detect)")
    parser.add_argument("--condition-on-prev", action="store_true",
                        help="Whisper condition_on_previous_text=True")
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    if args.all_selected:
        models = list(REGISTRY["asr_selected"])
    elif args.all_pilot:
        models = list(REGISTRY.get("asr_pilot", []))
    elif args.model:
        models = [asr_entry(args.model)]
    else:
        raise SystemExit("pass --model ID, --all-selected, or --all-pilot")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = OUT_DIR / f"asr-{stamp}.jsonl"
    meta_path = OUT_DIR / f"asr-{stamp}.meta.json"
    meta_path.write_text(json.dumps(decoding_config(args), indent=2), encoding="utf-8")
    with out_path.open("w", encoding="utf-8") as fh:
        for model in models:
            print(f"== {model['id']} ==")
            for row in rows:
                rec = run_one(model, row, args)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"  {row['id']}: {rec.get('wer_norm', rec.get('skipped'))}")
    print("wrote", out_path)
    print("meta ", meta_path)


if __name__ == "__main__":
    main()
