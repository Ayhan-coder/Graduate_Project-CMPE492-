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
import logging
import platform
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r".*triton not found.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
try:
    from transformers.utils import logging as tf_logging
    tf_logging.set_verbosity_error()
except Exception:
    pass

from fetch_models import load_registry, resolve_hub_cache
from gold_io import resolve_reference

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


def completed_pairs(path: Path) -> set[tuple[str, str]]:
    """(clip id, model id) already written without error. Used by --resume."""
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("error") or rec.get("skipped"):
            continue
        cid, mid = rec.get("id"), rec.get("model")
        if cid and mid:
            done.add((cid, mid))
    return done


def all_asr_entries() -> list[dict]:
    return list(REGISTRY.get("asr_selected", [])) + list(REGISTRY.get("asr_pilot", []))


def asr_entry(model_id: str) -> dict:
    for m in all_asr_entries():
        if m["id"] == model_id:
            return m
    raise SystemExit(f"{model_id} is not in models.json asr_selected or asr_pilot")


_PIPE_CACHE: dict[str, object] = {}
_WHISPER_CACHE: dict[str, tuple[object, object]] = {}
_QWEN_CACHE: dict[str, tuple[object, object]] = {}
_NEMO_CACHE: dict[str, tuple[object, object]] = {}
_SB_CACHE: dict[str, object] = {}
NEMOTRON_CHUNK_S = 60.0


def _device():
    import torch

    if torch.cuda.is_available():
        return 0, torch.float16
    return "cpu", torch.float32


def _load_wav_16k(audio_path: str):
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(audio_path, always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != 16000:
        raise SystemExit(f"{audio_path} is {sr} Hz; expected 16 kHz mono")
    return audio, sr


def _get_glossary_prompt(clip_id: str | None, lang: str | None, custom_text: str | None = None) -> str:
    """Build domain biasing prompt from slide terms or glossary."""
    if custom_text:
        return custom_text.strip()
    if not clip_id:
        return ""
    gold_path = ROOT / "data" / "gold_terms.json"
    if gold_path.exists():
        try:
            gold_data = json.loads(gold_path.read_text(encoding="utf-8"))
            if clip_id in gold_data:
                terms = [t["canonical"] for t in gold_data[clip_id].get("terms", [])[:25]]
                if lang and str(lang).lower().startswith("tr"):
                    return "Ders konusu ve teknik terimler: " + ", ".join(terms)
                else:
                    return "Lecture topic and technical terms: " + ", ".join(terms)
        except Exception:
            pass
    return ""


def transcribe_whisper(model_id: str, audio_path: str, lang: str | None, args, clip_id: str | None = None) -> str:
    """Sequential long-form generate() with optional VAD speech chunking and domain biasing."""
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    device, dtype = _device()
    packed = _WHISPER_CACHE.get(model_id)
    if packed is None:
        print(f"  loading whisper {model_id} on {device} ...", flush=True)
        cache = str(resolve_hub_cache(model_id))
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="cuda" if device == 0 else "cpu",
            cache_dir=cache,
        )
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache)
        if getattr(model.generation_config, "max_length", None):
            model.generation_config.max_length = None
        packed = (processor, model)
        _WHISPER_CACHE[model_id] = packed
    processor, model = packed

    audio, sr = _load_wav_16k(audio_path)
    target = "cuda" if device == 0 else "cpu"

    # Domain Biasing: Encode glossary terms as Whisper prompt_ids
    prompt_ids = None
    prompt_str = ""
    if getattr(args, "prompt_glossary", False) or getattr(args, "prompt_text", None):
        prompt_str = _get_glossary_prompt(clip_id, lang, getattr(args, "prompt_text", None))
        if prompt_str:
            try:
                p_ids = processor.get_prompt_ids(prompt_str, return_tensors="pt")
                prompt_ids = p_ids.to(target)
                print(f"    [Domain Biasing Active]: '{prompt_str[:70]}...'", flush=True)
            except Exception as exc:
                print(f"    [Domain Biasing Warning]: Could not encode prompt_ids: {exc}", flush=True)

    gen_kwargs = {
        "task": "transcribe",
        "return_timestamps": True,
        "condition_on_prev_tokens": bool(getattr(args, "condition_on_prev", False)),
        "temperature": 0.0,
        "num_beams": 1,
        "max_new_tokens": 440,
    }
    if prompt_ids is not None:
        gen_kwargs["prompt_ids"] = prompt_ids
    if not getattr(args, "language_detect", False) and lang in {"tr", "en"}:
        gen_kwargs["language"] = lang

    # VAD Long-form handling: segment audio on silence boundaries to prevent looping/hallucinations
    if getattr(args, "use_vad", False):
        from vad_chunker import segment_audio_vad

        intervals = segment_audio_vad(audio, sr=sr, min_chunk_s=12.0, max_chunk_s=28.0)
        print(f"    [VAD Enabled]: segmented {len(audio)/sr:.1f}s audio into {len(intervals)} speech chunks ...", flush=True)
        parts = []
        for i, (st, en) in enumerate(intervals):
            st_idx = int(st * sr)
            en_idx = min(len(audio), int(en * sr))
            piece = audio[st_idx:en_idx]
            if len(piece) < int(0.4 * sr):
                continue
            inputs = processor(
                piece,
                sampling_rate=sr,
                return_tensors="pt",
                truncation=False,
                padding="longest",
                return_attention_mask=True,
            )
            inputs = {k: v.to(target) for k, v in inputs.items()}
            if "input_features" in inputs:
                inputs["input_features"] = inputs["input_features"].to(dtype=dtype)
            with torch.inference_mode():
                ids = model.generate(**inputs, **gen_kwargs)
            txt = processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            if txt:
                parts.append(txt)
        return " ".join(parts).strip()

    # Standard sequential generate
    print(f"    sequential generate ({len(audio) / sr:.1f}s audio) ...", flush=True)
    inputs = processor(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
        truncation=False,
        padding="longest",
        return_attention_mask=True,
    )
    inputs = {k: v.to(target) for k, v in inputs.items()}
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(dtype=dtype)

    with torch.inference_mode():
        ids = model.generate(**inputs, **gen_kwargs)
    return processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def transcribe_hf_ctc(model_id: str, audio_path: str) -> str:
    """Wav2Vec2-style CTC / seq2seq checkpoints via transformers."""
    from transformers import pipeline

    device, _ = _device()
    pipe = _PIPE_CACHE.get(model_id)
    if pipe is None:
        print(f"  loading ctc {model_id} on {device} ...", flush=True)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            model_kwargs={"cache_dir": str(resolve_hub_cache(model_id))},
        )
        _PIPE_CACHE[model_id] = pipe
    out = pipe(audio_path)
    if isinstance(out, dict):
        return (out.get("text") or "").strip()
    return str(out).strip()


def transcribe_speechbrain(model_id: str, audio_path: str) -> str:
    from speechbrain.inference.ASR import EncoderDecoderASR

    asr = _SB_CACHE.get(model_id)
    if asr is None:
        print(f"  loading speechbrain {model_id} ...", flush=True)
        asr = EncoderDecoderASR.from_hparams(source=model_id, savedir=str(ROOT / "hf_cache" / "speechbrain"))
        _SB_CACHE[model_id] = asr
    return asr.transcribe_file(audio_path).strip()


def transcribe_qwen3(model_id: str, audio_path: str, lang: str | None, args=None) -> str:
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForMultimodalLM as _Model
    except ImportError:
        from transformers import AutoModel as _Model

    packed = _QWEN_CACHE.get(model_id)
    if packed is None:
        print(f"  loading qwen3 {model_id} ...", flush=True)
        cache = str(resolve_hub_cache(model_id))
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache)
        model = _Model.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            cache_dir=cache,
        )
        if not torch.cuda.is_available():
            model = model.to("cpu")
        packed = (processor, model)
        _QWEN_CACHE[model_id] = packed
    processor, model = packed

    audio, sr = _load_wav_16k(audio_path)
    lang_name = {"tr": "Turkish", "en": "English"}.get(lang or "", "")
    chunk = int(30.0 * sr)
    parts: list[str] = []
    n_chunks = max(1, (len(audio) + chunk - 1) // chunk)

    for i, start in enumerate(range(0, len(audio), chunk)):
        piece = audio[start : start + chunk]
        if len(piece) < sr:
            continue
        print(f"    qwen3 chunk {i + 1}/{n_chunks} ({len(piece) / sr:.1f}s) ...", flush=True)
        msgs = []
        if lang_name and not (args and getattr(args, "language_detect", False)):
            msgs.append({"role": "system", "content": f"Language: {lang_name}"})
        msgs.append({"role": "user", "content": [{"type": "audio", "audio": piece}]})

        text_prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text_prompt, audio=piece, return_tensors="pt")
        inputs = {
            k: v.to(model.device, dtype=model.dtype)
            if getattr(v, "dtype", None) in (torch.float32, torch.float16, torch.bfloat16)
            else (v.to(model.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=440)
        gen_ids = out[:, inputs["input_ids"].shape[1] :]
        decoded = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        if "<asr_text>" in decoded:
            decoded = decoded.split("<asr_text>", 1)[-1].strip()
        if decoded:
            parts.append(decoded)

    return " ".join(parts).strip()


def transcribe_nemotron(model_id: str, audio_path: str, lang: str | None, args) -> str:
    """NVIDIA Nemotron 3.5 ASR (transformers>=5.13 AutoModelForRNNT).

    Long lecture wavs are split into non-overlapping 60 s windows so a
    laptop GPU does not hold a 10+ minute encoder state. Language is
    forced to tr-TR / en-US unless --language-detect is set.
    """
    import torch
    from transformers import AutoModelForRNNT, AutoProcessor

    packed = _NEMO_CACHE.get(model_id)
    if packed is None:
        print(f"  loading nemotron {model_id} ...", flush=True)
        cache = str(resolve_hub_cache(model_id))
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForRNNT.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            cache_dir=cache,
        )
        if not torch.cuda.is_available():
            model = model.to("cpu")
        packed = (processor, model)
        _NEMO_CACHE[model_id] = packed
    processor, model = packed

    audio, sr = _load_wav_16k(audio_path)
    if args.language_detect or lang not in {"tr", "en"}:
        locale = "auto"
    else:
        locale = {"tr": "tr-TR", "en": "en-US"}[lang]

    chunk = int(NEMOTRON_CHUNK_S * sr)
    parts: list[str] = []
    n_chunks = max(1, (len(audio) + chunk - 1) // chunk)
    for i, start in enumerate(range(0, len(audio), chunk)):
        piece = audio[start : start + chunk]
        print(f"    nemotron chunk {i + 1}/{n_chunks} ({len(piece) / sr:.1f}s, lang={locale}) ...", flush=True)
        inputs = processor(piece, sampling_rate=sr, language=locale, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device, dtype=model.dtype)
        else:
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            if "input_features" in inputs:
                inputs["input_features"] = inputs["input_features"].to(dtype=model.dtype)
        with torch.inference_mode():
            output = model.generate(**inputs, return_dict_in_generate=True)
        seq = output.sequences if hasattr(output, "sequences") else output
        if hasattr(processor, "batch_decode"):
            text = processor.batch_decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            text = text[0] if isinstance(text, list) else text
        else:
            text = processor.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        text = (text or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


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
        "longform": "sequential_generate",
        "return_timestamps": True,
        "condition_on_prev_tokens": bool(args.condition_on_prev),
        "compression_ratio_threshold": 1.35,
        "logprob_threshold": -1.0,
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
    print(f"  {row.get('id')} start ({row.get('duration_s')}s) ...", flush=True)
    t0 = time.perf_counter()
    backend = model["backend"]
    if backend == "whisper":
        hyp = transcribe_whisper(model["id"], audio, lang, args, clip_id=row.get("id"))
    elif backend == "qwen3_asr":
        hyp = transcribe_qwen3(model["id"], audio, lang, args)
    elif backend == "nemotron_asr":
        hyp = transcribe_nemotron(model["id"], audio, lang, args)
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
        "confirmed": bool(row.get("confirmed")),
        "reference_source": row.get("reference_source") or "missing",
    }
    if rec["audio_duration_s"]:
        rec["rtf"] = elapsed / rec["audio_duration_s"]

    ref_text = resolve_reference(row)
    if ref_text:
        rec.update(pair_metrics(ref_text, hyp, lang=lang))
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.jsonl")
    parser.add_argument("--model", help="single model id (see models.json)")
    parser.add_argument("--all-selected", action="store_true",
                        help="run all refresh models officially covering TR+EN (Table tab:asr-2026)")
    parser.add_argument("--all-pilot", action="store_true",
                        help="run the 4 paper pilot systems (Table tab:asr-pilot)")
    parser.add_argument("--chunk-s", type=float, default=30.0,
                        help="Nemotron window size (Whisper ignores; stays sequential)")
    parser.add_argument("--language-detect", action="store_true",
                        help="do not force language=tr|en (Whisper auto-detect)")
    parser.add_argument("--condition-on-prev", action="store_true",
                        help="Whisper condition_on_prev_tokens=True (openai-whisper: condition_on_previous_text)")
    parser.add_argument("--use-vad", action="store_true",
                        help="use Voice Activity Detection pause chunking (prevents long-form hallucination/looping)")
    parser.add_argument("--prompt-glossary", action="store_true",
                        help="inject domain terms from gold_terms.json into Whisper initial prompt (Domain Biasing)")
    parser.add_argument("--prompt-text", type=str, default=None,
                        help="custom initial prompt text to bias Whisper domain decoding")
    parser.add_argument("--ids", nargs="*", default=[],
                        help="restrict to these clip ids (e.g. TR-1 EN-2)")
    parser.add_argument("--output", type=Path,
                        help="write this JSONL instead of a new outputs/asr-TIMESTAMP.jsonl")
    parser.add_argument("--resume", action="store_true",
                        help="skip (id, model) pairs already present without error in --output")
    args = parser.parse_args()
    if args.resume and not args.output:
        raise SystemExit("--resume requires --output PATH")

    rows = load_manifest(args.manifest)
    if args.ids:
        want = set(args.ids)
        rows = [r for r in rows if r.get("id") in want]
        if not rows:
            raise SystemExit(f"no manifest rows match --ids {sorted(want)}")
    if args.all_selected:
        models = list(REGISTRY["asr_selected"])
    elif args.all_pilot:
        models = list(REGISTRY.get("asr_pilot", []))
    elif args.model:
        models = [asr_entry(args.model)]
    else:
        raise SystemExit("pass --model ID, --all-selected, or --all-pilot")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = args.output if args.output.is_absolute() else (ROOT / args.output)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_path = OUT_DIR / f"asr-{stamp}.jsonl"
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(decoding_config(args), indent=2), encoding="utf-8")
    already = completed_pairs(out_path) if args.resume else set()
    if already:
        print(f"resume: {len(already)} completed pairs in {out_path}", flush=True)
    n_ok = n_err = n_skip = 0
    mode = "a" if args.resume and out_path.exists() else "w"
    with out_path.open(mode, encoding="utf-8") as fh:
        for model in models:
            print(f"== {model['id']} ==", flush=True)
            for row in rows:
                key = (row.get("id"), model["id"])
                if key in already:
                    print(f"  {row['id']}: resume-skip", flush=True)
                    n_skip += 1
                    continue
                try:
                    rec = run_one(model, row, args)
                except Exception as exc:
                    rec = {
                        "id": row.get("id"),
                        "lang": row.get("lang"),
                        "model": model["id"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    print(f"  {row['id']}: ERROR {rec['error']}", flush=True)
                    n_err += 1
                else:
                    if rec.get("skipped"):
                        n_skip += 1
                    else:
                        n_ok += 1
                    print(f"  {row['id']}: {rec.get('wer_norm', rec.get('skipped'))}", flush=True)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
    print("wrote", out_path, f"(ok={n_ok} skip={n_skip} err={n_err})")
    print("meta ", meta_path)


if __name__ == "__main__":
    main()
