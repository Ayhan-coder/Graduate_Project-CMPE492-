"""Run the paper's fixed terminology prompt on selected LLMs.

Manifest (JSONL):
    {"id": "TR-1", "lang": "Turkish", "transcript": "..."}

Usage:
    python run_llm.py --manifest data/transcripts.jsonl --model Qwen/Qwen3.5-4B
    python run_llm.py --manifest data/transcripts.jsonl --all-open
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
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

from fetch_models import hub_slug, load_registry, resolve_hub_cache

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"

# EVREN answers need the full JSON array inside one completion; the verbose models exceed
# 2048 tokens on the longer lectures and get truncated mid-array.
EVREN_MAX_TOKENS = 4096


def _local_model_path(model_id: str) -> str:
    cache = resolve_hub_cache(model_id)
    snaps = list((cache / hub_slug(model_id) / "snapshots").glob("*"))
    for snap in snaps:
        if any(snap.glob("*.safetensors")) or any(snap.glob("*.bin")):
            return str(snap)
    return model_id

PROMPT = (
    "You are a terminology assistant for {LANG} educational content. "
    "Step 1: From the transcript below, extract domain-specific terms "
    "(including multi-word terms). Step 2: For each term, write one short "
    "explanation at CEFR B1 level that a beginner can understand. Use only "
    "information consistent with the transcript context. Return strictly a "
    "JSON array of objects with fields \"term\" and \"explanation\". "
    "Transcript: {TRANSCRIPT}"
)


def load_manifest(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def parse_terms(text: str) -> list[dict]:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict) and "term" in x]


def _hf_model_class(model_id: str, cache: str):
    """Causal LM for Llama/Qwen2.5/Qwen3/Qwen3.5-9B; image-text class for Qwen3.5 multimodal."""
    if any(k in model_id.lower() for k in ("9b", "llama")):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM
    from transformers import AutoConfig

    path = _local_model_path(model_id)
    cfg = AutoConfig.from_pretrained(path, cache_dir=cache)
    arch = ""
    if getattr(cfg, "architectures", None):
        arch = cfg.architectures[0]
    if "ForCausalLM" in arch:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM


def _strip_think(text: str) -> str:
    marker = "</think>"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


_HF_LLM_CACHE: dict[str, tuple[object, object]] = {}
_ACTIVE_LLM_MODEL: str | None = None


def clear_llm_cache() -> None:
    """Clear cached HuggingFace LLM models and free CUDA memory."""
    import gc
    import torch

    global _HF_LLM_CACHE, _ACTIVE_LLM_MODEL
    _HF_LLM_CACHE.clear()
    _ACTIVE_LLM_MODEL = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_LLM_LOAD_LOCK = threading.Lock()


def get_or_load_llm(model_id: str = "Qwen/Qwen3.5-4B") -> tuple[object, object]:
    """Retrieve cached LLM tokenizer & model or load into memory once."""
    import gc
    import torch
    import safetensors
    from transformers import AutoTokenizer

    # On Windows, enforce pread backend for safetensors to avoid mmap OS error 1455 / pagefile overflow
    import transformers.modeling_utils
    if hasattr(transformers.modeling_utils, "safe_open") and not getattr(transformers.modeling_utils, "_pread_patched", False):
        _orig_mu_safe_open = transformers.modeling_utils.safe_open

        def _mu_pread_safe_open(*args, **kwargs):
            kwargs["backend"] = "pread"
            return _orig_mu_safe_open(*args, **kwargs)

        transformers.modeling_utils.safe_open = _mu_pread_safe_open
        safetensors.safe_open = _mu_pread_safe_open
        transformers.modeling_utils._pread_patched = True
        safetensors._pread_patched = True

    global _ACTIVE_LLM_MODEL
    with _LLM_LOAD_LOCK:
        packed = _HF_LLM_CACHE.get(model_id)
        if packed is not None:
            return packed

        # If switching models, evict previous LLM to free GPU VRAM
        if _ACTIVE_LLM_MODEL is not None and _ACTIVE_LLM_MODEL != model_id:
            clear_llm_cache()

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        cache = str(resolve_hub_cache(model_id))
        path = _local_model_path(model_id)
        tok = AutoTokenizer.from_pretrained(path, cache_dir=cache)
        Model = _hf_model_class(model_id, cache)
        model_kwargs = {"cache_dir": cache}
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            # Use normal RAM (CPU) fallback if GPU VRAM is full
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["dtype"] = torch.float32
            model_kwargs["device_map"] = "cpu"
        model = Model.from_pretrained(path, **model_kwargs)
        packed = (tok, model)
        _HF_LLM_CACHE[model_id] = packed
        _ACTIVE_LLM_MODEL = model_id
        return packed


def generate_hf(model_id: str, prompt: str, max_new_tokens: int = 1024) -> str:
    from evren_llm import is_evren_model, generate_evren

    if is_evren_model(model_id):
        return generate_evren(model_id, prompt, max_tokens=max_new_tokens)

    import torch

    tok, model = get_or_load_llm(model_id)
    messages = [{"role": "user", "content": prompt}]
    tmpl_kwargs = {
        "return_tensors": "pt",
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        inputs = tok.apply_chat_template(messages, **tmpl_kwargs)
    except TypeError:
        tmpl_kwargs.pop("enable_thinking", None)
        inputs = tok.apply_chat_template(messages, **tmpl_kwargs)
    if isinstance(inputs, dict) or hasattr(inputs, "data"):
        gen_in = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        input_len = gen_in["input_ids"].shape[-1]
    else:
        inputs = inputs.to(model.device)
        input_len = inputs.shape[-1]
        gen_in = {"input_ids": inputs}
    with torch.inference_mode():
        out = model.generate(**gen_in, max_new_tokens=max_new_tokens, do_sample=False)
    text = tok.decode(out[0][input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return _strip_think(text)


def generate_openai(model_id: str, prompt: str) -> str:
    import os
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--all-open", action="store_true", help="all Hugging Face LLMs in the registry")
    args = parser.parse_args()

    reg = load_registry()
    if args.all_open:
        models = [m for m in reg["llm_selected"] if m.get("provider") == "huggingface"]
    elif args.model:
        models = [m for m in reg["llm_selected"] if m["id"] == args.model]
        if not models:
            raise SystemExit(f"{args.model} is not in models.json llm_selected")
    else:
        raise SystemExit("pass --model ID or --all-open")

    rows = load_manifest(args.manifest)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"llm-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    bad_by_model: dict[str, tuple[int, int]] = {}
    with out_path.open("w", encoding="utf-8") as fh:
        for model in models:
            print(f"== {model['id']} ==")
            empty = unparsed = 0
            for row in rows:
                prompt = PROMPT.format(LANG=row.get("lang", "English"), TRANSCRIPT=row["transcript"])
                t0 = time.perf_counter()
                is_evren = model.get("provider") == "evren" or is_evren_model(model.get("id", ""))
                if model.get("provider") == "openai_api":
                    text = generate_openai(model["id"], prompt)
                elif is_evren:
                    from evren_llm import generate_evren

                    text = generate_evren(model["id"], prompt, max_tokens=EVREN_MAX_TOKENS, temperature=0.0)
                else:
                    text = generate_hf(model["id"], prompt)
                elapsed = time.perf_counter() - t0
                terms = parse_terms(text)
                if not text.strip():
                    empty += 1
                elif not terms:
                    unparsed += 1
                rec = {
                    "id": row["id"],
                    "lang": row.get("lang"),
                    "model": model["id"],
                    "latency_s": elapsed,
                    "raw": text,
                    "n_terms": len(terms),
                    "terms": terms,
                }
                if is_evren:
                    rec["reasoning_effort"] = "none"
                    rec["max_tokens"] = EVREN_MAX_TOKENS
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"  {row['id']}: {len(terms)} terms in {elapsed:.1f}s")
            bad_by_model[model["id"]] = (empty, unparsed)
    print("wrote", out_path)
    failed = {m: (e, u) for m, (e, u) in bad_by_model.items() if e or u}
    if failed:
        raise SystemExit(
            "UNUSABLE OUTPUT: "
            + "; ".join(
                f"{m}: {e} of {len(rows)} empty, {u} of {len(rows)} answer-with-zero-parsed-terms"
                for m, (e, u) in failed.items()
            )
            + f" — results in {out_path.name} are not usable."
        )


if __name__ == "__main__":
    main()
