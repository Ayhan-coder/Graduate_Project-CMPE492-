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
import time
from pathlib import Path

from fetch_models import hub_slug, load_registry, resolve_hub_cache

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"


def _local_model_path(model_id: str) -> str:
    cache = resolve_hub_cache(model_id)
    snaps = list((cache / hub_slug(model_id) / "snapshots").glob("*"))
    if snaps:
        return str(snaps[0])
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
    """Causal LM for Qwen2.5/Qwen3; image-text class for Qwen3.5+."""
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


def generate_hf(model_id: str, prompt: str) -> str:
    import torch
    from transformers import AutoTokenizer

    packed = _HF_LLM_CACHE.get(model_id)
    if packed is None:
        cache = str(resolve_hub_cache(model_id))
        path = _local_model_path(model_id)
        tok = AutoTokenizer.from_pretrained(path, cache_dir=cache)
        Model = _hf_model_class(model_id, cache)
        model_kwargs = {"cache_dir": cache}
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["dtype"] = torch.float32
        model = Model.from_pretrained(path, **model_kwargs)
        if not torch.cuda.is_available():
            model = model.to("cpu")
        packed = (tok, model)
        _HF_LLM_CACHE[model_id] = packed
    tok, model = packed
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
        out = model.generate(**gen_in, max_new_tokens=2048, do_sample=False)
    text = tok.decode(out[0][input_len:], skip_special_tokens=True)
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
    with out_path.open("w", encoding="utf-8") as fh:
        for model in models:
            print(f"== {model['id']} ==")
            for row in rows:
                prompt = PROMPT.format(LANG=row.get("lang", "English"), TRANSCRIPT=row["transcript"])
                t0 = time.perf_counter()
                if model.get("provider") == "openai_api":
                    text = generate_openai(model["id"], prompt)
                else:
                    text = generate_hf(model["id"], prompt)
                elapsed = time.perf_counter() - t0
                terms = parse_terms(text)
                rec = {
                    "id": row["id"],
                    "lang": row.get("lang"),
                    "model": model["id"],
                    "latency_s": elapsed,
                    "raw": text,
                    "n_terms": len(terms),
                    "terms": terms,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"  {row['id']}: {len(terms)} terms in {elapsed:.1f}s")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
