"""Measure small-model rows on the same BENCH-* probe the A40 report used.

Scoring is imported from finetune/run_hpc_scaling_law.py (UNSEEN_BENCHMARKS,
SYSTEM_PROMPT, extract_json, calculate_atesman, run_benchmark_for_model), so the
numbers are directly comparable with outputs/scaling_law_grand_report.json.
Re-running Qwen2.5-0.5B/-1.5B is the equivalence check (expect F1 0.3947/0.6188).

Usage:
    python benchmark_small_models_local.py --models qwen05 qwen15 qwen3 llama1 smol17
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

import run_hpc_scaling_law as canon  # noqa: E402
from fetch_models import hub_slug, resolve_hub_cache  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SmallModelLocal")

OUT_DEFAULT = ROOT / "outputs" / "small_models_local_report.json"

MODELS = {
    "qwen05": {"label": "Qwen2.5-0.5B-Instruct", "repo": "Qwen/Qwen2.5-0.5B-Instruct", "a40_f1": 0.3947},
    "qwen15": {"label": "Qwen2.5-1.5B-Instruct", "repo": "Qwen/Qwen2.5-1.5B-Instruct", "a40_f1": 0.6188},
    "qwen3": {"label": "Qwen2.5-3B-Instruct", "repo": "Qwen/Qwen2.5-3B-Instruct", "a40_f1": None},
    "llama1": {"label": "Llama-3.2-1B-Instruct", "repo": "unsloth/Llama-3.2-1B-Instruct", "a40_f1": None},
    "smol17": {"label": "SmolLM2-1.7B-Instruct", "repo": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "a40_f1": None},
}


def local_snapshot(repo: str) -> str:
    cache = resolve_hub_cache(repo)
    for snap in sorted((cache / hub_slug(repo) / "snapshots").glob("*")):
        if any(snap.glob("*.safetensors")):
            return str(snap)
    raise FileNotFoundError(f"no local weights for {repo}; run download_small_models.py first")


def run_model(name: str, path: str, *, quant: str = "bf16", prompt_format: str = "chatml") -> dict:
    """Same probe and scoring, with a choice of dtype and prompt format.

    chatml: the literal <|im_start|> string used by run_hpc_scaling_law (only correct
            for Qwen/ChatML tokenizers).
    auto:   the checkpoint's own chat template via apply_chat_template. Required for
            Llama-3.2, which otherwise receives malformed control tokens and scores
            near-zero JSON validity for reasons that have nothing to do with the model.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = {"device_map": "cuda:0", "trust_remote_code": True}
    if quant == "nf4":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
    model.eval()

    case_results = []
    total_tokens = 0
    total_latency = 0.0
    for bench in canon.UNSEEN_BENCHMARKS:
        if prompt_format == "auto":
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": canon.SYSTEM_PROMPT},
                 {"role": "user", "content": f"Metin: {bench['text']}"}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = (
                f"<|im_start|>system\n{canon.SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\nMetin: {bench['text']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        input_len = inputs["input_ids"].shape[1]
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        latency = time.time() - t0

        gen_tokens = outputs.shape[1] - input_len
        total_tokens += gen_tokens
        total_latency += latency
        resp = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

        parsed = canon.extract_json(resp)
        gold_terms = [t.lower() for t in bench["gold_terms"]]
        extracted, is_json_valid, atesman, bloom_pred = [], False, 0.0, ""
        if parsed and isinstance(parsed, dict):
            is_json_valid = True
            raw = parsed.get("kavramlar", [])
            if isinstance(raw, list):
                extracted = [str(k).strip() for k in raw]
            atesman = canon.calculate_atesman(str(parsed.get("pedagojik_ozet", "")))
            bloom_pred = str(parsed.get("bloom_seviyesi", "")).strip()
        else:
            atesman = canon.calculate_atesman(resp)

        ext_lower = [t.lower() for t in extracted]
        combined = " ".join(ext_lower) + " " + resp.lower()
        matched = sum(
            1 for gt in gold_terms
            if any(gt in e or e in gt for e in ext_lower) or gt in combined
        )
        recall = matched / len(gold_terms) if gold_terms else 1.0
        precision = (
            sum(1 for e in ext_lower if any(e in gt or gt in e for gt in gold_terms)) / len(extracted)
            if extracted else 0.0
        )
        case_results.append({
            "id": bench["id"],
            "domain": bench["domain"],
            "gold_terms": bench["gold_terms"],
            "extracted_terms": extracted,
            "matched_gold": matched,
            "recall": round(recall, 3),
            "precision": round(precision, 3),
            "atesman": round(atesman, 1),
            "b1_compliant": 50.0 <= atesman <= 75.0,
            "is_json_valid": is_json_valid,
            "gold_bloom": bench["gold_bloom"],
            "pred_bloom": bloom_pred,
            "bloom_match": bloom_pred.lower() == bench["gold_bloom"].lower(),
            "latency_s": round(latency, 3),
            "tokens_generated": gen_tokens,
            "throughput_tok_s": round(gen_tokens / max(0.001, latency), 1),
        })

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    avg_recall = sum(c["recall"] for c in case_results) / len(case_results)
    avg_precision = sum(c["precision"] for c in case_results) / len(case_results)
    f1 = 2 * avg_recall * avg_precision / max(1e-6, avg_recall + avg_precision)
    result = {
        "model_name": name,
        "parameter_count_raw": None,
        "parameter_count_b": None,
        "is_finetuned": False,
        "peak_vram_gb": round(peak_vram_gb, 2),
        "avg_latency_s": round(total_latency / len(case_results), 3),
        "avg_throughput_tok_s": round(total_tokens / max(0.001, total_latency), 1),
        "avg_recall": round(avg_recall, 4),
        "avg_precision": round(avg_precision, 4),
        "f1_score": round(f1, 4),
        "avg_atesman": round(sum(c["atesman"] for c in case_results) / len(case_results), 2),
        "b1_compliance_pct": round(100 * sum(c["b1_compliant"] for c in case_results) / len(case_results), 1),
        "json_validity_pct": round(100 * sum(c["is_json_valid"] for c in case_results) / len(case_results), 1),
        "bloom_accuracy_pct": round(100 * sum(c["bloom_match"] for c in case_results) / len(case_results), 1),
        "quantization": quant,
        "prompt_format": prompt_format,
        "case_results": case_results,
    }
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Local small-model measurement on the BENCH probe.")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--quant", choices=["bf16", "nf4"], default="bf16")
    parser.add_argument("--prompt-format", choices=["chatml", "auto"], default="chatml")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    gpu = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info(0)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "probe": {
            "cases": [b["id"] for b in canon.UNSEEN_BENCHMARKS],
            "prompt": "run_hpc_scaling_law.SYSTEM_PROMPT",
            "scoring": "run_hpc_scaling_law.run_benchmark_for_model (imported, identical)",
        },
        "hardware": f"{gpu} ({total / 1024 ** 3:.2f} GB, {free / 1024 ** 3:.2f} GB free at start)",
        "quantization": args.quant,
        "results": [],
    }
    if args.out.exists():
        prior = json.loads(args.out.read_text(encoding="utf-8"))
        report["results"] = prior.get("results", [])
        report["runs"] = prior.get("runs", [])

    for key in args.models:
        spec = MODELS[key]
        row = {"key": key, **spec}
        try:
            path = local_snapshot(spec["repo"])
            row["local_snapshot"] = path
            logger.info("benchmarking %s (%s, %s)", spec["label"], args.quant, args.prompt_format)
            if args.quant == "bf16" and args.prompt_format == "chatml":
                res = canon.run_benchmark_for_model(spec["label"], path, device="cuda:0", max_new_tokens=256)
                res["quantization"], res["prompt_format"] = "bf16", "chatml (canonical)"
            else:
                res = run_model(spec["label"], path, quant=args.quant, prompt_format=args.prompt_format)
            row["result"] = res
            row["status"] = "measured"
        except torch.cuda.OutOfMemoryError as exc:
            row.update(status="oom", error=str(exc)[:300])
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:400])
        logger.info("%s -> %s", spec["label"], row["status"])

        report["results"] = [r for r in report["results"] if r.get("key") != key] + [row]
        report.setdefault("runs", []).append(
            {"key": key, "quant": args.quant, "status": row["status"], "at": time.strftime("%H:%M:%S")}
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n| model | F1 | R | P | JSON | peak VRAM | latency | status |")
    for r in report["results"]:
        res = r.get("result")
        if res:
            print(f"| {r['label']} | {res['f1_score']:.4f} | {res['avg_recall']:.3f} | "
                  f"{res['avg_precision']:.3f} | {res['json_validity_pct']:.1f}% | "
                  f"{res['peak_vram_gb']:.2f} GB | {res['avg_latency_s']:.2f}s | {r['status']} |")
        else:
            print(f"| {r['label']} | - | - | - | - | - | - | {r['status']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
