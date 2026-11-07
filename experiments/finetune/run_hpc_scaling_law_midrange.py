"""Empirical Mid-Range (27B & 32B) Scaling Law Benchmark on Boğaziçi HPC NVIDIA A40 (48GB).

Fills the scaling gap between 14B and 72B:
- Gemma-2-27B-it (27.23B params) in 4-bit NF4
- Qwen2.5-32B-Instruct (32.76B params) in 4-bit NF4

Evaluates across the standardized 8 multi-domain unseen zero-leakage benchmarks:
BENCH-EE, BENCH-BIO, BENCH-CS, BENCH-TOUR, BENCH-ISG, BENCH-KVKK, BENCH-SEC, BENCH-BNK.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Compatibility patch for PyTorch 2.1 dynamo mark_static_address (required for Gemma 2 HybridCache)
try:
    import torch._dynamo
    if not hasattr(torch._dynamo, "mark_static_address"):
        torch._dynamo.mark_static_address = lambda *args, **kwargs: None
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ScalingLawMidRange")

# Multi-domain unseen test benchmarks (Strict zero-leakage test set)
UNSEEN_BENCHMARKS = [
    {
        "id": "BENCH-EE",
        "domain": "Elektrik-Elektronik",
        "text": "Düğüm gerilimleri analizinde bağımsız düğüm noktaları toprağa göre referans alınarak KCL akım denklemleri kurulur.",
        "gold_terms": ["Düğüm Gerilimleri Analizi", "Düğüm Noktası", "KCL Akım Denklemleri"],
        "gold_bloom": "Analiz"
    },
    {
        "id": "BENCH-BIO",
        "domain": "Moleküler Biyoloji",
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

SYSTEM_PROMPT = (
    "Sen Türkçe akademik ve kurumsal eğitim içeriklerini analiz eden uzman bir pedagojik asistansın. "
    "Verilen metindeki teknik terimleri çıkar, B1 seviyesinde sadeleştirilmiş pedagojik bir özet yaz ve Bloom taksonomisi seviyesini belirle. "
    "Yanıtını SADECE aşağıdaki JSON formatında ver, başka hiçbir metin ekleme:\n"
    "{\n"
    '  "kavramlar": ["Terim 1", "Terim 2"],\n'
    '  "pedagojik_ozet": "CEFR B1 sade tanım ve açıklama.",\n'
    '  "bloom_seviyesi": "Hatırlama|Anlama|Uygulama|Analiz|Değerlendirme|Yaratma"\n'
    "}"
)


def calculate_atesman(text: str) -> float:
    """Ateşman readability formula for Turkish."""
    if not text or not text.strip():
        return 0.0
    vowels = set("aeıioöuüAEIİOÖUÜ")
    words = re.findall(r"\b\w+\b", text)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    if not words:
        return 60.0
    syllables = sum(sum(1 for c in w if c in vowels) for w in words)
    spw = syllables / max(1, len(words))
    wps = len(words) / max(1, len(sentences))
    score = 198.825 - (40.175 * spw) - (2.610 * wps)
    return max(0.0, min(100.0, score))


def extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Robust JSON extraction from LLM output."""
    raw = raw_text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    brace_match = re.search(r"(\{.*\})", raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except Exception:
            pass
    try:
        return json.loads(raw)
    except Exception:
        return None


def format_prompt(tokenizer, system_prompt: str, user_text: str) -> str:
    """Format prompt with model-native chat template (handling Gemma vs Qwen differences)."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Metin: {user_text}"}
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Gemma 2 chat template does not accept a distinct system role
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\nMetin: {user_text}"}
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\nMetin: {user_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )


def run_benchmark_for_model(
    model_id_or_path: str,
    model_name: str,
    category: str,
    exact_params: int,
    device: str = "cuda:0",
    max_new_tokens: int = 256
) -> Dict[str, Any]:
    """Execute benchmark for an INT4-NF4 quantized model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info("=" * 80)
    logger.info(f"LOADING MODEL: {model_name} from {model_id_or_path}")
    logger.info("Precision: 4-bit NF4 with bfloat16 compute")
    logger.info("=" * 80)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    load_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Patch SDPA checks if running in older PyTorch / NGC container
    try:
        import transformers.modeling_utils
        import transformers.utils.import_utils
        transformers.utils.import_utils.is_torch_version = lambda op, ver: True
        transformers.modeling_utils.PreTrainedModel._check_and_enable_sdpa = staticmethod(
            lambda config, hard_check_only=False: setattr(config, "_attn_implementation", "sdpa") or config
        )
    except Exception as e:
        logger.warning(f"Note on SDPA patch: {e}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True
    )
    if hasattr(model, "generation_config"):
        model.generation_config.cache_implementation = None
    model.eval()

    load_time = time.time() - load_start
    total_params = exact_params or sum(p.numel() for p in model.parameters())
    param_b = round(total_params / 1e9, 2)
    logger.info(f"Model {model_name} loaded in {load_time:.2f}s | Params: {total_params:,} ({param_b}B)")

    case_results = []
    total_generated_tokens = 0
    total_latency = 0.0

    for bench in UNSEEN_BENCHMARKS:
        prompt = format_prompt(tokenizer, SYSTEM_PROMPT, bench["text"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            except Exception as gen_err:
                logger.warning(f"Default generate failed ({gen_err}), retrying with use_cache=False...")
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
        torch.cuda.synchronize()
        latency = time.time() - t0

        gen_tokens = outputs.shape[1] - input_len
        total_generated_tokens += gen_tokens
        total_latency += latency

        resp_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        parsed = extract_json(resp_text)

        gold_terms = [t.lower() for t in bench["gold_terms"]]
        extracted_terms = []
        is_json_valid = False
        atesman = 0.0
        bloom_pred = ""

        if parsed and isinstance(parsed, dict):
            is_json_valid = True
            raw_kavramlar = parsed.get("kavramlar", [])
            if isinstance(raw_kavramlar, list):
                extracted_terms = [str(k).strip() for k in raw_kavramlar]
            pedagojik_ozet = str(parsed.get("pedagojik_ozet", ""))
            atesman = calculate_atesman(pedagojik_ozet)
            bloom_pred = str(parsed.get("bloom_seviyesi", "")).strip()
        else:
            atesman = calculate_atesman(resp_text)

        extracted_lower = [t.lower() for t in extracted_terms]
        extracted_combined_text = " ".join(extracted_lower) + " " + resp_text.lower()
        matched_gold = 0
        for gt in gold_terms:
            if any(gt in ext or ext in gt for ext in extracted_lower) or gt in extracted_combined_text:
                matched_gold += 1
        recall = matched_gold / len(gold_terms) if gold_terms else 1.0

        if extracted_terms:
            valid_extracted = sum(1 for ext in extracted_lower if any(ext in gt or gt in ext for gt in gold_terms))
            precision = valid_extracted / len(extracted_terms)
        else:
            precision = 0.0

        bloom_match = (bloom_pred.lower() == bench["gold_bloom"].lower())

        logger.info(f"  [{bench['id']}] {bench['domain']}: Recall {recall*100:.0f}%, Prec {precision*100:.0f}%, Bloom: {bloom_pred} (Match: {bloom_match}), Latency: {latency:.2f}s")

        case_results.append({
            "id": bench["id"],
            "domain": bench["domain"],
            "gold_terms": bench["gold_terms"],
            "extracted_terms": extracted_terms,
            "matched_gold": matched_gold,
            "recall": round(recall, 3),
            "precision": round(precision, 3),
            "atesman": round(atesman, 1),
            "b1_compliant": (50.0 <= atesman <= 75.0),
            "is_json_valid": is_json_valid,
            "gold_bloom": bench["gold_bloom"],
            "pred_bloom": bloom_pred,
            "bloom_match": bloom_match,
            "latency_s": round(latency, 3),
            "tokens_generated": gen_tokens,
            "throughput_tok_s": round(gen_tokens / max(0.001, latency), 1)
        })

    peak_vram_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
    avg_recall = sum(c["recall"] for c in case_results) / len(case_results)
    avg_precision = sum(c["precision"] for c in case_results) / len(case_results)
    f1 = 2 * (avg_recall * avg_precision) / max(1e-6, (avg_recall + avg_precision))
    avg_atesman = sum(c["atesman"] for c in case_results) / len(case_results)
    b1_compliance = sum(1 for c in case_results if c["b1_compliant"]) / len(case_results)
    json_validity = sum(1 for c in case_results if c["is_json_valid"]) / len(case_results)
    bloom_acc = sum(1 for c in case_results if c["bloom_match"]) / len(case_results)
    avg_latency = total_latency / len(case_results)
    avg_throughput = total_generated_tokens / max(0.001, total_latency)

    logger.info("-" * 80)
    logger.info(f"SUMMARY FOR {model_name}:")
    logger.info(f"  Peak VRAM: {peak_vram_gb:.2f} GB | Avg Latency: {avg_latency:.2f}s | Throughput: {avg_throughput:.1f} tok/s")
    logger.info(f"  Recall: {avg_recall*100:.1f}% | Precision: {avg_precision*100:.1f}% | F1-Score: {f1*100:.1f}%")
    logger.info(f"  Bloom Accuracy: {bloom_acc*100:.1f}% | B1 Compliance: {b1_compliance*100:.1f}%")

    # Clean memory
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model_name": model_name,
        "parameter_count_raw": total_params,
        "parameter_count_b": param_b,
        "is_finetuned": False,
        "peak_vram_gb": round(peak_vram_gb, 2),
        "avg_latency_s": round(avg_latency, 3),
        "avg_throughput_tok_s": round(avg_throughput, 1),
        "avg_recall": round(avg_recall, 4),
        "avg_precision": round(avg_precision, 4),
        "f1_score": round(f1, 4),
        "avg_atesman": round(avg_atesman, 2),
        "b1_compliance_pct": round(b1_compliance * 100, 1),
        "json_validity_pct": round(json_validity * 100, 1),
        "bloom_accuracy_pct": round(bloom_acc * 100, 1),
        "category": category,
        "case_results": case_results
    }


def main():
    parser = argparse.ArgumentParser(description="Run Mid-Range 27B and 32B Scaling Law Benchmarks.")
    parser.add_argument("--base-report", type=str, default="experiments/outputs/scaling_law_full_72b_report.json", help="Existing scaling law report to merge")
    parser.add_argument("--output-json", type=str, default="experiments/outputs/scaling_law_grand_report.json", help="Merged output report path")
    parser.add_argument("--run-only", type=str, default="all", choices=["all", "27b", "32b"], help="Select which model(s) to benchmark")
    args = parser.parse_args()

    models_to_run = []
    if args.run_only in ("all", "27b"):
        models_to_run.append({
            "model_id_or_path": "unsloth/gemma-2-27b-it-bnb-4bit",
            "model_name": "Gemma-2-27B-it (INT4-NF4)",
            "category": "Geniş Model (27B)",
            "exact_params": 27228812800
        })
    if args.run_only in ("all", "32b"):
        models_to_run.append({
            "model_id_or_path": "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
            "model_name": "Qwen2.5-32B-Instruct (INT4-NF4)",
            "category": "Büyük Model (32B)",
            "exact_params": 32763656192
        })

    new_results = []
    for item in models_to_run:
        res = run_benchmark_for_model(
            model_id_or_path=item["model_id_or_path"],
            model_name=item["model_name"],
            category=item["category"],
            exact_params=item["exact_params"]
        )
        new_results.append(res)

    # Merge with base report
    base_file = Path(args.base_report)
    existing_results = []
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NVIDIA A40"
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if torch.cuda.is_available() else 44.38

    if base_file.exists():
        with open(base_file, "r", encoding="utf-8") as f:
            base_data = json.load(f)
            existing_results = base_data.get("results", [])

    # Filter out duplicate entries if re-running
    existing_names = {r["model_name"] for r in new_results}
    merged = [r for r in existing_results if r["model_name"] not in existing_names]
    merged.extend(new_results)

    # Sort in ascending order by parameter count
    merged.sort(key=lambda r: (r.get("parameter_count_b", 0), r.get("is_finetuned", False)))

    final_report = {
        "benchmark_title": "EduTerm Grand Scaling Law Analysis (0.5B -> 1.5B -> 7B -> 14B -> 27B -> 32B -> 72B)",
        "hardware": {
            "gpu": gpu_name,
            "total_vram_gb": round(vram_total, 2),
            "node": "Boğaziçi HPC (ultramarine)"
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_evaluated_count": len(merged),
        "results": merged
    }

    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    logger.info(f"Merged Grand Scaling Law Report successfully saved to {out_file}")

    print("\n" + "=" * 95)
    print(f"{'Model Architecture':<32} | {'Params':<7} | {'Peak VRAM':<10} | {'Latency':<8} | {'Recall':<8} | {'Prec':<7} | {'F1':<7} | {'Bloom':<7}")
    print("-" * 95)
    for r in merged:
        print(
            f"{r['model_name']:<32} | {r['parameter_count_b']:>5.2f}B | {r['peak_vram_gb']:>8.2f}GB | "
            f"{r['avg_latency_s']:>6.2f}s | {r['avg_recall']*100:>6.1f}% | {r['avg_precision']*100:>5.1f}% | "
            f"{r['f1_score']*100:>5.1f}% | {r['bloom_accuracy_pct']:>5.1f}%"
        )
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()

