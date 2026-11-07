"""Empirical 72B Scaling Law Benchmark on Boğaziçi HPC NVIDIA A40 (48GB).

Completes the Grand Scaling Law Curve:
0.49B (Edge SLM) -> 1.54B (Mobile SLM) -> 7.61B (HPC Baseline) -> 14.7B (Mid HPC) -> 72.7B (Flagship LLM)

Loads Qwen2.5-72B-Instruct in 4-bit NF4 to fit within the 44.38 GiB (48 GB) A40 memory ceiling.
Evaluates across 8 multi-domain unseen zero-leakage benchmarks.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ScalingLaw72B")

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


def run_benchmark_72b(
    model_id_or_path: str,
    device: str = "cuda:0",
    max_new_tokens: int = 256
) -> Dict[str, Any]:
    """Execute benchmark for Qwen2.5-72B-Instruct using 4-bit NF4 quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info("=" * 80)
    logger.info(f"LOADING FLAGSHIP MODEL: Qwen2.5-72B-Instruct from {model_id_or_path}")
    logger.info("Precision: 4-bit NF4 with bfloat16 compute (Designed for 48GB A40 VRAM)")
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

    import transformers.modeling_utils
    import transformers.utils.import_utils
    transformers.utils.import_utils.is_torch_version = lambda op, ver: True
    transformers.modeling_utils.PreTrainedModel._check_and_enable_sdpa = staticmethod(
        lambda config, hard_check_only=False: setattr(config, "_attn_implementation", "sdpa") or config
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()

    load_time = time.time() - load_start
    total_params = 72706988032  # Qwen2.5-72B exact parameter count
    logger.info(f"Flagship 72B Model successfully loaded in {load_time:.2f}s | Params: {total_params:,} (72.71B)")

    case_results = []
    total_generated_tokens = 0
    total_latency = 0.0

    for bench in UNSEEN_BENCHMARKS:
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nMetin: {bench['text']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
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
    logger.info(f"SUMMARY FOR Qwen2.5-72B-Instruct (4-bit NF4):")
    logger.info(f"  Peak VRAM: {peak_vram_gb:.2f} GB | Avg Latency: {avg_latency:.2f}s | Throughput: {avg_throughput:.1f} tok/s")
    logger.info(f"  Recall: {avg_recall*100:.1f}% | Precision: {avg_precision*100:.1f}% | F1-Score: {f1*100:.1f}%")
    logger.info(f"  Bloom Accuracy: {bloom_acc*100:.1f}% | B1 Compliance: {b1_compliance*100:.1f}%")

    return {
        "model_name": "Qwen2.5-72B-Instruct (INT4-NF4)",
        "parameter_count_raw": total_params,
        "parameter_count_b": round(total_params / 1e9, 2),
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
        "category": "Bayrak Gemisi (Flagship 72B)",
        "case_results": case_results
    }


def main():
    parser = argparse.ArgumentParser(description="Run 72B Scaling Law Benchmark.")
    parser.add_argument("--model-path", type=str, default="unsloth/Qwen2.5-72B-Instruct-bnb-4bit", help="Hugging Face model ID or path")
    parser.add_argument("--base-report", type=str, default="experiments/outputs/scaling_law_report.json", help="Existing scaling law report")
    parser.add_argument("--output-json", type=str, default="experiments/outputs/scaling_law_full_72b_report.json", help="Full merged report")
    args = parser.parse_args()

    res_72b = run_benchmark_72b(args.model_path)

    # Merge with existing scaling law baseline
    base_file = Path(args.base_report)
    merged_results = []
    if base_file.exists():
        with open(base_file, "r", encoding="utf-8") as f:
            base_data = json.load(f)
            merged_results = base_data.get("results", [])

    merged_results.append(res_72b)

    final_report = {
        "benchmark_title": "EduTerm Grand Scaling Law Analysis (0.5B -> 1.5B -> 7B -> 14B -> 72B)",
        "hardware": {
            "gpu": "NVIDIA A40",
            "total_vram_gb": 44.38,
            "node": "Boğaziçi HPC (ultramarine)"
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_evaluated_count": len(merged_results),
        "results": merged_results
    }

    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    logger.info(f"Grand Scaling Law Report saved to {out_file}")


if __name__ == "__main__":
    main()
