"""Empirical Quantization Degradation Benchmark on Boğaziçi HPC NVIDIA A40 (48GB).

Evaluates EduTerm-7B-TR across Quantization Levels:
1. FP16 / BF16 (16-bit Full Precision - Baseline)
2. INT8 (BitsAndBytes 8-bit / LLM.int8())
3. INT4-NF4 (BitsAndBytes NormalFloat4 with double quantization)
4. INT4-FP4 (BitsAndBytes Standard 4-bit Float)

Measures:
- Memory footprint: Model weight footprint & peak inference VRAM (GB)
- Compression factor: (1.0x -> 1.8x -> ~3.3x)
- Latency & Generation Throughput: Wall-clock seconds & tokens/sec
- Terminology Extraction Recall, Precision & F1 Score (%) across 10 multi-domain benchmarks
- Hallucination Rate (%)
- Ateşman Readability Score & CEFR B1 in-band compliance share (%)

Key Research Question:
"4-bit kuantalama (NF4) yerel dağıtımda %70+ VRAM tasarrufu sağlarken,
pedagojik sadeleştirme kalitesinde ve terim yakalama F1 skorunda
ne kadarlık bir ödünleşime (Pareto trade-off) yol açmaktadır?"
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("QuantizationCurveBenchmark")


def calculate_atesman(text: str) -> float:
    """Calculate Ateşman readability index for Turkish text."""
    vowels = "aeıioöuü"
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    if not words:
        return 60.0
    syllables = sum(sum(1 for c in w if c in vowels) for w in words)
    spw = syllables / max(1, len(words))
    wps = len(words) / max(1, len(sentences))
    score = 198.825 - (40.175 * spw) - (2.610 * wps)
    return max(0.0, min(100.0, score))


# 10 Multi-domain Benchmarks (Strict Zero-Leakage)
BENCHMARKS = [
    {
        "id": "BENCH-EE",
        "domain": "Elektrik-Elektronik",
        "lang": "tr",
        "text": "Düğüm gerilimleri analizinde bağımsız düğüm noktaları toprağa göre referans alınarak KCL akım denklemleri kurulur.",
        "gold_terms": ["Düğüm Gerilimleri Analizi", "Düğüm Noktası", "KCL Akım Denklemleri"],
        "gold_bloom": "Analiz"
    },
    {
        "id": "BENCH-BIO",
        "domain": "Moleküler Biyoloji",
        "lang": "tr",
        "text": "DNA replikasyonu sırasında helikaz enzimi çift zinciri açar ve okazaki parçaları ligaz enzimi ile birleştirilir.",
        "gold_terms": ["DNA Replikasyonu", "Helikaz Enzimi", "Okazaki Parçaları", "Ligaz Enzimi"],
        "gold_bloom": "Anlama"
    },
    {
        "id": "BENCH-CS",
        "domain": "Bilgisayar Mühendisliği",
        "lang": "tr",
        "text": "İkili arama ağaçlarında silme ve arama işlemleri dengeli ağaç yapısında logaritmik O(log n) karmaşıklık sunar.",
        "gold_terms": ["İkili Arama Ağacı", "Logaritmik Karmaşıklık", "Dengeli Ağaç"],
        "gold_bloom": "Uygulama"
    },
    {
        "id": "BENCH-TOUR",
        "domain": "Turizm & Seyahat (Tatilsepeti)",
        "lang": "tr",
        "text": "Dinamik paketleme motorunda uçak bileti ve otel kontenjanı birleştirilirken no-show riski için erken rezervasyon sigortası uygulanır.",
        "gold_terms": ["Dinamik Paketleme", "No-Show Riski", "Erken Rezervasyon Sigortası"],
        "gold_bloom": "Uygulama"
    },
    {
        "id": "BENCH-ISG",
        "domain": "İş Sağlığı ve Güvenliği",
        "lang": "tr",
        "text": "Şantiyede meydana gelen ramak kala olayları kayıt altına alınmalı ve acil durum eylem planı yenilenmelidir.",
        "gold_terms": ["Ramak Kala Olayı", "Acil Durum Eylem Planı"],
        "gold_bloom": "Uygulama"
    },
    {
        "id": "BENCH-KVKK",
        "domain": "Kişisel Verilerin Korunması",
        "lang": "tr",
        "text": "Veri sorumlusu aydınlatma yükümlülüğünü yerine getirdikten sonra ilgili kişiden açık rıza temin etmelidir.",
        "gold_terms": ["Veri Sorumlusu", "Aydınlatma Yükümlülüğü", "Açık Rıza"],
        "gold_bloom": "Anlama"
    },
    {
        "id": "BENCH-BANK",
        "domain": "Bankacılık & Finans",
        "lang": "tr",
        "text": "Merkez bankası politika faizini artırdığında likidite tuzağından kaçınmak için zorunlu karşılık oranları revize edilir.",
        "gold_terms": ["Politika Faizi", "Likidite Tuzağı", "Zorunlu Karşılık Oranı"],
        "gold_bloom": "Analiz"
    },
    {
        "id": "BENCH-ESG",
        "domain": "Sürdürülebilirlik & ESG",
        "lang": "tr",
        "text": "Karbon ayak izi hesaplamasında Kapsam 1 doğrudan emisyonları, Kapsam 2 satın alınan elektriği, Kapsam 3 ise tedarik zincirini kapsar.",
        "gold_terms": ["Karbon Ayak İzi", "Kapsam 1", "Kapsam 2", "Kapsam 3"],
        "gold_bloom": "Anlama"
    },
    {
        "id": "TR-1-PM",
        "domain": "Proje Yönetimi (EVM)",
        "lang": "tr",
        "text": "Kazanılmış değer yönetiminde maliyet sapması gerçekleşen maliyet ile kazanılmış değer arasındaki farktır. Çizelge performans indeksi ise planlanan değere bölünerek hesaplanır.",
        "gold_terms": ["Kazanılmış Değer Yönetimi", "Maliyet Sapması", "Gerçekleşen Maliyet", "Kazanılmış Değer", "Çizelge Performans İndeksi", "Planlanan Değer"],
        "gold_bloom": "Uygulama"
    },
    {
        "id": "EN-1-AGILE",
        "domain": "Agile Software Engineering",
        "lang": "en",
        "text": "In the Scrum framework, the product owner prioritizes the product backlog while the development team commits to user stories during sprint planning with cross-functional members.",
        "gold_terms": ["Scrum", "Product Owner", "Product Backlog", "Development Team", "Sprint Planning", "Cross-Functional Team"],
        "gold_bloom": "Uygulama"
    }
]

SYSTEM_PROMPT = (
    "Sen uzman bir eğitim ve mühendislik asistanısın. "
    "Verilen metindeki teknik terimleri çıkar, B1 seviyesinde sadeleştirilmiş pedagojik bir özet yaz ve Bloom taksonomisi seviyesini belirle. "
    "Çıktıyı kesinlikle geçerli bir JSON objesi olarak ver:\n"
    "{\n"
    '  "kavramlar": ["Terim1", "Terim2"],\n'
    '  "pedagojik_ozet": "Sade anlatım...",\n'
    '  "bloom_seviyesi": "Hatırlama|Anlama|Uygulama|Analiz|Değerlendirme|Yaratma"\n'
    "}"
)


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


def evaluate_quantization_tier(
    tier_name: str,
    quant_config: Any,
    torch_dtype: torch.dtype,
    model_path: str,
    adapter_path: Optional[str] = None
) -> Dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info("=" * 80)
    logger.info(f"BENCHMARKING QUANTIZATION TIER: {tier_name}")
    logger.info("=" * 80)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_start = time.perf_counter()
    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True
    }
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
    else:
        load_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    
    if adapter_path and os.path.exists(adapter_path):
        logger.info(f"Attaching LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)

    load_duration = time.perf_counter() - load_start
    load_vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    logger.info(f"Model Loaded in {load_duration:.2f}s | Baseline Allocated VRAM: {load_vram_gb:.2f} GB")

    model.eval()

    total_tokens = 0
    total_gen_time = 0.0
    case_results = []
    all_extracted = []
    all_golds = []
    total_matched_gold = 0
    total_gold_count = 0
    total_valid_extracted = 0
    total_extracted_count = 0
    atesman_list = []

    for bench in BENCHMARKS:
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nMetin: {bench['text']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]

        torch.cuda.synchronize()
        start_t = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=280,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_t

        output_tokens = outputs[0][input_len:]
        num_new_tokens = len(output_tokens)
        resp_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()

        total_tokens += num_new_tokens
        total_gen_time += elapsed

        parsed = extract_json(resp_text)
        gold_terms = [t.lower() for t in bench["gold_terms"]]
        extracted_terms = []
        atesman = 0.0

        if parsed and isinstance(parsed, dict):
            raw_kav = parsed.get("kavramlar", [])
            if isinstance(raw_kav, list):
                extracted_terms = [str(k).strip() for k in raw_kav]
            pedagojik_ozet = str(parsed.get("pedagojik_ozet", ""))
            atesman = calculate_atesman(pedagojik_ozet)
        else:
            # Fallback regex extraction
            lines = resp_text.split("\n")
            for line in lines:
                m = re.search(r"[-*•]\s*([A-Za-zÇĞİÖŞÜçğıöşü\s]+)", line)
                if m:
                    extracted_terms.append(m.group(1).strip())
            atesman = calculate_atesman(resp_text)

        atesman_list.append(atesman)

        # Recall calculation
        extracted_lower = [t.lower() for t in extracted_terms]
        extracted_combined_text = " ".join(extracted_lower) + " " + resp_text.lower()
        matched_gold = 0
        for gt in gold_terms:
            if any(gt in ext or ext in gt for ext in extracted_lower) or gt in extracted_combined_text:
                matched_gold += 1

        recall = (matched_gold / len(gold_terms)) if gold_terms else 1.0

        # Precision calculation
        if extracted_terms:
            valid_extracted = sum(1 for ext in extracted_lower if any(ext in gt or gt in ext for gt in gold_terms))
            precision = valid_extracted / len(extracted_terms)
        else:
            precision = 0.0

        total_matched_gold += matched_gold
        total_gold_count += len(gold_terms)
        total_valid_extracted += (valid_extracted if extracted_terms else 0)
        total_extracted_count += len(extracted_terms)

        case_results.append({
            "id": bench["id"],
            "domain": bench["domain"],
            "elapsed_s": round(elapsed, 3),
            "tokens_per_s": round(num_new_tokens / max(0.001, elapsed), 1),
            "extracted_terms": extracted_terms,
            "matched_gold": matched_gold,
            "recall": round(recall * 100.0, 1),
            "precision": round(precision * 100.0, 1),
            "atesman": round(atesman, 1),
            "b1_compliant": (50.0 <= atesman <= 75.0)
        })

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    macro_recall = (total_matched_gold / max(1, total_gold_count)) * 100.0
    macro_precision = (total_valid_extracted / max(1, total_extracted_count)) * 100.0
    macro_f1 = (2 * macro_precision * macro_recall / max(0.001, (macro_precision + macro_recall)))
    hallucination = 100.0 - macro_precision

    in_b1_count = sum(1 for a in atesman_list if 50.0 <= a <= 75.0)
    b1_share_pct = round(in_b1_count / max(1, len(atesman_list)) * 100.0, 1)
    avg_atesman = round(sum(atesman_list) / max(1, len(atesman_list)), 1)
    throughput = round(total_tokens / max(0.001, total_gen_time), 1)
    avg_latency = round(total_gen_time / len(BENCHMARKS), 2)

    logger.info(
        f"Tier {tier_name} Complete: Peak VRAM: {peak_vram_gb:.2f} GB | "
        f"Throughput: {throughput} tok/s | F1: {macro_f1:.1f}% | "
        f"Recall: {macro_recall:.1f}% | Precision: {macro_precision:.1f}% | "
        f"Ateşman: {avg_atesman} (B1 Share: {b1_share_pct}%)"
    )

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "tier": tier_name,
        "load_vram_gb": round(load_vram_gb, 2),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "avg_latency_s": avg_latency,
        "throughput_tokens_per_s": throughput,
        "overall_recall_pct": round(macro_recall, 1),
        "overall_precision_pct": round(macro_precision, 1),
        "overall_f1_pct": round(macro_f1, 1),
        "hallucination_pct": round(hallucination, 1),
        "mean_atesman": avg_atesman,
        "cefr_b1_compliance_pct": b1_share_pct,
        "cases": case_results
    }


def main():
    parser = argparse.ArgumentParser(description="Empirical Quantization Degradation Benchmark")
    parser.add_argument("--model-path", type=str, default="/users/ali.gunder/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--sft-adapter", type=str, default="/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_7b_tr_sft")
    parser.add_argument("--output-json", type=str, default="experiments/outputs/quantization_degradation_report.json")
    args = parser.parse_args()

    from transformers import BitsAndBytesConfig

    tiers = [
        {
            "name": "FP16 (Full Precision)",
            "config": None,
            "dtype": torch.bfloat16
        },
        {
            "name": "INT8 (LLM.int8())",
            "config": BitsAndBytesConfig(load_in_8bit=True),
            "dtype": torch.bfloat16
        },
        {
            "name": "INT4-NF4 (NormalFloat4 + DQ)",
            "config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            ),
            "dtype": torch.bfloat16
        },
        {
            "name": "INT4-FP4 (Standard Float4)",
            "config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_use_double_quant=False,
                bnb_4bit_compute_dtype=torch.bfloat16
            ),
            "dtype": torch.bfloat16
        }
    ]

    results = []
    fp16_vram = None

    for t in tiers:
        try:
            res = evaluate_quantization_tier(
                tier_name=t["name"],
                quant_config=t["config"],
                torch_dtype=t["dtype"],
                model_path=args.model_path,
                adapter_path=args.sft_adapter if os.path.exists(args.sft_adapter) else None
            )
            if fp16_vram is None:
                fp16_vram = res["load_vram_gb"]
            res["compression_ratio"] = round(fp16_vram / max(0.1, res["load_vram_gb"]), 2)
            results.append(res)
        except Exception as e:
            logger.error(f"Error evaluating tier {t['name']}: {e}", exc_info=True)

    report = {
        "benchmark": "BOUN HPC NVIDIA A40 Quantization Degradation Analysis",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "vram_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        },
        "model": "EduTerm-7B-TR (Qwen2.5-7B + SFT LoRA)",
        "tiers": results
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Quantization Degradation Report saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
