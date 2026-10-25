"""Empirical Scaling Law Benchmark on Boğaziçi HPC NVIDIA A40 (48GB).

Evaluates across parameter scales:
- Qwen2.5-0.5B-Instruct (0.49B params) - Edge SLM
- Qwen2.5-1.5B-Instruct (1.54B params) - Uç Cihaz Hedefi
- Qwen2.5-7B-Instruct (7.61B params) - HPC Standard Base
- EduTerm-7B-TR (7.61B + LoRA SFT) - Dikey Alan Uyarlanmış
- Qwen2.5-14B-Instruct (14.7B params) - Büyük HPC Modeli

Key Research Question:
"Model boyutu arttıkça pedagojik sadeleştirme kalitesi, terim doğruluğu
ve donanım maliyetleri ampirik olarak nasıl değişiyor?"
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
logger = logging.getLogger("ScalingLawBenchmark")

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
    "Çıktıyı kesinlikle geçerli bir JSON objesi olarak ver:\n"
    "{\n"
    '  "kavramlar": ["Terim1", "Terim2"],\n'
    '  "pedagojik_ozet": "Sade anlatım...",\n'
    '  "bloom_seviyesi": "Hatırlama|Anlama|Uygulama|Analiz|Değerlendirme|Yaratma"\n'
    "}"
)


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


def run_benchmark_for_model(
    model_name: str,
    model_path: str,
    lora_path: Optional[str] = None,
    device: str = "cuda:0",
    max_new_tokens: int = 256
) -> Dict[str, Any]:
    """Execute benchmark across unseen multi-domain test prompts for a single model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info("=" * 70)
    logger.info(f"LOADING MODEL: {model_name} from {model_path}")
    if lora_path:
        logger.info(f"APPLYING LORA ADAPTER: {lora_path}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )

    if lora_path and os.path.exists(lora_path):
        model = PeftModel.from_pretrained(model, lora_path)
        logger.info(f"Successfully attached LoRA adapter {lora_path}")

    model.eval()
    load_time = time.time() - load_start
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model {model_name} loaded in {load_time:.2f}s | Params: {total_params:,} ({total_params/1e9:.2f}B)")

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

        # Term recall & precision
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
            # Fallback text search
            atesman = calculate_atesman(resp_text)

        # Recall calculation
        extracted_lower = [t.lower() for t in extracted_terms]
        extracted_combined_text = " ".join(extracted_lower) + " " + resp_text.lower()
        matched_gold = 0
        for gt in gold_terms:
            if any(gt in ext or ext in gt for ext in extracted_lower) or gt in extracted_combined_text:
                matched_gold += 1
        recall = matched_gold / len(gold_terms) if gold_terms else 1.0

        # Precision calculation
        if extracted_terms:
            valid_extracted = sum(1 for ext in extracted_lower if any(ext in gt or gt in ext for gt in gold_terms))
            precision = valid_extracted / len(extracted_terms)
        else:
            precision = 0.0

        bloom_match = (bloom_pred.lower() == bench["gold_bloom"].lower())

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

    # Summary metrics
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    avg_recall = sum(c["recall"] for c in case_results) / len(case_results)
    avg_precision = sum(c["precision"] for c in case_results) / len(case_results)
    f1 = 2 * (avg_recall * avg_precision) / max(1e-6, (avg_recall + avg_precision))
    avg_atesman = sum(c["atesman"] for c in case_results) / len(case_results)
    b1_compliance = sum(1 for c in case_results if c["b1_compliant"]) / len(case_results)
    json_validity = sum(1 for c in case_results if c["is_json_valid"]) / len(case_results)
    bloom_acc = sum(1 for c in case_results if c["bloom_match"]) / len(case_results)
    avg_latency = total_latency / len(case_results)
    avg_throughput = total_generated_tokens / max(0.001, total_latency)

    logger.info(
        f"RESULT [{model_name}]: Recall: {avg_recall*100:.1f}% | Precision: {avg_precision*100:.1f}% | "
        f"F1: {f1*100:.1f}% | Ateşman: {avg_atesman:.1f} | B1: {b1_compliance*100:.1f}% | "
        f"JSON: {json_validity*100:.1f}% | Peak VRAM: {peak_vram_gb:.2f} GB | Latency: {avg_latency:.2f}s"
    )

    # Free memory
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model_name": model_name,
        "parameter_count_raw": total_params,
        "parameter_count_b": round(total_params / 1e9, 2),
        "is_finetuned": bool(lora_path),
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
        "case_results": case_results
    }


def main():
    parser = argparse.ArgumentParser(description="HPC Scaling Law Benchmark across Model Sizes")
    parser.add_argument("--models-dir", type=str, default="/users/ali.gunder/models", help="Directory containing downloaded models")
    parser.add_argument("--lora-dir", type=str, default="/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_7b_tr_sft", help="LoRA checkpoint")
    parser.add_argument("--output-json", type=str, default="/users/ali.gunder/eduterm-finetune/outputs/scaling_law_report.json")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    lora_dir = Path(args.lora_dir)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if torch.cuda.is_available() else 0.0

    logger.info(f"Target GPU: {gpu_name} ({total_gpu_vram:.2f} GB VRAM)")

    # Define model suite for Scaling Law analysis
    model_suite = [
        {
            "name": "Qwen2.5-0.5B-Instruct",
            "path": str(models_dir / "Qwen2.5-0.5B-Instruct"),
            "lora": None,
            "category": "Edge SLM"
        },
        {
            "name": "Qwen2.5-1.5B-Instruct",
            "path": str(models_dir / "Qwen2.5-1.5B-Instruct"),
            "lora": None,
            "category": "Uç Cihaz Hedefi"
        },
        {
            "name": "Qwen2.5-7B-Instruct",
            "path": str(models_dir / "Qwen2.5-7B-Instruct"),
            "lora": None,
            "category": "HPC Standard Base"
        },
        {
            "name": "EduTerm-7B-TR (LoRA SFT)",
            "path": str(models_dir / "Qwen2.5-7B-Instruct"),
            "lora": str(lora_dir) if lora_dir.exists() else None,
            "category": "Dikey Alan SFT"
        },
        {
            "name": "Qwen2.5-14B-Instruct",
            "path": str(models_dir / "Qwen2.5-14B-Instruct"),
            "lora": None,
            "category": "Büyük HPC Modeli"
        }
    ]

    all_results = []
    for item in model_suite:
        p = Path(item["path"])
        if not p.exists():
            logger.warning(f"Model path {p} not found, skipping {item['name']}...")
            continue

        res = run_benchmark_for_model(
            model_name=item["name"],
            model_path=item["path"],
            lora_path=item["lora"]
        )
        res["category"] = item["category"]
        all_results.append(res)

    final_report = {
        "benchmark_title": "EduTerm Empirical Model Scaling Law Analysis (0.5B -> 1.5B -> 7B -> 14B)",
        "hardware": {
            "gpu": gpu_name,
            "total_vram_gb": round(total_gpu_vram, 2),
            "node": "Boğaziçi HPC (ultramarine)"
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_evaluated_count": len(all_results),
        "results": all_results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    logger.info(f"Scaling Law Report saved to {output_path}")
    print("\n" + "=" * 90)
    print(f"{'Model Name':<28} | {'Params':<7} | {'VRAM (GB)':<9} | {'Latency':<8} | {'Recall':<8} | {'Precision':<9} | {'Ateşman':<8} | {'JSON Valid':<10}")
    print("-" * 90)
    for r in all_results:
        print(
            f"{r['model_name']:<28} | {r['parameter_count_b']:>5}B | {r['peak_vram_gb']:>7.2f}GB | "
            f"{r['avg_latency_s']:>6.2f}s | {r['avg_recall']*100:>6.1f}% | {r['avg_precision']*100:>7.1f}% | "
            f"{r['avg_atesman']:>7.1f} | {r['json_validity_pct']:>8.1f}%"
        )
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
