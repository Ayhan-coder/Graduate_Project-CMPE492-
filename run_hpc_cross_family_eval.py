#!/usr/bin/env python3
"""Comprehensive 4-Way Cross-Family Evaluation Benchmark on HPC A40.

Evaluates 4 distinct model architectures under identical conditions on the 8 unseen benchmarks:
1. Kumru-2B (Zero-Shot Base)
2. Kumru-2B (EduTerm-2B CEFR SFT Adapter)
3. Qwen2.5-14B (Zero-Shot Base)
4. EduTerm-14B-TR (Full 3-Epoch CEFR SFT Adapter)

Measures:
- Extraction Recall, Precision, F1
- Bloom Taxonomy Classification Accuracy
- Multi-Level CEFR Readability (Ateşman B1, B2, C1)
- Latency (sec) and GPU Peak Memory
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cross_family_eval")

BENCHMARKS: List[Dict[str, Any]] = [
    {
        "id": "BENCH-EE",
        "domain": "Elektrik ve Elektronik",
        "text": "Düğüm gerilimleri analizinde bağımsız düğüm noktaları toprağa göre referans alınarak KCL akım denklemleri kurulur.",
        "gold_terms": ["Düğüm Gerilimleri Analizi", "Düğüm Noktası", "KCL Akım Denklemleri"],
        "gold_bloom": "Analiz"
    },
    {
        "id": "BENCH-BIO",
        "domain": "Biyomühendislik & Biyoloji",
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

def calculate_atesman_score(text: str) -> float:
    if not text or not text.strip():
        return 60.0
    vowels = "aeıioöuü"
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    if not words:
        return 60.0
    syllables = sum(sum(1 for c in w if c in vowels) for w in words)
    spw = syllables / max(1, len(words))
    wps = len(words) / max(1, len(sentences))
    score = 198.825 - (40.175 * spw) - (2.610 * wps)
    return max(0.0, min(100.0, round(score, 1)))

def normalize_str(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s.lower()).strip()

def format_prompt(model_family: str, text: str) -> str:
    system_prompt = (
        "Sen Türkçe teknik eğitimde terminoloji tespiti, pedagojik sadeleştirme ve öğrenci düzeyine "
        "göre (CEFR B1, B2 veya C1) kontrollü karmaşıklıkta açıklama üreten uzman bir asistansın."
    )
    instruction = (
        "Aşağıdaki ders notu metninden teknik terimleri tespit et. "
        "Her terim için Bloom taksonomisi belirle ve CEFR B1 (Sade), B2 (Teknik) ve C1 (Akademik) "
        "düzeylerinde açıklamalar üret. Sonucu geçerli JSON olarak ver."
    )
    user_block = f"{instruction}\n\nDers Notu Metni:\n{text}"

    if model_family == "kumru":
        return (
            f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_block}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    else:  # chatml for qwen
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_block}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

def evaluate_model_arm(
    arm_name: str,
    base_model_path: str,
    adapter_path: str | None,
    model_family: str,
    device: str = "cuda"
) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info("EVALUATING ARM: %s (Base: %s, Adapter: %s)", arm_name, base_model_path, adapter_path)
    logger.info("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<PAD>"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True
    )

    if adapter_path and Path(adapter_path).exists():
        logger.info("Attaching LoRA adapter from %s...", adapter_path)
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model

    model.eval()

    # Determine stop ids
    stop_ids = [tokenizer.eos_token_id]
    if model_family == "kumru":
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot is not None:
            stop_ids.append(eot)
    else:
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None:
            stop_ids.append(im_end)

    results = []
    level_compliance = {"B1": 0, "B2": 0, "C1": 0}

    for bench in BENCHMARKS:
        prompt = format_prompt(model_family, bench["text"])
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=450,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_ids
            )
        elapsed = time.time() - t0

        raw_gen = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        # JSON parsing
        parsed_json = None
        json_match = re.search(r"```json\s*(.*?)\s*```", raw_gen, re.DOTALL)
        candidate = json_match.group(1) if json_match else raw_gen
        try:
            parsed_json = json.loads(candidate)
        except Exception:
            try:
                first_b = candidate.find("{")
                last_b = candidate.rfind("}")
                if first_b != -1 and last_b != -1:
                    parsed_json = json.loads(candidate[first_b:last_b+1])
            except Exception:
                parsed_json = None

        extracted_terms = []
        cefr_readabilities = {"B1": [], "B2": [], "C1": []}
        predicted_bloom = "Bilinmiyor"

        if isinstance(parsed_json, dict):
            terms_obj = parsed_json.get("terimler") or parsed_json.get("terms") or []
            if isinstance(terms_obj, list):
                for t in terms_obj:
                    if isinstance(t, dict):
                        t_name = t.get("terim") or t.get("term") or ""
                        if t_name:
                            extracted_terms.append(t_name)
                        b_val = t.get("bloom_seviyesi") or t.get("bloom_level") or t.get("bloom") or ""
                        if b_val and predicted_bloom == "Bilinmiyor":
                            predicted_bloom = b_val
                        for lvl in ["B1", "B2", "C1"]:
                            expl = t.get(lvl) or t.get(f"{lvl.lower()}_aciklama") or ""
                            if expl:
                                cefr_readabilities[lvl].append(calculate_atesman_score(expl))
            elif isinstance(terms_obj, dict):
                for k in terms_obj.keys():
                    extracted_terms.append(k)

        norm_gold = [normalize_str(t) for t in bench["gold_terms"]]
        norm_pred = [normalize_str(t) for t in extracted_terms if t]
        tp = sum(1 for p in norm_pred if any(p in g or g in p for g in norm_gold))
        precision = tp / max(1, len(norm_pred)) if norm_pred else 0.0
        recall = tp / max(1, len(norm_gold)) if norm_gold else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        avg_b1 = sum(cefr_readabilities["B1"]) / max(1, len(cefr_readabilities["B1"])) if cefr_readabilities["B1"] else 0.0
        avg_b2 = sum(cefr_readabilities["B2"]) / max(1, len(cefr_readabilities["B2"])) if cefr_readabilities["B2"] else 0.0
        avg_c1 = sum(cefr_readabilities["C1"]) / max(1, len(cefr_readabilities["C1"])) if cefr_readabilities["C1"] else 0.0
        is_steering = (avg_b1 >= avg_b2 >= avg_c1) if (avg_b1 > 0 and avg_b2 > 0 and avg_c1 > 0) else False

        bloom_corr = normalize_str(predicted_bloom) in normalize_str(bench["gold_bloom"]) if predicted_bloom != "Bilinmiyor" else False

        results.append({
            "id": bench["id"],
            "domain": bench["domain"],
            "latency_s": round(elapsed, 2),
            "json_valid": parsed_json is not None,
            "extracted_terms": extracted_terms,
            "gold_terms": bench["gold_terms"],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "predicted_bloom": predicted_bloom,
            "bloom_correct": bloom_corr,
            "readability_atesman": {
                "B1": round(avg_b1, 1),
                "B2": round(avg_b2, 1),
                "C1": round(avg_c1, 1)
            },
            "cefr_steering_compliant": is_steering
        })

    avg_f1 = sum(r["f1"] for r in results) / len(results)
    avg_p = sum(r["precision"] for r in results) / len(results)
    avg_r = sum(r["recall"] for r in results) / len(results)
    json_acc = sum(1 for r in results if r["json_valid"]) / len(results)
    bloom_acc = sum(1 for r in results if r["bloom_correct"]) / len(results)
    steering_acc = sum(1 for r in results if r["cefr_steering_compliant"]) / len(results)
    vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)

    logger.info("Summary [%s]: F1=%.2f%%, Precision=%.2f%%, Recall=%.2f%%, JSON=%.1f%%, VRAM=%.2fGB",
                arm_name, avg_f1 * 100, avg_p * 100, avg_r * 100, json_acc * 100, vram_peak)

    # Clean up GPU memory for next arm
    del model
    if adapter_path:
        del base_model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "arm_name": arm_name,
        "base_model": base_model_path,
        "adapter_path": adapter_path,
        "peak_vram_gb": round(vram_peak, 2),
        "macro_precision": round(avg_p, 4),
        "macro_recall": round(avg_r, 4),
        "macro_f1": round(avg_f1, 4),
        "json_schema_validity": round(json_acc, 4),
        "bloom_accuracy": round(bloom_acc, 4),
        "cefr_steering_rate": round(steering_acc, 4),
        "benchmarks": results
    }

def main():
    parser = argparse.ArgumentParser(description="Cross-Family Benchmark Evaluation on HPC A40")
    parser.add_argument("--output", type=str, default="/users/ali.gunder/eduterm-finetune/outputs/cross_family_comprehensive_benchmark.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Starting Cross-Family Comprehensive Benchmark on %s...", device)

    arms = [
        {
            "name": "Kumru-2B (Zero-Shot Base)",
            "base": "/users/ali.gunder/models/Kumru-2B",
            "adapter": None,
            "family": "kumru"
        },
        {
            "name": "Kumru-2B (EduTerm-2B SFT)",
            "base": "/users/ali.gunder/models/Kumru-2B",
            "adapter": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_kumru_2b_cefr",
            "family": "kumru"
        },
        {
            "name": "Qwen2.5-14B (Zero-Shot Base)",
            "base": "/users/ali.gunder/models/Qwen2.5-14B-Instruct",
            "adapter": None,
            "family": "qwen"
        },
        {
            "name": "EduTerm-14B-TR (Full CEFR SFT)",
            "base": "/users/ali.gunder/models/Qwen2.5-14B-Instruct",
            "adapter": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora",
            "family": "qwen"
        }
    ]

    all_results = {}
    for arm in arms:
        res = evaluate_model_arm(
            arm_name=arm["name"],
            base_model_path=arm["base"],
            adapter_path=arm["adapter"],
            model_family=arm["family"],
            device=device
        )
        all_results[arm["name"]] = res

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info("=" * 80)
    logger.info("CROSS-FAMILY BENCHMARK COMPLETED SUCCESSFULLY!")
    logger.info("Saved full comparative report to: %s", args.output)
    logger.info("=" * 80)

if __name__ == "__main__":
    main()

