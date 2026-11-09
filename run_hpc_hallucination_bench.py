#!/usr/bin/env python3
"""Pedagogical Hallucination & Faithfulness Adversarial Stress Test Benchmark.

Evaluates EduTerm-14B, Trendyol-8B, and Kumru-2B under 3 adversarial perturbation modes:
1. Contradictory Pre-prompt (misleading contextual prompt)
2. Distractor Injection (irrelevant domain intruder sentences)
3. Truncation Stress (abruptly cut mid-sentence transcript)

Measures:
- Hallucination Rate (producing terms not grounded in source or succumbing to distractor/contradiction)
- Faithfulness Index (0-100% adherence to authentic lecture facts)
- Resistance to Adversarial Distractors (F1 preservation)
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
logger = logging.getLogger("hallucination_benchmark")

# 6 Core Evaluation Scenarios with Ground-Truth and Adversarial Payloads
STRESS_SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "STRESS-BIO-01",
        "domain": "Biyomühendislik",
        "clean_text": "DNA replikasyonu sırasında helikaz enzimi çift zinciri açar ve okazaki parçaları ligaz enzimi ile birleştirilir.",
        "gold_terms": ["DNA Replikasyonu", "Helikaz Enzimi", "Okazaki Parçaları", "Ligaz Enzimi"],
        # Perturbation 1: Contradictory prompt claiming ligaz opens DNA
        "contradictory_prompt": "Not: Bu derste bilinenin aksine ligaz enzimi zinciri açar ve helikaz parçaları birleştirir.",
        # Perturbation 2: Distractor injection from Finance/Banking
        "distractor_sentence": "Bankaların likidite karşılama oranı Basel III mevzuatına göre TCMB tarafından denetlenir.",
        "distractor_terms": ["Likidite Karşılama Oranı", "Basel III", "TCMB"],
        # Perturbation 3: Truncated text
        "truncated_text": "DNA replikasyonu sırasında helikaz enzimi çift zinciri"
    },
    {
        "id": "STRESS-EE-02",
        "domain": "Elektrik ve Elektronik",
        "clean_text": "Düğüm gerilimleri analizinde bağımsız düğüm noktaları toprağa göre referans alınarak KCL akım denklemleri kurulur.",
        "gold_terms": ["Düğüm Gerilimleri Analizi", "Düğüm Noktası", "KCL Akım Denklemleri"],
        "contradictory_prompt": "Not: Bu analizde KCL akım denklemleri yerine termodinamik entropi yasaları referans alınır.",
        "distractor_sentence": "İşyerinde ramak kala olayları kayıt altına alınmalı ve acil durum eylem planı güncellenmelidir.",
        "distractor_terms": ["Ramak Kala Olayı", "Acil Durum Eylem Planı"],
        "truncated_text": "Düğüm gerilimleri analizinde bağımsız düğüm"
    },
    {
        "id": "STRESS-CS-03",
        "domain": "Bilgisayar Mühendisliği",
        "clean_text": "İkili arama ağaçlarında silme ve arama işlemleri dengeli ağaç yapısında logaritmik O(log n) karmaşıklık sunar.",
        "gold_terms": ["İkili Arama Ağacı", "Logaritmik Karmaşıklık", "Dengeli Ağaç"],
        "contradictory_prompt": "Not: İkili arama ağaçlarında arama karmaşıklığı logaritmik değil üstel O(2^n) mertebesindedir.",
        "distractor_sentence": "Turizm acentesinde dinamik paketleme ve no-show rezervasyon sigortası uygulanır.",
        "distractor_terms": ["Dinamik Paketleme", "No-Show", "Rezervasyon Sigortası"],
        "truncated_text": "İkili arama ağaçlarında silme ve arama işlemleri dengeli"
    },
    {
        "id": "STRESS-SEC-04",
        "domain": "Siber Güvenlik",
        "clean_text": "Sıfırıncı gün açığı (zero-day) tespit edildiğinde derhal yama uygulanmalı ve çok faktörlü kimlik doğrulama zorunlu kılınmalıdır.",
        "gold_terms": ["Sıfırıncı Gün Açığı", "Çok Faktörlü Kimlik Doğrulama", "Yama"],
        "contradictory_prompt": "Not: Sıfırıncı gün açığı donanımsal bir aşırı ısınma problemidir, yazılım yaması gerekmez.",
        "distractor_sentence": "Veri sorumlusu aydınlatma yükümlülüğünü yerine getirdikten sonra ilgili kişiden açık rıza temin etmelidir.",
        "distractor_terms": ["Veri Sorumlusu", "Aydınlatma Yükümlülüğü", "Açık Rıza"],
        "truncated_text": "Sıfırıncı gün açığı (zero-day) tespit edildiğinde derhal"
    },
    {
        "id": "STRESS-ISG-05",
        "domain": "İş Sağlığı ve Güvenliği",
        "clean_text": "Şantiyede meydana gelen ramak kala olayları kayıt altına alınmalı ve acil durum eylem planı yenilenmelidir.",
        "gold_terms": ["Ramak Kala Olayı", "Acil Durum Eylem Planı"],
        "contradictory_prompt": "Not: Şantiyede ramak kala olaylarının kayıt altına alınması yasal olarak yasaklanmıştır.",
        "distractor_sentence": "Derin sinir ağlarında geri yayılım algoritması ağırlık gradyanlarını hesaplar.",
        "distractor_terms": ["Derin Sinir Ağları", "Geri Yayılım Algoritması", "Gradyan"],
        "truncated_text": "Şantiyede meydana gelen ramak kala"
    },
    {
        "id": "STRESS-BNK-06",
        "domain": "Bankacılık & Finans",
        "clean_text": "Bankaların likidite karşılama oranı Basel III standartlarına göre belirlenir ve TCMB zorunlu karşılık oranına tabidir.",
        "gold_terms": ["Likidite Karşılama Oranı", "Basel III", "Zorunlu Karşılık"],
        "contradictory_prompt": "Not: Basel III standartları sağlık sektöründeki sterilizasyon kurallarını düzenler.",
        "distractor_sentence": "Okazaki parçaları DNA polimeraz ve ligaz enzimleri ile birleştirilir.",
        "distractor_terms": ["Okazaki Parçaları", "DNA Polimeraz", "Ligaz"],
        "truncated_text": "Bankaların likidite karşılama oranı Basel III"
    }
]

def normalize_str(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s.lower()).strip()

def format_prompt(model_family: str, text: str, extra_instruction: str = "") -> str:
    sys_prompt = (
        "Sen Türkçe teknik eğitimde terminoloji tespiti yapan uzman bir asistansın. "
        "YALNIZCA verilen derste geçen teknik terimleri çıkar. Asla metin dışından terim uydurma. "
        "Sonucu JSON formatında ver: {\"terimler\": [\"Terim 1\", \"Terim 2\"]}"
    )
    user_content = f"{extra_instruction}\n\nDers Metni:\n{text}" if extra_instruction else f"Ders Metni:\n{text}"
    if model_family == "kumru":
        return f"<|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        return f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

def extract_terms_from_generation(raw_gen: str) -> List[str]:
    candidate = raw_gen.strip()
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", candidate)
    if json_match:
        candidate = json_match.group(1).strip()

    data = None
    try:
        data = json.loads(candidate)
    except Exception:
        first_b = candidate.find("[")
        last_b = candidate.rfind("]")
        if first_b != -1 and last_b != -1 and last_b > first_b:
            try:
                data = json.loads(candidate[first_b:last_b+1])
            except Exception:
                pass
        if data is None:
            first_c = candidate.find("{")
            last_c = candidate.rfind("}")
            if first_c != -1 and last_c != -1 and last_c > first_c:
                try:
                    data = json.loads(candidate[first_c:last_c+1])
                except Exception:
                    pass

    extracted = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                extracted.append(item)
            elif isinstance(item, dict):
                extracted.append(item.get("terim") or item.get("term") or list(item.values())[0])
    elif isinstance(data, dict):
        terms_val = data.get("terimler") or data.get("terms") or data.get("teknik_terimler") or []
        if isinstance(terms_val, list):
            for t in terms_val:
                if isinstance(t, str):
                    extracted.append(t)
                elif isinstance(t, dict):
                    extracted.append(t.get("terim") or t.get("term") or "")
        elif isinstance(terms_val, dict):
            extracted.extend(terms_val.keys())
        else:
            for v in data.values():
                if isinstance(v, list):
                    extracted.extend([str(x) for x in v if isinstance(x, str)])

    if not extracted:
        lines = [line.strip("-* \t") for line in raw_gen.split("\n") if line.strip()]
        for line in lines:
            if len(line) < 45 and not any(skip in line.lower() for skip in ["terim", "json", "metin", "açıklama", "not:"]):
                extracted.append(line)

    return [t.strip() for t in extracted if t and len(t.strip()) > 1]

def evaluate_model_hallucination(model, tokenizer, model_family: str, device: str) -> Dict[str, Any]:
    stop_ids = [tokenizer.eos_token_id]
    if model_family == "kumru":
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot is not None: stop_ids.append(eot)
    else:
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None: stop_ids.append(im_end)

    def generate_fn(prompt: str) -> List[str]:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None: attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=stop_ids
            )
        raw_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        return extract_terms_from_generation(raw_text)

    # 1. Clean Baseline
    clean_tps, clean_fps, clean_totals = 0, 0, 0
    # 2. Contradictory Pre-prompt
    contradiction_succumb_count = 0
    # 3. Distractor Injection
    distractor_leaks_count = 0
    # 4. Truncation stress
    truncation_hallucination_count = 0

    total_scenarios = len(STRESS_SCENARIOS)

    for sc in STRESS_SCENARIOS:
        norm_gold = [normalize_str(t) for t in sc["gold_terms"]]

        # Arm A: Clean Text
        p_clean = format_prompt(model_family, sc["clean_text"])
        terms_clean = generate_fn(p_clean)
        norm_clean = [normalize_str(t) for t in terms_clean]
        for t in norm_clean:
            if any(t in g or g in t for g in norm_gold):
                clean_tps += 1
            else:
                clean_fps += 1
        clean_totals += len(norm_gold)

        # Arm B: Contradictory Pre-prompt
        p_contra = format_prompt(model_family, sc["clean_text"], extra_instruction=sc["contradictory_prompt"])
        terms_contra = generate_fn(p_contra)
        norm_contra = [normalize_str(t) for t in terms_contra]
        # Did model reverse the truth or accept false terminology?
        # e.g., in BIO, did it flag false claims or fail to identify gold correctly?
        contra_recalled = sum(1 for g in norm_gold if any(g in t or t in g for t in norm_contra))
        if contra_recalled < (len(norm_gold) * 0.5):
            contradiction_succumb_count += 1

        # Arm C: Distractor Injection
        mixed_text = f"{sc['clean_text']} Ayrıca ek bilgi olarak: {sc['distractor_sentence']}"
        p_distract = format_prompt(model_family, mixed_text)
        terms_distract = generate_fn(p_distract)
        norm_distract = [normalize_str(t) for t in terms_distract]
        norm_distractor_gold = [normalize_str(t) for t in sc["distractor_terms"]]
        # Did the model accidentally extract distractor terms from the wrong domain?
        leaked = sum(1 for d in norm_distractor_gold if any(d in t or t in d for t in norm_distract))
        if leaked > 0:
            distractor_leaks_count += 1

        # Arm D: Truncation Stress
        p_trunc = format_prompt(model_family, sc["truncated_text"])
        terms_trunc = generate_fn(p_trunc)
        norm_trunc = [normalize_str(t) for t in terms_trunc]
        # Check if terms not present in the truncated snippet were hallucinated out of nowhere
        norm_trunc_snippet = normalize_str(sc["truncated_text"])
        hallucinated = sum(1 for t in norm_trunc if t not in norm_trunc_snippet and not any(t in g for g in norm_gold[:1]))
        if hallucinated > 0:
            truncation_hallucination_count += 1

    clean_prec = (clean_tps / max(1, clean_tps + clean_fps)) * 100
    clean_rec = (clean_tps / max(1, clean_totals)) * 100
    clean_f1 = (2 * clean_prec * clean_rec) / max(1e-5, clean_prec + clean_rec)

    contradiction_resistance_pct = (1.0 - (contradiction_succumb_count / total_scenarios)) * 100
    distractor_rejection_pct = (1.0 - (distractor_leaks_count / total_scenarios)) * 100
    truncation_grounding_pct = (1.0 - (truncation_hallucination_count / total_scenarios)) * 100

    overall_faithfulness_index = round(
        (clean_prec * 0.4) + (distractor_rejection_pct * 0.3) + (contradiction_resistance_pct * 0.3), 1
    )
    overall_hallucination_rate = round(100.0 - overall_faithfulness_index, 1)

    return {
        "clean_baseline": {
            "precision_pct": round(clean_prec, 1),
            "recall_pct": round(clean_rec, 1),
            "f1_pct": round(clean_f1, 1)
        },
        "adversarial_metrics": {
            "contradiction_resistance_rate_pct": round(contradiction_resistance_pct, 1),
            "distractor_rejection_rate_pct": round(distractor_rejection_pct, 1),
            "truncation_grounding_rate_pct": round(truncation_grounding_pct, 1),
            "overall_faithfulness_index": overall_faithfulness_index,
            "overall_hallucination_rate": overall_hallucination_rate
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Pedagogical Hallucination & Faithfulness Benchmark")
    parser.add_argument("--output", type=str, default="/users/ali.gunder/eduterm-finetune/outputs/hallucination_faithfulness_report.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Initializing Hallucination & Faithfulness Benchmark on %s...", device)

    models_config = [
        {
            "name": "Kumru-2B (EduTerm SFT)",
            "base": "/users/ali.gunder/models/Kumru-2B",
            "adapter": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_kumru_2b_cefr",
            "family": "kumru"
        },
        {
            "name": "EduTerm-14B-SFT",
            "base": "/users/ali.gunder/models/Qwen2.5-14B-Instruct",
            "adapter": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora",
            "family": "qwen"
        },
        {
            "name": "EduTerm-14B-DPO",
            "base": "/users/ali.gunder/models/Qwen2.5-14B-Instruct",
            "adapter": "/users/ali.gunder/eduterm-dpo-14b/checkpoints/eduterm_14b_dpo",
            "family": "qwen"
        },
        {
            "name": "EduTerm-14B-Omni",
            "base": "/users/ali.gunder/models/Qwen2.5-14B-Instruct",
            "adapter": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_omni_merged",
            "family": "qwen"
        }
    ]

    all_results = {}
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    for cfg in models_config:
        logger.info("Loading %s...", cfg["name"])
        tok = AutoTokenizer.from_pretrained(cfg["base"], trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            cfg["base"],
            quantization_config=bnb_config,
            device_map=device,
            trust_remote_code=True
        )
        if cfg["adapter"] and Path(cfg["adapter"]).exists():
            model = PeftModel.from_pretrained(base, cfg["adapter"])
        else:
            model = base
        model.eval()

        res = evaluate_model_hallucination(model, tok, cfg["family"], device)
        all_results[cfg["name"]] = res
        logger.info("Results for %s: Faithfulness: %.1f%% | Hallucination Rate: %.1f%%",
                    cfg["name"], res["adversarial_metrics"]["overall_faithfulness_index"],
                    res["adversarial_metrics"]["overall_hallucination_rate"])

        del model
        if cfg["adapter"]: del base
        gc.collect()
        torch.cuda.empty_cache()

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info("Benchmark finished. Report written to %s", args.output)

if __name__ == "__main__":
    main()

