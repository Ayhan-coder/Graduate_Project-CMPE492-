# -*- coding: utf-8 -*-
"""
Standalone Evaluation of EduTerm-14B-CEFR Checkpoint on Boğaziçi HPC A40.
Loads the fine-tuned LoRA checkpoint and evaluates 8 unseen disciplines for:
  - Terminology Extraction Recall / Precision / F1
  - CEFR Multi-Level (B1, B2, C1) complexity compliance (Ateşman readability)
  - Bloom Taxonomy Classification Accuracy
  - Structural JSON Compliance
"""

import os
import sys
import json
import time
import re
import logging
from pathlib import Path
from typing import Dict, List, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_cefr_14b")

BENCHMARKS: List[Dict[str, Any]] = [
    {
        "id": "BENCH-EE",
        "domain": "Elektrik ve Elektronik",
        "text": "Düğüm gerilimleri yöntemi, Kirchhoff'un akım kanununa (KCL) dayanır. Her bağımsız düğüm için akım denklemleri yazılarak devre analiz edilir.",
        "gold_terms": ["Düğüm Gerilimleri Analizi", "Düğüm Noktası", "KCL Akım Denklemleri"],
        "gold_bloom": "Analiz"
    },
    {
        "id": "BENCH-BIO",
        "domain": "Biyomühendislik & Biyoloji",
        "text": "DNA replikasyonunda helikaz enzimi çift sarmalı açar. Kesintili zincirde Okazaki parçaları oluşur ve DNA ligaz bu parçaları birleştirir.",
        "gold_terms": ["DNA Replikasyonu", "Helikaz Enzimi", "Okazaki Parçaları", "Ligaz Enzimi"],
        "gold_bloom": "Anlama"
    },
    {
        "id": "BENCH-CS",
        "domain": "Bilgisayar Mühendisliği",
        "text": "İkili arama ağacında (BST) arama karmaşıklığı ortalama durumda O(log n)'dir. Ağacın dengeli kalması için AVL veya Kırmızı-Siyah ağaç rotasyonları uygulanır.",
        "gold_terms": ["İkili Arama Ağacı", "Logaritmik Karmaşıklık", "Dengeli Ağaç"],
        "gold_bloom": "Uygulama"
    },
    {
        "id": "BENCH-TOUR",
        "domain": "Turizm & Seyahat (Tatilsepeti)",
        "text": "Dinamik paketleme algoritması uçak bileti ve otel kontenjanını eşzamanlı birleştirir. No-show riskine karşı erken rezervasyon iptal sigortası devreye alınır.",
        "gold_terms": ["Dinamik Paketleme", "No-Show Riski", "Erken Rezervasyon Sigortası"],
        "gold_bloom": "Uygulama"
    },
    {
        "id": "BENCH-ISG",
        "domain": "İş Sağlığı ve Güvenliği",
        "text": "Ramak kala olayları proaktif risk analizi için kayıt altına alınmalıdır. İş güvenliği uzmanı acil durum eylem planını periyodik tatbikatlarla denetler.",
        "gold_terms": ["Ramak Kala Olayı", "Risk Analizi", "Acil Durum Eylem Planı"],
        "gold_bloom": "Değerlendirme"
    },
    {
        "id": "BENCH-KVKK",
        "domain": "Veri Koruma Hukuku & KVKK",
        "text": "Veri sorumlusu, açık rıza metninde aydınlatma yükümlülüğünü eksiksiz yerine getirmelidir. Özel nitelikli kişisel veriler açık rıza olmaksızın işlenemez.",
        "gold_terms": ["Veri Sorumlusu", "Aydınlatma Yükümlülüğü", "Açık Rıza", "Özel Nitelikli Kişisel Veri"],
        "gold_bloom": "Anlama"
    },
    {
        "id": "BENCH-CYBER",
        "domain": "Siber Güvenlik",
        "text": "Sıfırıncı gün açığı (zero-day exploit) yama yayınlanana kadar tehdit oluşturur. Saldırganların yetki yükseltme denemeleri SIEM logları üzerinden izlenir.",
        "gold_terms": ["Sıfırıncı Gün Açığı", "Yetki Yükseltme", "SIEM Analizi"],
        "gold_bloom": "Analiz"
    },
    {
        "id": "BENCH-FIN",
        "domain": "Bankacılık & Finans",
        "text": "Basel III standartları bankaların sermaye yeterlilik oranını ve likidite karşılama oranını (LKO) sıkı finansal stres testleriyle denetler.",
        "gold_terms": ["Basel III", "Sermaye Yeterlilik Oranı", "Likidite Karşılama Oranı", "Stres Testi"],
        "gold_bloom": "Değerlendirme"
    }
]

def calculate_atesman(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    vowels = set("aeıioöuüAEIİOÖUÜ")
    words = re.findall(r"\b[\w'-]+\b", text)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    total_words = len(words)
    total_sentences = max(1, len(sentences))
    if total_words == 0:
        return 0.0
    total_syllables = sum(1 for char in text if char in vowels)
    avg_syllables_per_word = total_syllables / total_words
    avg_words_per_sentence = total_words / total_sentences
    score = 198.825 - (40.175 * avg_syllables_per_word) - (2.610 * avg_words_per_sentence)
    return round(score, 2)

def main():
    model_path = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
    adapter_path = "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora"
    out_file = "/users/ali.gunder/eduterm-finetune/outputs/eduterm_14b_cefr_sft_report.json"

    logger.info("Initializing Evaluation for EduTerm-14B-CEFR Checkpoint...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    logger.info("Loading Base Model in 4-bit NF4...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True
    )

    logger.info("Loading LoRA adapter from %s...", adapter_path)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    results = []
    level_compliance = {"B1": 0, "B2": 0, "C1": 0}

    logger.info("=" * 75)
    logger.info("RUNNING BENCHMARK EVALUATION (8 UNSEEN DISCIPLINES)")
    logger.info("=" * 75)

    for bench in BENCHMARKS:
        instruction = (
            "Verilen eğitim metnini analiz et. Teknik terimleri çıkar, Bloom taksonomisi düzeyini belirle. "
            "Her terim için üç farklı hedef kitleye uygun açıklama üret:\n"
            "- 'b1_aciklama': Lisans 1 için günlük benzetimlerle sade ve anlaşılır (Ateşman >= 50)\n"
            "- 'b2_aciklama': Lisans 3-4 için teknik ve uygulamalı (Ateşman 35-50)\n"
            "- 'c1_aciklama': Yüksek lisans için akademik ve kuramsal (Ateşman < 35)\n\n"
            "Çıktıyı kesinlikle şu JSON şemasında ver:\n"
            "{\n"
            '  "bloom_seviyesi": "<Hatırlama|Anlama|Uygulama|Analiz|Değerlendirme|Yaratma>",\n'
            '  "terimler": [\n'
            '    {\n'
            '      "terim": "<terim adı>",\n'
            '      "b1_aciklama": "<sade açıklama>",\n'
            '      "b2_aciklama": "<teknik açıklama>",\n'
            '      "c1_aciklama": "<akademik açıklama>"\n'
            '    }\n'
            '  ]\n'
            "}"
        )

        user_content = f"<instruction>\n{instruction}\n</instruction>\n\n<context>\n{bench['text']}\n</context>"
        chat_prompt = (
            f"<|im_start|>system\nSen uzman bir pedagojik terminoloji çıkarım ve uyarlanabilir eğitim asistanısın.<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inp = tokenizer(chat_prompt, return_tensors="pt").to(device)
        t0 = time.time()
        with torch.no_grad():
            gen_out = model.generate(
                **inp,
                max_new_tokens=600,
                temperature=0.05,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_time = time.time() - t0
        raw_text = tokenizer.decode(gen_out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # JSON parsing
        json_valid = False
        parsed_terms = []
        bloom_pred = ""
        score_b1 = 0.0
        score_b2 = 0.0
        score_c1 = 0.0

        try:
            m = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                json_valid = True
                bloom_pred = data.get("bloom_seviyesi", "")
                terms_data = data.get("terimler", [])
                for td in terms_data:
                    t_name = td.get("terim", "")
                    if t_name:
                        parsed_terms.append(t_name)
                    # Check CEFR levels
                    b1_text = td.get("b1_aciklama", "")
                    b2_text = td.get("b2_aciklama", "")
                    c1_text = td.get("c1_aciklama", "")
                    if b1_text:
                        s1 = calculate_atesman(b1_text)
                        score_b1 = max(score_b1, s1)
                        if s1 >= 50.0:
                            level_compliance["B1"] += 1
                    if b2_text:
                        s2 = calculate_atesman(b2_text)
                        score_b2 = max(score_b2, s2)
                        if 35.0 <= s2 < 50.0:
                            level_compliance["B2"] += 1
                    if c1_text:
                        s3 = calculate_atesman(c1_text)
                        score_c1 = max(score_c1, s3)
                        if s3 < 35.0:
                            level_compliance["C1"] += 1
        except Exception as e:
            logger.warning("JSON parse error for %s: %s", bench["id"], e)

        # Evaluate extraction metrics
        gold = [g.lower() for g in bench["gold_terms"]]
        extr_lower = [p.lower() for p in parsed_terms]
        matched = sum(1 for g in gold if any(g in p or p in g for p in extr_lower))
        recall = matched / len(gold) if gold else 0.0
        precision = (sum(1 for p in extr_lower if any(p in g or g in p for g in gold)) / len(parsed_terms)) if parsed_terms else 0.0
        f1 = (2 * recall * precision) / max(1e-6, (recall + precision))
        bloom_match = (bloom_pred.lower() == bench["gold_bloom"].lower()) if bloom_pred else False

        logger.info("  [%s] %s: Recall %.0f%% | Prec %.0f%% | F1 %.0f%% | Bloom: %s (Match: %s) | Ateşman: B1=%.1f B2=%.1f C1=%.1f",
                    bench["id"], bench["domain"], recall * 100, precision * 100, f1 * 100, bloom_pred, bloom_match, score_b1, score_b2, score_c1)

        results.append({
            "id": bench["id"],
            "domain": bench["domain"],
            "json_valid": json_valid,
            "terms_extracted": parsed_terms,
            "gold_terms": bench["gold_terms"],
            "recall": round(recall, 3),
            "precision": round(precision, 3),
            "f1": round(f1, 3),
            "bloom_match": bloom_match,
            "bloom_pred": bloom_pred,
            "gold_bloom": bench["gold_bloom"],
            "atesman_b1": round(score_b1, 1),
            "atesman_b2": round(score_b2, 1),
            "atesman_c1": round(score_c1, 1),
            "latency_s": round(gen_time, 2)
        })

    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_f1 = (2 * avg_recall * avg_precision) / max(1e-6, (avg_recall + avg_precision))
    bloom_acc = (sum(1 for r in results if r["bloom_match"]) / len(results)) * 100
    json_valid_pct = (sum(1 for r in results if r["json_valid"]) / len(results)) * 100
    cefr_ctrl_pct = ((level_compliance["B1"] + level_compliance["B2"] + level_compliance["C1"]) / (max(1, len(results) * 3))) * 100

    report = {
        "model_name": "EduTerm-14B-CEFR-Controllable (QLoRA SFT)",
        "base_model": model_path,
        "adapter_path": adapter_path,
        "epochs_trained": "2.5 (Checkpoint Epoch 2)",
        "hardware": {
            "gpu": "NVIDIA A40",
            "device": device
        },
        "evaluation_summary": {
            "avg_recall_pct": round(avg_recall * 100, 1),
            "avg_precision_pct": round(avg_precision * 100, 1),
            "f1_score_pct": round(avg_f1 * 100, 1),
            "bloom_accuracy_pct": round(bloom_acc, 1),
            "json_compliance_pct": round(json_valid_pct, 1),
            "cefr_complexity_control_pct": round(cefr_ctrl_pct, 1),
            "level_compliance": level_compliance
        },
        "benchmarks": results
    }

    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("=" * 75)
    logger.info("EVALUATION FINISHED SUCCESSFULLY!")
    logger.info("Recall: %.1f%% | Precision: %.1f%% | F1: %.1f%% | Bloom Acc: %.1f%% | CEFR Control: %.1f%% | JSON: %.1f%%",
                avg_recall * 100, avg_precision * 100, avg_f1 * 100, bloom_acc, cefr_ctrl_pct, json_valid_pct)
    logger.info("Saved final report to %s", out_file)

if __name__ == "__main__":
    main()

