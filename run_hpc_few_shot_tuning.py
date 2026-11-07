#!/usr/bin/env python3
"""
HPC Few-Shot Dynamic In-Context Exemplar Tuning Benchmark
Evaluates EduTerm-14B and Kumru-2B across 0-shot, 1-shot, 3-shot, and 5-shot
in-context exemplars for CEFR-B1 pedagogical simplification and terminology mining.
Measures Ateşman Readability, F1 Gain, Token Budget Saturation, and Prompt Overhead.
"""

import os
import sys
import json
import logging
import time
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_CONFIG = [
    {
        "id": "Kumru-2B",
        "family": "kumru",
        "base_path": "/users/ali.gunder/models/Kumru-2B",
        "adapter_path": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_kumru_2b_cefr",
        "load_kwargs": {"torch_dtype": torch.float16, "device_map": "auto"}
    },
    {
        "id": "EduTerm-14B",
        "family": "qwen",
        "base_path": "/users/ali.gunder/models/Qwen2.5-14B-Instruct",
        "adapter_path": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora",
        "load_kwargs": {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    }
]

EXEMPLAR_BANK = [
    {
        "text": "Yapay zeka modellerinde aşırı öğrenme (overfitting) eğitim verisine aşırı uyum sağlayıp yeni verilerde başarısız olma durumudur. Düzenlileştirme (regularization) bunu engeller.",
        "output": "aşırı öğrenme: Modelin ezberlemesi ve genelleme yapamaması durumu.\ndüzenlileştirme: Ezberlemeyi önleyen matematiksel denetim tekniği."
    },
    {
        "text": "Veritabanlarında ACID prensiplerinden atomiklik bir işlemin ya tamamen yapılmasını ya da hiç yapılmamasını ifade eder.",
        "output": "atomiklik: İşlemin bölünemez tek bir parça olarak tamamlanması kuralı."
    },
    {
        "text": "Ekonomide fırsat maliyeti bir seçeneği tercih ettiğimizde vazgeçtiğimiz en iyi alternatifin değeridir.",
        "output": "fırsat maliyeti: Seçilmeyen diğer en iyi kararın vazgeçilen kazancı."
    },
    {
        "text": "Kuantum süperpozisyonu bir parçacığın aynı anda birden fazla kuantum durumunda bulunabilmesidir.",
        "output": "kuantum süperpozisyonu: Bir sistemin aynı anda birden çok olasılık durumunu barındırması."
    },
    {
        "text": "Biyolojide osmoz su moleküllerinin yarı geçirgen bir zardan derişimin az olduğu ortamdan çok olduğu ortama kendiliğinden geçişidir.",
        "output": "osmoz: Suyun hücre zarından otomatik dengelenme geçişi."
    }
]

TEST_CASES = [
    {
        "text": "Bilgisayar mimarisinde boru hattı (pipelining) tekniği buyrukların aşamalarını çakıştırarak mikroişlemci verimini artırır.",
        "gold_terms": ["boru hattı", "pipelining", "mikroişlemci verimi"]
    },
    {
        "text": "Makine öğrenmesinde k-en yakın komşu algoritması denetimli bir sınıflandırma yöntemidir.",
        "gold_terms": ["k-en yakın komşu algoritması", "denetimli sınıflandırma"]
    }
]

def calculate_atesman(text):
    words = re.findall(r'\b\w+\b', text)
    if not words:
        return 50.0
    num_words = len(words)
    num_sentences = max(1, len(re.split(r'[.!?]+', text)) - 1)
    # Turkish syllable approx
    vowels = "aeıioöuüAEIİOÖUÜ"
    num_syllables = sum(sum(1 for char in word if char in vowels) for word in words)

    asl = num_words / num_sentences
    asw = num_syllables / num_words
    score = 198.825 - (40.175 * asl) - (2.610 * asw)
    return max(0.0, min(100.0, score))

def build_prompt(model_family, k_shots, test_text):
    shots_text = ""
    for i in range(k_shots):
        ex = EXEMPLAR_BANK[i]
        shots_text += f"\nÖrnek Metin: {ex['text']}\nBasit Açıklama:\n{ex['output']}\n"

    if model_family == "kumru":
        return (
            f"<|im_start|>user\nAşağıdaki ders metnindeki teknik terimleri CEFR-B1 düzeyinde sade bir dille açıkla:{shots_text}\n"
            f"Metin: {test_text}\nBasit Açıklama:<|im_end|>\n<|im_start|>assistant\n"
        )
    else:
        return (
            f"<|im_start|>system\nSen uzman bir pedagojik öğretmensin. Metindeki teknik terimleri başlangıç seviyesine (CEFR-B1) uygun, anlaşılır şekilde açıkla.<|im_end|>\n"
            f"<|im_start|>user\n{shots_text}\nMetin: {test_text}\nBasit Açıklama:<|im_end|>\n<|im_start|>assistant\n"
        )

def benchmark_few_shot(cfg):
    logging.info(f"Loading {cfg['id']} for Dynamic Few-Shot ICL Benchmark...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg["base_path"], trust_remote_code=True, **cfg["load_kwargs"])
    if os.path.exists(cfg["adapter_path"]):
        logging.info(f"  Loading LoRA adapter: {cfg['adapter_path']}")
        model = PeftModel.from_pretrained(model, cfg["adapter_path"])
    model.eval()

    shot_levels = [0, 1, 3, 5]
    results = {}

    for k in shot_levels:
        prompt_lens = []
        atesman_scores = []
        latencies = []
        hit_counts = 0
        total_golds = 0

        for tc in TEST_CASES:
            prompt = build_prompt(cfg["family"], k, tc["text"])
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            input_ids = inputs["input_ids"]
            prompt_lens.append(input_ids.shape[1])

            gen_kwargs = {
                "max_new_tokens": 80,
                "do_sample": False,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id
            }
            if "attention_mask" in inputs:
                gen_kwargs["attention_mask"] = inputs["attention_mask"]

            t0 = time.time()
            with torch.no_grad():
                out = model.generate(input_ids, **gen_kwargs)
            latencies.append(time.time() - t0)

            resp = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
            atesman_scores.append(calculate_atesman(resp))

            # Check term coverage
            resp_lower = resp.lower()
            for g in tc["gold_terms"]:
                total_golds += 1
                if g.lower() in resp_lower:
                    hit_counts += 1

        avg_prompt_tokens = sum(prompt_lens) / len(prompt_lens)
        avg_atesman = sum(atesman_scores) / len(atesman_scores)
        avg_latency = sum(latencies) / len(latencies)
        term_recall_pct = (hit_counts / max(1, total_golds)) * 100.0

        results[f"{k}-shot"] = {
            "k_shots": k,
            "avg_prompt_tokens": round(avg_prompt_tokens, 1),
            "atesman_readability_score": round(avg_atesman, 1),
            "cefr_b1_adherence": "Compliant (B1)" if avg_atesman >= 50.0 else "Non-compliant",
            "term_coverage_recall_pct": round(term_recall_pct, 1),
            "latency_sec": round(avg_latency, 2)
        }
        logging.info(f"[{cfg['id']}] {k}-shot -> Prompt: {avg_prompt_tokens:.0f} tok, Ateşman: {avg_atesman:.1f}, Recall: {term_recall_pct:.1f}%")

    del model
    del tokenizer
    torch.cuda.empty_cache()
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/few_shot_icl_tuning_report.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    all_reports = {}

    for cfg in MODELS_CONFIG:
        all_reports[cfg["id"]] = benchmark_few_shot(cfg)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)

    logging.info(f"Few-shot ICL benchmark complete. Saved to {args.output}")

if __name__ == "__main__":
    main()

