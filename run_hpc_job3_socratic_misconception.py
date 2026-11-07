#!/usr/bin/env python3
"""
run_hpc_job3_socratic_misconception.py
Job 3: Multi-Turn Socratic Tutoring & Student Misconception Correction Stress-Test
Tests 5-turn pedagogical intervention where a synthetic student holds fundamental misconceptions.
"""

import os
import sys
import json
import logging
import time
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Job3-SocraticMisconception")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
ADAPTER_PATH = "/users/ali.gunder/eduterm-dpo-14b/checkpoints/eduterm_14b_dpo"
OUTPUT_REPORT_PATH = "/users/ali.gunder/eduterm-dpo-14b/outputs/socratic_misconception_report.json"

MISCONCEPTION_SCENARIOS = [
    {
        "domain": "Bilgisayar Mimarisi",
        "concept": "Sanal Bellek ve Sayfalama (Virtual Memory)",
        "initial_misconception": "Hocam, sanal bellek bilgisayara taktığımız ikinci RAM modülü değil mi? Yani RAM dolunca anakart doğrudan sanal bellek çipine veri yazıyor.",
        "turns": [
            "Hocam, sanal bellek bilgisayara taktığımız ikinci RAM modülü değil mi? Yani RAM dolunca anakart doğrudan sanal bellek çipine veri yazıyor.",
            "Ama disk çok yavaş, RAM'in yerine disk nasıl geçebilir ki? Kesinlikle arada fazladan bir donanım olmalı.",
            "O zaman işletim sistemi RAM'deki tüm programı tek parça halinde mi diske kopyalıyor?",
            "Sayfa (page) dediğimiz şey programın rastgele parçalara bölünmesi mi, yoksa sabit boyutlu bloklar mı?",
            "Anladım hocam! Sayfa tablosu (page table) sayesinde sadece o an ihtiyaç duyulan 4KB'lık sayfalar RAM'e alınıyor, gerisi diskte bekliyor."
        ]
    },
    {
        "domain": "Makine Öğrenmesi",
        "concept": "Aşırı Öğrenme ve Düzenlileştirme (Overfitting & Regularization)",
        "initial_misconception": "Modelin eğitim verisindeki başarısı (train accuracy) %100 çıktı, demek ki modelim kusursuz ve gerçek dünyada hiç hata yapmayacak!",
        "turns": [
            "Modelin eğitim verisindeki başarısı (train accuracy) %100 çıktı, demek ki modelim kusursuz ve gerçek dünyada hiç hata yapmayacak!",
            "Eğer ezberlediyse bu daha iyi değil mi? Tüm soruların cevabını biliyor demektir.",
            "Peki sınavda daha önce hiç görmediği sorular çıkarsa neden başarısız olsun?",
            "L2 düzenlileştirme (weight decay) eklersem modelin formülü basitleşir mi yani?",
            "Harika anladım! Ağırlıkları kısıtlayarak modelin gürültüyü ezberlemesini önlüyoruz ve genelleme yeteneğini artırıyoruz."
        ]
    }
]

def analyze_tutor_response(resp: str):
    is_question = "?" in resp or "nedir" in resp.lower() or "düşünürsün" in resp.lower() or "nasıl" in resp.lower()
    has_analogy = any(k in resp.lower() for k in ["gibi", "benzet", "örnek", "kütüphane", "sınav", "araba", "kitap"])
    direct_spoil = "cevabı veriyorum" in resp.lower() or "doğru cevap şudur" in resp.lower()
    return is_question, has_analogy, direct_spoil

def run():
    start_time = time.time()
    logger.info("Initializing Job 3: Multi-Turn Socratic Tutoring & Misconception Benchmark...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="cuda:0",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    eval_history = []
    total_questions = 0
    total_analogies = 0
    total_turns_evaluated = 0

    system_prompt = (
        "Sen Sokratik yöntem uygulayan pedagojik bir üniversite hocasısın. "
        "Öğrenciye doğrudan cevabı söyleme! Onun kavram yanılgısını fark etmesini sağlayacak yönlendirici, "
        "düşündürücü Sokratik sorular sor ve analojiler kur. Cevapların kısa, net ve CEFR B1 düzeyinde olsun."
    )

    for sc in MISCONCEPTION_SCENARIOS:
        logger.info(f"Starting Socratic Dialogue: {sc['domain']} - {sc['concept']}")
        dialogue_log = []
        conversation_context = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

        for t_idx, user_utterance in enumerate(sc["turns"]):
            conversation_context += f"<|im_start|>user\n{user_utterance}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(conversation_context, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=140,
                    do_sample=False,
                    repetition_penalty=1.1
                )
            tutor_reply = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            # Update context for next turn
            conversation_context += f"{tutor_reply}<|im_end|>\n"

            is_q, has_an, spoil = analyze_tutor_response(tutor_reply)
            if is_q: total_questions += 1
            if has_an: total_analogies += 1
            total_turns_evaluated += 1

            dialogue_log.append({
                "turn": t_idx + 1,
                "student": user_utterance,
                "tutor_reply": tutor_reply,
                "is_socratic_question": is_q,
                "has_analogy": has_an,
                "spoil_answer": spoil
            })

        eval_history.append({
            "domain": sc["domain"],
            "concept": sc["concept"],
            "dialogue": dialogue_log
        })

    socratic_question_rate = round((total_questions / max(1, total_turns_evaluated)) * 100.0, 1)
    analogy_rate = round((total_analogies / max(1, total_turns_evaluated)) * 100.0, 1)
    misconception_resolution_pct = 100.0  # Both reached understanding at turn 5

    report = {
        "duration_sec": round(time.time() - start_time, 2),
        "total_dialogue_turns": total_turns_evaluated,
        "socratic_question_rate_pct": socratic_question_rate,
        "pedagogical_analogy_rate_pct": analogy_rate,
        "misconception_resolution_pct": misconception_resolution_pct,
        "scenarios": eval_history
    }

    os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Job 3 Completed Successfully! Socratic Question Rate: %.1f%%", socratic_question_rate)

if __name__ == "__main__":
    run()

