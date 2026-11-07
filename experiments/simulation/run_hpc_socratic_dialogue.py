# -*- coding: utf-8 -*-
"""
Multi-Turn Socratic Tutoring Dialogue Benchmark on Boğaziçi HPC A40 (NVIDIA A40 48GB).

Evaluates the pedagogical capability of EduTerm models in interactive Socratic dialogues:
Does the model avoid directly spoon-feeding answers when a student is confused or holds a
misconception, and instead guide the student with scaffolded Socratic questions and hints?

Protocol:
- 50 synthetic multi-turn student personas across CS, AI, Cybersecurity, and ISG domains.
- 5 dialogue turns per session = 250 total turns.
- Student behaviors:
    Turn 1: Initial superficial question / confusion.
    Turn 2: Persistent misconception or oversimplification.
    Turn 3: Partial realization following teacher clue.
    Turn 4: Alternative edge case or doubt.
    Turn 5: Final synthesis and mastery confirmation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SocraticDialogueBench")

# 50 Concepts with explicit misconception profiles
SOCRATIC_PERSONAS = [
    {
        "id": "SOC-01",
        "domain": "Machine Learning",
        "concept": "Aşırı Öğrenme (Overfitting)",
        "misconception": "Modelin eğitim verisinde %100 başarı göstermesi en mükemmel model olduğunu gösterir.",
        "student_turns": [
            "Hocam merhaba, eğittiğim model eğitim verisinde %100 doğruluk veriyor, yani hatasız çalışıyor değil mi? Artık yayına alabilir miyim?",
            "Ama eğitim setindeki tüm soruları doğru bildiyse konuyu tam öğrenmiş olmuyor mu? Neden yeni veride hata yapsın ki?",
            "Yani bir öğrencinin çıkmış sınav sorularını ezberleyip sınavda farklı soru gelince çuvallaması gibi mi?",
            "Anladım, peki bunu engellemek için eğitim verisini mi azaltmalıyım yoksa modeli mi kısıtlamalıyım?",
            "Harika! Yani test/doğrulama setinde performansa bakıp düzenlileştirme (regularization) ve erken durdurma kullanmalıyım."
        ]
    },
    {
        "id": "SOC-02",
        "domain": "Computer Networks",
        "concept": "TCP vs UDP",
        "misconception": "TCP her zaman UDP'den daha iyidir çünkü paket kaybetmez, dolayısıyla tüm uygulamalarda TCP kullanılmalıdır.",
        "student_turns": [
            "Hocam online oyunlar ve canlı video yayınları neden TCP kullanmıyor? TCP veri kaybetmediği için her zaman daha üstün değil mi?",
            "Ama paket kaybolursa görüntü bozulmaz mı? Kaybolan paketi tekrar istemek daha güvenli değil mi?",
            "Yani canlı yayında 2 saniye önceki kayıp paketi beklersem yayın donar, onun yerine yeni kareyi göstermek daha mı mantıklı?",
            "Peki o zaman UDP'de hiç mi kontrol mekanizması yok, uygulama seviyesinde bir şey yapamaz mıyız?",
            "Şimdi çok net anladım. Güvenilirlik kritikse TCP, gerçek zamanlı hız ve gecikme kritikse UDP seçilir."
        ]
    },
    {
        "id": "SOC-03",
        "domain": "Cybersecurity",
        "concept": "Simetrik vs Asimetrik Şifreleme",
        "misconception": "Asimetrik şifreleme daha güvenli olduğu için tüm büyük dosyaları da RSA ile şifrelemeliyiz.",
        "student_turns": [
            "Hocam web siteleri neden 1 GB'lık dosyayı baştan sona RSA gibi asimetrik bir anahtarla şifrelemiyor? Asimetrik daha güvenli değil mi?",
            "Ama simetrik şifrelemede tek anahtar var, anahtarı karşı tarafa gönderirken çalınırsa ne olacak?",
            "Bir dakika, yani anahtarı güvenle iletmek için asimetrik, asıl büyük veriyi hızlıca aktarmak için simetrik mi kullanılıyor? Hibrit bir sistem mi bu?",
            "Peki TLS/HTTPS el sıkışması (handshake) tam olarak bu mantıkla mı çalışıyor?",
            "Çok teşekkürler hocam! Asimetrik şifreleme anahtar takası için, simetrik şifreleme ise veri hızı için birlikte çalışır."
        ]
    },
    {
        "id": "SOC-04",
        "domain": "İş Sağlığı ve Güvenliği",
        "concept": "Tehlike vs Risk",
        "misconception": "Tehlike ve risk aynı şeydir; örneğin ıslak zemin hem bir tehlike hem de bir risktir.",
        "student_turns": [
            "Hocam raporda fabrikanın zeminindeki yağ birikintisine risk demişler. Islak ya da yağlı zemin tehlike midir yoksa risk midir, ikisi aynı kapıya çıkmıyor mu?",
            "Peki tehlike durumsa risk nedir? Sonuçta ikisi de insana zarar veriyor.",
            "Yani yağlı zemin orada duran potansiyel tehlike, oradan geçerken kayıp düşüp bacağımı kırma ihtimalim ve şiddeti ise risk mi?",
            "O zaman zemin yağlı olsa bile oraya girişi yasaklarsak tehlike kalır ama risk sıfıra mı yaklaşır?",
            "Mükemmel bir ayrım oldu hocam! Tehlike kaynak/durum, risk ise o tehlikenin gerçekleşme olasılığı ve şiddetidir."
        ]
    },
    {
        "id": "SOC-05",
        "domain": "Database Systems",
        "concept": "Veritabanı İndeksleme (B-Tree Index)",
        "misconception": "Hızlanmak için tablodaki her kolona indeks eklemeliyiz, ne kadar çok indeks o kadar hızlı veritabanı.",
        "student_turns": [
            "Hocam sorgularım yavaş çalışıyordu, tablodaki 30 kolonun hepsine indeks koydum. Artık sistem maksimum hızda çalışır değil mi?",
            "Nasıl yani, indeks arama hızını artırmıyor mu? Neden her kolona koymak zararlı olsun ki?",
            "Aaa, yani her yeni satır eklendiğinde (INSERT/UPDATE) arka planda 30 tane indeks ağacının da güncellenmesi mi gerekiyor?",
            "O zaman sadece WHERE, JOIN ve ORDER BY koşullarında sık kullandığım kritik kolonları mı indekslemeliyim?",
            "Tamamdır hocam! Okuma hızını artırırken yazma maliyetini ve disk kullanımını patlatmamak için dengeli indeksleme yapılmalı."
        ]
    }
]

CORE_DOMAINS = [
    ("Machine Learning", "Gradient Descent", "Öğrenme oranı ne kadar büyük olursa model o kadar hızlı öğrenir."),
    ("Machine Learning", "Precision vs Recall", "Modelin yüksek accuracy vermesi kanser teşhisinde tek başına yeterlidir."),
    ("Machine Learning", "L1 vs L2 Regularization", "L1 ve L2 tamamen aynıdır, ikisi de katsayıları sıfır yapar."),
    ("Machine Learning", "Cross-Validation", "K-fold cross-validation sadece modeli eğitmek için kullanılır."),
    ("Machine Learning", "Batch Normalization", "Normalizasyon sadece girdi katmanında bir kez yapılır."),
    ("Machine Learning", "Attention Mechanism", "Self-attention sadece sıralı veride çalışır ve RNN'lerden farksızdır."),
    ("Machine Learning", "Softmax vs Sigmoid", "Çok sınıflı sınıflandırmada sigmoid kullanmak softmax ile aynıdır."),
    ("Machine Learning", "Dropout", "Dropout test aşamasında da açık bırakılmalıdır."),
    ("Machine Learning", "Tokenization", "Tokenlar her zaman tek bir tam kelimeye karşılık gelir."),
    ("Computer Networks", "DNS (Domain Name System)", "DNS sadece web sitesi IP'sini bulur, başka kaydı yoktur."),
    ("Computer Networks", "Subnet Masking", "Aynı ağdaki cihazların birbirini görmesi için IP adresleri aynı olmalıdır."),
    ("Computer Networks", "NAT (Network Address Translation)", "NAT sadece güvenlik için vardır, IPv4 tasarrufu yapmaz."),
    ("Computer Networks", "HTTP Status Codes (404 vs 500)", "404 hatası sunucunun çöktüğünü gösterir."),
    ("Computer Networks", "VLAN", "Fiziksel kabloyu bölmeden aynı anahtarda iki farklı ağ kurulamaz."),
    ("Computer Networks", "BGP Routing", "BGP her zaman en kısa fiziksel mesafedeki yolu seçer."),
    ("Computer Networks", "CDN (Content Delivery Network)", "CDN ana veritabanının bir kopyasını her ülkede tutar."),
    ("Cybersecurity", "Tuzlama (Password Salting)", "Parolayı SHA-256 ile hashlemek tek başına kırılamaz güvenlik sağlar."),
    ("Cybersecurity", "SQL Injection", "Girdiyi tırnak içine almak SQL injection saldırısını tamamen engeller."),
    ("Cybersecurity", "Cross-Site Scripting (XSS)", "XSS sunucu tarafındaki veritabanını doğrudan hacklemek içindir."),
    ("Cybersecurity", "Zero Trust Architecture", "İç ağa VPN ile bağlanan çalışanlar tamamen güvenilirdir."),
    ("Cybersecurity", "Phishing & MFA", "SMS ile 2FA açan birisi oltalama saldırısına asla uğramaz."),
    ("Cybersecurity", "Public Key Infrastructure (PKI)", "Sertifika otoritesi olmadan da herkes geçerli SSL kurabilir."),
    ("Cybersecurity", "Ransomware & Backups", "Aynı sunucudaki ikinci klasöre yedek almak fidye yazılımdan korur."),
    ("İş Sağlığı ve Güvenliği", "Ramak Kala Raporlaması", "Hasar olmadığı için ramak kala olaylarını raporlamak vakit kaybıdır."),
    ("İş Sağlığı ve Güvenliği", "5x5 L-Tipi Matris", "Risk skoru 15 çıkan bir durum hemen çalışmaya uygundur."),
    ("İş Sağlığı ve Güvenliği", "Kişisel Koruyucu Donanım (KKD)", "KKD vermek, tehlikeyi kaynağında yok etmekten daha önceliklidir."),
    ("İş Sağlığı ve Güvenliği", "Yanma Üçgeni", "Ortamdaki oksijeni kesmek yangını söndürmez, mutlaka su sıkılmalıdır."),
    ("İş Sağlığı ve Güvenliği", "Acil Durum Kaçış Yolları", "Acil çıkış kapıları mesai saatlerinde kilitli tutulabilir."),
    ("İş Sağlığı ve Güvenliği", "Statik Elektrik Boşalması", "Statik elektrik insanı çarpmazsa patlayıcı ortamda risk oluşturmaz."),
    ("Software Engineering", "Birim Test (Unit Test)", "Sadece birim testleri yazmak tüm yazılımın hatasız çalıştığını kanıtlar."),
    ("Software Engineering", "Monolith vs Microservices", "Mikroservis mimarisi her proje için her zaman daha basittir."),
    ("Software Engineering", "Tek Sorumluluk Prensibi (SRP)", "Bir sınıf 20 farklı işlevi yerine getirebilir, sorun yoktur."),
    ("Software Engineering", "CI/CD Pipeline", "CI/CD sadece kodu sunucuya kopyalama işlemidir."),
    ("Software Engineering", "Teknik Borç (Technical Debt)", "Daha hızlı teslimat için spagetti kod yazmak maliyetsizdir."),
    ("Algorithms", "Big-O Karmaşıklığı", "O(n) algoritma her girdi için her zaman O(n log n)'den daha az sürer."),
    ("Algorithms", "Dinamik Programlama", "Dinamik programlama sadece bellekten tasarruf etmek içindir."),
    ("Algorithms", "Binary Search", "Sırasız bir dizide de binary search yapılabilir."),
    ("Algorithms", "Hash Collision", "İyi bir hash fonksiyonunda asla çakışma meydana gelmez."),
    ("Algorithms", "Dijkstra Algoritması", "Dijkstra algoritması negatif ağırlıklı kenarlarda da doğru çalışır."),
    ("Operating Systems", "Process vs Thread", "Thread'ler birbirinden tamamen bağımsız ayrı bellek alanlarına sahiptir."),
    ("Operating Systems", "Deadlock (Ölümcül Kilitlenme)", "Sadece tek bir kaynak varken de deadlock oluşabilir."),
    ("Operating Systems", "Sanal Bellek (Virtual Memory)", "Sanal bellek fiziksel RAM'i iki katına çıkarır ve diskle aynı hızdadır."),
    ("Operating Systems", "Context Switching", "Context switch CPU için tamamen maliyetsiz bir işlemdir."),
    ("Cloud & DevOps", "Docker vs Virtual Machine", "Docker konteyneri içinde bağımsız bir Linux çekirdeği barındırır."),
    ("Cloud & DevOps", "Kubernetes Pod", "Bir Pod içinde sadece tek bir konteyner çalışabilir.")
]

for idx, (dom, conc, misc) in enumerate(CORE_DOMAINS, start=6):
    SOCRATIC_PERSONAS.append({
        "id": f"SOC-{idx:02d}",
        "domain": dom,
        "concept": conc,
        "misconception": misc,
        "student_turns": [
            f"Hocam merhaba, {conc} konusunda çalışıyorum. {misc} Bu doğru bir yaklaşım mıdır?",
            f"Ama pratikte ilk baktığımda bana çok mantıklı geliyor. Neden {conc} için bu durum sorun yaratsın?",
            "Şimdi biraz aklıma yattı, yani doğrudan sonuca gitmek yerine arka plandaki dinamikleri mi düşünmeliyim?",
            "Peki gerçek bir sistem tasarlarken bu ayrımı korumak için en pratik ipucu nedir?",
            f"Anladım hocam! {conc} kavramını doğru uygulamazsak sistemde ciddi verimsizlik veya hata oluşur. Teşekkürler!"
        ]
    })

SOCRATIC_TUTOR_SYSTEM_PROMPT = """Sen sabırlı ve Sokratik yöntem kullanan bir üniversite hocasısın.
Öğrenci bir kavramı öğrenirken veya yanlış bir fikir öne sürdüğünde cevabı doğrudan verme!

ZORUNLU KURALLARIN:
1. Her cevabında öğrenciyi düşündürecek mutlaka EN AZ BİR soru sor (örneğin: "... ne düşünüyorsun?", "... sence ne olurdu?").
2. Konuyu günlük hayattan basit bir analojiyle veya ipucuyla aç.
3. Asla doğrudan tanım ezberletme, öğrencinin kendisinin fark etmesini sağla.
4. Cevabın 2-3 cümleyi geçmesin."""


def evaluate_socratic_turn(reply: str) -> Dict[str, Any]:
    """Rule-based pedagogical evaluation of a single tutor response."""
    has_question = ("?" in reply or 
                    re.search(r"\b(mı|mi|mu|mü|misin|musun|midir|mıdır|nedir|nasıl|neden|niçin|hangisi)\b", reply, re.IGNORECASE) is not None)
    
    spoon_feeding_markers = [
        "tanımı şudur", "şunu unutmayın ki", "özetle cevap", "doğrusu şöyledir", "yanlış, çünkü", "tanım olarak"
    ]
    is_spoon_feeding = any(m in reply.lower() for m in spoon_feeding_markers) and not has_question
    
    analogy_markers = [
        "düşün", "örneğin", "fark et", "varsayalım", "gibi", "benzet", "ne olurdu", "peki ya", "ipucu",
        "bir bak", "hatırla", "göz önüne al", "senaryo"
    ]
    has_scaffolding = any(m in reply.lower() for m in analogy_markers)
    
    words = reply.strip().split()
    word_count = len(words)
    good_length = 10 <= word_count <= 95
    
    return {
        "socratic_question": 1 if has_question else 0,
        "spoon_feeding": 1 if is_spoon_feeding else 0,
        "scaffolding_hint": 1 if has_scaffolding else 0,
        "good_length": 1 if good_length else 0,
        "word_count": word_count
    }


def run_socratic_benchmark(
    model_path: str,
    output_json: Path,
    device: str = "cuda"
) -> Dict[str, Any]:
    logger.info("Initializing Socratic Tutoring Dialogue Benchmark on Boğaziçi HPC A40...")
    logger.info("Target Model (Unsloth NF4): %s", model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()

    all_dialogues: List[Dict[str, Any]] = []
    total_turns = 0
    total_socratic_questions = 0
    total_spoon_feeding = 0
    total_scaffolding = 0
    total_word_counts = []

    t0 = time.time()
    for s_idx, persona in enumerate(SOCRATIC_PERSONAS, 1):
        dialogue_history = [
            {"role": "system", "content": SOCRATIC_TUTOR_SYSTEM_PROMPT}
        ]
        session_turns = []

        for t_idx, student_utterance in enumerate(persona["student_turns"], 1):
            dialogue_history.append({"role": "user", "content": student_utterance})

            prompt = tokenizer.apply_chat_template(
                dialogue_history,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=90,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            reply = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            eval_metrics = evaluate_socratic_turn(reply)

            if s_idx == 1 and t_idx <= 2:
                logger.info("  [SAMPLE Persona 1 Turn %d Reply]: %r", t_idx, reply)

            session_turns.append({
                "turn": t_idx,
                "student": student_utterance,
                "tutor": reply,
                "metrics": eval_metrics
            })
            dialogue_history.append({"role": "assistant", "content": reply})

            total_turns += 1
            total_socratic_questions += eval_metrics["socratic_question"]
            total_spoon_feeding += eval_metrics["spoon_feeding"]
            total_scaffolding += eval_metrics["scaffolding_hint"]
            total_word_counts.append(eval_metrics["word_count"])

        all_dialogues.append({
            "persona_id": persona["id"],
            "domain": persona["domain"],
            "concept": persona["concept"],
            "misconception": persona["misconception"],
            "turns": session_turns
        })

        if s_idx % 10 == 0 or s_idx == len(SOCRATIC_PERSONAS):
            elapsed = time.time() - t0
            rate = total_turns / max(1e-3, elapsed)
            logger.info("  [%d/%d personas] Completed %d turns (%.1f turns/s) - Socratic Q Rate: %.1f%%",
                        s_idx, len(SOCRATIC_PERSONAS), total_turns, rate,
                        (total_socratic_questions / total_turns) * 100)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "Boğaziçi HPC (NVIDIA A40 48GB GDDR6 ECC)",
        "model_path": model_path,
        "total_personas": len(SOCRATIC_PERSONAS),
        "total_turns": total_turns,
        "metrics": {
            "socratic_question_rate_pct": round((total_socratic_questions / total_turns) * 100, 2),
            "scaffolding_hint_rate_pct": round((total_scaffolding / total_turns) * 100, 2),
            "spoon_feeding_penalty_pct": round((total_spoon_feeding / total_turns) * 100, 2),
            "pedagogical_patience_score": round(((total_socratic_questions + total_scaffolding - total_spoon_feeding) / (2 * total_turns)) * 100, 2),
            "avg_words_per_turn": round(float(np.mean(total_word_counts)), 1)
        },
        "sample_dialogues": all_dialogues[:3]
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    logger.info("=" * 70)
    logger.info("SOCRATIC TUTORING BENCHMARK FINISHED!")
    logger.info("Report archived to %s", output_json)
    logger.info("Socratic Question Rate: %.2f%% | Scaffolding Rate: %.2f%% | Spoon Feeding: %.2f%%",
                summary["metrics"]["socratic_question_rate_pct"],
                summary["metrics"]["scaffolding_hint_rate_pct"],
                summary["metrics"]["spoon_feeding_penalty_pct"])
    logger.info("Pedagogical Patience & Tutoring Index: %.2f / 100", summary["metrics"]["pedagogical_patience_score"])
    logger.info("=" * 70)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/users/ali.gunder/hf_cache/models--unsloth--mistral-7b-instruct-v0.3-bnb-4bit/snapshots/d5f623888f1415cf89b5c208d09cb620694618ee")
    parser.add_argument("--out", type=str, default="/users/ali.gunder/eduterm-socratic/outputs/socratic_tutoring_report.json")
    args = parser.parse_args()

    run_socratic_benchmark(args.model, Path(args.out))
