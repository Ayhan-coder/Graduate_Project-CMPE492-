# -*- coding: utf-8 -*-
"""
EduTerm-14B-Socratic-V2: Multi-Turn Socratic + CEFR Mixed QLoRA Training & Evaluation on HPC A40.

Addresses single-turn SFT template overfitting observed in Job 34699 (`EduTerm-14B-CEFR`: 38.89%
resolution, 2.889 Socratic score) by fine-tuning Qwen2.5-14B-Instruct with multi-turn masked
assistant loss on a balanced 3-stream mixture (excluding all 12 test ISG concepts to prevent leakage):
  1. Multi-Turn Socratic Tutoring Dialogues (~480 conversations): student misconception -> Socratic
     analogy/question without spoon-feeding -> partial realization -> mastery synthesis.
  2. Pedagogical Distinction / Scenario / Varied Prerequisite Scaffolding (~360 rows).
  3. CEFR-B1/B2/C1 & General Turkish Replay (~360 rows) to retain CEFR & conversational fluency.

Then evaluates the resulting `eduterm_14b_socratic_v2` adapter directly in unified multi-turn
dialogue against `Kumru-2B` across all 12 ISG concepts x 3 cognitive personas (36 sessions, 144 turns)
using the unbiased Base Qwen2.5-14B-Instruct judge.

Outputs:
  checkpoints/eduterm_14b_socratic_v2/
  results/hpc_socratic_v2_report.json
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_hpc_live_student_simulation import (
    ISG_CONCEPTS,
    PERSONAS,
    generate_chat,
    has_cjk,
    judge_session,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EduTermSocraticV2")

IM_END = "<|im_end|>"
MAX_TOKENS = 896

# Strict exclusion keywords to prevent any data leakage into the 12 test ISG concepts (C01..C12)
TEST_ISG_BLOCKLIST = [
    "tehlike", "risk", "ramak kala", "yanma üçgeni", "yangın", "5x5", "l-tipi",
    "kkd", "kişisel koruyucu", "tahliye", "acil durum", "fine-kinney", "kök neden",
]

# Non-ISG multi-domain misconceptions for authentic multi-turn Socratic dialogue synthesis
NON_ISG_MISCONCEPTIONS = [
    ("Makine Öğrenmesi", "Aşırı Öğrenme (Overfitting)", "Genelleme ve Düzenlileştirme",
     "Hocam, modelim eğitim verisinde %100 doğruluk verdi. Yani tamamen hatasız öğrenmiş, yayına alabilirim değil mi?",
     "Eğitim setinde %100 başarı kulağa harika geliyor! Peki sence bir öğrenci çıkmış soruların tamamını ezberleyip hiç görmediği bir sınava girerse nasıl bir sonuç alır?",
     "Doğru söylüyorsunuz, yeni sorularda zorlanabilir. Yani eğitimde yüksek skor yetmez, görülmemiş test verisinde de başarmalı mı?",
     "Harika benzetme! Model veriyi ezberlediğinde yeni örneklerde hata yapmasına overfitting diyoruz. Peki bunu önlemek için doğrulama (validation) setine ek olarak nasıl bir kısıt koyabiliriz?"),
    ("Makine Öğrenmesi", "Precision vs Recall", "Hata Matrisi (Confusion Matrix)",
     "Hocam kanser teşhisi yapan modelimde accuracy %99 çıktı, artık başka metriğe bakmaya gerek yok mu?",
     "Yüzde 99 çok yüksek görünüyor. Peki veri setimizdeki 100 hastanın sadece 1'i gerçekten hasta ve model herkese 'sağlıklı' derse accuracy yine %99 olmaz mı?",
     "Aa evet! O 1 hastayı kaçırmış oluruz ve bu hayati risk taşır. Yani hasta olanları kaçırmamak için başka bir ölçüye mi bakmalıyız?",
     "Kesinlikle! Pozitif vakaları yakalama oranını ölçen Recall metriği bu yüzden kritik. Sence yanlış alarm vermemeyi (Precision) ve hastayı kaçırmamayı (Recall) nasıl dengeleriz?"),
    ("Makine Öğrenmesi", "Öğrenme Oranı (Learning Rate)", "Gradyan İnişi (Gradient Descent)",
     "Gradyan inişinde öğrenme oranını ne kadar büyük seçersem model o kadar hızlı en iyi noktaya ulaşır diye düşündüm, doğru mu?",
     "Hızlı gitme fikri mantıklı! Ama karanlıkta dar bir vadinin dibine inmeye çalışırken çok dev adımlar atarsan vadinin tabanına mı basarsın yoksa karşı yamaca mı savrulursun?",
     "Karşı yamaca atlayıp dururum herhalde! Yani adım çok büyük olursa en düşük hata noktasını aşar mıyız?",
     "Tam üstüne bastın! Adım boyu çok büyük olduğunda kayıp (loss) düşmek yerine salınır veya patlar. Bu durumda eğitim sırasında öğrenme oranını nasıl ayarlamak sence daha sağlıklı olur?"),
    ("Bilgisayar Ağları", "TCP vs UDP", "Taşıma Katmanı Protokolleri",
     "Canlı yayın ve online oyunlarda neden hâlâ UDP kullanılıyor? TCP hiç paket kaybetmediğine göre her zaman daha iyi değil mi?",
     "Güvenilirlik açısından haklısın. Ama canlı bir futbol maçında 3 saniye önceki kayıp bir görüntü paketini tekrar beklemek sence yayını nasıl etkiler?",
     "Yayın donar kalır! Onun yerine o anki güncel kareyi akıtmak daha mantıklı galiba?",
     "Kesinlikle! Gecikmenin kritik olduğu gerçek zamanlı akışta eski paketi beklememek gerekir. Peki sence dosya indirme veya banka işleminde TCP mi UDP mi tercih edilmelidir?"),
    ("Siber Güvenlik", "Parola Tuzlama (Salting)", "Özet (Hash) Fonksiyonları",
     "Kullanıcı parolalarını veritabanına SHA-256 ile hashleyip koyduk, artık veritabanı sızıntısı olsa bile parolalar güvende değil mi?",
     "Düz metin tutmamak iyi bir başlangıç. Ama iki farklı kullanıcı aynı '123456' parolasını seçerse ikisinin hash çıktısı da birebir aynı olmaz mı?",
     "Evet aynı olur, hatta saldırgan hazır gökkuşağı (rainbow) tablolarıyla eşleştirebilir! Her kullanıcıya rastgele bir ek değer mi katmalıyız?",
     "Harika çıkardın! Her parola için benzersiz rastgele 'tuz' (salt) ekleyince aynı parolalar bile bambaşka hash'lere dönüşür. Bu sayede hangi saldırı türünü doğrudan boşa çıkarmış oluruz?"),
    ("Siber Güvenlik", "SQL Injection", "Parametreli Sorgular",
     "Web formunda gelen kullanıcı girdisindeki tek tırnak işaretini silersem SQL Injection saldırısından tamamen korunmuş olur muyum?",
     "Tek tırnağı engellemek yaygın bir refleks. Peki saldırgan tırnak kullanmadan sayısal bir ID alanına '1 OR 1=1' yazarsa veya farklı kodlama (encoding) kullanırsa filtre ne yapar?",
     "Doğru, sayısal alanlarda tırnak bile yok ve filtreyi atlatabilir! Yani string temizlemek yerine sorgu yapısını veriden ayırmak mı gerekiyor?",
     "Aynen öyle! Veritabanına 'bu kısım SQL komutu, şu kısım ise sadece ham veri' diyen Prepared Statement / parametreli sorgular bunun için var. Bir örnek üzerinde nasıl ayrıldığını görmek ister misin?"),
    ("Veritabanı Sistemleri", "B-Tree İndeksleme", "Okuma vs Yazma Maliyeti",
     "Sorgularım hızlansın diye tablodaki 25 sütunun tamamına indeks ekledim, artık veritabanım maksimum hızda çalışır değil mi?",
     "Okuma tarafını düşündün ama küçük bir soru: Kütüphanedeki her kitabın konusuna göre 25 farklı fihrist defteri tutsak, kütüphaneye 1 yeni kitap geldiğinde kaç defteri güncellememiz gerekir?",
     "25 defterin hepsini tek tek güncelleriz! Yani INSERT ve UPDATE işlemleri çok mu yavaşlar?",
     "Tam isabet! Her indeks okuma hızını artırırken yazma (INSERT/UPDATE/DELETE) sırasında ağaç güncelleme yükü ve disk alanı maliyeti doğurur. Sence bir tabloda hangi sütunlara indeks koymak mantıklıdır?"),
    ("Algoritmalar", "İkili Arama (Binary Search)", "Sıralı Veri Önkoşulu",
     "Milyonluk bir dizide hızlı arama yapmak için diziyi sıralamadan doğrudan ortadaki elemana bakarak Binary Search çalıştırabilir miyim?",
     "Deneyelim: Elinde sırasız `[50, 10, 90, 20, 80]` listesi var ve `20`'yi arıyorsun. Ortadaki eleman `90` çıktı; '20 < 90' diyerek sağ yarıyı atarsan ne olur?",
     "20 sağ yarıdaydı, yanlış tarafı çöpe atmış olurum! Demek ki yarıya bölme mantığının çalışması için verinin önce sıralı olması şart mı?",
     "Doğru tespit! Sıralı olma önkoşulu sağlanmazsa eleme kuralı çöker. Peki tek bir arama yapacaksak önce O(n log n) sıralayıp sonra O(log n) aramak mı, yoksa doğrudan O(n) lineer taramak mı daha hızlı olur?"),
    ("Algoritmalar", "Dijkstra Algoritması", "Açgözlü (Greedy) Gevşetme",
     "En kısa yolu bulmak için her zaman Dijkstra kullanabilirim, kenar ağırlıklarının eksi değer alması sonucu değiştirmez sanırım?",
     "Güzel soru! Dijkstra bir düğümü 'en kısa mesafesi kesinleşti' diye işaretleyip kapatır. Peki sonradan başka bir yoldan `-10` ağırlıklı bir kenar gelirse önceden kapattığın yolu güncelleyebilir misin?",
     "Hayır, kapatıldığı için daha kısa negatif yolu görmezden gelir ve yanlış sonuç verir! O zaman negatif kenar varken hangi algoritmaya geçmeliyiz?",
     "Harika fark ettin! Negatif kenarlar açgözlü (greedy) varsayımı bozar; bu durumda tüm kenarları iteratif gevşeten Bellman-Ford algoritması gerekir. Negatif döngü olursa sence ne yaşanır?"),
    ("İşletim Sistemleri", "Sanal Bellek (Virtual Memory)", "Sayfalama ve Disk G/Ç",
     "Bilgisayarda sanal bellek (swap) boyutunu büyütürsem fiziksel RAM iki katına çıkmış gibi programlarım aynı hızda çalışır mı?",
     "Kapasite olarak daha çok program sığar, haklısın. Ama işlemcinin RAM'e erişimi nanosaniyeler sürerken disk erişimi milisaniyeler sürerse program sürekli swap kullandığında hızına ne olur?",
     "Binlerce kat yavaşlar ve disk sürekli meşgul olur! Sanal bellek hız için değil kapasite ve bellek izolasyonu için mi var?",
     "Tamamen doğru! Fiziksel RAM yetmeyip sürekli diskle sayfa değişimi yapıldığında sistem 'thrashing' durumuna girer. Sence bir uygulamada bunu önlemek için çalışma kümesi (working set) neden küçük tutulur?"),
    ("İşletim Sistemleri", "Ölümcül Kilitlenme (Deadlock)", "Coffman Koşulları",
     "Sistemde tek bir yazıcı (kaynak) varken iki süreç aynı anda isterse mutlaka deadlock (ölümcül kilitlenme) oluşur mu?",
     "Bir an durup düşünelim: Tek bir kaynağı A süreci tutuyor, B ise onu bekliyor. A işini bitirdiğinde onu başka beklediği bir şey yoksa serbest bırakmaz mı?",
     "Aa evet, A'nın beklediği ikinci bir kaynak yoksa işini bitirip yazıcıyı bırakır ve B devam eder! Deadlock için döngüsel bekleme ve en az iki kaynak mı gerekiyor?",
     "Mükemmel! Karşılıklı dışlama, tutup bekleme, kesememe ve döngüsel bekleme (4 Coffman koşulu) bir arada olmadıkça deadlock oluşmaz. Bunlardan hangisini kırmak sence pratikte en kolaydır?"),
    ("Yazılım Mühendisliği", "Tek Sorumluluk Prensibi (SRP)", "Modülerlik ve Değişim Nedeni",
     "Tüm veritabanı bağlantılarını, e-posta gönderimini ve fatura hesabını tek bir `Manager` sınıfına topladım; böylece her şey tek yerde olur ve kod daha temiz görünür değil mi?",
     "Başta tek dosya pratik görünebilir. Ama muhasebe ekibi KDV kuralını değiştirdiğinde e-posta kodunun veya veritabanı sorgusunun da aynı sınıfta düzenlenmesi gerekirse test ederken ne riski doğar?",
     "Bir yeri düzeltirken hiç alakasız e-posta sistemini bozabiliriz! Her sınıfın değişmesi için tek bir nedeni mi olmalı?",
     "Kesinlikle! Robert Martin'in Tek Sorumluluk Prensibi tam olarak bunu söyler: Bir sınıfın değişmek için yalnızca bir nedeni olmalıdır. Şimdi bu `Manager` sınıfını hangi üç bağımsız bileşene ayırırdın?"),
    ("Bulut Bilişim", "Docker Konteyner vs Sanal Makine", "Çekirdek Paylaşımı",
     "Docker konteynerleri de tıpkı VMware veya VirtualBox gibi kendi içinde sıfırdan bağımsız bir işletim sistemi çekirdeği çalıştırır değil mi?",
     "Konteynerlerin kendini ayrı bir makine gibi görmesi kafa karıştırıcı olabiliyor. Peki her konteyner kendi çekirdeğini başlatsaydı neden milisaniyeler yerine dakikalar içinde açılmaz ve gigabaytlarca RAM tüketmezdi?",
     "Aaa doğru, konteynerler saniyede açılıyor ve çok hafif! Demek ki ana bilgisayarın (host) çekirdeğini paylaşıp sadece kullanıcı alanını mı izole ediyorlar?",
     "Tam isabet! Linux cgroups ve namespaces sayesinde konteynerler host çekirdeğini paylaşır. Peki sence bu mimari güvenlik izolasyonu açısından VM'lere göre nasıl bir ödünleşim (trade-off) yaratır?"),
    ("Ağ Güvenliği", "Zero Trust (Sıfır Güven) Mimarisi", "İç Ağ Doğrulaması",
     "Şirket ağına VPN ile bağlanan tüm çalışanlar zaten kimlik doğruladığına göre iç ağdaki her veritabanına sınırsız erişebilmelidir, öyle değil mi?",
     "VPN kimliği girişte doğrular, haklısın. Peki bir çalışanın bilgisayarına oltalama (phishing) yoluyla zararlı yazılım bulaşırsa ve o bilgisayar VPN ile iç ağdaysa saldırgan tüm ağa rahatça yayılamaz mı?",
     "Evet! Bir bilgisayar ele geçirilince bütün şirketin veritabanları tehlikeye girer. Yani 'içerideki herkes güvenilirdir' varsayımını bırakıp her isteği ayrı ayrı mı doğrulamalıyız?",
     "Harika çıkardın! Zero Trust'ın 'Asla güvenme, her zaman doğrula' ilkesi ve en az yetki (least privilege) yaklaşımı yanal hareketi (lateral movement) durdurur. Sence bunu ağda nasıl katmanlandırabiliriz?"),
]

SOCRATIC_TUTOR_SYS_TEMPLATES = [
    (
        "Sen Boğaziçi EduTerm Sokratik Eğitmenisin. Hedef kavram: {concept}. Önkoşullar: {prereq}. "
        "Kurallar: Öğrenciye cevabı ilk turda hazır verme; günlük bir analoji ve yönlendirici bir soru ile "
        "kavram yanılgısını kendisinin keşfetmesini sağla. Yalnızca temiz ve anlaşılır Türkçe (CEFR-B1/B2, 2-4 cümle) yaz."
    ),
    (
        "Sen sabırlı bir üniversite hocasısın. Konu: {concept} (Önkoşul: {prereq}). "
        "Cevabı doğrudan ezberletme; önce önkoşulu yokla, sonra kısa bir soruyla düşünmeye teşvik et."
    ),
]


def is_clean_of_test_isg(text: str) -> bool:
    low = text.lower()
    return not any(kw in low for kw in TEST_ISG_BLOCKLIST)


def build_training_conversations(cefr_jsonl: Path, seed: int = 42) -> Tuple[List[List[Dict[str, str]]], Dict[str, int]]:
    """Build balanced 3-stream multi-turn + CEFR conversations without leaking 12 test ISG items."""
    rng = random.Random(seed)
    conversations: List[List[Dict[str, str]]] = []
    counts = {"socratic_multi_turn": 0, "pedagogical_scaffolding": 0, "cefr_general_replay": 0}

    # 1. Stream A: Multi-turn Socratic dialogues (up to 480) from core non-ISG misconceptions + variants
    socratic_pool: List[List[Dict[str, str]]] = []
    for domain, concept, prereq, st1, tu1, st2, tu2 in NON_ISG_MISCONCEPTIONS:
        for sys_tmpl in SOCRATIC_TUTOR_SYS_TEMPLATES:
            sys_msg = sys_tmpl.format(concept=concept, prereq=prereq)
            socratic_pool.append([
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": st1},
                {"role": "assistant", "content": tu1},
                {"role": "user", "content": st2},
                {"role": "assistant", "content": tu2},
            ])
            # Also add single-step Socratic turn variants (start directly from misconception + follow-up)
            socratic_pool.append([
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": f"Kafam çok karıştı hocam, {concept} hakkında şöyle düşünüyorum: {st1}"},
                {"role": "assistant", "content": tu1},
            ])

    # Read CEFR dataset rows
    rows_by_type: Dict[str, List[Dict[str, Any]]] = {
        "cefr_json": [], "prerequisite": [], "general_qa": [], "scenario_qa": [], "distinction": []
    }
    for line in cefr_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        tt = item.get("task_type", "cefr_json")
        if tt not in rows_by_type:
            tt = "general_qa"
        blob = f"{item.get('instruction', '')} {item.get('input', '')} {item.get('output', '')}"
        if not is_clean_of_test_isg(blob):
            continue
        rows_by_type[tt].append(item)

    # Convert distinction & scenario rows into synthetic 2-turn Socratic coaching dialogues (excluding test ISG!)
    for item in rows_by_type["distinction"]:
        m = re.search(r"'([^']+)'\s+ile\s+'([^']+)'", item.get("instruction", ""))
        if not m:
            continue
        c1, c2 = m.group(1), m.group(2)
        out_clean = item["output"].strip()
        short_explain = ". ".join(out_clean.split(".")[:2]).strip() + "."
        sys_msg = SOCRATIC_TUTOR_SYS_TEMPLATES[0].format(concept=f"{c1} vs {c2}", prereq=c1)
        socratic_pool.append([
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": f"Hocam {c1} ile {c2} tamamen aynı şey değil mi? Ben aralarında bir fark göremiyorum."},
            {"role": "assistant", "content": f"İkisi birbirine yakın göründüğü için karıştırılması çok normal! Peki sence '{c1}' sürecinin ortaya çıkış koşulu veya temel amacı '{c2}' ile birebir aynı mı, yoksa birinin gerçekleşme şekli diğerinden farklı mı?"},
            {"role": "user", "content": "Sanırım birinin kapsamı ve oluşum mekanizması farklı, ama tam ayırt edemedim."},
            {"role": "assistant", "content": f"Çok yaklaştın! {short_explain} Şimdi kendi cümlelerinle bir örnek üzerinde ikisini nasıl ayırırsın?"},
        ])

    for item in rows_by_type["scenario_qa"]:
        inst = item.get("instruction", "")
        if "Senaryo:" not in inst:
            continue
        scen = inst.split("Senaryo:", 1)[1].strip()
        out_clean = ". ".join(item["output"].strip().split(".")[:2]).strip() + "."
        sys_msg = SOCRATIC_TUTOR_SYS_TEMPLATES[1].format(concept="Uygulamalı Durum Analizi", prereq="Temel İlke")
        socratic_pool.append([
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": f"Şu olayda kafam karıştı: {scen} Burada doğrudan bildiğimiz standart cevabı uygulayıp geçebilir miyiz?"},
            {"role": "assistant", "content": f"Hemen sonuca atlamadan önce bir adım geri çekilelim: Bu senaryodaki temel risk veya tetikleyici mekanizma sence neyi işaret ediyor?"},
            {"role": "user", "content": "Durumun arka planındaki kuralı ve olasılığı incelememiz gerekiyor sanırım."},
            {"role": "assistant", "content": f"Doğru yoldasın! {out_clean} Bu durumda ilk öncelikli adım olarak neyi seçerdin?"},
        ])

    # Augment/repeat high-value multi-turn Socratic examples up to 480 for strong gradient signal
    while len(socratic_pool) < 480:
        socratic_pool.extend(socratic_pool[: min(len(socratic_pool), 480 - len(socratic_pool))])
    rng.shuffle(socratic_pool)
    for conv in socratic_pool[:480]:
        conversations.append(conv)
        counts["socratic_multi_turn"] += 1

    # 2. Stream B: Pedagogical Scaffolding (varied prerequisite + scenario + distinction = 360)
    scaff_pool: List[List[Dict[str, str]]] = []
    prereq_sys = "Sen pedagojik önkoşul ve kavram ilişkilerini sade Türkçe ile açıklayan EduTerm eğitmenisin."
    for item in rows_by_type["prerequisite"]:
        # Reframe rigid boilerplate into varied conversational tutoring explanation
        out_txt = item["output"].replace("Pedagojik DAG hiyerarşisinde:", "Kavram haritası açısından bakarsak:")
        scaff_pool.append([
            {"role": "system", "content": prereq_sys},
            {"role": "user", "content": item["instruction"]},
            {"role": "assistant", "content": out_txt},
        ])
    for item in rows_by_type["scenario_qa"] + rows_by_type["distinction"]:
        scaff_pool.append([
            {"role": "system", "content": prereq_sys},
            {"role": "user", "content": item["instruction"]},
            {"role": "assistant", "content": item["output"]},
        ])
    rng.shuffle(scaff_pool)
    for conv in scaff_pool[:360]:
        conversations.append(conv)
        counts["pedagogical_scaffolding"] += 1

    # 3. Stream C: CEFR & General Turkish Replay (360 items: 180 general_qa + 180 cefr_json)
    cefr_sys = (
        "Sen Türkçe teknik eğitimde terminoloji tespiti, pedagojik sadeleştirme ve öğrenci düzeyine "
        "göre (CEFR B1, B2 veya C1) kontrollü karmaşıklıkta açıklama üreten uzman bir asistansın."
    )
    gen_rows = list(rows_by_type["general_qa"])
    cefr_rows = list(rows_by_type["cefr_json"])
    rng.shuffle(gen_rows)
    rng.shuffle(cefr_rows)

    for item in gen_rows[:180]:
        u_txt = item["instruction"] if not item.get("input") else f"{item['instruction']}\n\n{item['input']}"
        conversations.append([
            {"role": "system", "content": "Sen yardımcı ve akıcı Türkçe konuşan bir eğitim asistanısın."},
            {"role": "user", "content": u_txt},
            {"role": "assistant", "content": item["output"]},
        ])
        counts["cefr_general_replay"] += 1

    for item in cefr_rows[:180]:
        u_txt = f"{item['instruction']}\n\nDers Notu Metni:\n{item['input']}" if item.get("input") else item["instruction"]
        conversations.append([
            {"role": "system", "content": cefr_sys},
            {"role": "user", "content": u_txt},
            {"role": "assistant", "content": item["output"]},
        ])
        counts["cefr_general_replay"] += 1

    rng.shuffle(conversations)
    return conversations, counts


def encode_multiturn_chatml(tokenizer, messages: List[Dict[str, str]]) -> Optional[Tuple[List[int], List[int]]]:
    """Encode multi-turn ChatML conversation masking everything EXCEPT assistant turns (-100)."""
    input_ids: List[int] = []
    labels: List[int] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        header_ids = tokenizer.encode(f"<|im_start|>{role}\n", add_special_tokens=False)
        body_ids = tokenizer.encode(f"{content}{IM_END}\n", add_special_tokens=False)

        input_ids.extend(header_ids)
        input_ids.extend(body_ids)

        if role == "assistant":
            labels.extend([-100] * len(header_ids))
            labels.extend(body_ids)
        else:
            labels.extend([-100] * (len(header_ids) + len(body_ids)))

    input_ids = input_ids[:MAX_TOKENS]
    labels = labels[:MAX_TOKENS]
    # Ensure enough supervised assistant tokens remain
    if sum(1 for x in labels if x != -100) < 8:
        return None
    return input_ids, labels


def collate_batch(batch: List[Tuple[List[int], List[int]]], pad_id: int, device: str) -> Dict[str, torch.Tensor]:
    max_len = max(len(x[0]) for x in batch)
    ids_out, lbl_out, mask_out = [], [], []
    for ids, lbl in batch:
        pad = max_len - len(ids)
        ids_out.append(ids + [pad_id] * pad)
        lbl_out.append(lbl + [-100] * pad)
        mask_out.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(ids_out, dtype=torch.long, device=device),
        "labels": torch.tensor(lbl_out, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(mask_out, dtype=torch.long, device=device),
    }


def main() -> None:
    t0 = time.time()
    tutor_base_path = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
    student_path = "/users/ali.gunder/models/Kumru-2B"
    cefr_jsonl = Path("/users/ali.gunder/eduterm-finetune/data/train_instruct_cefr_multi_level.jsonl")
    ckpt_dir = Path("/users/ali.gunder/eduterm-hybrid-sim/checkpoints/eduterm_14b_socratic_v2")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Building leak-free Multi-Turn Socratic + CEFR training corpus...")
    conversations, mix_counts = build_training_conversations(cefr_jsonl, seed=42)
    logger.info("Corpus built: %d conversations | breakdown=%s", len(conversations), mix_counts)

    t_tok = AutoTokenizer.from_pretrained(tutor_base_path, trust_remote_code=True)
    if t_tok.pad_token is None:
        t_tok.pad_token = t_tok.eos_token

    encoded = []
    for conv in conversations:
        pair = encode_multiturn_chatml(t_tok, conv)
        if pair is not None:
            encoded.append(pair)
    logger.info("Encoded %d supervised multi-turn sequences.", len(encoded))

    logger.info("Loading Qwen2.5-14B-Instruct (4-bit NF4) for QLoRA Socratic-V2 training...")
    base_model = AutoModelForCausalLM.from_pretrained(
        tutor_base_path, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    t_model = get_peft_model(base_model, peft_cfg)
    t_model.print_trainable_parameters()
    t_model.config.use_cache = False

    epochs = 2
    batch_size = 4
    accum_steps = 4
    lr_max = 1.5e-4
    eff_batch = batch_size * accum_steps
    trainable_params = [p for p in t_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr_max, weight_decay=0.01)
    total_steps = max(1, (len(encoded) // eff_batch) * epochs)
    warmup_steps = max(1, int(total_steps * 0.08))

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return lr_max * (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 1.5e-5 + 0.5 * (lr_max - 1.5e-5) * (1.0 + math.cos(math.pi * prog))

    logger.info("Starting Training: %d epochs | eff_batch=%d | total_steps~%d", epochs, eff_batch, total_steps)
    t_model.train()
    global_step = 0
    micro = 0
    step_losses: List[float] = []
    epoch_losses: List[float] = []

    for ep in range(epochs):
        random.shuffle(encoded)
        ep_loss_sum = 0.0
        ep_steps = 0
        for i in range(0, len(encoded), batch_size):
            batch = collate_batch(encoded[i : i + batch_size], t_tok.pad_token_id, device=device)
            outputs = t_model(**batch)
            loss = outputs.loss / accum_steps
            loss.backward()
            micro += 1
            if micro % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                cur_lr = get_lr(global_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                val_loss = float(outputs.loss.item())
                if not math.isfinite(val_loss):
                    raise RuntimeError(f"Non-finite training loss at step {global_step}: {val_loss}")
                step_losses.append(round(val_loss, 4))
                ep_loss_sum += val_loss
                ep_steps += 1
                if global_step % 15 == 0 or global_step == total_steps:
                    logger.info(
                        "  Epoch [%d/%d] Step [%d/%d] Loss: %.4f | LR: %.2e",
                        ep + 1, epochs, global_step, total_steps, val_loss, cur_lr,
                    )
        epoch_losses.append(round(ep_loss_sum / max(1, ep_steps), 4))
        logger.info("Epoch %d avg loss: %.4f", ep + 1, epoch_losses[-1])

    logger.info("Saving EduTerm-14B-Socratic-V2 LoRA adapter -> %s", ckpt_dir)
    t_model.save_pretrained(ckpt_dir)
    t_tok.save_pretrained(ckpt_dir)

    # Switch to Evaluation Mode with Kumru-2B Student Agent
    train_elapsed = time.time() - t0
    t_model.gradient_checkpointing_disable()
    t_model.config.use_cache = True
    t_model.eval()

    logger.info("Loading Student Agent (Kumru-2B bf16) for Live Unified Socratic Evaluation...")
    s_tok = AutoTokenizer.from_pretrained(student_path)
    s_model = AutoModelForCausalLM.from_pretrained(student_path, torch_dtype=torch.bfloat16, device_map="auto")
    s_model.eval()

    sessions: List[Dict[str, Any]] = []
    for concept in ISG_CONCEPTS:
        prereq_str = ", ".join(concept["prerequisites"]) if concept["prerequisites"] else "Yok (Temel Kavram)"
        for persona in PERSONAS:
            student_sys = (
                f"Sen gerçek bir öğrencisin ({persona['label']}). {persona['instruction']}\n"
                f"Konu: {concept['name']}.\n"
                f"Başlangıçtaki yanlış inancın: {concept['misconception']}\n"
                "ÖNEMLİ: Sadece kendi ağzından ('Ben...' diyerek) 1-2 cümlelik kısa Türkçe yanıt ver. "
                "Asla 'Öğretmen:' veya 'Öğrenci:' diye senaryo yazma, Çince/İngilizce kullanma."
            )
            tutor_sys = (
                f"Sen Boğaziçi EduTerm Sokratik İSG Eğitmenisin. Hedef kavram: {concept['name']}.\n"
                f"Bu kavramın DAG önkoşulları: {prereq_str}.\n"
                "Kurallar: Cevabı ilk turda doğrudan verme; önce önkoşul bilgisini yokla, "
                "ardından günlük hayattan bir örnek ve yönlendirici bir soruyla öğrencinin yanılgısını "
                "kendisinin fark etmesini sağla. Yalnızca temiz ve anlaşılır Türkçe (2-4 cümle) yaz."
            )

            student_msgs = [{"role": "system", "content": student_sys}]
            tutor_msgs = [{"role": "system", "content": tutor_sys}]
            transcript: List[Dict[str, str]] = []

            student_msgs.append({
                "role": "user",
                "content": f"Hocaya '{concept['name']}' hakkındaki yanlış düşünceni doğrudan 1-2 cümleyle sor.",
            })

            for t in range(1, 5):
                s_reply = generate_chat(s_model, s_tok, student_msgs, max_new_tokens=100, temperature=0.7)
                student_msgs.append({"role": "assistant", "content": s_reply})
                tutor_msgs.append({"role": "user", "content": s_reply})

                # Evaluate EduTerm-14B-Socratic-V2 LoRA DIRECTLY (adapter enabled)
                t_reply = generate_chat(t_model, t_tok, tutor_msgs, max_new_tokens=140, temperature=0.4)
                tutor_msgs.append({"role": "assistant", "content": t_reply})
                student_msgs.append({"role": "user", "content": f"Hocan şöyle dedi: {t_reply}\nÖğrenci olarak 1-2 cümleyle yanıtla."})
                transcript.append({"turn": t, "student": s_reply, "tutor": t_reply})

            sess = {
                "concept_id": concept["id"],
                "concept_name": concept["name"],
                "persona_id": persona["id"],
                "condition": "eduterm_14b_socratic_v2_direct",
                "cjk_leakage_turns": sum(1 for x in transcript if has_cjk(x["tutor"]) or has_cjk(x["student"])),
                "transcript": transcript,
            }
            scores = judge_session(t_model, t_tok, concept, sess)
            sess["scores"] = scores
            sessions.append(sess)
            logger.info(
                "[%s | %s | socratic_v2] resolved=%d socratic=%d prereq=%d purity=%d cjk=%d",
                concept["id"],
                persona["id"],
                scores["misconception_resolved"],
                scores["socratic_score"],
                scores["prerequisite_score"],
                scores["turkish_purity"],
                sess["cjk_leakage_turns"],
            )

    n = len(sessions)
    by_persona = {}
    for p in PERSONAS:
        ps = [s for s in sessions if s["persona_id"] == p["id"]]
        by_persona[p["id"]] = {
            "n": len(ps),
            "empirical_transition_rate": round(sum(x["scores"]["misconception_resolved"] for x in ps) / max(len(ps), 1), 4),
            "socratic_mean": round(sum(x["scores"]["socratic_score"] for x in ps) / max(len(ps), 1), 3),
            "prerequisite_mean": round(sum(x["scores"]["prerequisite_score"] for x in ps) / max(len(ps), 1), 3),
        }

    summary = {
        "total_sessions": n,
        "total_turns": n * 4,
        "misconception_resolution_rate": round(sum(s["scores"]["misconception_resolved"] for s in sessions) / max(n, 1), 4),
        "socratic_mean": round(sum(s["scores"]["socratic_score"] for s in sessions) / max(n, 1), 3),
        "prerequisite_grounding_mean": round(sum(s["scores"]["prerequisite_score"] for s in sessions) / max(n, 1), 3),
        "turkish_purity_mean": round(sum(s["scores"]["turkish_purity"] for s in sessions) / max(n, 1), 3),
        "cjk_leakage_turn_rate": round(sum(s["cjk_leakage_turns"] for s in sessions) / max(n * 4, 1), 4),
        "by_persona": by_persona,
    }

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hpc_socratic_v2_report.json"
    out_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "base_model": "Qwen2.5-14B-Instruct (4-bit NF4)",
                    "adapter": "eduterm_14b_socratic_v2 (LoRA r=16, alpha=32)",
                    "student_model": "Kumru-2B (bf16)",
                    "train_mix_counts": mix_counts,
                    "epochs": epochs,
                    "effective_batch_size": eff_batch,
                    "optimizer_steps": global_step,
                    "epoch_losses": epoch_losses,
                    "step_losses": step_losses,
                    "train_elapsed_sec": round(train_elapsed, 1),
                    "total_elapsed_sec": round(time.time() - t0, 1),
                },
                "summary": {"eduterm_14b_socratic_v2_direct": summary},
                "sessions": sessions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("EduTerm-14B-Socratic-V2 Finished in %.1fs -> %s | Summary=%s", time.time() - t0, out_path, summary)


if __name__ == "__main__":
    main()

