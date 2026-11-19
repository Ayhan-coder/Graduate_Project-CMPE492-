# -*- coding: utf-8 -*-
"""
Experiment 8: Out-of-Domain (OOD) Multi-Turn Tutoring Transfer
on Boğaziçi HPC (A40 48GB).

Evaluates 3 tutor conditions:
  - prompt_dag_rag_base_14b
  - task_separated_hybrid
  - onpolicy_dpo_v2
across 12 unseen OOD Turkish university concepts (CS/Algorithms, KVKK/Privacy,
Banking/Finance, Electrical Engineering; 3 concepts each) x 3 personas
= 36 paired sessions/arm (108 sessions, 432 turns) with Kumru-2B student
and full Non-Latin logit blocking.

Outputs:
  results/hpc_ood_tutoring_report.json
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_hpc_alignment_tax_cefr import resolve_dpo_v2_path
from run_hpc_cjk_blocked_dag_ablation import (
    CJKBlockLogitsProcessor,
    bootstrap_ci,
    exact_mcnemar_p,
    generate_chat_nocjk,
    judge_session_blinded,
)
from run_hpc_dpo_3seed_ablation import generate_cefr_card
from run_hpc_dpo_r2_nonlatin import build_nonlatin_token_ids, has_nonlatin
from run_hpc_live_student_simulation import PERSONAS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp8OODTutoring")

OOD_CONCEPTS: List[Dict[str, Any]] = [
    # CS / Algorithms
    {
        "id": "OOD01",
        "domain": "CS",
        "name": "Dinamik Programlama ve Örtüşen Alt Problemler",
        "prerequisites": ["Özyineleme (Recursion)"],
        "misconception": "Dinamik programlama ile böl ve yönet (Divide & Conquer) aynı şeydir; ikisi de alt problemleri bağımsız çözer.",
        "gold_principle": "Dinamik programlama, örtüşen alt problemlerin sonuçlarını bellekte saklayarak tekrar hesaplamayı önler; böl ve yönet ise ayrık alt problemlere uygulanır.",
    },
    {
        "id": "OOD02",
        "domain": "CS",
        "name": "Hash Tablolarında Çakışma (Collision)",
        "prerequisites": ["Dizi İndeksleme"],
        "misconception": "İyi bir hash fonksiyonu seçilirse farklı iki anahtar asla aynı indekse düşmez.",
        "gold_principle": "Güvercin yuvası ilkesi gereği anahtar uzayı tablo boyutundan büyük olduğundan çakışma kaçınılmazdır ve zincirleme veya açık adresleme ile çözülür.",
    },
    {
        "id": "OOD03",
        "domain": "CS",
        "name": "Veritabanı İndeksleme (B-Tree) Maliyeti",
        "prerequisites": ["Tablo ve Sorgu"],
        "misconception": "Bir tablodaki tüm sütunlara indeks eklemek her zaman tüm veritabanı işlemlerini hızlandırır.",
        "gold_principle": "İndeksler okuma (SELECT) sorgularını hızlandırırken, her INSERT/UPDATE/DELETE işleminde indeks ağacını güncelleme maliyeti ve ek disk alanı yaratır.",
    },
    # KVKK / Data Privacy
    {
        "id": "OOD04",
        "domain": "KVKK",
        "name": "Açık Rıza ve Hukuki Sebepler",
        "prerequisites": ["Kişisel Veri Tanımı"],
        "misconception": "KVKK kapsamında kişisel veri işleyebilmek için tek geçerli şart her zaman ilgili kişinin açık rızasını almaktır.",
        "gold_principle": "Sözleşmenin ifası, kanunlarda açıkça öngörülme veya meşru menfaat gibi KVKK Madde 5/2 şartları varsa açık rıza aranmaksızın veri işlenebilir.",
    },
    {
        "id": "OOD05",
        "domain": "KVKK",
        "name": "Anonimleştirme ve Pseudonimizasyon",
        "prerequisites": ["Kişisel Veri Tanımı"],
        "misconception": "İsimleri çıkarıp yerine rastgele kimlik numarası (ID) koymak veriyi tamamen anonim hale getirir ve KVKK dışına çıkarır.",
        "gold_principle": "Ek bilgiyle kişi yeniden belirlenebiliyorsa bu pseudonimizasyondur ve hâlâ kişisel veridir; anonimleştirme geri döndürülemez olmalıdır.",
    },
    {
        "id": "OOD06",
        "domain": "KVKK",
        "name": "Veri Minimizasyonu İlkesi",
        "prerequisites": ["İşleme Amacı"],
        "misconception": "İleride lazım olabileceği düşüncesiyle kullanıcıdan mümkün olan en fazla veriyi baştan toplamak daha doğrudur.",
        "gold_principle": "Kişisel veriler yalnızca işlenme amacıyla bağlantılı, sınırlı ve ölçülü olmalı; gereksiz veri toplanmamalıdır.",
    },
    # Banking / Finance
    {
        "id": "OOD07",
        "domain": "Banking",
        "name": "Bileşik Faiz ve Enflasyon Etkisi",
        "prerequisites": ["Nominal Faiz"],
        "misconception": "Nominal mevduat faizi pozitif olduğu sürece paranın satın alma gücü mutlaka artar.",
        "gold_principle": "Reel getiri enflasyondan arındırılmış getiridir; nominal faiz enflasyonun altındaysa satın alma gücü reel olarak azalır.",
    },
    {
        "id": "OOD08",
        "domain": "Banking",
        "name": "Kaldıraç ve Likidite Riski",
        "prerequisites": ["Özkaynak ve Borç"],
        "misconception": "Yüksek kârlılığa sahip bir kurum borçlarını vadesinde mutlaka ödeyebilir; kârlılık likiditeyi garanti eder.",
        "gold_principle": "Kârlılık uzun vadeli gelir-gider dengesidir; likidite ise kısa vadeli nakit yükümlülüklerini vadesinde karşılayabilme gücüdür.",
    },
    {
        "id": "OOD09",
        "domain": "Banking",
        "name": "Çeşitlendirme ve Sistematik Risk",
        "prerequisites": ["Portföy Riski"],
        "misconception": "Portföye yeterince farklı hisse senedi eklenirse piyasadaki tüm risk tamamen sıfırlanabilir.",
        "gold_principle": "Çeşitlendirme yalnızca firmaya özgü (sistematik olmayan) riski azaltır; faiz veya kriz gibi piyasa geneli sistematik riski yok etmez.",
    },
    # Electrical Engineering
    {
        "id": "OOD10",
        "domain": "EE",
        "name": "Nyquist Örnekleme Teoremi",
        "prerequisites": ["Sinyal Frekansı"],
        "misconception": "Bir sinyali dijital olarak hatasız geri oluşturmak için sinyalin en yüksek frekansıyla aynı hızda örneklemek yeterlidir.",
        "gold_principle": "Örtüşmeyi (aliasing) önlemek için örnekleme frekansı, sinyalin içerdiği en yüksek frekansın kesinlikle iki katından büyük olmalıdır.",
    },
    {
        "id": "OOD11",
        "domain": "EE",
        "name": "Güç Faktörü ve Reaktif Güç",
        "prerequisites": ["Alternatif Akım (AC)"],
        "misconception": "Endüktif yüklerde çekilen reaktif güç tamamen yok olduğu için şebekeden hiçbir akım çekmez.",
        "gold_principle": "Reaktif güç iş yapmasa da manyetik alan kurulumu için kaynakla yük arasında gidip gelir, hat akımını ve iletim kayıplarını artırır.",
    },
    {
        "id": "OOD12",
        "domain": "EE",
        "name": "Op-Amp Geri Besleme ve Sanal Kısa Devre",
        "prerequisites": ["Gerilim Bölücü"],
        "misconception": "Op-Amp girişlerindeki sanal kısa devre özelliği tüm devrelerde, geri besleme olmasa bile geçerlidir.",
        "gold_principle": "Eviren ve evirmeyen giriş gerilimlerinin eşitlenmesi yalnızca negatif geri besleme altında ve doğrusal çalışma bölgesinde geçerlidir.",
    },
]


def run_ood_session(
    tutor_model, tutor_tok, tutor_proc, student_model, student_tok, student_proc,
    concept: Dict[str, Any], persona: Dict[str, Any], condition: str, cefr_card: str
) -> Dict[str, Any]:
    prereq_str = ", ".join(concept["prerequisites"]) if concept["prerequisites"] else "Yok (Temel Kavram)"
    if condition == "prompt_dag_rag_base_14b":
        tutor_sys = (
            "Sen Sokratik bir üniversite öğretmenisin. Doğrudan uzun ders anlatımı yapma; önce öğrencinin "
            "ön koşul bilgisini yokla ve adım adım sorularla kavram yanılgısını düzelt. Sadece Türkçe yaz.\n"
            f"Kavram: {concept['name']} (Alan: {concept['domain']})\nÖn koşullar: {prereq_str}\nHedef ilke: {concept['gold_principle']}"
        )
    elif condition == "task_separated_hybrid":
        tutor_sys = (
            "Sen Sokratik ve sabırlı bir üniversite öğretmenisin. Öğrenciye CEFR-B1 düzeyinde net sorular sorarak "
            "doğru ilkeyi kendisinin keşfetmesini sağla. Sadece akıcı Türkçe yaz.\n"
            f"Kavram: {concept['name']}\nÖn koşul özeti: {cefr_card}"
        )
    else:  # onpolicy_dpo_v2
        tutor_sys = (
            "Sen Sokratik ve destekleyici bir Türkçe üniversite öğretmenisin.\n"
            f"Kavram: {concept['name']}\nÖn koşullar: {prereq_str}\nHedef ilke: {concept['gold_principle']}\n"
            "Öğrencinin kavram yanılgısını tespit et, doğrudan cevabı vermek yerine ön koşula dayalı yönlendirici sorular sor."
        )

    student_sys = (
        f"Sen bir üniversite öğrencisisin ({persona['instruction']}).\n"
        f"Başlangıçtaki yanlış düşüncen: {concept['misconception']}\n"
        "Kısa ve doğal Türkçe ile yanıt ver. Öğretmen iyi sorularla seni ikna ederse hatanı kabul edip doğru ilkeyi kendi cümlelerinle özetle."
    )

    t_msgs = [{"role": "system", "content": tutor_sys}]
    s_msgs = [
        {"role": "system", "content": student_sys},
        {"role": "user", "content": f"Merhaba, bugün '{concept['name']}' konusunu konuşacağız. Bu konuda ne düşünüyorsun?"},
    ]

    transcript: List[Dict[str, Any]] = []
    nonlatin_turns = 0
    q_turns = 0
    for turn in range(1, 5):
        s_reply = generate_chat_nocjk(student_model, student_tok, s_msgs, cjk_processor=student_proc, max_new_tokens=120, temperature=0.7)
        s_msgs.append({"role": "assistant", "content": s_reply})
        t_msgs.append({"role": "user", "content": s_reply})

        t_reply = generate_chat_nocjk(tutor_model, tutor_tok, t_msgs, cjk_processor=tutor_proc, max_new_tokens=200, temperature=0.4)
        t_msgs.append({"role": "assistant", "content": t_reply})
        s_msgs.append({"role": "user", "content": t_reply})

        if has_nonlatin(t_reply):
            nonlatin_turns += 1
        if "?" in t_reply:
            q_turns += 1
        transcript.append({"turn": turn, "student": s_reply, "tutor": t_reply})

    return {
        "concept_id": concept["id"],
        "domain": concept["domain"],
        "concept_name": concept["name"],
        "persona_id": persona["id"],
        "condition": condition,
        "nonlatin_leakage_turns": nonlatin_turns,
        "tutor_question_turns": q_turns,
        "transcript": transcript,
    }


def main():
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    base_path = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
    t_tok = AutoTokenizer.from_pretrained(base_path)
    t_base = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb, device_map="auto")
    t_model = PeftModel.from_pretrained(t_base, "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora", adapter_name="cefr")
    t_model.load_adapter(resolve_dpo_v2_path(), adapter_name="dpo_v2")
    t_model.eval()
    t_proc = CJKBlockLogitsProcessor(build_nonlatin_token_ids(t_tok))

    s_path = "/users/ali.gunder/models/Kumru-2B"
    s_tok = AutoTokenizer.from_pretrained(s_path)
    s_model = AutoModelForCausalLM.from_pretrained(s_path, torch_dtype=torch.bfloat16, device_map="auto")
    s_model.eval()
    s_proc = CJKBlockLogitsProcessor(build_nonlatin_token_ids(s_tok))

    cefr_cards: Dict[str, str] = {}
    for c in OOD_CONCEPTS:
        cefr_cards[c["id"]] = generate_cefr_card(t_model, t_tok, t_proc, c)

    conditions = ["prompt_dag_rag_base_14b", "task_separated_hybrid", "onpolicy_dpo_v2"]
    sessions: List[Dict[str, Any]] = []
    for cond in conditions:
        for c in OOD_CONCEPTS:
            for p in PERSONAS:
                if cond == "prompt_dag_rag_base_14b":
                    with t_model.disable_adapter():
                        s_out = run_ood_session(t_model, t_tok, t_proc, s_model, s_tok, s_proc, c, p, cond, "")
                elif cond == "task_separated_hybrid":
                    with t_model.disable_adapter():
                        s_out = run_ood_session(t_model, t_tok, t_proc, s_model, s_tok, s_proc, c, p, cond, cefr_cards[c["id"]])
                else:
                    t_model.set_adapter("dpo_v2")
                    s_out = run_ood_session(t_model, t_tok, t_proc, s_model, s_tok, s_proc, c, p, cond, "")
                sessions.append(s_out)
                if len(sessions) % 18 == 0:
                    logger.info("OOD simulation progress: %d/108", len(sessions))

    del t_model, t_base, s_model
    gc.collect()
    torch.cuda.empty_cache()

    Path("results/hpc_ood_tutoring_raw_sessions.json").write_text(json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8")

    j_path = "/users/ali.gunder/models/Qwen2.5-7B-Instruct"
    j_tok = AutoTokenizer.from_pretrained(j_path)
    j_model = AutoModelForCausalLM.from_pretrained(j_path, torch_dtype=torch.bfloat16, device_map="auto")
    j_model.eval()
    j_proc = CJKBlockLogitsProcessor(build_nonlatin_token_ids(j_tok))

    concept_by_id = {c["id"]: c for c in OOD_CONCEPTS}
    for s in sessions:
        s["scores"] = judge_session_blinded(j_model, j_tok, j_proc, concept_by_id[s["concept_id"]], s)

    summary: Dict[str, Any] = {}
    by_cond: Dict[str, List[Dict[str, Any]]] = {}
    by_key: Dict[tuple, Dict[str, Dict[str, Any]]] = {}
    for s in sessions:
        by_cond.setdefault(s["condition"], []).append(s)
        by_key.setdefault((s["concept_id"], s["persona_id"]), {})[s["condition"]] = s

    for cond, items in by_cond.items():
        n = len(items)
        res = [x["scores"]["misconception_resolved"] for x in items]
        summary[cond] = {
            "total_sessions": n,
            "total_turns": n * 4,
            "misconception_resolution_rate": round(sum(res) / n, 4),
            "resolution_bootstrap_95ci": list(bootstrap_ci(res)),
            "socratic_mean": round(sum(x["scores"]["socratic_score"] for x in items) / n, 3),
            "prerequisite_grounding_mean": round(sum(x["scores"]["prerequisite_score"] for x in items) / n, 3),
            "turkish_purity_mean": round(sum(x["scores"]["turkish_purity"] for x in items) / n, 3),
            "nonlatin_leakage_turn_rate": round(sum(x["nonlatin_leakage_turns"] for x in items) / (n * 4), 4),
            "tutor_question_turn_rate": round(sum(x["tutor_question_turns"] for x in items) / (n * 4), 4),
        }

    paired: Dict[str, Any] = {}
    for a, b in [("onpolicy_dpo_v2", "task_separated_hybrid"), ("onpolicy_dpo_v2", "prompt_dag_rag_base_14b")]:
        a_vals = [cmap[a]["scores"]["misconception_resolved"] for cmap in by_key.values()]
        b_vals = [cmap[b]["scores"]["misconception_resolved"] for cmap in by_key.values()]
        paired[f"{a}__vs__{b}"] = exact_mcnemar_p(a_vals, b_vals)

    report = {
        "metadata": {"domains": ["CS", "KVKK", "Banking", "EE"], "elapsed_sec": round(time.time() - t0, 1)},
        "summary": summary,
        "paired_comparisons": paired,
        "sessions": sessions,
    }
    Path("results/hpc_ood_tutoring_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Exp 8 OOD Summary:\n%s", json.dumps({"summary": summary, "paired_comparisons": paired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
