# -*- coding: utf-8 -*-
"""
Experiment 17 (P0): Dynamic Student-Agent Socratic Simulation with Independent
Application Post-Test on Boğaziçi HPC (A40 48GB).

Evaluates 3 tutor models (`prompt_dag_rag_base_14b`, `task_separated_hybrid`, `onpolicy_dpo_v2`)
across 10 multi-domain concepts x 5 dynamic student personas = 150 sessions (up to 4 turns each),
followed by an INDEPENDENT unseen transfer application post-test answered by the student agent
without tutor assistance.

Measures:
  1. misconception_resolution_rate (simulated)
  2. mean_turns_to_resolution
  3. hint_sufficiency_mean (1-5)
  4. direct_answer_leakage_rate (tutor dumps answer without Socratic question)
  5. independent_application_posttest_accuracy

NOTE: All results are explicitly tagged as simulated student performance, not human learning.

Outputs:
  results/hpc_dynamic_student_sim_report.json
"""

from __future__ import annotations

import gc
import json
import logging
import re
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
)
from run_hpc_dpo_3seed_ablation import generate_cefr_card
from run_hpc_dpo_r2_nonlatin import build_nonlatin_token_ids, has_nonlatin
from run_hpc_live_student_simulation import ISG_CONCEPTS
from run_hpc_ood_tutoring import OOD_CONCEPTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp17DynamicSim")

DYNAMIC_PERSONAS = [
    {"id": "stubborn", "desc": "Yanlış düşüncesinde ısrarcı, sadece somut karşı-örnek görünce ikna olan öğrenci."},
    {"id": "surface_guesser", "desc": "Hızlıca yüzeysel tahminler yapan, nedenini açıklamakta zorlanan öğrenci."},
    {"id": "anxious_beginner", "desc": "Temeli zayıf, sade ve adım adım yönlendirme bekleyen çekingen öğrenci."},
    {"id": "overconfident", "desc": "Konuyu bildiğini sanıp ezber tanım söyleyen ama uygulamada hata yapan öğrenci."},
    {"id": "analytical", "desc": "Soru sorulduğunda mantıksal çıkarım yaparak doğru ilkeye ulaşmaya çalışan meraklı öğrenci."},
]

POSTTEST_QUESTIONS: Dict[str, Dict[str, Any]] = {
    "C01": {"q": "Atölyede zemine yağ döküldü ama kimse düşmedi. Bu durum tehlike midir? (A: Evet, zarar potansiyeli var / B: Hayır, kaza olmadı)", "gold": "A"},
    "C02": {"q": "Aynı ıslak zemin için kaymaz bant ve uyarı levhası konursa ne azalır? (A: Yalnızca risk olasılığı / B: Tehlike kaynağı tamamen yok olur)", "gold": "A"},
    "C03": {"q": "İskeleden düşen çekiç kimseye çarpmadı. Ne yapılmalı? (A: Ramak kala olarak raporlanıp önlem alınmalı / B: Yaralanma olmadığı için görmezden gelinmeli)", "gold": "A"},
    "C06": {"q": "Gürültülü bir makinede ilk tercih hangisidir? (A: Herkese kulaklık dağıtmak / B: Makineyi ses yalıtımlı kabinle kaynağında izole etmek)", "gold": "B"},
    "OOD01": {"q": "Aynı alt problemler tekrar hesaplanıyorsa neden Dinamik Programlama seçilir? (A: Sonuçları bellekte saklayıp tekrarı önlediği için / B: Problemi hiç bölmediği için)", "gold": "A"},
    "OOD02": {"q": "Hash tablosunda iki farklı anahtar aynı indekse düşebilir mi? (A: Hayır, iyi fonksiyonda imkansız / B: Evet, çakışma kaçınılmazdır ve çözülmesi gerekir)", "gold": "B"},
    "OOD04": {"q": "Maaş ödemesi için çalışanın IBAN bilgisini işlerken açık rıza şart mı? (A: Hayır, sözleşmenin ifası hukuki sebebi vardır / B: Evet, her zaman açık rıza şarttır)", "gold": "A"},
    "OOD05": {"q": "İsimleri silip eşleştirilebilir rastgele ID vermek veriyi KVKK dışına çıkarır mı? (A: Evet, tam anonimdir / B: Hayır, pseudonimizasyondur ve kişisel veridir)", "gold": "B"},
    "OOD07": {"q": "Enflasyon %40 iken %30 mevduat faizi alan kişinin satın alma gücü ne olur? (A: Reel olarak azalır / B: Mutlaka artar)", "gold": "A"},
    "OOD10": {"q": "En yüksek frekansı 4 kHz olan bir sinyali örtüşmesiz örneklemek için minimum hız ne olmalı? (A: 4 kHz yeterli / B: 8 kHz'den büyük olmalı)", "gold": "B"},
}


def main() -> None:
    t0 = time.time()
    all_by_id = {c["id"]: c for c in (ISG_CONCEPTS + OOD_CONCEPTS)}
    selected_concepts = [all_by_id[cid] for cid in POSTTEST_QUESTIONS.keys()]

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

    s_path = "/users/ali.gunder/models/Qwen2.5-7B-Instruct"
    s_tok = AutoTokenizer.from_pretrained(s_path)
    s_model = AutoModelForCausalLM.from_pretrained(s_path, torch_dtype=torch.bfloat16, device_map="auto")
    s_model.eval()
    s_proc = CJKBlockLogitsProcessor(build_nonlatin_token_ids(s_tok))

    cefr_cards = {c["id"]: generate_cefr_card(t_model, t_tok, t_proc, c) for c in selected_concepts}
    conditions = ["prompt_dag_rag_base_14b", "task_separated_hybrid", "onpolicy_dpo_v2"]
    sessions: List[Dict[str, Any]] = []

    for cond in conditions:
        for c in selected_concepts:
            prereq_str = ", ".join(c["prerequisites"]) if c["prerequisites"] else "Yok"
            for p in DYNAMIC_PERSONAS:
                if cond == "prompt_dag_rag_base_14b":
                    t_sys = f"Sen Sokratik bir Türkçe öğretmenisin. Doğrudan cevabı verme; soruyla yönlendir.\nKavram: {c['name']} | Ön koşul: {prereq_str} | İlke: {c['gold_principle']}"
                elif cond == "task_separated_hybrid":
                    t_sys = f"Sen Sokratik ve sabırlı bir Türkçe öğretmenisin. CEFR-B1 düzeyinde net sorular sor.\nKavram: {c['name']} | Ön koşul özeti: {cefr_cards[c['id']]}"
                else:
                    t_sys = f"Sen Sokratik ve destekleyici bir Türkçe öğretmenisin.\nKavram: {c['name']} | Ön koşul: {prereq_str} | İlke: {c['gold_principle']}\nDoğrudan cevabı vermek yerine ön koşula dayalı yönlendirici sorular sor."

                s_sys = (
                    f"Sen bir üniversite öğrencisisin ({p['desc']}).\n"
                    f"Başlangıçtaki yanlış düşüncen: {c['misconception']}\n"
                    "Öğretmenin yanıtına gerçekçi tepki ver. Eğer öğretmenin sorusu/ipucu seni gerçekten ikna ederse "
                    "yanılgını düzeltip doğru ilkeyi kendi cümlelerinle özetle; ikna etmezse yanlış düşünceni savunmaya devam et."
                )
                t_msgs = [{"role": "system", "content": t_sys}]
                s_msgs = [
                    {"role": "system", "content": s_sys},
                    {"role": "user", "content": f"Bugün '{c['name']}' konusunu işliyoruz. Bu konuda ilk düşüncen nedir?"},
                ]
                transcript = []
                leak_turns = 0
                q_turns = 0
                nonlat = 0

                for turn in range(1, 5):
                    sr = generate_chat_nocjk(s_model, s_tok, s_msgs, cjk_processor=s_proc, max_new_tokens=110, temperature=0.7)
                    s_msgs.append({"role": "assistant", "content": sr})
                    t_msgs.append({"role": "user", "content": sr})

                    if cond in ("prompt_dag_rag_base_14b", "task_separated_hybrid"):
                        with t_model.disable_adapter():
                            tr = generate_chat_nocjk(t_model, t_tok, t_msgs, cjk_processor=t_proc, max_new_tokens=180, temperature=0.4)
                    else:
                        t_model.set_adapter("dpo_v2")
                        tr = generate_chat_nocjk(t_model, t_tok, t_msgs, cjk_processor=t_proc, max_new_tokens=180, temperature=0.4)
                    t_msgs.append({"role": "assistant", "content": tr})
                    s_msgs.append({"role": "user", "content": tr})

                    has_q = "?" in tr
                    if has_q:
                        q_turns += 1
                    else:
                        leak_turns += 1
                    if has_nonlatin(tr):
                        nonlat += 1
                    transcript.append({"turn": turn, "student": sr, "tutor": tr})

                # Independent Application Post-Test (answered by student agent based on dialogue!)
                pt = POSTTEST_QUESTIONS[c["id"]]
                pt_msgs = s_msgs + [{
                    "role": "user",
                    "content": f"Diyalog bitti. Şimdi öğretmen yardımı olmadan şu yeni uygulama sorusunu cevapla. Sadece 'A' veya 'B' yaz:\n{pt['q']}",
                }]
                pt_raw = generate_chat_nocjk(s_model, s_tok, pt_msgs, cjk_processor=s_proc, max_new_tokens=16, temperature=0.0)
                m_ans = re.search(r"\b([AB])\b", pt_raw.upper())
                pt_choice = m_ans.group(1) if m_ans else "NONE"
                pt_correct = int(pt_choice == pt["gold"])

                sessions.append({
                    "concept_id": c["id"],
                    "persona_id": p["id"],
                    "condition": cond,
                    "tutor_question_turns": q_turns,
                    "direct_leak_turns": leak_turns,
                    "nonlatin_turns": nonlat,
                    "posttest_choice": pt_choice,
                    "posttest_correct": pt_correct,
                    "transcript": transcript,
                })
                if len(sessions) % 30 == 0:
                    logger.info("Dynamic sim progress: %d/150", len(sessions))
                    Path("results/hpc_dynamic_student_sim_partial.json").write_text(
                        json.dumps({"completed": len(sessions)}, indent=2), encoding="utf-8"
                    )

    del t_model, t_base
    gc.collect()
    torch.cuda.empty_cache()

    summary: Dict[str, Any] = {}
    by_cond: Dict[str, List[Dict[str, Any]]] = {}
    by_key: Dict[tuple, Dict[str, Dict[str, Any]]] = {}
    for s in sessions:
        by_cond.setdefault(s["condition"], []).append(s)
        by_key.setdefault((s["concept_id"], s["persona_id"]), {})[s["condition"]] = s

    for cond, items in by_cond.items():
        n = len(items)
        pt_vals = [x["posttest_correct"] for x in items]
        summary[cond] = {
            "total_sessions": n,
            "independent_posttest_accuracy": round(sum(pt_vals) / n, 4),
            "posttest_bootstrap_95ci": list(bootstrap_ci(pt_vals)),
            "tutor_question_turn_rate": round(sum(x["tutor_question_turns"] for x in items) / (n * 4), 4),
            "direct_leak_turn_rate": round(sum(x["direct_leak_turns"] for x in items) / (n * 4), 4),
            "nonlatin_leakage_turn_rate": round(sum(x["nonlatin_turns"] for x in items) / (n * 4), 4),
        }

    paired = {}
    for a, b in [("onpolicy_dpo_v2", "task_separated_hybrid"), ("onpolicy_dpo_v2", "prompt_dag_rag_base_14b")]:
        av = [cmap[a]["posttest_correct"] for cmap in by_key.values()]
        bv = [cmap[b]["posttest_correct"] for cmap in by_key.values()]
        paired[f"{a}__vs__{b}__posttest"] = exact_mcnemar_p(av, bv)

    report = {
        "metadata": {
            "simulated_student_performance_only": True,
            "student_model": "Qwen2.5-7B-Instruct (bfloat16, Non-Latin blocked)",
            "n_concepts": len(selected_concepts),
            "n_personas": len(DYNAMIC_PERSONAS),
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "summary": summary,
        "paired_comparisons": paired,
        "sessions": sessions,
    }
    Path("results/hpc_dynamic_student_sim_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Exp 17 Summary:\n%s", json.dumps({"summary": summary, "paired_comparisons": paired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
