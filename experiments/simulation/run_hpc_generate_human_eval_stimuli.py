# -*- coding: utf-8 -*-
"""
Experiment 15: Controlled Same-Input Stimulus Generation for Human Evaluation V2
on Boğaziçi HPC (A40 48GB).

Generates 100% verbatim, unedited, deterministic (temperature=0.0, seed=42, Non-Latin blocked)
model outputs for all previously missing or simulator-confounded Human Eval V2 modules:
  1. Section 2 (Explanation Quality): Base 14B vs EduTerm (cefr, onpolicy_dpo_v2) across 4 domains
     (ISG, KVKK, CS, Finance) with Atesman readability scores.
  2. Section 4 (Real Multi-Tier CEFR B1/B2/C1): Same concept at B1, B2, C1 levels (non-mock).
  3. Section 5 (Controlled Same-Input Socratic Tutoring): Identical clean Turkish student
     misconception prompts across ISG (2), KVKK (1), and CS (1) for
     `prompt_dag_rag_base_14b`, `task_separated_hybrid`, and `onpolicy_dpo_v2`.
  4. Section 6 (Verified Zero-Overlap OOD): Maritime Law & Quantum Cryptography
     (absent from ISG/KVKK/CS/Banking training corpora) for `prompt_dag_rag_base_14b` vs `onpolicy_dpo_v2`.

Outputs:
  results/hpc_human_eval_v2_stimuli.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_hpc_alignment_tax_cefr import calculate_atesman_score, evaluate_cefr_compliance, resolve_dpo_v2_path
from run_hpc_cjk_blocked_dag_ablation import CJKBlockLogitsProcessor, generate_chat_nocjk
from run_hpc_dpo_3seed_ablation import generate_cefr_card
from run_hpc_dpo_r2_nonlatin import build_nonlatin_token_ids, has_nonlatin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp15HumanStimuli")


def main() -> None:
    t0 = time.time()
    torch.manual_seed(42)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    base_path = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
    cefr_path = "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora"
    dpo_v2_path = resolve_dpo_v2_path()

    tok = AutoTokenizer.from_pretrained(base_path)
    base = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb, device_map="auto")
    model = PeftModel.from_pretrained(base, cefr_path, adapter_name="cefr")
    model.load_adapter(dpo_v2_path, adapter_name="dpo_v2")
    model.eval()
    proc = CJKBlockLogitsProcessor(build_nonlatin_token_ids(tok))

    def gen(adapter: str | None, msgs: List[Dict[str, str]], max_new_tokens: int = 220) -> str:
        if adapter is None:
            with model.disable_adapter():
                return generate_chat_nocjk(model, tok, msgs, cjk_processor=proc, max_new_tokens=max_new_tokens, temperature=0.0)
        model.set_adapter(adapter)
        return generate_chat_nocjk(model, tok, msgs, cjk_processor=proc, max_new_tokens=max_new_tokens, temperature=0.0)

    # 1. Section 2: Standalone Explanations (Base 14B vs EduTerm DPO-v2 & CEFR) across 4 domains
    exp_concepts = [
        {"id": "EXP_01", "domain": "ISG", "concept": "Tehlike Kavramı"},
        {"id": "EXP_02", "domain": "KVKK", "concept": "Açık Rıza ve Hukuki Sebepler"},
        {"id": "EXP_03", "domain": "CS", "concept": "Dinamik Programlama ve Örtüşen Alt Problemler"},
        {"id": "EXP_04", "domain": "Finance", "concept": "Nominal Faiz ve Enflasyon Etkisi"},
    ]
    section_2: List[Dict[str, Any]] = []
    for item in exp_concepts:
        msgs = [
            {"role": "system", "content": "Sen Türkçe bir üniversite eğitim asistanısın. Kavramı başlangıç seviyesindeki bir öğrenci için açık, doğru ve anlaşılır biçimde 3-4 cümleyle açıkla."},
            {"role": "user", "content": f"'{item['concept']}' kavramını örnek vererek Türkçe açıkla."},
        ]
        out_base = gen(None, msgs, 200)
        out_dpo = gen("dpo_v2", msgs, 200)
        out_cefr = gen("cefr", msgs, 200)
        section_2.append({
            **item,
            "prompt": msgs,
            "outputs": {
                "base_14b": {"text": out_base, "atesman": calculate_atesman_score(out_base), "nonlatin": has_nonlatin(out_base)},
                "onpolicy_dpo_v2": {"text": out_dpo, "atesman": calculate_atesman_score(out_dpo), "nonlatin": has_nonlatin(out_dpo)},
                "eduterm_14b_cefr": {"text": out_cefr, "atesman": calculate_atesman_score(out_cefr), "nonlatin": has_nonlatin(out_cefr)},
            },
        })
        logger.info("Generated Section 2 %s", item["id"])

    # 2. Section 4: Real Multi-Tier CEFR (B1 / B2 / C1) for the same concept
    cefr_concept = "Tehlike ve Risk Kavramı"
    cefr_levels = [
        ("B1", "Sade, günlük örneklerle lise/1. sınıf düzeyinde anlaşılır kısa cümlelerle açıkla."),
        ("B2", "Teknik üniversite düzeyinde, mesleki terminoloji kullanarak yapılandırılmış biçimde açıkla."),
        ("C1", "Akademik, mevzuat ve mühendislik kavramlarıyla yoğun ve ileri düzeyde analiz ederek açıkla."),
    ]
    section_4: Dict[str, Any] = {"concept": cefr_concept, "model": "onpolicy_dpo_v2", "levels": {}}
    for lvl, instr in cefr_levels:
        msgs = [
            {"role": "system", "content": f"Sen EduTerm İSG uzmanısın. Hedef okunabilirlik seviyesi: CEFR-{lvl}. {instr}"},
            {"role": "user", "content": f"'{cefr_concept}' konusunu CEFR-{lvl} düzeyinde 3-4 cümleyle Türkçe açıkla."},
        ]
        txt = gen("dpo_v2", msgs, 200)
        at = calculate_atesman_score(txt)
        section_4["levels"][lvl] = {
            "text": txt,
            "atesman": at,
            "cefr_compliant": evaluate_cefr_compliance(lvl, at),
            "nonlatin": has_nonlatin(txt),
        }
    logger.info("Generated Section 4 real B1/B2/C1 levels.")

    # 3. Section 5: Controlled Same-Input Socratic Tutoring (ISG x2, KVKK x1, CS x1)
    soc_cases = [
        {
            "scenario_id": "SOC_01",
            "domain": "ISG",
            "name": "Tehlike Kavramı",
            "prerequisites": [],
            "gold_principle": "Tehlike, henüz zarar oluşmasa da zarar verme potansiyeli taşıyan kaynak veya durumdur.",
            "student_prompt": "Hocam, tehlike sadece işyerinde aktif bir kaza veya yaralanma olduğunda ortaya çıkmaz mı? Duran bir makine veya ıslak zemin kimseye zarar vermediyse tehlike sayılır mı?",
        },
        {
            "scenario_id": "SOC_02",
            "domain": "ISG",
            "name": "Ramak Kala Olayı",
            "prerequisites": ["Tehlike Kavramı"],
            "gold_principle": "Ramak kala olayında yaralanma veya hasar oluşmaz ama tekrarında kazaya yol açabileceği için raporlanıp önlem alınmalıdır.",
            "student_prompt": "Hocam, ramak kala olayında kimse yaralanmadıysa ve maddi hasar yoksa bunu raporlayıp önlem almaya gerçekten gerek var mı?",
        },
        {
            "scenario_id": "SOC_03",
            "domain": "KVKK",
            "name": "Anonimleştirme ve Pseudonimizasyon",
            "prerequisites": ["Kişisel Veri Tanımı"],
            "gold_principle": "Ek bilgiyle kişi yeniden belirlenebiliyorsa bu pseudonimizasyondur ve hâlâ kişisel veridir; anonimleştirme geri döndürülemez olmalıdır.",
            "student_prompt": "Hocam, bir veri tablosundan isimleri silip yerine rastgele müşteri ID numarası koyarsak veri tamamen anonim hale gelip KVKK dışına çıkar değil mi?",
        },
        {
            "scenario_id": "SOC_04",
            "domain": "CS",
            "name": "Dinamik Programlama ve Örtüşen Alt Problemler",
            "prerequisites": ["Özyineleme (Recursion)"],
            "gold_principle": "Dinamik programlama örtüşen alt problemlerin sonuçlarını bellekte saklayarak tekrar hesaplamayı önler; böl ve yönet ayrık alt problemlere uygulanır.",
            "student_prompt": "Hocam, dinamik programlama da problemi küçük parçalara bölüp çözüyor; o zaman Böl ve Yönet (Divide & Conquer) yöntemiyle tamamen aynı şey değil mi?",
        },
    ]
    section_5: List[Dict[str, Any]] = []
    for c in soc_cases:
        prereq_str = ", ".join(c["prerequisites"]) if c["prerequisites"] else "Yok (Temel Kavram)"
        cefr_card = generate_cefr_card(model, tok, proc, c)
        msgs_base = [
            {"role": "system", "content": f"Sen Sokratik bir üniversite öğretmenisin. Doğrudan uzun ders anlatımı yapma; önce öğrencinin ön koşul bilgisini yokla ve adım adım sorularla kavram yanılgısını düzelt. Sadece Türkçe yaz.\nKavram: {c['name']}\nÖn koşullar: {prereq_str}\nHedef ilke: {c['gold_principle']}"},
            {"role": "user", "content": c["student_prompt"]},
        ]
        msgs_hyb = [
            {"role": "system", "content": f"Sen Sokratik ve sabırlı bir üniversite öğretmenisin. Öğrenciye CEFR-B1 düzeyinde net sorular sorarak doğru ilkeyi kendisinin keşfetmesini sağla. Sadece akıcı Türkçe yaz.\nKavram: {c['name']}\nÖn koşul özeti: {cefr_card}"},
            {"role": "user", "content": c["student_prompt"]},
        ]
        msgs_dpo = [
            {"role": "system", "content": f"Sen Sokratik ve destekleyici bir Türkçe üniversite öğretmenisin.\nKavram: {c['name']}\nÖn koşullar: {prereq_str}\nHedef ilke: {c['gold_principle']}\nÖğrencinin kavram yanılgısını tespit et, doğrudan cevabı vermek yerine ön koşula dayalı yönlendirici sorular sor."},
            {"role": "user", "content": c["student_prompt"]},
        ]
        section_5.append({
            **c,
            "same_input_controlled": True,
            "cefr_card_used_for_hybrid": cefr_card,
            "outputs": {
                "prompt_dag_rag_base_14b": gen(None, msgs_base, 200),
                "task_separated_hybrid": gen(None, msgs_hyb, 200),
                "onpolicy_dpo_v2": gen("dpo_v2", msgs_dpo, 200),
            },
        })
        logger.info("Generated Section 5 %s", c["scenario_id"])

    # 4. Section 6: Verified Zero-Overlap OOD (Maritime Law & Quantum Cryptography)
    ood_cases = [
        {
            "scenario_id": "OOD_01",
            "domain": "Maritime Law (Zero-Overlap OOD)",
            "name": "Müşterek Avarya (General Average)",
            "prerequisites": ["Deniz Ticareti Sorumluluğu"],
            "gold_principle": "Gemi ve yükü ortak tehlikeden kurtarmak için bilerek yapılan fedakârlık ve masraflar, kurtarılan gemi ve tüm yük sahipleri arasında orantılı olarak paylaşılır.",
            "student_prompt": "Hocam, bir gemide yangın çıkıp yükün bir kısmı gemiyi kurtarmak için bilerek denize atıldığında, bu zararı yalnızca denize atılan yükün sahibi mi karşılar?",
        },
        {
            "scenario_id": "OOD_02",
            "domain": "Quantum Cryptography (Zero-Overlap OOD)",
            "name": "BB84 Kuantum Anahtar Dağıtımı ve Gözlemci Etkisi",
            "prerequisites": ["Kuantum Ölçümü"],
            "gold_principle": "Kuantum durumları bozulmadan kopyalanamadığı (no-cloning) için üçüncü bir kişinin dinleme girişimi hata oranını artırır ve iletişim taraflarınca tespit edilir.",
            "student_prompt": "Hocam, kuantum anahtar dağıtımında (BB84) bir saldırgan hattı gizlice dinlerse klasik şifrelemedeki gibi fark edilmeden tüm anahtarı kopyalayabilir mi?",
        },
    ]
    section_6: List[Dict[str, Any]] = []
    for c in ood_cases:
        prereq_str = ", ".join(c["prerequisites"])
        msgs_base = [
            {"role": "system", "content": f"Sen Sokratik bir üniversite öğretmenisin. Doğrudan uzun ders anlatımı yapma; adım adım sorularla kavram yanılgısını düzelt. Sadece Türkçe yaz.\nKavram: {c['name']}\nÖn koşullar: {prereq_str}\nHedef ilke: {c['gold_principle']}"},
            {"role": "user", "content": c["student_prompt"]},
        ]
        msgs_dpo = [
            {"role": "system", "content": f"Sen Sokratik ve destekleyici bir Türkçe üniversite öğretmenisin.\nKavram: {c['name']}\nÖn koşullar: {prereq_str}\nHedef ilke: {c['gold_principle']}\nÖğrencinin kavram yanılgısını tespit et, doğrudan cevabı vermek yerine ön koşula dayalı yönlendirici sorular sor."},
            {"role": "user", "content": c["student_prompt"]},
        ]
        section_6.append({
            **c,
            "same_input_controlled": True,
            "outputs": {
                "prompt_dag_rag_base_14b": gen(None, msgs_base, 200),
                "onpolicy_dpo_v2": gen("dpo_v2", msgs_dpo, 200),
            },
        })
        logger.info("Generated Section 6 %s", c["scenario_id"])

    report = {
        "metadata": {
            "base_model": base_path,
            "cefr_adapter": cefr_path,
            "dpo_v2_adapter": dpo_v2_path,
            "decoding": {"temperature": 0.0, "do_sample": False, "seed": 42, "nonlatin_blocked": True},
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "section_2_explanations": section_2,
        "section_4_cefr_levels": section_4,
        "section_5_socratic_same_input": section_5,
        "section_6_verified_ood": section_6,
    }
    Path("results/hpc_human_eval_v2_stimuli.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved results/hpc_human_eval_v2_stimuli.json in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
