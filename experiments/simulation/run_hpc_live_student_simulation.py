# -*- coding: utf-8 -*-
"""
Live Dual-Agent LLM-vs-LLM Synthetic Student Simulation on Boğaziçi HPC (NVIDIA A40 48GB).

Implements the Agent4Edu (AAAI 2025) + Embracing Imperfection / Student_100 (ACL 2025)
paradigm on our 12-node İSG Prerequisite DAG:
- Student Agent: Qwen2.5-7B-Instruct conditioned on 3 cognitive personas
  (At-Risk/Misconception-Prone, Standard, Inattentive) generating unscripted Turkish turns.
- Tutor Agent: Compares Base Qwen2.5-14B-Instruct (disable_adapter) vs.
  EduTerm-14B-DPO (LoRA enabled + DAG prerequisite context).
- Judge: Base Qwen2.5-14B-Instruct (disable_adapter) evaluates misconception resolution,
  Socratic scaffolding (no spoon-feeding), and prerequisite grounding.

Outputs:
  results/hpc_live_student_simulation_report.json
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HPCLiveStudentSim")

# 12 İSG Prerequisite DAG concepts with realistic student misconceptions
ISG_CONCEPTS = [
    {
        "id": "C01",
        "name": "Tehlike Kavramı",
        "prerequisites": [],
        "misconception": "Tehlike sadece aktif bir kaza veya yaralanma gerçekleştiğinde ortaya çıkar; duran bir makine veya ıslak zemin tehlike sayılmaz.",
        "gold_principle": "Tehlike, işyerinde var olan veya dışarıdan gelebilecek, zarar verme potansiyeli taşıyan kaynak veya durumdur (henüz zarar oluşmasa da vardır).",
    },
    {
        "id": "C02",
        "name": "Risk ve Zarar Olasılığı",
        "prerequisites": ["Tehlike Kavramı"],
        "misconception": "Tehlike ve risk tamamen aynı anlamdadır; her tehlikeli durumun riski de aynıdır.",
        "gold_principle": "Risk, tehlikeden kaynaklanacak kayıp/yaralanma olasılığı ile zarar şiddetinin bileşimidir; önlemle risk azaltılabilir ama tehlike kaynağı kalabilir.",
    },
    {
        "id": "C03",
        "name": "Ramak Kala Olayı",
        "prerequisites": ["Tehlike Kavramı"],
        "misconception": "Kimse yaralanmadığı ve maddi hasar olmadığı için ramak kala olaylarını raporlamaya gerek yoktur.",
        "gold_principle": "Ramak kala olaylar yaralanma veya hasar olmadan atlatılan tehlikeli olaylardır ve gelecekteki büyük iş kazalarının en önemli erken uyarı habercisidir.",
    },
    {
        "id": "C04",
        "name": "Yanma Üçgeni ve Yangın Fiziği",
        "prerequisites": [],
        "misconception": "Yangının çıkması ve sürmesi için sadece yanıcı madde ve ısı yeterlidir; oksijen olmasa da yanma devam eder.",
        "gold_principle": "Yanma üçgeni yanıcı madde, oksijen (oksitleyici) ve ısı/tutuşturucu kaynak gerektirir; bu üç bileşenden birini kesmek yangını söndürür.",
    },
    {
        "id": "C05",
        "name": "Olasılık ve Şiddet Skalası",
        "prerequisites": ["Risk ve Zarar Olasılığı", "Ramak Kala Olayı"],
        "misconception": "Risk skorunda yalnızca olayın olma olasılığına bakılır; sonucun ne kadar ağır olduğu skoru değiştirmez.",
        "gold_principle": "Risk skoru hem gerçekleşme olasılığının hem de olası şiddetin sayısal skalalarla çarpımı/bileşimiyle objektif derecelendirilir.",
    },
    {
        "id": "C06",
        "name": "Yangın Sınıfları ve Söndürücüler",
        "prerequisites": ["Yanma Üçgeni ve Yangın Fiziği"],
        "misconception": "Su en güçlü söndürücü olduğu için elektrik ve akaryakıt yangınları dahil tüm yangınlarda su kullanılmalıdır.",
        "gold_principle": "A, B, C, D, F yangın sınıflarına göre söndürücü seçilir; elektrik ve sıvı yakıt yangınlarında su kullanmak iletkenlik ve sıçrama nedeniyle ölümcüldür.",
    },
    {
        "id": "C07",
        "name": "L-Tipi Risk Değerlendirme Matrisi (5x5)",
        "prerequisites": ["Olasılık ve Şiddet Skalası"],
        "misconception": "5x5 matriste orta veya düşük çıkan riskler için hiçbir kontrol adımı veya izleme planı gerekmez.",
        "gold_principle": "5x5 L-Tipi matris (Olasılık x Şiddet) kabul edilemez risklere acil önlem önceliği verirken orta riskleri de planlı iyileştirme ve izlemeye alır.",
    },
    {
        "id": "C08",
        "name": "Kişisel Koruyucu Donanım (KKD) Seçimi",
        "prerequisites": ["Risk ve Zarar Olasılığı", "L-Tipi Risk Değerlendirme Matrisi (5x5)"],
        "misconception": "İşyerinde güvenlik sağlamak için ilk ve en etkili adım doğrudan işçilere baret ve gözlük (KKD) dağıtmaktır.",
        "gold_principle": "Kontrol hiyerarşisinde KKD en son çaredir; önce tehlike kaynağında yok edilmeli veya toplu koruma önlemleri alınmalıdır.",
    },
    {
        "id": "C09",
        "name": "Yangın Tahliye ve Müdahale Protokolü",
        "prerequisites": ["Yangın Sınıfları ve Söndürücüler"],
        "misconception": "Yangın alarmı çaldığında üst katlardan en hızlı iniş için asansör kullanmak en doğru tahliye yöntemidir.",
        "gold_principle": "Tahliyede asansörler kesinlikle kullanılmaz; işaretli acil çıkış merdivenleri ve rüzgar yönü gözetilerek toplanma alanına geçilir.",
    },
    {
        "id": "C10",
        "name": "Acil Durum Eylem Planı",
        "prerequisites": ["L-Tipi Risk Değerlendirme Matrisi (5x5)", "Yangın Tahliye ve Müdahale Protokolü"],
        "misconception": "Acil durum eylem planı sadece yasal denetim için dosyada duran sabit bir belgedir; çalışanların bilmesine gerek yoktur.",
        "gold_principle": "Acil durum eylem planı arama-kurtarma, tahliye ve ilk yardım ekiplerinin görev dağılımını tanımlar ve düzenli tatbikatlarla güncellenir.",
    },
    {
        "id": "C11",
        "name": "Fine-Kinney Risk Analiz Yöntemi",
        "prerequisites": ["L-Tipi Risk Değerlendirme Matrisi (5x5)"],
        "misconception": "Fine-Kinney yöntemi de 5x5 matris gibi sadece Olasılık ve Şiddet olmak üzere iki parametreyi çarpar.",
        "gold_principle": "Fine-Kinney yöntemi İhtimal, Frekans (maruziyet sıklığı) ve Şiddet olmak üzere üç parametreyi çarparak (R = I x F x S) ayrıntılı risk skoru üretir.",
    },
    {
        "id": "C12",
        "name": "İş Kazası Kök Neden Analizi (Kaza Çizgesi)",
        "prerequisites": ["Ramak Kala Olayı", "Acil Durum Eylem Planı", "Fine-Kinney Risk Analiz Yöntemi"],
        "misconception": "Bir iş kazası olduğunda sadece kaza anındaki işçinin dikkatsizliğini bulup cezalandırmak kök neden analizidir.",
        "gold_principle": "Kök neden analizi (ör. 5-Neden) yüzeysel işçi hatasının ötesindeki sistemik, eğitimsel ve teknik aksaklıkları ortaya çıkarıp tekrarını önler.",
    },
]

PERSONAS = [
    {
        "id": "at_risk",
        "label": "Temeli Zayıf / Risk Grubu (At-Risk)",
        "instruction": (
            "Sen ön bilgisi çok zayıf, kafası karışık ve yanlış kavramlara inatla tutunan bir öğrencisin. "
            "Kısa ve günlük Türkçe konuş. Hocanın verdiği ilk ipucunda hemen ikna olma, kendi yanlış fikrini savun; "
            "ancak hoca somut ve anlaşılır bir örnek verirse yavaşça doğruyu kavramaya başla."
        ),
    },
    {
        "id": "standard",
        "label": "Standart Öğrenci (Standard)",
        "instruction": (
            "Sen ortalama ön bilgiye sahip meraklı bir lisans öğrencisisin. "
            "Başlangıçta yaygın bir kavram yanılgısına sahipsin ama hocanın yönlendirici sorularını "
            "mantık yürüterek takip edebiliyorsun. Doğal ve kısa Türkçe cümleler kur."
        ),
    },
    {
        "id": "inattentive",
        "label": "Dikkatsiz / Yüzeysel Öğrenen (Inattentive)",
        "instruction": (
            "Sen dikkati çabuk dağılan, terimleri birbirine karıştıran ve hızlıca kestirme cevap arayan bir öğrencisin. "
            "Kısa konuş, bazen konunun bir detayını atla; ancak hoca dikkatini doğru önkoşul kavrama çekerse toparlan."
        ),
    },
]


def generate_chat(model, tokenizer, messages: List[Dict[str, str]], max_new_tokens: int = 160, temperature: float = 0.6) -> str:
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback for Mistral templates without explicit system role
        merged = []
        sys_buf = ""
        for m in messages:
            if m["role"] == "system":
                sys_buf += m["content"] + "\n\n"
            elif m["role"] == "user" and sys_buf:
                merged.append({"role": "user", "content": sys_buf + m["content"]})
                sys_buf = ""
            else:
                merged.append(m)
        text = tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    inputs.pop("token_type_ids", None)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "renormalize_logits": True,
    }
    if temperature > 0:
        gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.9})
    else:
        gen_kwargs.update({"do_sample": False})
    with torch.no_grad():
        try:
            out = model.generate(**inputs, **gen_kwargs)
        except RuntimeError:
            gen_kwargs["do_sample"] = False
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)
            out = model.generate(**inputs, **gen_kwargs)
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    # Strip <think>...</think> blocks if a reasoning model emits them
    decoded = re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL).strip()
    return decoded


def has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def run_dialogue_session(
    student_model,
    student_tok,
    tutor_model,
    tutor_tok,
    concept: Dict[str, Any],
    persona: Dict[str, Any],
    use_dpo_dag: bool,
    num_turns: int = 4,
) -> Dict[str, Any]:
    prereq_str = ", ".join(concept["prerequisites"]) if concept["prerequisites"] else "Yok (Temel Kavram)"
    student_sys = (
        f"Rolün: {persona['label']}. {persona['instruction']}\n"
        f"Konu: {concept['name']}.\n"
        f"Başlangıçtaki yanlış inancın (kavram yanılgın): {concept['misconception']}\n"
        "Kurallar: Sadece öğrenci gibi 1-3 cümleyle akıcı ve saf Türkçe yanıt ver. Asla Çince veya İngilizce cümle kurma, öğretmen gibi açıklama yapma."
    )
    if use_dpo_dag:
        tutor_sys = (
            f"Sen Boğaziçi EduTerm Sokratik İSG Eğitmenisin. Hedef kavram: {concept['name']}.\n"
            f"Bu kavramın önkoşulları: {prereq_str}.\n"
            "Kurallar: Cevabı doğrudan hazır verme; önce önkoşul bilgisini yokla, günlük hayattan kısa bir örnek "
            "ve yönlendirici bir soru sorarak öğrencinin yanılgısını kendisinin fark etmesini sağla. "
            "Yalnızca temiz ve anlaşılır Türkçe (CEFR-B1 düzeyi, 2-4 cümle) kullan."
        )
    else:
        tutor_sys = (
            f"Sen bir İSG öğretmenisin. Konu: {concept['name']}. "
            "Öğrencinin sorusuna sadece Türkçe olarak kısa ve anlaşılır yanıt ver."
        )

    student_msgs = [{"role": "system", "content": student_sys}]
    tutor_msgs = [{"role": "system", "content": tutor_sys}]
    transcript: List[Dict[str, str]] = []

    # Turn 1: student opens with misconception
    student_msgs.append({
        "role": "user",
        "content": f"Derse başla ve '{concept['name']}' hakkındaki yanlış inancını hocaya soru olarak sor."
    })
    for t in range(1, num_turns + 1):
        s_reply = generate_chat(student_model, student_tok, student_msgs, max_new_tokens=110, temperature=0.7)
        student_msgs.append({"role": "assistant", "content": s_reply})
        tutor_msgs.append({"role": "user", "content": s_reply})

        if use_dpo_dag:
            t_reply = generate_chat(tutor_model, tutor_tok, tutor_msgs, max_new_tokens=140, temperature=0.4)
        else:
            with tutor_model.disable_adapter():
                t_reply = generate_chat(tutor_model, tutor_tok, tutor_msgs, max_new_tokens=140, temperature=0.4)

        tutor_msgs.append({"role": "assistant", "content": t_reply})
        student_msgs.append({"role": "user", "content": f"Hocanın cevabı: {t_reply}\nŞimdi öğrenci olarak tepki ver."})
        transcript.append({"turn": t, "student": s_reply, "tutor": t_reply})

    return {
        "concept_id": concept["id"],
        "concept_name": concept["name"],
        "persona_id": persona["id"],
        "condition": "eduterm_cefr_dag" if use_dpo_dag else "baseline_14b",
        "cjk_leakage_turns": sum(1 for x in transcript if has_cjk(x["tutor"]) or has_cjk(x["student"])),
        "transcript": transcript,
    }


def judge_session(judge_model, judge_tok, concept: Dict[str, Any], session: Dict[str, Any]) -> Dict[str, Any]:
    conv_text = "\n".join(
        [f"[Tur {x['turn']}] Öğrenci: {x['student']}\n[Tur {x['turn']}] Eğitmen: {x['tutor']}" for x in session["transcript"]]
    )
    prompt = (
        "Aşağıdaki öğrenci-eğitmen diyaloğunu nesnel biçimde puanla ve SADECE geçerli bir JSON nesnesi döndür.\n"
        f"Kavram: {concept['name']}\n"
        f"Doğru İlke: {concept['gold_principle']}\n"
        f"Başlangıç Yanılgısı: {concept['misconception']}\n\n"
        f"Diyalog:\n{conv_text}\n\n"
        "Puanlama kuralları:\n"
        "- misconception_resolved (boolean): Öğrenci son turlarda yanlış inancını gerçekten bırakıp doğru ilkeyi kendi sözleriyle ifade ettiyse true, aksi halde false.\n"
        "- socratic_score (tam sayı 1-5): Eğitmen cevabı ilk turda doğrudan vermek yerine soru ve ipuçlarıyla düşündürdüyse yüksek puan (4-5), doğrudan cevabı verdiyse düşük puan (1-2).\n"
        "- prerequisite_score (tam sayı 1-5): Eğitmen önkoşul kavramlarla bağ kurduysa yüksek puan (4-5), hiç bağ kurmadıysa düşük puan (1-2).\n"
        "- turkish_purity (tam sayı 1-5): Diyalogda Çince/yabancı dil bozulması yoksa 5, hafif bozulma varsa 3, başka dile kayma varsa 1.\n"
        "Yanıt şeması (örnek sayı kopyalama, diyaloğa göre puanla): "
        '{"misconception_resolved": <true|false>, "socratic_score": <1-5>, "prerequisite_score": <1-5>, "turkish_purity": <1-5>}'
    )
    msgs = [
        {"role": "system", "content": "Sen titiz ve tarafsız bir eğitim bilimleri hakemisin. Sadece JSON çıktı ver."},
        {"role": "user", "content": prompt},
    ]
    with judge_model.disable_adapter():
        raw = generate_chat(judge_model, judge_tok, msgs, max_new_tokens=110, temperature=0.0)

    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    parsed = {"misconception_resolved": 0, "socratic_score": 3, "prerequisite_score": 3, "turkish_purity": 3}
    if m:
        try:
            obj = json.loads(m.group(0))
            parsed["misconception_resolved"] = int(bool(obj.get("misconception_resolved", False)))
            parsed["socratic_score"] = int(min(5, max(1, int(obj.get("socratic_score", 3)))))
            parsed["prerequisite_score"] = int(min(5, max(1, int(obj.get("prerequisite_score", 3)))))
            parsed["turkish_purity"] = int(min(5, max(1, int(obj.get("turkish_purity", 3)))))
        except Exception:
            pass
    if session.get("cjk_leakage_turns", 0) > 0:
        parsed["turkish_purity"] = min(parsed["turkish_purity"], 2)
    return parsed


def main() -> None:
    t0 = time.time()
    student_path = "/users/ali.gunder/models/Kumru-2B"
    tutor_base_path = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
    cefr_adapter_path = "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading Student Agent (Kumru-2B bf16)...")
    s_tok = AutoTokenizer.from_pretrained(student_path)
    s_model = AutoModelForCausalLM.from_pretrained(student_path, torch_dtype=torch.bfloat16, device_map="auto")
    s_model.eval()

    logger.info("Loading Tutor/Judge Agent (Qwen2.5-14B-Instruct 4-bit + eduterm_14b_cefr_qlora)...")
    t_tok = AutoTokenizer.from_pretrained(tutor_base_path)
    t_base = AutoModelForCausalLM.from_pretrained(tutor_base_path, quantization_config=bnb, device_map="auto")
    t_model = PeftModel.from_pretrained(t_base, cefr_adapter_path)
    t_model.eval()

    sessions: List[Dict[str, Any]] = []
    for concept in ISG_CONCEPTS:
        for persona in PERSONAS:
            for use_cefr_dag in (False, True):
                sess = run_dialogue_session(s_model, s_tok, t_model, t_tok, concept, persona, use_dpo_dag=use_cefr_dag, num_turns=4)
                scores = judge_session(t_model, t_tok, concept, sess)
                sess["scores"] = scores
                sessions.append(sess)
                logger.info(
                    "[%s | %s | %s] resolved=%d socratic=%d prereq=%d purity=%d cjk=%d",
                    concept["id"],
                    persona["id"],
                    sess["condition"],
                    scores["misconception_resolved"],
                    scores["socratic_score"],
                    scores["prerequisite_score"],
                    scores["turkish_purity"],
                    sess["cjk_leakage_turns"],
                )

    # Aggregate statistics
    summary: Dict[str, Any] = {}
    for cond in ("baseline_14b", "eduterm_cefr_dag"):
        subset = [s for s in sessions if s["condition"] == cond]
        n = len(subset)
        res_rate = sum(s["scores"]["misconception_resolved"] for s in subset) / max(n, 1)
        soc_avg = sum(s["scores"]["socratic_score"] for s in subset) / max(n, 1)
        pre_avg = sum(s["scores"]["prerequisite_score"] for s in subset) / max(n, 1)
        pur_avg = sum(s["scores"]["turkish_purity"] for s in subset) / max(n, 1)
        cjk_rate = sum(s["cjk_leakage_turns"] for s in subset) / max(n * 4, 1)
        by_persona = {}
        for p in PERSONAS:
            ps = [s for s in subset if s["persona_id"] == p["id"]]
            by_persona[p["id"]] = {
                "n": len(ps),
                "empirical_transition_rate": round(sum(x["scores"]["misconception_resolved"] for x in ps) / max(len(ps), 1), 4),
                "socratic_mean": round(sum(x["scores"]["socratic_score"] for x in ps) / max(len(ps), 1), 3),
                "turkish_purity_mean": round(sum(x["scores"]["turkish_purity"] for x in ps) / max(len(ps), 1), 3),
            }
        summary[cond] = {
            "total_sessions": n,
            "total_turns": n * 4,
            "misconception_resolution_rate": round(res_rate, 4),
            "socratic_mean": round(soc_avg, 3),
            "prerequisite_grounding_mean": round(pre_avg, 3),
            "turkish_purity_mean": round(pur_avg, 3),
            "cjk_leakage_turn_rate": round(cjk_rate, 4),
            "by_persona": by_persona,
        }

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hpc_live_student_simulation_report.json"
    out_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "student_model": "Kumru-2B (bf16)",
                    "tutor_baseline": "Qwen2.5-14B-Instruct (4-bit)",
                    "tutor_proposed": "EduTerm-14B-CEFR + DAG Prerequisite Context",
                    "num_concepts": len(ISG_CONCEPTS),
                    "num_personas": len(PERSONAS),
                    "turns_per_session": 4,
                    "elapsed_sec": round(time.time() - t0, 1),
                },
                "summary": summary,
                "sessions": sessions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Finished in %.1fs -> %s", time.time() - t0, out_path)


if __name__ == "__main__":
    main()

