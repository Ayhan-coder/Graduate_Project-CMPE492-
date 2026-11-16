# -*- coding: utf-8 -*-
"""
Task-Separated Hybrid Socratic Ablation on Boğaziçi HPC (NVIDIA A40 48GB).

Tests the architectural separation discovered in Job 34699:
- Condition A (`no_dag_base_14b`): Base Qwen2.5-14B-Instruct without DAG prerequisite context
  or EduTerm-14B-CEFR scaffolding.
- Condition B (`task_separated_hybrid`):
    1. Specialized `EduTerm-14B-CEFR` adapter generates a concise CEFR-B1 prerequisite
       micro-explanation card for the target concept's DAG parents.
    2. Conversational `Base Qwen2.5-14B-Instruct` (adapter disabled) uses that card to
       conduct multi-turn Socratic guidance without SFT template overfitting.

Student Agent: Kumru-2B (bf16) across 12 İSG concepts x 3 personas (72 sessions, 288 turns).
Outputs: results/hpc_task_separated_ablation_report.json
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_hpc_live_student_simulation import (
    ISG_CONCEPTS,
    PERSONAS,
    generate_chat,
    has_cjk,
    judge_session,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TaskSeparatedAblation")


def generate_cefr_prereq_card(tutor_model, tutor_tok, concept: Dict[str, Any]) -> str:
    """Use specialized EduTerm-14B-CEFR adapter to generate a CEFR-B1 prerequisite card."""
    prereqs = concept["prerequisites"]
    prereq_str = ", ".join(prereqs) if prereqs else concept["name"]
    msgs = [
        {
            "role": "system",
            "content": "Sen EduTerm CEFR-B1 eğitmenisin. Verilen önkoşul kavramı öğrenciye 2 kısa, sade Türkçe cümleyle özetle.",
        },
        {
            "role": "user",
            "content": (
                f"Hedef kavram: {concept['name']}\n"
                f"Önkoşul kavram(lar): {prereq_str}\n"
                "Bu önkoşul kavramı CEFR-B1 düzeyinde, günlük bir örnekle 2 cümlede açıkla."
            ),
        },
    ]
    return generate_chat(tutor_model, tutor_tok, msgs, max_new_tokens=95, temperature=0.3)


def run_ablation_session(
    student_model,
    student_tok,
    tutor_model,
    tutor_tok,
    concept: Dict[str, Any],
    persona: Dict[str, Any],
    use_task_separated_hybrid: bool,
    num_turns: int = 4,
) -> Dict[str, Any]:
    prereq_str = ", ".join(concept["prerequisites"]) if concept["prerequisites"] else "Yok (Temel Kavram)"
    student_sys = (
        f"Sen gerçek bir öğrencisin ({persona['label']}). {persona['instruction']}\n"
        f"Konu: {concept['name']}.\n"
        f"Başlangıçtaki yanlış inancın: {concept['misconception']}\n"
        "ÖNEMLİ: Sadece kendi ağzından ('Ben...' diyerek) 1-2 cümlelik kısa Türkçe yanıt ver. "
        "Asla 'Öğretmen:' veya 'Öğrenci:' diye senaryo yazma, Çince/İngilizce kullanma."
    )

    cefr_card = ""
    if use_task_separated_hybrid:
        cefr_card = generate_cefr_prereq_card(tutor_model, tutor_tok, concept)
        tutor_sys = (
            f"Sen Sokratik bir İSG eğitmenisin. Hedef kavram: {concept['name']}.\n"
            f"DAG Önkoşulları: {prereq_str}.\n"
            f"EduTerm CEFR-B1 Önkoşul Özeti: {cefr_card}\n"
            "Kurallar: Cevabı ilk turda doğrudan verme. Yukarıdaki önkoşul özetinden yararlanarak "
            "öğrenciye kısa, yönlendirici bir soru sor ve kavram yanılgısını kendisinin keşfetmesini sağla. "
            "Yalnızca temiz Türkçe (2-4 cümle) yaz."
        )
    else:
        tutor_sys = (
            f"Sen bir İSG öğretmenisin. Konu: {concept['name']}. "
            "Öğrencinin sorusuna sadece Türkçe olarak kısa ve anlaşılır yanıt ver."
        )

    student_msgs = [{"role": "system", "content": student_sys}]
    tutor_msgs = [{"role": "system", "content": tutor_sys}]
    transcript: List[Dict[str, str]] = []

    student_msgs.append({
        "role": "user",
        "content": f"Hocaya '{concept['name']}' hakkındaki yanlış düşünceni doğrudan 1-2 cümleyle sor.",
    })

    for t in range(1, num_turns + 1):
        s_reply = generate_chat(student_model, student_tok, student_msgs, max_new_tokens=100, temperature=0.7)
        student_msgs.append({"role": "assistant", "content": s_reply})
        tutor_msgs.append({"role": "user", "content": s_reply})

        # Both conditions use conversational Base 14B (adapter disabled) for multi-turn dialogue
        with tutor_model.disable_adapter():
            t_reply = generate_chat(tutor_model, tutor_tok, tutor_msgs, max_new_tokens=140, temperature=0.4)

        tutor_msgs.append({"role": "assistant", "content": t_reply})
        student_msgs.append({"role": "user", "content": f"Hocan şöyle dedi: {t_reply}\nÖğrenci olarak 1-2 cümleyle yanıtla."})
        transcript.append({"turn": t, "student": s_reply, "tutor": t_reply})

    return {
        "concept_id": concept["id"],
        "concept_name": concept["name"],
        "persona_id": persona["id"],
        "condition": "task_separated_hybrid" if use_task_separated_hybrid else "no_dag_base_14b",
        "cefr_prereq_card": cefr_card,
        "cjk_leakage_turns": sum(1 for x in transcript if has_cjk(x["tutor"]) or has_cjk(x["student"])),
        "transcript": transcript,
    }


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
            for use_hybrid in (False, True):
                sess = run_ablation_session(
                    s_model, s_tok, t_model, t_tok, concept, persona,
                    use_task_separated_hybrid=use_hybrid, num_turns=4,
                )
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

    summary: Dict[str, Any] = {}
    for cond in ("no_dag_base_14b", "task_separated_hybrid"):
        subset = [s for s in sessions if s["condition"] == cond]
        n = len(subset)
        by_persona = {}
        for p in PERSONAS:
            ps = [s for s in subset if s["persona_id"] == p["id"]]
            by_persona[p["id"]] = {
                "n": len(ps),
                "empirical_transition_rate": round(sum(x["scores"]["misconception_resolved"] for x in ps) / max(len(ps), 1), 4),
                "socratic_mean": round(sum(x["scores"]["socratic_score"] for x in ps) / max(len(ps), 1), 3),
                "prerequisite_mean": round(sum(x["scores"]["prerequisite_score"] for x in ps) / max(len(ps), 1), 3),
            }
        summary[cond] = {
            "total_sessions": n,
            "total_turns": n * 4,
            "misconception_resolution_rate": round(sum(s["scores"]["misconception_resolved"] for s in subset) / max(n, 1), 4),
            "socratic_mean": round(sum(s["scores"]["socratic_score"] for s in subset) / max(n, 1), 3),
            "prerequisite_grounding_mean": round(sum(s["scores"]["prerequisite_score"] for s in subset) / max(n, 1), 3),
            "turkish_purity_mean": round(sum(s["scores"]["turkish_purity"] for s in subset) / max(n, 1), 3),
            "cjk_leakage_turn_rate": round(sum(s["cjk_leakage_turns"] for s in subset) / max(n * 4, 1), 4),
            "by_persona": by_persona,
        }

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hpc_task_separated_ablation_report.json"
    out_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "student_model": "Kumru-2B (bf16)",
                    "condition_a": "no_dag_base_14b (Qwen2.5-14B-Instruct)",
                    "condition_b": "task_separated_hybrid (EduTerm-14B-CEFR prereq card + Qwen2.5-14B-Instruct Socratic chat)",
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

