# -*- coding: utf-8 -*-
"""
Option 2: Stronger Student Agent (Qwen2.5-7B-Instruct, CJK-blocked) + 4-Arm Paired Evaluation
including CJK-blocked EduTerm-14B-Socratic-V2.

Arms (36 paired sessions each = 144 sessions, 576 turns):
  1. no_dag_base_14b
  2. prompt_dag_rag_base_14b
  3. task_separated_hybrid (eduterm_14b_cefr_qlora card + Base 14B chat)
  4. socratic_v2_cjk_blocked (eduterm_14b_socratic_v2 LoRA with CJKBlockLogitsProcessor)

Outputs:
  results/hpc_7b_student_socratic_report.json
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_hpc_cjk_blocked_dag_ablation import (
    CJKBlockLogitsProcessor,
    bootstrap_ci,
    build_cjk_token_ids,
    exact_mcnemar_p,
    generate_chat_nocjk,
    judge_session_blinded,
)
from run_hpc_live_student_simulation import ISG_CONCEPTS, PERSONAS, has_cjk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Opt2Q7BStudent")


def generate_cefr_card(tutor_model, tutor_tok, cjk_proc: CJKBlockLogitsProcessor, concept: Dict[str, Any]) -> str:
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
    tutor_model.set_adapter("cefr")
    return generate_chat_nocjk(tutor_model, tutor_tok, msgs, cjk_processor=cjk_proc, max_new_tokens=95, temperature=0.3)


def run_session(
    s_model,
    s_tok,
    s_proc: CJKBlockLogitsProcessor,
    t_model,
    t_tok,
    t_proc: CJKBlockLogitsProcessor,
    concept: Dict[str, Any],
    persona: Dict[str, Any],
    condition: str,
    num_turns: int = 4,
) -> Dict[str, Any]:
    prereq_str = ", ".join(concept["prerequisites"]) if concept["prerequisites"] else "Yok (Temel Kavram)"
    student_sys = (
        f"Sen gerçek bir üniversite öğrencisisin ({persona['label']}). {persona['instruction']}\n"
        f"Konu: {concept['name']}.\n"
        f"Başlangıçtaki yanlış inancın: {concept['misconception']}\n"
        "KURALLAR: Yalnızca birinci tekil şahısla ('Ben...') 1-2 kısa Türkçe cümle kur. "
        "Asla yönergeyi tekrarlama, senaryo yazma veya Çince/İngilizce kelime kullanma."
    )

    cefr_card = ""
    if condition == "task_separated_hybrid":
        cefr_card = generate_cefr_card(t_model, t_tok, t_proc, concept)
        tutor_sys = (
            f"Sen Sokratik bir İSG eğitmenisin. Hedef kavram: {concept['name']}.\n"
            f"DAG Önkoşulları: {prereq_str}.\n"
            f"EduTerm CEFR-B1 Önkoşul Özeti: {cefr_card}\n"
            "Kurallar: Cevabı ilk turda doğrudan verme. Önkoşul özetinden yararlanarak "
            "öğrenciye kısa, yönlendirici bir soru sor. Yalnızca temiz Türkçe (2-4 cümle) yaz."
        )
    elif condition == "prompt_dag_rag_base_14b":
        tutor_sys = (
            f"Sen Sokratik bir İSG eğitmenisin. Hedef kavram: {concept['name']}.\n"
            f"DAG Önkoşulları: {prereq_str}.\n"
            "Kurallar: Cevabı ilk turda doğrudan verme. Önkoşul kavramlarla bağ kur, kısa bir örnek ve "
            "yönlendirici bir soru sor. Yalnızca temiz Türkçe (2-4 cümle) yaz."
        )
    elif condition == "socratic_v2_cjk_blocked":
        tutor_sys = (
            f"Sen EduTerm Sokratik İSG Eğitmenisin. Hedef kavram: {concept['name']}.\n"
            f"DAG Önkoşulları: {prereq_str}.\n"
            "Kurallar: Cevabı ilk turda doğrudan verme; yönlendirici Sokratik soru sor, 2-4 cümle temiz Türkçe yaz."
        )
    else:  # no_dag_base_14b
        tutor_sys = (
            f"Sen bir İSG öğretmenisin. Konu: {concept['name']}. "
            "Öğrencinin sorusuna sadece Türkçe olarak kısa ve anlaşılır yanıt ver."
        )

    student_msgs = [
        {"role": "system", "content": student_sys},
        {"role": "user", "content": f"Derse başla ve '{concept['name']}' hakkındaki yanlış düşünceni hocaya sor."},
    ]
    tutor_msgs = [{"role": "system", "content": tutor_sys}]
    transcript: List[Dict[str, str]] = []

    for t in range(1, num_turns + 1):
        s_reply = generate_chat_nocjk(s_model, s_tok, student_msgs, cjk_processor=s_proc, max_new_tokens=100, temperature=0.7)
        student_msgs.append({"role": "assistant", "content": s_reply})
        tutor_msgs.append({"role": "user", "content": s_reply})

        if condition == "socratic_v2_cjk_blocked":
            t_model.set_adapter("socratic_v2")
            t_reply = generate_chat_nocjk(t_model, t_tok, tutor_msgs, cjk_processor=t_proc, max_new_tokens=140, temperature=0.4)
        else:
            with t_model.disable_adapter():
                t_reply = generate_chat_nocjk(t_model, t_tok, tutor_msgs, cjk_processor=t_proc, max_new_tokens=140, temperature=0.4)

        tutor_msgs.append({"role": "assistant", "content": t_reply})
        student_msgs.append({"role": "user", "content": f"Hocan şöyle dedi: {t_reply}\nÖğrenci olarak 1-2 cümleyle yanıtla."})
        transcript.append({"turn": t, "student": s_reply, "tutor": t_reply})

    return {
        "concept_id": concept["id"],
        "concept_name": concept["name"],
        "persona_id": persona["id"],
        "condition": condition,
        "cefr_prereq_card": cefr_card,
        "cjk_leakage_turns": sum(1 for x in transcript if has_cjk(x["tutor"]) or has_cjk(x["student"])),
        "tutor_question_turns": sum(1 for x in transcript if "?" in x["tutor"]),
        "transcript": transcript,
    }


def main() -> None:
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading Student Agent (Qwen2.5-7B-Instruct 4-bit, CJK-blocked)...")
    s_tok = AutoTokenizer.from_pretrained("/users/ali.gunder/models/Qwen2.5-7B-Instruct")
    s_model = AutoModelForCausalLM.from_pretrained("/users/ali.gunder/models/Qwen2.5-7B-Instruct", quantization_config=bnb, device_map="auto")
    s_model.eval()
    s_proc = CJKBlockLogitsProcessor(build_cjk_token_ids(s_tok))

    logger.info("Loading Tutor/Judge (Qwen2.5-14B-Instruct 4-bit + cefr + socratic_v2 adapters)...")
    t_tok = AutoTokenizer.from_pretrained("/users/ali.gunder/models/Qwen2.5-14B-Instruct")
    t_base = AutoModelForCausalLM.from_pretrained("/users/ali.gunder/models/Qwen2.5-14B-Instruct", quantization_config=bnb, device_map="auto")
    t_model = PeftModel.from_pretrained(t_base, "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora", adapter_name="cefr")
    t_model.load_adapter("/users/ali.gunder/eduterm-hybrid-sim/checkpoints/eduterm_14b_socratic_v2", adapter_name="socratic_v2")
    t_model.eval()
    t_proc = CJKBlockLogitsProcessor(build_cjk_token_ids(t_tok))

    conditions = ("no_dag_base_14b", "prompt_dag_rag_base_14b", "task_separated_hybrid", "socratic_v2_cjk_blocked")
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / "hpc_7b_student_socratic_partial.json"

    sessions: List[Dict[str, Any]] = []
    total = len(ISG_CONCEPTS) * len(PERSONAS) * len(conditions)

    for c_idx, concept in enumerate(ISG_CONCEPTS):
        for p_idx, persona in enumerate(PERSONAS):
            pair_seed = 42 + c_idx * 100 + p_idx * 10
            for cond in conditions:
                torch.manual_seed(pair_seed)
                random.seed(pair_seed)
                sess = run_session(s_model, s_tok, s_proc, t_model, t_tok, t_proc, concept, persona, cond, num_turns=4)
                scores = judge_session_blinded(t_model, t_tok, t_proc, concept, sess, disable_adapter=True)
                sess["scores"] = scores
                sessions.append(sess)
                logger.info(
                    "[%d/%d] [%s | %s | %s] res=%d soc=%d pre=%d pur=%d cjk=%d",
                    len(sessions), total, concept["id"], persona["id"], cond,
                    scores["misconception_resolved"], scores["socratic_score"],
                    scores["prerequisite_score"], scores["turkish_purity"], sess["cjk_leakage_turns"],
                )
            if len(sessions) % 16 == 0:
                partial_path.write_text(json.dumps({"completed": len(sessions), "sessions": sessions}, ensure_ascii=False), encoding="utf-8")

    summary: Dict[str, Any] = {}
    cmap: Dict[str, Dict[tuple, Dict[str, Any]]] = {c: {} for c in conditions}
    for s in sessions:
        cmap[s["condition"]][(s["concept_id"], s["persona_id"])] = s

    for cond in conditions:
        sub = [s for s in sessions if s["condition"] == cond]
        n = len(sub)
        res_vals = [s["scores"]["misconception_resolved"] for s in sub]
        ci_lo, ci_hi = bootstrap_ci([float(v) for v in res_vals])
        summary[cond] = {
            "total_sessions": n,
            "total_turns": n * 4,
            "misconception_resolution_rate": round(sum(res_vals) / n, 4),
            "resolution_bootstrap_95ci": [ci_lo, ci_hi],
            "socratic_mean": round(sum(s["scores"]["socratic_score"] for s in sub) / n, 3),
            "prerequisite_grounding_mean": round(sum(s["scores"]["prerequisite_score"] for s in sub) / n, 3),
            "turkish_purity_mean": round(sum(s["scores"]["turkish_purity"] for s in sub) / n, 3),
            "cjk_leakage_turn_rate": round(sum(s["cjk_leakage_turns"] for s in sub) / (n * 4), 4),
            "tutor_question_turn_rate": round(sum(s["tutor_question_turns"] for s in sub) / (n * 4), 4),
        }

    keys = sorted(cmap[conditions[0]].keys())
    paired = {}
    for c_a, c_b in [
        ("task_separated_hybrid", "no_dag_base_14b"),
        ("socratic_v2_cjk_blocked", "no_dag_base_14b"),
        ("task_separated_hybrid", "socratic_v2_cjk_blocked"),
        ("task_separated_hybrid", "prompt_dag_rag_base_14b"),
    ]:
        va = [cmap[c_a][k]["scores"]["misconception_resolved"] for k in keys]
        vb = [cmap[c_b][k]["scores"]["misconception_resolved"] for k in keys]
        diffs = [float(a - b) for a, b in zip(va, vb)]
        d_lo, d_hi = bootstrap_ci(diffs)
        paired[f"{c_a}__vs__{c_b}"] = {
            "mean_diff_pp": round(100.0 * sum(diffs) / len(diffs), 2),
            "diff_pp_95ci": [round(100.0 * d_lo, 2), round(100.0 * d_hi, 2)],
            "mcnemar": exact_mcnemar_p(va, vb),
        }

    out_path = out_dir / "hpc_7b_student_socratic_report.json"
    out_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "student_model": "Qwen2.5-7B-Instruct (4-bit NF4, CJK-blocked)",
                    "tutor_base_model": "Qwen2.5-14B-Instruct (4-bit NF4, CJK-blocked)",
                    "elapsed_sec": round(time.time() - t0, 1),
                },
                "summary": summary,
                "paired_comparisons": paired,
                "sessions": sessions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Option 2 Finished in %.1fs -> %s", time.time() - t0, out_path)


if __name__ == "__main__":
    main()
