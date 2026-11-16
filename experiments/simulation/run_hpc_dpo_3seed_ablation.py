# -*- coding: utf-8 -*-
"""
Experiment 1: 3-Seed (n=108/arm) Validation of On-Policy DPO vs. Task-Separated Hybrid vs. Prompt DAG-RAG
with CJKBlockLogitsProcessor + Qwen2.5-7B-Instruct (bfloat16) Blinded Pairwise Cross-Judge.

Arms (3 seeds x 12 concepts x 3 personas = 108 paired sessions per arm, 324 total, 1,296 turns):
  1. prompt_dag_rag_base_14b
  2. task_separated_hybrid
  3. onpolicy_dpo (checkpoints/eduterm_14b_onpolicy_dpo)

Outputs:
  results/hpc_dpo_3seed_ablation_report.json
"""

from __future__ import annotations

import gc
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_hpc_cjk_blocked_dag_ablation import (
    SEEDS,
    CJKBlockLogitsProcessor,
    bootstrap_ci,
    build_cjk_token_ids,
    exact_mcnemar_p,
    generate_chat_nocjk,
    judge_session_blinded,
)
from run_hpc_cross_judge_pairwise import pairwise_judge, two_sided_sign_p
from run_hpc_live_student_simulation import ISG_CONCEPTS, PERSONAS, has_cjk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp1DPO3Seed")


def resolve_dpo_path() -> str:
    base = Path("checkpoints/eduterm_14b_onpolicy_dpo")
    if (base / "adapter_config.json").exists():
        return str(base)
    if (base / "dpo" / "adapter_config.json").exists():
        return str(base / "dpo")
    raise FileNotFoundError(f"Could not find adapter_config.json in {base}")


def generate_cefr_card(t_model, t_tok, t_proc: CJKBlockLogitsProcessor, concept: Dict[str, Any]) -> str:
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
    t_model.set_adapter("cefr")
    return generate_chat_nocjk(t_model, t_tok, msgs, cjk_processor=t_proc, max_new_tokens=95, temperature=0.3)


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
    seed: int,
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
    if condition == "task_separated_hybrid":
        cefr_card = generate_cefr_card(t_model, t_tok, t_proc, concept)
        tutor_sys = (
            f"Sen Sokratik bir İSG eğitmenisin. Hedef kavram: {concept['name']}.\n"
            f"DAG Önkoşulları: {prereq_str}.\n"
            f"EduTerm CEFR-B1 Önkoşul Özeti: {cefr_card}\n"
            "Kurallar: Cevabı ilk turda doğrudan verme. Yukarıdaki önkoşul özetinden yararlanarak "
            "öğrenciye kısa, yönlendirici bir soru sor. Yalnızca temiz Türkçe (2-4 cümle) yaz."
        )
    elif condition == "prompt_dag_rag_base_14b":
        tutor_sys = (
            f"Sen Sokratik bir İSG eğitmenisin. Hedef kavram: {concept['name']}.\n"
            f"DAG Önkoşulları: {prereq_str}.\n"
            "Kurallar: Cevabı ilk turda doğrudan verme. Önce yukarıdaki önkoşul kavramlarla bağ kur, "
            "günlük hayattan kısa bir örnek ve yönlendirici bir soru sor. Yalnızca temiz Türkçe (2-4 cümle) yaz."
        )
    elif condition == "onpolicy_dpo":
        tutor_sys = (
            f"Sen EduTerm Sokratik İSG Eğitmenisin. Hedef kavram: {concept['name']}.\n"
            f"DAG Önkoşulları: {prereq_str}.\n"
            "Kurallar: Cevabı ilk turda doğrudan verme; kısa, yönlendirici Sokratik soru sor ve temiz Türkçe yaz."
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")

    student_msgs = [
        {"role": "system", "content": student_sys},
        {"role": "user", "content": f"Hocaya '{concept['name']}' hakkındaki yanlış düşünceni doğrudan 1-2 cümleyle sor."},
    ]
    tutor_msgs = [{"role": "system", "content": tutor_sys}]
    transcript: List[Dict[str, str]] = []

    for t in range(1, num_turns + 1):
        s_reply = generate_chat_nocjk(s_model, s_tok, student_msgs, cjk_processor=s_proc, max_new_tokens=100, temperature=0.7)
        student_msgs.append({"role": "assistant", "content": s_reply})
        tutor_msgs.append({"role": "user", "content": s_reply})

        if condition == "onpolicy_dpo":
            t_model.set_adapter("dpo")
            t_reply = generate_chat_nocjk(t_model, t_tok, tutor_msgs, cjk_processor=t_proc, max_new_tokens=140, temperature=0.4)
        else:
            with t_model.disable_adapter():
                t_reply = generate_chat_nocjk(t_model, t_tok, tutor_msgs, cjk_processor=t_proc, max_new_tokens=140, temperature=0.4)

        tutor_msgs.append({"role": "assistant", "content": t_reply})
        student_msgs.append({"role": "user", "content": f"Hocan şöyle dedi: {t_reply}\nÖğrenci olarak 1-2 cümleyle yanıtla."})
        transcript.append({"turn": t, "student": s_reply, "tutor": t_reply})

    return {
        "seed": seed,
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

    logger.info("Loading Student Agent (Kumru-2B bf16, CJK-blocked)...")
    s_tok = AutoTokenizer.from_pretrained("/users/ali.gunder/models/Kumru-2B")
    s_model = AutoModelForCausalLM.from_pretrained("/users/ali.gunder/models/Kumru-2B", torch_dtype=torch.bfloat16, device_map="auto")
    s_model.eval()
    s_proc = CJKBlockLogitsProcessor(build_cjk_token_ids(s_tok))

    dpo_path = resolve_dpo_path()
    logger.info("Loading Tutor/Judge (Qwen2.5-14B-Instruct 4-bit + cefr + dpo=%s)...", dpo_path)
    t_tok = AutoTokenizer.from_pretrained("/users/ali.gunder/models/Qwen2.5-14B-Instruct")
    t_base = AutoModelForCausalLM.from_pretrained("/users/ali.gunder/models/Qwen2.5-14B-Instruct", quantization_config=bnb, device_map="auto")
    t_model = PeftModel.from_pretrained(t_base, "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora", adapter_name="cefr")
    t_model.load_adapter(dpo_path, adapter_name="dpo")
    t_model.eval()
    t_proc = CJKBlockLogitsProcessor(build_cjk_token_ids(t_tok))

    conditions = ("prompt_dag_rag_base_14b", "task_separated_hybrid", "onpolicy_dpo")
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / "hpc_dpo_3seed_ablation_partial.json"

    sessions: List[Dict[str, Any]] = []
    done_keys = set()
    if partial_path.exists():
        try:
            prev = json.loads(partial_path.read_text(encoding="utf-8"))
            sessions = prev.get("sessions", [])
            done_keys = {(s["seed"], s["concept_id"], s["persona_id"], s["condition"]) for s in sessions}
            logger.info("Resuming from %s (%d completed).", partial_path, len(sessions))
        except Exception:
            pass

    concept_by_id = {c["id"]: c for c in ISG_CONCEPTS}
    total = len(SEEDS) * len(ISG_CONCEPTS) * len(PERSONAS) * len(conditions)

    for seed in SEEDS:
        for c_idx, concept in enumerate(ISG_CONCEPTS):
            for p_idx, persona in enumerate(PERSONAS):
                pair_seed = seed + c_idx * 100 + p_idx * 10
                ran_any = False
                for cond in conditions:
                    key = (seed, concept["id"], persona["id"], cond)
                    if key in done_keys:
                        continue
                    ran_any = True
                    torch.manual_seed(pair_seed)
                    random.seed(pair_seed)
                    sess = run_session(s_model, s_tok, s_proc, t_model, t_tok, t_proc, concept, persona, cond, seed, num_turns=4)
                    scores = judge_session_blinded(t_model, t_tok, t_proc, concept, sess, disable_adapter=True)
                    sess["scores"] = scores
                    sessions.append(sess)
                    done_keys.add(key)
                    logger.info(
                        "[%d/%d] seed=%d [%s | %s | %s] res=%d soc=%d pre=%d pur=%d cjk=%d",
                        len(sessions), total, seed, concept["id"], persona["id"], cond,
                        scores["misconception_resolved"], scores["socratic_score"],
                        scores["prerequisite_score"], scores["turkish_purity"], sess["cjk_leakage_turns"],
                    )
                if ran_any and len(sessions) % 18 == 0:
                    partial_path.write_text(json.dumps({"completed": len(sessions), "sessions": sessions}, ensure_ascii=False), encoding="utf-8")

    # Free Student + 14B and run Qwen2.5-7B-Instruct (bfloat16) blinded pairwise cross-judge
    logger.info("Unloading 14B & Student; loading Qwen2.5-7B-Instruct (bfloat16) for blinded A/B pairwise validation...")
    del s_model, t_model, t_base
    gc.collect()
    torch.cuda.empty_cache()

    j7_tok = AutoTokenizer.from_pretrained("/users/ali.gunder/models/Qwen2.5-7B-Instruct")
    j7_model = AutoModelForCausalLM.from_pretrained("/users/ali.gunder/models/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
    j7_model.eval()
    j7_proc = CJKBlockLogitsProcessor(build_cjk_token_ids(j7_tok))

    cmap: Dict[str, Dict[Tuple[int, str, str], Dict[str, Any]]] = {c: {} for c in conditions}
    for s in sessions:
        cmap[s["condition"]][(s["seed"], s["concept_id"], s["persona_id"])] = s
    keys = sorted(cmap[conditions[0]].keys())

    rng = random.Random(2026)
    pw_7b: Dict[str, Any] = {}
    for c_a, c_b in [
        ("onpolicy_dpo", "task_separated_hybrid"),
        ("onpolicy_dpo", "prompt_dag_rag_base_14b"),
        ("task_separated_hybrid", "prompt_dag_rag_base_14b"),
    ]:
        wa, wb, ties = 0, 0, 0
        for k in keys:
            res = pairwise_judge(j7_model, j7_tok, j7_proc, concept_by_id[k[1]], cmap[c_a][k], cmap[c_b][k], rng)
            if res == "A":
                wa += 1
            elif res == "B":
                wb += 1
            else:
                ties += 1
        pw_7b[f"{c_a}__vs__{c_b}"] = {"wins_a": wa, "wins_b": wb, "ties": ties, "sign_p": two_sided_sign_p(wa, wb)}

    summary: Dict[str, Any] = {}
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

    paired = {}
    for c_a, c_b in [
        ("onpolicy_dpo", "task_separated_hybrid"),
        ("onpolicy_dpo", "prompt_dag_rag_base_14b"),
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

    out_path = out_dir / "hpc_dpo_3seed_ablation_report.json"
    out_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "student_model": "Kumru-2B (bf16, CJK-blocked)",
                    "pairwise_cross_judge": "Qwen2.5-7B-Instruct (bfloat16, CJK-blocked)",
                    "seeds": SEEDS,
                    "elapsed_sec": round(time.time() - t0, 1),
                },
                "summary": summary,
                "paired_comparisons": paired,
                "pairwise_7b_bf16": pw_7b,
                "sessions": sessions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Exp 1 Finished in %.1fs -> %s", time.time() - t0, out_path)


if __name__ == "__main__":
    main()
