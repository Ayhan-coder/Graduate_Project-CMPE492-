# -*- coding: utf-8 -*-
"""
Experiment 6: 3-Seed (n=108/arm) Validation on Stronger Qwen2.5-7B-Instruct (bfloat16) Student
with NonLatinBlockLogitsProcessor (eliminating Cyrillic & CJK script drift).

Compares `task_separated_hybrid` vs `onpolicy_dpo_v2` across 3 seeds x 12 concepts x 3 personas
(108 paired sessions per arm = 216 total sessions, 864 turns).

Outputs:
  results/hpc_7b_bf16_3seed_nonlatin_report.json
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_hpc_alignment_tax_cefr import resolve_dpo_v2_path
from run_hpc_cjk_blocked_dag_ablation import (
    SEEDS,
    CJKBlockLogitsProcessor,
    bootstrap_ci,
    exact_mcnemar_p,
    generate_chat_nocjk,
    judge_session_blinded,
)
from run_hpc_dpo_3seed_ablation import generate_cefr_card
from run_hpc_dpo_r2_nonlatin import build_nonlatin_token_ids, has_nonlatin
from run_hpc_live_student_simulation import ISG_CONCEPTS, PERSONAS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp6Q7B3Seed")


def run_session(
    s_model, s_tok, s_proc, t_model, t_tok, t_proc,
    concept: Dict[str, Any], persona: Dict[str, Any], condition: str, seed: int, num_turns: int = 4
) -> Dict[str, Any]:
    prereq_str = ", ".join(concept["prerequisites"]) if concept["prerequisites"] else "Yok (Temel Kavram)"
    student_sys = (
        f"Sen gerçek bir üniversite öğrencisisin ({persona['label']}). {persona['instruction']}\n"
        f"Konu: {concept['name']}.\n"
        f"Başlangıçtaki yanlış inancın: {concept['misconception']}\n"
        "KURALLAR: Yalnızca birinci tekil şahısla ('Ben...') 1-2 kısa Türkçe cümle kur. "
        "Asla yönergeyi tekrarlama, senaryo yazma veya yabancı dil kullanma."
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
    else:
        tutor_sys = (
            f"Sen EduTerm Sokratik İSG Eğitmenisin. Hedef kavram: {concept['name']}.\n"
            f"DAG Önkoşulları: {prereq_str}.\n"
            "Kurallar: Cevabı ilk turda doğrudan verme; kısa, yönlendirici Sokratik soru sor ve yalnızca saf Türkçe yaz."
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

        if condition == "onpolicy_dpo_v2":
            t_model.set_adapter("dpo_v2")
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
        "nonlatin_leakage_turns": sum(1 for x in transcript if has_nonlatin(x["tutor"]) or has_nonlatin(x["student"])),
        "tutor_question_turns": sum(1 for x in transcript if "?" in x["tutor"]),
        "transcript": transcript,
    }


def main() -> None:
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
    )

    logger.info("Loading Student Agent (Qwen2.5-7B-Instruct torch.bfloat16, Non-Latin blocked)...")
    s_tok = AutoTokenizer.from_pretrained("/users/ali.gunder/models/Qwen2.5-7B-Instruct")
    s_model = AutoModelForCausalLM.from_pretrained(
        "/users/ali.gunder/models/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
    )
    s_model.eval()
    s_proc = CJKBlockLogitsProcessor(build_nonlatin_token_ids(s_tok))

    dpo_v2_path = resolve_dpo_v2_path()
    logger.info("Loading Tutor/Judge (Qwen2.5-14B-Instruct 4-bit + cefr + dpo_v2=%s)...", dpo_v2_path)
    t_tok = AutoTokenizer.from_pretrained("/users/ali.gunder/models/Qwen2.5-14B-Instruct")
    t_base = AutoModelForCausalLM.from_pretrained("/users/ali.gunder/models/Qwen2.5-14B-Instruct", quantization_config=bnb, device_map="auto")
    t_model = PeftModel.from_pretrained(t_base, "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora", adapter_name="cefr")
    t_model.load_adapter(dpo_v2_path, adapter_name="dpo_v2")
    t_model.eval()
    t_proc = CJKBlockLogitsProcessor(build_nonlatin_token_ids(t_tok))

    conditions = ("task_separated_hybrid", "onpolicy_dpo_v2")
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / "hpc_7b_bf16_3seed_nonlatin_partial.json"

    sessions: List[Dict[str, Any]] = []
    done_keys = set()
    if partial_path.exists():
        try:
            prev = json.loads(partial_path.read_text(encoding="utf-8"))
            sessions = prev.get("sessions", [])
            done_keys = {(s["seed"], s["concept_id"], s["persona_id"], s["condition"]) for s in sessions}
        except Exception:
            pass

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
                        "[%d/%d] seed=%d [%s | %s | %s] res=%d soc=%d pre=%d pur=%d nonlat=%d",
                        len(sessions), total, seed, concept["id"], persona["id"], cond,
                        scores["misconception_resolved"], scores["socratic_score"],
                        scores["prerequisite_score"], scores["turkish_purity"], sess["nonlatin_leakage_turns"],
                    )
                if ran_any and len(sessions) % 18 == 0:
                    partial_path.write_text(json.dumps({"completed": len(sessions), "sessions": sessions}, ensure_ascii=False), encoding="utf-8")

    summary: Dict[str, Any] = {}
    cmap: Dict[str, Dict[Tuple[int, str, str], Dict[str, Any]]] = {c: {} for c in conditions}
    for s in sessions:
        cmap[s["condition"]][(s["seed"], s["concept_id"], s["persona_id"])] = s

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
            "nonlatin_leakage_turn_rate": round(sum(s["nonlatin_leakage_turns"] for s in sub) / (n * 4), 4),
            "tutor_question_turn_rate": round(sum(s["tutor_question_turns"] for s in sub) / (n * 4), 4),
        }

    keys = sorted(cmap[conditions[0]].keys())
    va = [cmap["onpolicy_dpo_v2"][k]["scores"]["misconception_resolved"] for k in keys]
    vb = [cmap["task_separated_hybrid"][k]["scores"]["misconception_resolved"] for k in keys]
    diffs = [float(a - b) for a, b in zip(va, vb)]
    d_lo, d_hi = bootstrap_ci(diffs)
    paired = {
        "onpolicy_dpo_v2__vs__task_separated_hybrid": {
            "mean_diff_pp": round(100.0 * sum(diffs) / len(diffs), 2),
            "diff_pp_95ci": [round(100.0 * d_lo, 2), round(100.0 * d_hi, 2)],
            "mcnemar": exact_mcnemar_p(va, vb),
        }
    }

    out_path = out_dir / "hpc_7b_bf16_3seed_nonlatin_report.json"
    out_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "student_model": "Qwen2.5-7B-Instruct (torch.bfloat16, Non-Latin blocked)",
                    "tutor_base_model": "Qwen2.5-14B-Instruct (4-bit NF4, Non-Latin blocked)",
                    "seeds": SEEDS,
                    "elapsed_sec": round(time.time() - t0, 1),
                },
                "summary": summary,
                "paired_comparison": paired,
                "sessions": sessions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Exp 6 Finished in %.1fs -> %s", time.time() - t0, out_path)


if __name__ == "__main__":
    main()
