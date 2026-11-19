# -*- coding: utf-8 -*-
"""
Experiment 12: Unseen Out-of-Domain (OOD) Transfer with Strong Qwen2.5-7B-Instruct (bfloat16)
Student Simulator on Boğaziçi HPC (A40 48GB).

Evaluates 12 OOD concepts (CS, KVKK, Banking, EE) x 3 personas x 3 conditions
= 108 sessions (432 turns) with Non-Latin blocking and Qwen2.5-7B-Instruct (bf16) judge
using `pointwise_judge`.

Outputs:
  results/hpc_ood_7b_bf16_report.json
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
)
from run_hpc_dpo_3seed_ablation import generate_cefr_card
from run_hpc_dpo_r2_nonlatin import build_nonlatin_token_ids
from run_hpc_live_student_simulation import PERSONAS
from run_hpc_ood_tutoring import OOD_CONCEPTS, run_ood_session
from run_hpc_trendyol8b_judge import pointwise_judge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp12OOD7B")


def main() -> None:
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

    s_path = "/users/ali.gunder/models/Qwen2.5-7B-Instruct"
    s_tok = AutoTokenizer.from_pretrained(s_path)
    s_model = AutoModelForCausalLM.from_pretrained(s_path, torch_dtype=torch.bfloat16, device_map="auto")
    s_model.eval()
    s_proc = CJKBlockLogitsProcessor(build_nonlatin_token_ids(s_tok))

    cefr_cards = {c["id"]: generate_cefr_card(t_model, t_tok, t_proc, c) for c in OOD_CONCEPTS}
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
                    logger.info("OOD 7B-bf16 progress: %d/108", len(sessions))

    del t_model, t_base
    gc.collect()
    torch.cuda.empty_cache()

    concept_by_id = {c["id"]: c for c in OOD_CONCEPTS}
    parse_ok = 0
    for s in sessions:
        sc, ok = pointwise_judge(s_model, s_tok, s_proc, concept_by_id[s["concept_id"]], s)
        if ok:
            parse_ok += 1
        s["scores"] = sc

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
        "metadata": {
            "student_model": "Qwen2.5-7B-Instruct (bfloat16, Non-Latin blocked)",
            "judge_parse_rate": round(parse_ok / len(sessions), 4),
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "summary": summary,
        "paired_comparisons": paired,
        "sessions": sessions,
    }
    Path("results/hpc_ood_7b_bf16_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Exp 12 OOD 7B-bf16 Summary:\n%s", json.dumps({"summary": summary, "paired_comparisons": paired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
