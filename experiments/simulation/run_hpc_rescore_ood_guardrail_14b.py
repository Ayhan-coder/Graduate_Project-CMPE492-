# -*- coding: utf-8 -*-
"""
Experiment 10: 14B Judge Re-Scoring of Saved OOD (108 sessions) & Guardrail (72 sessions)
Transcripts on Boğaziçi HPC (A40 48GB).

Avoids re-running the 5-hour simulations by re-scoring the saved transcripts from:
  - results/hpc_ood_tutoring_report.json (108 sessions)
  - results/hpc_dpo_guardrail_v3_report.json (72 sessions)
using Qwen2.5-14B-Instruct (4-bit NF4, Non-Latin blocked).

Outputs:
  results/hpc_rescore_ood_guardrail_14b_report.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_hpc_cjk_blocked_dag_ablation import (
    CJKBlockLogitsProcessor,
    bootstrap_ci,
    exact_mcnemar_p,
)
from run_hpc_dpo_r2_nonlatin import build_nonlatin_token_ids
from run_hpc_live_student_simulation import ISG_CONCEPTS
from run_hpc_ood_tutoring import OOD_CONCEPTS
from run_hpc_trendyol8b_judge import pointwise_judge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp10Rescore14B")


def rescore_dataset(
    model, tok, proc, report_path: Path, concept_by_id: Dict[str, Dict[str, Any]], pairs_to_compare: List[tuple]
) -> Dict[str, Any]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    sessions = data.get("sessions", [])
    by_cond: Dict[str, List[Dict[str, Any]]] = {}
    by_key: Dict[tuple, Dict[str, Dict[str, Any]]] = {}
    parse_ok = 0

    for idx, s in enumerate(sessions, start=1):
        sc, ok = pointwise_judge(model, tok, proc, concept_by_id[s["concept_id"]], s)
        if ok:
            parse_ok += 1
        item = {
            "concept_id": s["concept_id"],
            "persona_id": s["persona_id"],
            "condition": s["condition"],
            "nonlatin_leakage_turns": s.get("nonlatin_leakage_turns", 0),
            "tutor_question_turns": s.get("tutor_question_turns", 0),
            "scores": sc,
        }
        by_cond.setdefault(s["condition"], []).append(item)
        by_key.setdefault((s["concept_id"], s["persona_id"]), {})[s["condition"]] = item
        if idx % 36 == 0 or idx == len(sessions):
            logger.info("[%s] Rescored %d/%d (parse_ok=%d)", report_path.name, idx, len(sessions), parse_ok)

    summary: Dict[str, Any] = {}
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
    for a, b in pairs_to_compare:
        a_vals = [cmap[a]["scores"]["misconception_resolved"] for cmap in by_key.values() if a in cmap and b in cmap]
        b_vals = [cmap[b]["scores"]["misconception_resolved"] for cmap in by_key.values() if a in cmap and b in cmap]
        paired[f"{a}__vs__{b}"] = exact_mcnemar_p(a_vals, b_vals)

    return {
        "total_sessions": len(sessions),
        "parse_success_rate": round(parse_ok / max(len(sessions), 1), 4),
        "summary": summary,
        "paired_comparisons": paired,
    }


def main() -> None:
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    judge_path = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
    tok = AutoTokenizer.from_pretrained(judge_path)
    model = AutoModelForCausalLM.from_pretrained(judge_path, quantization_config=bnb, device_map="auto")
    model.eval()
    proc = CJKBlockLogitsProcessor(build_nonlatin_token_ids(tok))

    ood_res = rescore_dataset(
        model, tok, proc,
        Path("results/hpc_ood_tutoring_report.json"),
        {c["id"]: c for c in OOD_CONCEPTS},
        [("onpolicy_dpo_v2", "task_separated_hybrid"), ("onpolicy_dpo_v2", "prompt_dag_rag_base_14b")],
    )
    guard_res = rescore_dataset(
        model, tok, proc,
        Path("results/hpc_dpo_guardrail_v3_report.json"),
        {c["id"]: c for c in ISG_CONCEPTS},
        [("dpo_v3_guardrail", "onpolicy_dpo_v2")],
    )

    out = {
        "metadata": {"judge_model": "Qwen2.5-14B-Instruct (4-bit NF4, Non-Latin blocked)", "elapsed_sec": round(time.time() - t0, 1)},
        "ood_tutoring_14b_judge": ood_res,
        "guardrail_v3_live_14b_judge": guard_res,
    }
    Path("results/hpc_rescore_ood_guardrail_14b_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Exp 10 Rescore Report:\n%s", json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
