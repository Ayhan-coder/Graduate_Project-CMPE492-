# -*- coding: utf-8 -*-
"""
Qwen2.5-7B-Instruct (torch.bfloat16, unquantized) Cross-Judge Validation
on the 324 CJK-free sessions from Job 34740 (results/hpc_cjk_blocked_dag_ablation_report.json).

Fixes the 4-bit fp16 numeric instability observed in Jobs 34741 and 34742.
Runs pointwise scoring (324 sessions) + blinded randomized A/B pairwise comparison (324 pairs).

Outputs:
  results/hpc_7b_bf16_cross_judge_report.json
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_hpc_cjk_blocked_dag_ablation import (
    CJKBlockLogitsProcessor,
    build_cjk_token_ids,
    cohens_kappa,
)
from run_hpc_cross_judge_pairwise import (
    pairwise_judge,
    pointwise_judge,
    two_sided_sign_p,
)
from run_hpc_live_student_simulation import ISG_CONCEPTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CrossJudge7BBF16")


def main() -> None:
    t0 = time.time()
    src_path = Path("results/hpc_cjk_blocked_dag_ablation_report.json")
    data = json.loads(src_path.read_text(encoding="utf-8"))
    sessions = data["sessions"]
    concept_by_id = {c["id"]: c for c in ISG_CONCEPTS}

    logger.info("Loading Qwen2.5-7B-Instruct in torch.bfloat16 (unquantized)...")
    j7_tok = AutoTokenizer.from_pretrained("/users/ali.gunder/models/Qwen2.5-7B-Instruct")
    j7_model = AutoModelForCausalLM.from_pretrained(
        "/users/ali.gunder/models/Qwen2.5-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    j7_model.eval()
    j7_proc = CJKBlockLogitsProcessor(build_cjk_token_ids(j7_tok))

    parse_ok = 0
    for idx, s in enumerate(sessions, start=1):
        sc, ok = pointwise_judge(j7_model, j7_tok, j7_proc, concept_by_id[s["concept_id"]], s)
        s["scores_7b_bf16"] = sc
        parse_ok += int(ok)
        if idx % 54 == 0:
            logger.info("7B-bf16 pointwise progress: %d/%d (parse_ok=%d)", idx, len(sessions), parse_ok)

    by_cond_map: Dict[str, Dict[Tuple[int, str, str], Dict[str, Any]]] = {}
    for s in sessions:
        by_cond_map.setdefault(s["condition"], {})[(s["seed"], s["concept_id"], s["persona_id"])] = s
    keys = sorted(by_cond_map["no_dag_base_14b"].keys())
    pairs = [
        ("task_separated_hybrid", "no_dag_base_14b"),
        ("task_separated_hybrid", "prompt_dag_rag_base_14b"),
        ("prompt_dag_rag_base_14b", "no_dag_base_14b"),
    ]

    rng = random.Random(2026)
    pw_7b: Dict[str, Dict[str, Any]] = {}
    for c_a, c_b in pairs:
        wa, wb, ties = 0, 0, 0
        for k in keys:
            res = pairwise_judge(
                j7_model, j7_tok, j7_proc, concept_by_id[k[1]], by_cond_map[c_a][k], by_cond_map[c_b][k], rng
            )
            if res == "A":
                wa += 1
            elif res == "B":
                wb += 1
            else:
                ties += 1
        pw_7b[f"{c_a}__vs__{c_b}"] = {"wins_a": wa, "wins_b": wb, "ties": ties, "sign_p": two_sided_sign_p(wa, wb)}
    logger.info("7B-bf16 pairwise: %s", json.dumps(pw_7b, ensure_ascii=False))

    pointwise_7b_summary = {}
    for cond in ("no_dag_base_14b", "prompt_dag_rag_base_14b", "task_separated_hybrid"):
        sub = [s for s in sessions if s["condition"] == cond]
        n = len(sub)
        pointwise_7b_summary[cond] = {
            "n": n,
            "resolution_rate_7b_bf16": round(sum(x["scores_7b_bf16"]["misconception_resolved"] for x in sub) / n, 4),
            "socratic_mean_7b_bf16": round(sum(x["scores_7b_bf16"]["socratic_score"] for x in sub) / n, 3),
            "prerequisite_mean_7b_bf16": round(sum(x["scores_7b_bf16"]["prerequisite_score"] for x in sub) / n, 3),
        }

    y14 = [s["scores"]["misconception_resolved"] for s in sessions]
    y7 = [s["scores_7b_bf16"]["misconception_resolved"] for s in sessions]
    inter_judge = {
        "n_sessions": len(sessions),
        "parse_ok_7b_bf16": parse_ok,
        "raw_agreement": round(sum(1 for a, b in zip(y14, y7) if a == b) / len(sessions), 4),
        "cohens_kappa": cohens_kappa(y14, y7),
    }

    out_path = Path("results/hpc_7b_bf16_cross_judge_report.json")
    out_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "judge_model": "Qwen2.5-7B-Instruct (torch.bfloat16, CJK-blocked)",
                    "source_report": str(src_path),
                    "elapsed_sec": round(time.time() - t0, 1),
                },
                "pointwise_7b_bf16_summary": pointwise_7b_summary,
                "inter_judge_14b_vs_7b_bf16": inter_judge,
                "pairwise_7b_bf16": pw_7b,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Finished in %.1fs -> %s", time.time() - t0, out_path)


if __name__ == "__main__":
    main()
