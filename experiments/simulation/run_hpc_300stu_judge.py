# -*- coding: utf-8 -*-
"""
Experiment 27: 14B Rubric Scoring + Spearman Correlation on the 300-Session
Dynamic Student Simulation (`results/hpc_300_synthetic_student_report.json`).

Scores all 300 saved sessions with Qwen2.5-14B-Instruct (4-bit NF4, Non-Latin blocked)
using `pointwise_judge`, and computes Spearman correlation between rubric dimensions
(socratic_score, prerequisite_score, turkish_purity) and independent post-test success.

Outputs:
  results/hpc_300stu_14b_rubric_report.json
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_hpc_cjk_blocked_dag_ablation import CJKBlockLogitsProcessor, bootstrap_ci
from run_hpc_dpo_r2_nonlatin import build_nonlatin_token_ids
from run_hpc_live_student_simulation import ISG_CONCEPTS
from run_hpc_ood_tutoring import OOD_CONCEPTS
from run_hpc_trendyol8b_judge import pointwise_judge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp27_300StuJudge")


def spearman(x: List[float], y: List[float]) -> float:
    def rvec(v: List[float]) -> List[float]:
        idx = sorted(enumerate(v), key=lambda t: t[1])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and idx[j + 1][1] == idx[i][1]:
                j += 1
            ar = (i + 1 + j + 1) / 2.0
            for k in range(i, j + 1):
                r[idx[k][0]] = ar
            i = j + 1
        return r
    rx, ry = rvec(x), rvec(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(num / den, 4) if den > 0 else 0.0


def main() -> None:
    t0 = time.time()
    all_by_id = {c["id"]: c for c in (ISG_CONCEPTS + OOD_CONCEPTS)}
    data = json.loads(Path("results/hpc_300_synthetic_student_report.json").read_text(encoding="utf-8"))
    sessions = data["sessions"]

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    judge_path = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
    tok = AutoTokenizer.from_pretrained(judge_path)
    model = AutoModelForCausalLM.from_pretrained(judge_path, quantization_config=bnb, device_map="auto")
    model.eval()
    proc = CJKBlockLogitsProcessor(build_nonlatin_token_ids(tok))

    parse_ok = 0
    by_cond: Dict[str, List[Dict[str, Any]]] = {}
    soc_all, pre_all, pur_all, pt_all = [], [], [], []

    for idx, s in enumerate(sessions, start=1):
        sc, ok = pointwise_judge(model, tok, proc, all_by_id[s["concept_id"]], s)
        if ok:
            parse_ok += 1
        rec = {**sc, "posttest_correct": s["posttest_correct"]}
        by_cond.setdefault(s["condition"], []).append(rec)
        soc_all.append(float(sc["socratic_score"]))
        pre_all.append(float(sc["prerequisite_score"]))
        pur_all.append(float(sc["turkish_purity"]))
        pt_all.append(float(s["posttest_correct"]))
        if idx % 60 == 0:
            logger.info("Scored %d/%d (parse_ok=%d)", idx, len(sessions), parse_ok)

    summary = {}
    for cond, items in by_cond.items():
        n = len(items)
        res = [x["misconception_resolved"] for x in items]
        summary[cond] = {
            "n": n,
            "14b_resolution_rate": round(sum(res) / n, 4),
            "resolution_95ci": list(bootstrap_ci(res)),
            "socratic_mean": round(sum(x["socratic_score"] for x in items) / n, 3),
            "prerequisite_mean": round(sum(x["prerequisite_score"] for x in items) / n, 3),
            "turkish_purity_mean": round(sum(x["turkish_purity"] for x in items) / n, 3),
        }

    report = {
        "metadata": {"n_sessions": len(sessions), "parse_success_rate": round(parse_ok / len(sessions), 4), "elapsed_sec": round(time.time() - t0, 1)},
        "summary_by_condition": summary,
        "spearman_vs_posttest": {
            "socratic_score": spearman(soc_all, pt_all),
            "prerequisite_score": spearman(pre_all, pt_all),
            "turkish_purity": spearman(pur_all, pt_all),
        },
    }
    Path("results/hpc_300stu_14b_rubric_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Exp 27 Summary:\n%s", json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
