# -*- coding: utf-8 -*-
"""
Experiment 21: Additional Local HPC Judges on the Exact Same 600-Sample
Controlled Negative-Class Benchmark (`results/hpc_judge_negatives_600_dataset.json`).

Evaluates:
  1. Kumru-2B (bfloat16, Turkish-native 2B — cross-family vs Qwen)
  2. Qwen2.5-1.5B-Instruct (bfloat16 — small-scale boundary)
  3. EduTerm-14B-DPO-v2 (4-bit NF4 — checks whether pedagogical DPO improves/degrades judging)

Outputs:
  results/hpc_judge_negatives_600_extra_report.json
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_hpc_alignment_tax_cefr import resolve_dpo_v2_path
from run_hpc_cjk_blocked_dag_ablation import CJKBlockLogitsProcessor
from run_hpc_dpo_r2_nonlatin import build_nonlatin_token_ids
from run_hpc_judge_negatives_600 import compute_metrics, eval_judge, judge_explanation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp21Neg600Extra")


def eval_dpo_v2_judge(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    base_path = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
    tok = AutoTokenizer.from_pretrained(base_path)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    base = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb, device_map="auto")
    model = PeftModel.from_pretrained(base, resolve_dpo_v2_path(), adapter_name="dpo_v2")
    model.set_adapter("dpo_v2")
    model.eval()
    proc = CJKBlockLogitsProcessor(build_nonlatin_token_ids(tok))

    preds: List[Tuple[int, bool]] = []
    for idx, item in enumerate(dataset, start=1):
        preds.append(judge_explanation(model, tok, proc, item))
        if idx % 100 == 0:
            logger.info("[EduTerm-14B-DPO-v2] Scored %d/%d", idx, len(dataset))

    res = compute_metrics(dataset, preds)
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main() -> None:
    t0 = time.time()
    dataset = json.loads(Path("results/hpc_judge_negatives_600_dataset.json").read_text(encoding="utf-8"))

    results = {
        "Kumru-2B_bf16": eval_judge("Kumru-2B", "/users/ali.gunder/models/Kumru-2B", False, dataset),
        "Qwen2.5-1.5B-Instruct_bf16": eval_judge("Qwen2.5-1.5B-Instruct", "/users/ali.gunder/models/Qwen2.5-1.5B-Instruct", False, dataset),
        "EduTerm-14B-DPO-v2_nf4": eval_dpo_v2_judge(dataset),
    }
    report = {
        "metadata": {"n_items": len(dataset), "elapsed_sec": round(time.time() - t0, 1)},
        "results": results,
    }
    Path("results/hpc_judge_negatives_600_extra_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Exp 21 Extra Judges Report:\n%s", json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
