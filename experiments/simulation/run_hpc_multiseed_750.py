# -*- coding: utf-8 -*-
"""
Experiment 23 (P1): 50-Prompt x 3-Model x 5-Seed = 750-Generation Stochastic Stability Benchmark
on Boğaziçi HPC (A40 48GB).

Evaluates stochastic decoding stability (temperature=0.4, do_sample=True,
seeds=[42, 101, 202, 303, 404]) across 50 diverse Turkish educational prompts
(24 ISG+OOD concepts + 26 student misconception / CEFR prompts) for:
  - base_14b
  - socratic_v2 (SFT)
  - onpolicy_dpo_v2 (DPO)

Computes per-seed and cross-seed Mean, SD, and 95% Bootstrap CI for:
  - Socratic question rate
  - Atesman readability score
  - Non-Latin leakage rate
  - Mean response character length

Outputs:
  results/hpc_multiseed_750_report.json
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_hpc_alignment_tax_cefr import calculate_atesman_score, resolve_dpo_v2_path
from run_hpc_cjk_blocked_dag_ablation import CJKBlockLogitsProcessor, bootstrap_ci, generate_chat_nocjk
from run_hpc_dpo_r2_nonlatin import build_nonlatin_token_ids, has_nonlatin
from run_hpc_live_student_simulation import ISG_CONCEPTS
from run_hpc_ood_tutoring import OOD_CONCEPTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp23MultiSeed750")

SEEDS = [42, 101, 202, 303, 404]


def build_50_prompts() -> List[Dict[str, str]]:
    prompts: List[Dict[str, str]] = []
    all_concepts = ISG_CONCEPTS + OOD_CONCEPTS  # 24 concepts
    for c in all_concepts:
        prompts.append({
            "id": f"M_{c['id']}",
            "type": "misconception",
            "sys": f"Sen Sokratik bir Türkçe öğretmenisin. Kavram: {c['name']}. Doğrudan cevabı verme, soruyla yönlendir.",
            "user": f"Hocam, '{c['name']}' konusunda şöyle düşünüyorum: {c['misconception']} Doğru mu?",
        })
    for idx, c in enumerate(all_concepts):
        prompts.append({
            "id": f"C_{c['id']}",
            "type": "cefr_explain",
            "sys": "Sen EduTerm Türkçe eğitim asistanısın. Kavramı CEFR-B1 düzeyinde sade ve anlaşılır Türkçe ile açıkla.",
            "user": f"'{c['name']}' kavramını başlangıç seviyesindeki bir öğrenciye 3 cümleyle anlatır mısın?",
        })
    # Add 2 extra cross-domain reasoning prompts to reach exactly 50
    prompts.append({
        "id": "X_01",
        "type": "misconception",
        "sys": "Sen Sokratik bir Türkçe öğretmenisin.",
        "user": "Hocam bir projede zaman çizelgesi sapması (Schedule Variance) pozitifse proje gecikmiş mi demektir?",
    })
    prompts.append({
        "id": "X_02",
        "type": "cefr_explain",
        "sys": "Sen EduTerm Türkçe eğitim asistanısın.",
        "user": "Siber güvenlikte sıfırıncı gün (zero-day) açığı ne demektir, günlük hayattan bir örnekle açıklar mısın?",
    })
    assert len(prompts) == 50
    return prompts


def main() -> None:
    t0 = time.time()
    prompts = build_50_prompts()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    base_path = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
    tok = AutoTokenizer.from_pretrained(base_path)
    base = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb, device_map="auto")
    model = PeftModel.from_pretrained(base, "/users/ali.gunder/eduterm-hybrid-sim/checkpoints/eduterm_14b_socratic_v2", adapter_name="socratic_v2")
    model.load_adapter(resolve_dpo_v2_path(), adapter_name="dpo_v2")
    model.eval()
    proc = CJKBlockLogitsProcessor(build_nonlatin_token_ids(tok))

    models = [("base_14b", None), ("socratic_v2", "socratic_v2"), ("onpolicy_dpo_v2", "dpo_v2")]
    summary: Dict[str, Any] = {}
    total_gens = 0

    for mname, adp in models:
        per_seed_stats: List[Dict[str, Any]] = []
        all_q, all_at, all_nl = [], [], []
        for seed in SEEDS:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            q_list, at_list, nl_list, len_list = [], [], [], []
            for p in prompts:
                msgs = [{"role": "system", "content": p["sys"]}, {"role": "user", "content": p["user"]}]
                if adp is None:
                    with model.disable_adapter():
                        out = generate_chat_nocjk(model, tok, msgs, cjk_processor=proc, max_new_tokens=160, temperature=0.4)
                else:
                    model.set_adapter(adp)
                    out = generate_chat_nocjk(model, tok, msgs, cjk_processor=proc, max_new_tokens=160, temperature=0.4)
                q = 1 if "?" in out else 0
                at = calculate_atesman_score(out)
                nl = 1 if has_nonlatin(out) else 0
                q_list.append(q)
                at_list.append(at)
                nl_list.append(nl)
                len_list.append(len(out))
                total_gens += 1
            per_seed_stats.append({
                "seed": seed,
                "question_rate": round(sum(q_list) / len(q_list), 4),
                "mean_atesman": round(sum(at_list) / len(at_list), 2),
                "nonlatin_rate": round(sum(nl_list) / len(nl_list), 4),
                "mean_chars": round(sum(len_list) / len(len_list), 1),
            })
            all_q.extend(q_list)
            all_at.extend(at_list)
            all_nl.extend(nl_list)
            logger.info("[%s | seed=%d] Q-rate=%.3f Atesman=%.2f (total_gens=%d/750)",
                        mname, seed, per_seed_stats[-1]["question_rate"], per_seed_stats[-1]["mean_atesman"], total_gens)

        seed_q = [x["question_rate"] for x in per_seed_stats]
        seed_at = [x["mean_atesman"] for x in per_seed_stats]
        summary[mname] = {
            "n_generations": len(all_q),
            "question_rate_mean": round(statistics.mean(seed_q), 4),
            "question_rate_seed_sd": round(statistics.stdev(seed_q), 4),
            "question_rate_bootstrap_95ci": list(bootstrap_ci(all_q)),
            "atesman_mean": round(statistics.mean(seed_at), 2),
            "atesman_seed_sd": round(statistics.stdev(seed_at), 2),
            "nonlatin_leakage_rate": round(sum(all_nl) / len(all_nl), 4),
            "per_seed": per_seed_stats,
        }

    report = {
        "metadata": {
            "n_prompts": len(prompts),
            "n_models": len(models),
            "seeds": SEEDS,
            "total_generations": total_gens,
            "decoding": {"temperature": 0.4, "do_sample": True, "nonlatin_blocked": True},
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "summary": summary,
    }
    Path("results/hpc_multiseed_750_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Exp 23 Finished in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
