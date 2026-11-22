# -*- coding: utf-8 -*-
"""
Experiment 26: Student Simulator Capacity Scaling (Qwen2.5-1.5B vs Kumru-2B vs Qwen2.5-7B bf16)
on Boğaziçi HPC (A40 48GB).

Evaluates `task_separated_hybrid` vs `onpolicy_dpo_v2` across 12 ISG concepts x 3 student models
= 72 live 4-turn sessions + independent post-test, using token_type_ids-safe generation.

Outputs:
  results/hpc_student_scale_comparison_report.json
"""

from __future__ import annotations

import gc
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_hpc_alignment_tax_cefr import resolve_dpo_v2_path
from run_hpc_cjk_blocked_dag_ablation import CJKBlockLogitsProcessor, bootstrap_ci, exact_mcnemar_p
from run_hpc_dpo_3seed_ablation import generate_cefr_card
from run_hpc_dpo_r2_nonlatin import build_nonlatin_token_ids, has_nonlatin
from run_hpc_live_student_simulation import ISG_CONCEPTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp26StudentScale")


def gen_safe(model, tok, proc, msgs: List[Dict[str, str]], max_new_tokens: int = 140, temperature: float = 0.4) -> str:
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    enc.pop("token_type_ids", None)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=0.9 if temperature > 0 else None,
            logits_processor=[proc],
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    base_path = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
    t_tok = AutoTokenizer.from_pretrained(base_path)
    t_base = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb, device_map="auto")
    t_model = PeftModel.from_pretrained(
        t_base, "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora", adapter_name="cefr"
    )
    t_model.load_adapter(resolve_dpo_v2_path(), adapter_name="dpo_v2")
    t_model.eval()
    t_proc = CJKBlockLogitsProcessor(build_nonlatin_token_ids(t_tok))

    cefr_cards = {c["id"]: generate_cefr_card(t_model, t_tok, t_proc, c) for c in ISG_CONCEPTS}

    student_models = [
        ("Qwen2.5-1.5B", "/users/ali.gunder/models/Qwen2.5-1.5B-Instruct"),
        ("Kumru-2B", "/users/ali.gunder/models/Kumru-2B"),
        ("Qwen2.5-7B-bf16", "/users/ali.gunder/models/Qwen2.5-7B-Instruct"),
    ]

    summary: Dict[str, Any] = {}
    for s_name, s_path in student_models:
        s_tok = AutoTokenizer.from_pretrained(s_path)
        s_model = AutoModelForCausalLM.from_pretrained(s_path, torch_dtype=torch.bfloat16, device_map="auto")
        s_model.eval()
        s_proc = CJKBlockLogitsProcessor(build_nonlatin_token_ids(s_tok))

        by_cond: Dict[str, List[Dict[str, Any]]] = {"task_separated_hybrid": [], "onpolicy_dpo_v2": []}
        for cond in ("task_separated_hybrid", "onpolicy_dpo_v2"):
            for c in ISG_CONCEPTS:
                prereq = ", ".join(c["prerequisites"]) if c["prerequisites"] else "Yok"
                t_sys = (
                    f"Sen Sokratik bir Türkçe İSG öğretmenisin. Kavram: {c['name']}. Ön koşul özeti: {cefr_cards[c['id']]}"
                    if cond == "task_separated_hybrid"
                    else f"Sen Sokratik ve destekleyici bir Türkçe İSG öğretmenisin. Kavram: {c['name']} | Ön koşullar: {prereq} | İlke: {c['gold_principle']}\nDoğrudan cevap yerine soruyla yönlendir."
                )
                s_sys = (
                    f"Sen ön bilgisi zayıf bir İSG öğrencisisin. Yanlış düşüncen: {c['misconception']}\n"
                    "Kısa ve doğal Türkçe ile yanıt ver; öğretmen ikna ederse doğru ilkeyi kabul et."
                )
                t_msgs = [{"role": "system", "content": t_sys}]
                s_msgs = [{"role": "system", "content": s_sys}, {"role": "user", "content": f"Bugün '{c['name']}' konusunu konuşuyoruz. Ne düşünüyorsun?"}]
                q_turns = nl = 0
                for _t in range(1, 5):
                    sr = gen_safe(s_model, s_tok, s_proc, s_msgs, max_new_tokens=100, temperature=0.7)
                    s_msgs.append({"role": "assistant", "content": sr})
                    t_msgs.append({"role": "user", "content": sr})
                    if cond == "task_separated_hybrid":
                        with t_model.disable_adapter():
                            tr = gen_safe(t_model, t_tok, t_proc, t_msgs, max_new_tokens=160, temperature=0.4)
                    else:
                        t_model.set_adapter("dpo_v2")
                        tr = gen_safe(t_model, t_tok, t_proc, t_msgs, max_new_tokens=160, temperature=0.4)
                    t_msgs.append({"role": "assistant", "content": tr})
                    s_msgs.append({"role": "user", "content": tr})
                    if "?" in tr:
                        q_turns += 1
                    if has_nonlatin(tr):
                        nl += 1

                pt_msgs = s_msgs + [{"role": "user", "content": f"Şimdi öğretmensiz cevapla. Aşağıdaki ifade DOĞRU mu YANLIŞ mı? Sadece 'DOĞRU' veya 'YANLIŞ' yaz:\n'{c['gold_principle']}'"}]
                pt_raw = gen_safe(s_model, s_tok, s_proc, pt_msgs, max_new_tokens=12, temperature=0.0)
                pt_ok = int("DOĞRU" in pt_raw.upper() and "YANLIŞ" not in pt_raw.upper())
                by_cond[cond].append({"concept_id": c["id"], "q_turns": q_turns, "nl": nl, "posttest_correct": pt_ok})

        del s_model
        gc.collect()
        torch.cuda.empty_cache()

        s_out = {}
        for cond, items in by_cond.items():
            pts = [x["posttest_correct"] for x in items]
            s_out[cond] = {
                "n": len(items),
                "posttest_accuracy": round(sum(pts) / len(pts), 4),
                "posttest_95ci": list(bootstrap_ci(pts)),
                "tutor_question_rate": round(sum(x["q_turns"] for x in items) / (len(items) * 4), 4),
                "nonlatin_rate": round(sum(x["nl"] for x in items) / (len(items) * 4), 4),
            }
        s_out["mcnemar_dpo_vs_hybrid"] = exact_mcnemar_p(
            [x["posttest_correct"] for x in by_cond["onpolicy_dpo_v2"]],
            [x["posttest_correct"] for x in by_cond["task_separated_hybrid"]],
        )
        summary[s_name] = s_out
        logger.info("Student [%s] summary: %s", s_name, s_out)

    report = {"metadata": {"elapsed_sec": round(time.time() - t0, 1)}, "summary_by_student_scale": summary}
    Path("results/hpc_student_scale_comparison_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
