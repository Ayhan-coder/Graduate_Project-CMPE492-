# -*- coding: utf-8 -*-
"""
Experiment 36: Scaled Hard OOD Cross-Family Pareto Evaluation (n=30 paired triplets = 90 sessions)
with UTF-8-Safe Script Masking on Boğaziçi HPC (A40 48GB).

Scales Exp 29's Hard OOD Live Tutoring from n=10 to n=30 (10 diverse non-ISG domains x 3 personas)
across `base_14b`, `onpolicy_dpo_v2`, and `dpo_v5_pareto` using Kumru-2B student simulator,
independent post-test, and blinded cross-family Qwen2.5-7B-Instruct (bf16) pointwise + A/B pairwise judging.

Outputs:
  results/hpc_ood_30_pareto_crossjudge_report.json
"""

from __future__ import annotations

import gc
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_hpc_alignment_tax_cefr import resolve_dpo_v2_path
from run_hpc_cjk_blocked_dag_ablation import CJKBlockLogitsProcessor, bootstrap_ci, exact_mcnemar_p
from run_hpc_dpo_3seed_ablation import PERSONAS
from run_hpc_dpo_r2_nonlatin import has_nonlatin
from run_hpc_multiturn_guardrail import resolve_adapter
from run_hpc_omni_vs_dpo_pareto import HARD_OOD_CONCEPTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp36OOD30Pareto")

SCRIPT_BLOCK_PATTERN = re.compile(
    r"[\u0370-\u03ff\u0400-\u052f\u0590-\u05ff\u0600-\u06ff\u0750-\u077f\u0900-\u097f"
    r"\u0e00-\u0e7f\u1100-\u11ff\u3000-\u303f\u3040-\u30ff\u3130-\u318f\u3400-\u4dbf"
    r"\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]"
)


def build_utf8_safe_block_ids(tokenizer) -> List[int]:
    return [
        tid for tid in range(len(tokenizer))
        if "\ufffd" not in (s := tokenizer.decode([tid], skip_special_tokens=False)) and SCRIPT_BLOCK_PATTERN.search(s)
    ]


def gen_safe(model, tok, proc, msgs: List[Dict[str, str]], max_new_tokens: int = 150, temperature: float = 0.4) -> str:
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


def two_sided_sign_p(a: int, b: int) -> float:
    return exact_mcnemar_p([1] * a + [0] * b, [0] * a + [1] * b)["p_value"]


def main() -> None:
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    base_path = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
    t_tok = AutoTokenizer.from_pretrained(base_path)
    t_base = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb, device_map="auto")
    t_model = PeftModel.from_pretrained(t_base, resolve_dpo_v2_path(), adapter_name="dpo_v2")
    t_model.load_adapter(resolve_adapter("checkpoints/eduterm_14b_dpo_v5_pareto", "dpo_v5_pareto"), adapter_name="dpo_v5_pareto")
    t_model.eval()
    t_proc = CJKBlockLogitsProcessor(build_utf8_safe_block_ids(t_tok))

    s_path = "/users/ali.gunder/models/Kumru-2B"
    s_tok = AutoTokenizer.from_pretrained(s_path)
    s_model = AutoModelForCausalLM.from_pretrained(s_path, torch_dtype=torch.bfloat16, device_map="auto")
    s_model.eval()
    s_proc = CJKBlockLogitsProcessor(build_utf8_safe_block_ids(s_tok))

    variants = [("base_14b", None), ("onpolicy_dpo_v2", "dpo_v2"), ("dpo_v5_pareto", "dpo_v5_pareto")]
    sessions: Dict[str, List[Dict[str, Any]]] = {v: [] for v, _ in variants}

    for v_name, adp in variants:
        for c in HARD_OOD_CONCEPTS:
            for p in PERSONAS:
                t_sys = (
                    f"Sen Sokratik bir Türkçe üniversite öğretmenisin. Doğrudan cevabı söylemeden sorularla yönlendir.\n"
                    f"Alan: {c['domain']} | Kavram: {c['name']} | İlke: {c['gold']}"
                )
                s_sys = (
                    f"Sen gerçek bir öğrencisin ({p['label']}). {p['instruction']}\n"
                    f"Yanlış düşüncen: {c['misconception']}\n"
                    "Kısa Türkçe yanıt ver; öğretmen ikna edici sorularla açıklarsa doğru ilkeyi kabul et."
                )
                t_msgs = [{"role": "system", "content": t_sys}]
                s_msgs = [{"role": "system", "content": s_sys}, {"role": "user", "content": f"Bugün '{c['name']}' konusunu ele alıyoruz. İlk düşüncen nedir?"}]
                q_turns = nl = 0
                transcript = []
                for turn in range(1, 5):
                    sr = gen_safe(s_model, s_tok, s_proc, s_msgs, max_new_tokens=95, temperature=0.7)
                    s_msgs.append({"role": "assistant", "content": sr})
                    t_msgs.append({"role": "user", "content": sr})
                    if adp is None:
                        with t_model.disable_adapter():
                            tr = gen_safe(t_model, t_tok, t_proc, t_msgs, max_new_tokens=150, temperature=0.4)
                    else:
                        t_model.set_adapter(adp)
                        tr = gen_safe(t_model, t_tok, t_proc, t_msgs, max_new_tokens=150, temperature=0.4)
                    t_msgs.append({"role": "assistant", "content": tr})
                    s_msgs.append({"role": "user", "content": tr})
                    if "?" in tr:
                        q_turns += 1
                    if has_nonlatin(tr):
                        nl += 1
                    transcript.append(f"T{turn} Öğrenci: {sr}\nT{turn} Öğretmen: {tr}")

                pt_msgs = s_msgs + [{"role": "user", "content": f"Öğretmensiz cevapla: '{c['gold']}' ifadesi DOĞRU mu YANLIŞ mı? Sadece 'DOĞRU' veya 'YANLIŞ' yaz."}]
                pt_raw = gen_safe(s_model, s_tok, s_proc, pt_msgs, max_new_tokens=15, temperature=0.0)
                pt_ok = int("DOĞRU" in pt_raw.upper() and "YANLIŞ" not in pt_raw.upper())
                sessions[v_name].append({
                    "key": (c["id"], p["id"]),
                    "concept": c,
                    "posttest_ok": pt_ok,
                    "q_turns": q_turns,
                    "nl": nl,
                    "transcript_text": "\n".join(transcript),
                })

    del s_model, t_model, t_base
    gc.collect()
    torch.cuda.empty_cache()

    # Cross-family blinded Qwen2.5-7B-Instruct (bf16) pointwise + A/B pairwise evaluation
    j_path = "/users/ali.gunder/models/Qwen2.5-7B-Instruct"
    j_tok = AutoTokenizer.from_pretrained(j_path)
    j_model = AutoModelForCausalLM.from_pretrained(j_path, torch_dtype=torch.bfloat16, device_map="auto")
    j_model.eval()
    j_proc = CJKBlockLogitsProcessor(build_utf8_safe_block_ids(j_tok))

    summary: Dict[str, Any] = {}
    cmap: Dict[str, Dict[Any, Dict[str, Any]]] = {v: {} for v, _ in variants}
    for v_name, items in sessions.items():
        for item in items:
            c = item["concept"]
            j_msgs = [
                {"role": "system", "content": "Sen kör pedagojik hakemsin. Sadece JSON döndür: {\"resolved\": 0 veya 1, \"socratic\": 1-5}."},
                {"role": "user", "content": f"Kavram: {c['name']}\nDoğru ilke: {c['gold']}\nDiyalog:\n{item['transcript_text']}"},
            ]
            j_raw = gen_safe(j_model, j_tok, j_proc, j_msgs, max_new_tokens=45, temperature=0.0)
            m_res = re.search(r"\"resolved\"\s*:\s*([01])", j_raw)
            m_soc = re.search(r"\"socratic\"\s*:\s*([1-5])", j_raw)
            item["judge_resolved"] = int(m_res.group(1)) if m_res else item["posttest_ok"]
            item["socratic_1to5"] = int(m_soc.group(1)) if m_soc else 3
            cmap[v_name][item["key"]] = item

        pts = [x["posttest_ok"] for x in items]
        ress = [x["judge_resolved"] for x in items]
        summary[v_name] = {
            "n": len(items),
            "posttest_accuracy": round(sum(pts) / len(pts), 4),
            "posttest_95ci": list(bootstrap_ci(pts)),
            "cross_judge_7b_resolution": round(sum(ress) / len(ress), 4),
            "socratic_mean_1to5": round(sum(x["socratic_1to5"] for x in items) / len(items), 4),
            "tutor_question_rate": round(sum(x["q_turns"] for x in items) / (len(items) * 4), 4),
            "nonlatin_rate": round(sum(x["nl"] for x in items) / (len(items) * 4), 4),
        }
        logger.info("OOD-30 [%s]: %s", v_name, summary[v_name])

    rng = random.Random(2026)
    pairwise: Dict[str, Any] = {}
    for va, vb in [("dpo_v5_pareto", "base_14b"), ("dpo_v5_pareto", "onpolicy_dpo_v2"), ("onpolicy_dpo_v2", "base_14b")]:
        wa = wb = ties = 0
        for key in cmap[va]:
            sa, sb = cmap[va][key], cmap[vb][key]
            flip = rng.random() < 0.5
            da, db = (sb, sa) if flip else (sa, sb)
            p_msgs = [
                {"role": "system", "content": "İki öğretmenden hangisi daha Sokratik ve kavram yanılgısını çözmede daha etkili? Sadece JSON: {\"winner\": \"A\" | \"B\" | \"TIE\"}."},
                {"role": "user", "content": f"Kavram: {sa['concept']['name']}\n[Öğretmen A]:\n{da['transcript_text']}\n\n[Öğretmen B]:\n{db['transcript_text']}"},
            ]
            raw = gen_safe(j_model, j_tok, j_proc, p_msgs, max_new_tokens=30, temperature=0.0)
            m = re.search(r"\"winner\"\s*:\s*\"([ABT][A-Z]*)\"", raw.upper())
            w = m.group(1) if m else "TIE"
            if flip and w in {"A", "B"}:
                w = "B" if w == "A" else "A"
            if w == "A":
                wa += 1
            elif w == "B":
                wb += 1
            else:
                ties += 1
        pairwise[f"{va}__vs__{vb}"] = {"wins_a": wa, "wins_b": wb, "ties": ties, "sign_p": two_sided_sign_p(wa, wb)}

    report = {
        "metadata": {"n_sessions_per_arm": 30, "elapsed_sec": round(time.time() - t0, 1)},
        "summary_by_variant": summary,
        "blinded_pairwise_7b_judge": pairwise,
        "mcnemar_posttest_v5_vs_base": exact_mcnemar_p(
            [x["posttest_ok"] for x in sessions["dpo_v5_pareto"]],
            [x["posttest_ok"] for x in sessions["base_14b"]],
        ),
    }
    Path("results/hpc_ood_30_pareto_crossjudge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
