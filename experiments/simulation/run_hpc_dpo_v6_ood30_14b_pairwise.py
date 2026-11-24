# -*- coding: utf-8 -*-
"""
Experiment 41: Scaled Hard OOD (n=30 triplets = 90 sessions) with 14B-Base Blinded
A/B Pairwise Judging (`dpo_v6_multidomain` vs `dpo_v5_pareto` vs `base_14b`)
on Boğaziçi HPC (A40 48GB).

Resolves the 7B-judge JSON-parse failure (0.0% parse in Exps 36 & 37) by using
`Qwen2.5-14B-Instruct` (adapter disabled, NF4) as the blinded randomized A/B pairwise
judge alongside independent post-testing on Kumru-2B under `utf8_exact_allowlist`.

Outputs:
  results/hpc_dpo_v6_ood30_14b_pairwise_report.json
"""

from __future__ import annotations

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

from run_hpc_cjk_blocked_dag_ablation import CJKBlockLogitsProcessor, bootstrap_ci, exact_mcnemar_p
from run_hpc_dpo_3seed_ablation import PERSONAS
from run_hpc_dpo_r2_nonlatin import NON_LATIN_PATTERN, has_nonlatin
from run_hpc_multiturn_guardrail import resolve_adapter
from run_hpc_omni_vs_dpo_pareto import HARD_OOD_CONCEPTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp41OOD30PW14B")


def build_utf8_exact_allowlist_ids(tok) -> List[int]:
    return [
        tid for tid in range(len(tok))
        if "\ufffd" not in (s := tok.decode([tid], skip_special_tokens=False)) and NON_LATIN_PATTERN.search(s)
    ]


def gen_safe(model, tok, proc, msgs: List[Dict[str, str]], max_new_tokens: int = 145, temperature: float = 0.4) -> str:
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
    t_model = PeftModel.from_pretrained(
        t_base, resolve_adapter("checkpoints/eduterm_14b_dpo_v5_pareto", "dpo_v5_pareto"), adapter_name="dpo_v5_pareto"
    )
    t_model.load_adapter(
        resolve_adapter("checkpoints/eduterm_14b_dpo_v6_multidomain", "dpo_v6_multidomain"), adapter_name="dpo_v6_multidomain"
    )
    t_model.eval()
    t_proc = CJKBlockLogitsProcessor(build_utf8_exact_allowlist_ids(t_tok))

    s_path = "/users/ali.gunder/models/Kumru-2B"
    s_tok = AutoTokenizer.from_pretrained(s_path)
    s_model = AutoModelForCausalLM.from_pretrained(s_path, torch_dtype=torch.bfloat16, device_map="auto")
    s_model.eval()
    s_proc = CJKBlockLogitsProcessor(build_utf8_exact_allowlist_ids(s_tok))

    variants = [("base_14b", None), ("dpo_v5_pareto", "dpo_v5_pareto"), ("dpo_v6_multidomain", "dpo_v6_multidomain")]
    sessions: Dict[str, List[Dict[str, Any]]] = {v: [] for v, _ in variants}
    cmap: Dict[str, Dict[Any, Dict[str, Any]]] = {v: {} for v, _ in variants}

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
                    sr = gen_safe(s_model, s_tok, s_proc, s_msgs, max_new_tokens=90, temperature=0.7)
                    s_msgs.append({"role": "assistant", "content": sr})
                    t_msgs.append({"role": "user", "content": sr})
                    if adp is None:
                        with t_model.disable_adapter():
                            tr = gen_safe(t_model, t_tok, t_proc, t_msgs, max_new_tokens=145, temperature=0.4)
                    else:
                        t_model.set_adapter(adp)
                        tr = gen_safe(t_model, t_tok, t_proc, t_msgs, max_new_tokens=145, temperature=0.4)
                    t_msgs.append({"role": "assistant", "content": tr})
                    s_msgs.append({"role": "user", "content": tr})
                    q_turns += int("?" in tr)
                    nl += int(has_nonlatin(tr))
                    transcript.append(f"T{turn} Öğrenci: {sr}\nT{turn} Öğretmen: {tr}")

                pt_msgs = s_msgs + [{"role": "user", "content": f"Öğretmensiz cevapla: '{c['gold']}' ifadesi DOĞRU mu YANLIŞ mı? Sadece 'DOĞRU' veya 'YANLIŞ' yaz."}]
                pt_raw = gen_safe(s_model, s_tok, s_proc, pt_msgs, max_new_tokens=15, temperature=0.0)
                pt_ok = int("DOĞRU" in pt_raw.upper() and "YANLIŞ" not in pt_raw.upper())
                rec = {
                    "key": (c["id"], p["id"]),
                    "concept": c,
                    "posttest_ok": pt_ok,
                    "q_turns": q_turns,
                    "nl": nl,
                    "transcript_text": "\n".join(transcript),
                }
                sessions[v_name].append(rec)
                cmap[v_name][rec["key"]] = rec

    # Blinded 14B-base pointwise + randomized A/B pairwise evaluation
    summary: Dict[str, Any] = {}
    for v_name, items in sessions.items():
        parsed = 0
        for item in items:
            c = item["concept"]
            j_msgs = [
                {"role": "system", "content": "Sen kör pedagojik hakemsin. Yalnızca geçerli JSON döndür: {\"resolved\": 0 veya 1, \"socratic\": 1-5}."},
                {"role": "user", "content": f"Kavram: {c['name']}\nDoğru ilke: {c['gold']}\nDiyalog:\n{item['transcript_text']}"},
            ]
            with t_model.disable_adapter():
                j_raw = gen_safe(t_model, t_tok, t_proc, j_msgs, max_new_tokens=60, temperature=0.0)
            m_res = re.search(r"\"resolved\"\s*:\s*([01])", j_raw)
            m_soc = re.search(r"\"socratic\"\s*:\s*([1-5])", j_raw)
            if m_res and m_soc:
                parsed += 1
            item["judge_resolved"] = int(m_res.group(1)) if m_res else item["posttest_ok"]
            item["socratic_1to5"] = int(m_soc.group(1)) if m_soc else 3

        pts = [x["posttest_ok"] for x in items]
        summary[v_name] = {
            "n": len(items),
            "posttest_accuracy": round(sum(pts) / len(pts), 4),
            "posttest_95ci": list(bootstrap_ci(pts)),
            "judge_14b_resolution": round(sum(x["judge_resolved"] for x in items) / len(items), 4),
            "socratic_mean_1to5": round(sum(x["socratic_1to5"] for x in items) / len(items), 4),
            "judge_parse_rate": round(parsed / len(items), 4),
            "tutor_question_rate": round(sum(x["q_turns"] for x in items) / (len(items) * 4), 4),
            "nonlatin_rate": round(sum(x["nl"] for x in items) / (len(items) * 4), 4),
        }
        logger.info("Exp41 OOD-30 [%s]: %s", v_name, summary[v_name])

    rng = random.Random(2026)
    pairwise: Dict[str, Any] = {}
    for va, vb in [("dpo_v6_multidomain", "base_14b"), ("dpo_v6_multidomain", "dpo_v5_pareto"), ("dpo_v5_pareto", "base_14b")]:
        wa = wb = ties = pw_parsed = 0
        for key in cmap[va]:
            sa, sb = cmap[va][key], cmap[vb][key]
            flip = rng.random() < 0.5
            da, db = (sb, sa) if flip else (sa, sb)
            p_msgs = [
                {
                    "role": "system",
                    "content": "İki öğretmenden hangisi daha Sokratik soru soruyor ve kavram yanılgısını çözmede daha başarılı? Mümkünse A veya B seç. Yalnızca JSON döndür: {\"winner\": \"A\" | \"B\" | \"TIE\"}.",
                },
                {"role": "user", "content": f"Kavram: {sa['concept']['name']}\n[Öğretmen A]:\n{da['transcript_text']}\n\n[Öğretmen B]:\n{db['transcript_text']}"},
            ]
            with t_model.disable_adapter():
                raw = gen_safe(t_model, t_tok, t_proc, p_msgs, max_new_tokens=50, temperature=0.0)
            m = re.search(r"\"winner\"\s*:\s*\"(A|B|TIE)\"", raw.upper())
            if m:
                pw_parsed += 1
                w = m.group(1)
            else:
                w = "TIE"
            if flip and w in {"A", "B"}:
                w = "B" if w == "A" else "A"
            if w == "A":
                wa += 1
            elif w == "B":
                wb += 1
            else:
                ties += 1
        pairwise[f"{va}__vs__{vb}"] = {
            "wins_a": wa, "wins_b": wb, "ties": ties, "parse_rate": round(pw_parsed / len(cmap[va]), 4), "sign_p": two_sided_sign_p(wa, wb)
        }
        logger.info("Pairwise [%s vs %s]: %s", va, vb, pairwise[f"{va}__vs__{vb}"])

    report = {
        "metadata": {"n_sessions_per_arm": 30, "elapsed_sec": round(time.time() - t0, 1)},
        "summary_by_variant": summary,
        "blinded_pairwise_14b_judge": pairwise,
        "mcnemar_v6_vs_v5_posttest": exact_mcnemar_p(
            [x["posttest_ok"] for x in sessions["dpo_v6_multidomain"]],
            [x["posttest_ok"] for x in sessions["dpo_v5_pareto"]],
        ),
        "mcnemar_v6_vs_base_posttest": exact_mcnemar_p(
            [x["posttest_ok"] for x in sessions["dpo_v6_multidomain"]],
            [x["posttest_ok"] for x in sessions["base_14b"]],
        ),
    }
    Path("results/hpc_dpo_v6_ood30_14b_pairwise_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
