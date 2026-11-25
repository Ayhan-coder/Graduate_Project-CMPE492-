# -*- coding: utf-8 -*-
"""
Experiment 45: Dual-Judge Blinded A/B Pairwise Preference (`14B-base` + `7B-bf16`
Single-Letter Judge) & Inter-Judge Cohen's Kappa (`dpo_v6_multidomain` vs `dpo_v5_pareto` vs `base_14b`)
on Boğaziçi HPC (A40 48GB).

Fixes the case-sensitivity regex bug from Exp 41 and bypasses 7B JSON failures via
direct single-letter forced selection (`A` / `B` / `TIE`) across 20 paired concepts
(10 Hard OOD + 10 ISG = 60 live 4-turn sessions, 60 pairwise comparisons per judge).

Outputs:
  results/hpc_dual_judge_pairwise_v6_report.json
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

from run_hpc_cjk_blocked_dag_ablation import CJKBlockLogitsProcessor, exact_mcnemar_p
from run_hpc_dpo_r2_nonlatin import NON_LATIN_PATTERN, has_nonlatin
from run_hpc_live_student_simulation import ISG_CONCEPTS
from run_hpc_multiturn_guardrail import resolve_adapter
from run_hpc_omni_vs_dpo_pareto import HARD_OOD_CONCEPTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp45DualPairwise")


def build_utf8_exact_allowlist_ids(tok) -> List[int]:
    return [
        tid for tid in range(len(tok))
        if "\ufffd" not in (s := tok.decode([tid], skip_special_tokens=False)) and NON_LATIN_PATTERN.search(s)
    ]


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


def two_sided_sign_p(a: int, b: int) -> float:
    return exact_mcnemar_p([1] * a + [0] * b, [0] * a + [1] * b)["p_value"]


def cohen_kappa(y1: List[str], y2: List[str]) -> float:
    n = len(y1)
    if n == 0:
        return 0.0
    labels = sorted(set(y1) | set(y2))
    po = sum(1 for a, b in zip(y1, y2) if a == b) / n
    pe = sum((y1.count(l) / n) * (y2.count(l) / n) for l in labels)
    return round((po - pe) / (1.0 - pe), 4) if pe < 1.0 else 1.0


def parse_winner(raw: str) -> tuple[str, int]:
    m = re.search(r"\"?winner\"?\s*[:=]\s*\"?(A|B|TIE)\b", raw, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper(), 1
    m2 = re.match(r"^\s*(A|B|TIE)\b", raw.strip(), flags=re.IGNORECASE)
    if m2:
        return m2.group(1).upper(), 1
    m3 = re.search(r"\b(A|B|TIE)\b", raw, flags=re.IGNORECASE)
    if m3:
        return m3.group(1).upper(), 1
    return "TIE", 0


def main() -> None:
    t0 = time.time()
    # Unified 20-concept benchmark (10 Hard OOD + 10 ISG)
    concepts = [
        {"id": c["id"], "name": c["name"], "misconception": c["misconception"], "gold": c["gold"]}
        for c in HARD_OOD_CONCEPTS
    ] + [
        {"id": c["id"], "name": c["name"], "misconception": c["misconception"], "gold": c["gold_principle"]}
        for c in ISG_CONCEPTS[:10]
    ]

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

    pk = "/users/ali.gunder/models/Kumru-2B"
    s_tok = AutoTokenizer.from_pretrained(pk)
    s_model = AutoModelForCausalLM.from_pretrained(pk, torch_dtype=torch.bfloat16, device_map="auto")
    s_model.eval()
    s_proc = CJKBlockLogitsProcessor(build_utf8_exact_allowlist_ids(s_tok))

    variants = [("base_14b", None), ("dpo_v5_pareto", "dpo_v5_pareto"), ("dpo_v6_multidomain", "dpo_v6_multidomain")]
    cmap: Dict[str, Dict[str, Dict[str, Any]]] = {v: {} for v, _ in variants}

    for v_name, adp in variants:
        for c in concepts:
            t_sys = f"Sen Sokratik bir Türkçe üniversite öğretmenisin. Doğrudan cevabı söylemeden sorularla yönlendir.\nKavram: {c['name']} | İlke: {c['gold']}"
            s_sys = f"Sen şüpheci bir öğrencisin. Yanlış düşüncen: {c['misconception']}\nKısa Türkçe yanıt ver; öğretmen ikna edici sorularla açıklarsa doğru ilkeyi kabul et."
            t_msgs = [{"role": "system", "content": t_sys}]
            s_msgs = [{"role": "system", "content": s_sys}, {"role": "user", "content": f"Bugün '{c['name']}' konusunu konuşuyoruz. İlk düşüncen nedir?"}]
            transcript = []
            for turn in range(1, 5):
                sr = gen_safe(s_model, s_tok, s_proc, s_msgs, max_new_tokens=85, temperature=0.7)
                s_msgs.append({"role": "assistant", "content": sr})
                t_msgs.append({"role": "user", "content": sr})
                if adp is None:
                    with t_model.disable_adapter():
                        tr = gen_safe(t_model, t_tok, t_proc, t_msgs, max_new_tokens=140, temperature=0.4)
                else:
                    t_model.set_adapter(adp)
                    tr = gen_safe(t_model, t_tok, t_proc, t_msgs, max_new_tokens=140, temperature=0.4)
                t_msgs.append({"role": "assistant", "content": tr})
                s_msgs.append({"role": "user", "content": tr})
                transcript.append(f"T{turn} Öğrenci: {sr}\nT{turn} Öğretmen: {tr}")
            cmap[v_name][c["id"]] = {"concept": c, "transcript": "\n".join(transcript)}

    # Build deterministic randomized flip map shared by both 14B and 7B judges
    rng = random.Random(2026)
    pair_specs = [("dpo_v6_multidomain", "base_14b"), ("dpo_v6_multidomain", "dpo_v5_pareto"), ("dpo_v5_pareto", "base_14b")]
    flips = {(va, vb, c["id"]): (rng.random() < 0.5) for va, vb in pair_specs for c in concepts}

    # Judge 1: 14B-base (adapter disabled)
    j14_res: Dict[str, Any] = {}
    j14_labels: List[str] = []
    for va, vb in pair_specs:
        wa = wb = ties = parsed = 0
        for c in concepts:
            flip = flips[(va, vb, c["id"])]
            da, db = (cmap[vb][c["id"]], cmap[va][c["id"]]) if flip else (cmap[va][c["id"]], cmap[vb][c["id"]])
            msgs = [
                {
                    "role": "system",
                    "content": "İki öğretmenden hangisi daha iyi Sokratik sorular sorup kavram yanılgısını daha etkili çözüyor? Sadece tek bir kelime yaz: A, B veya TIE.",
                },
                {"role": "user", "content": f"Kavram: {c['name']}\n[Öğretmen A]:\n{da['transcript']}\n\n[Öğretmen B]:\n{db['transcript']}\n\nKazanan (A / B / TIE):"},
            ]
            with t_model.disable_adapter():
                raw = gen_safe(t_model, t_tok, t_proc, msgs, max_new_tokens=16, temperature=0.0)
            w, ok = parse_winner(raw)
            parsed += ok
            if flip and w in {"A", "B"}:
                w = "B" if w == "A" else "A"
            j14_labels.append(w)
            if w == "A":
                wa += 1
            elif w == "B":
                wb += 1
            else:
                ties += 1
        j14_res[f"{va}__vs__{vb}"] = {"wins_a": wa, "wins_b": wb, "ties": ties, "parse_rate": round(parsed / len(concepts), 4), "sign_p": two_sided_sign_p(wa, wb)}
        logger.info("Judge-14B [%s vs %s]: %s", va, vb, j14_res[f"{va}__vs__{vb}"])

    del s_model, t_model, t_base
    gc.collect()
    torch.cuda.empty_cache()

    # Judge 2: Qwen2.5-7B-Instruct (bf16) single-letter forced judge
    p7 = "/users/ali.gunder/models/Qwen2.5-7B-Instruct"
    j7_tok = AutoTokenizer.from_pretrained(p7)
    j7_model = AutoModelForCausalLM.from_pretrained(p7, torch_dtype=torch.bfloat16, device_map="auto")
    j7_model.eval()
    j7_proc = CJKBlockLogitsProcessor(build_utf8_exact_allowlist_ids(j7_tok))

    j7_res: Dict[str, Any] = {}
    j7_labels: List[str] = []
    for va, vb in pair_specs:
        wa = wb = ties = parsed = 0
        for c in concepts:
            flip = flips[(va, vb, c["id"])]
            da, db = (cmap[vb][c["id"]], cmap[va][c["id"]]) if flip else (cmap[va][c["id"]], cmap[vb][c["id"]])
            msgs = [
                {
                    "role": "system",
                    "content": "İki öğretmenden hangisi daha iyi Sokratik sorular sorup kavram yanılgısını daha etkili çözüyor? Sadece tek bir kelime yaz: A, B veya TIE.",
                },
                {"role": "user", "content": f"Kavram: {c['name']}\n[Öğretmen A]:\n{da['transcript']}\n\n[Öğretmen B]:\n{db['transcript']}\n\nKazanan (A / B / TIE):"},
            ]
            raw = gen_safe(j7_model, j7_tok, j7_proc, msgs, max_new_tokens=16, temperature=0.0)
            w, ok = parse_winner(raw)
            parsed += ok
            if flip and w in {"A", "B"}:
                w = "B" if w == "A" else "A"
            j7_labels.append(w)
            if w == "A":
                wa += 1
            elif w == "B":
                wb += 1
            else:
                ties += 1
        j7_res[f"{va}__vs__{vb}"] = {"wins_a": wa, "wins_b": wb, "ties": ties, "parse_rate": round(parsed / len(concepts), 4), "sign_p": two_sided_sign_p(wa, wb)}
        logger.info("Judge-7B [%s vs %s]: %s", va, vb, j7_res[f"{va}__vs__{vb}"])

    agree = round(sum(1 for a, b in zip(j14_labels, j7_labels) if a == b) / max(len(j14_labels), 1), 4)
    kappa = cohen_kappa(j14_labels, j7_labels)
    report = {
        "metadata": {"n_concepts": len(concepts), "n_pairwise_total": len(j14_labels), "elapsed_sec": round(time.time() - t0, 1)},
        "judge_14b_pairwise": j14_res,
        "judge_7b_pairwise": j7_res,
        "inter_judge_agreement": {"raw_agreement": agree, "cohens_kappa": kappa},
    }
    Path("results/hpc_dual_judge_pairwise_v6_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
