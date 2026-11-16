# -*- coding: utf-8 -*-
"""
Option 1: Fixed Cross-Judge + Blinded Pairwise A/B Evaluation + Deterministic Metrics
on the 324 CJK-free sessions from Job 34740 (results/hpc_cjk_blocked_dag_ablation_report.json).

1. Deterministic lexical/pedagogical metrics on all 324 transcripts (no LLM needed).
2. Fixed Qwen2.5-7B-Instruct pointwise judging (larger token budget, robust JSON/regex fallback).
3. Randomized-order Blinded Pairwise A/B comparison across all 108 paired triplets
   for all 3 arm pairs (324 pairs) using Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct.

Outputs:
  results/hpc_cross_judge_pairwise_report.json
"""

from __future__ import annotations

import gc
import json
import logging
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_hpc_cjk_blocked_dag_ablation import (
    CJKBlockLogitsProcessor,
    build_cjk_token_ids,
    cohens_kappa,
    generate_chat_nocjk,
)
from run_hpc_live_student_simulation import ISG_CONCEPTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CrossJudgePairwise")


def compute_deterministic_metrics(sess: Dict[str, Any], concept: Dict[str, Any]) -> Dict[str, Any]:
    tr = sess["transcript"]
    t1_tutor = tr[0]["tutor"] if tr else ""
    tutor_turns = [x["tutor"] for x in tr]
    student_turns = [x["student"] for x in tr]

    q_turns = sum(1 for t in tutor_turns if "?" in t)
    t1_has_question = int("?" in t1_tutor)
    avg_tutor_words = round(sum(len(t.split()) for t in tutor_turns) / max(len(tutor_turns), 1), 1)
    avg_student_words = round(sum(len(s.split()) for s in student_turns) / max(len(student_turns), 1), 1)

    prereq_tokens = []
    for p in concept["prerequisites"]:
        prereq_tokens.extend([w.lower() for w in re.findall(r"\w+", p) if len(w) >= 4])
    tutor_all_low = " ".join(tutor_turns).lower()
    prereq_hits = sum(1 for tok in set(prereq_tokens) if tok in tutor_all_low)
    prereq_lexical_hit = int(prereq_hits > 0) if prereq_tokens else 1

    student_echo = int(
        any("hakkındaki yanlış düşünceni doğrudan sor" in s.lower() for s in student_turns)
    )
    return {
        "q_turns": q_turns,
        "t1_has_question": t1_has_question,
        "avg_tutor_words": avg_tutor_words,
        "avg_student_words": avg_student_words,
        "prereq_lexical_hit": prereq_lexical_hit,
        "student_prompt_echo": student_echo,
    }


def parse_judge_json(raw: str) -> Tuple[Dict[str, int], bool]:
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    parsed = {"misconception_resolved": 0, "socratic_score": 3, "prerequisite_score": 3, "turkish_purity": 5}
    ok = False
    if m:
        candidate = m.group(0)
        # Fix Python booleans if emitted
        candidate = re.sub(r"\bTrue\b", "true", candidate)
        candidate = re.sub(r"\bFalse\b", "false", candidate)
        try:
            obj = json.loads(candidate)
            parsed["misconception_resolved"] = int(bool(obj.get("misconception_resolved", False)))
            parsed["socratic_score"] = int(min(5, max(1, int(obj.get("socratic_score", 3)))))
            parsed["prerequisite_score"] = int(min(5, max(1, int(obj.get("prerequisite_score", 3)))))
            parsed["turkish_purity"] = int(min(5, max(1, int(obj.get("turkish_purity", 5)))))
            ok = True
        except Exception:
            pass
    if not ok:
        m_res = re.search(r"misconception_resolved[\"']?\s*[:=]\s*(true|false|1|0)", cleaned, re.I)
        if m_res:
            parsed["misconception_resolved"] = 1 if m_res.group(1).lower() in ("true", "1") else 0
            ok = True
        for key in ("socratic_score", "prerequisite_score", "turkish_purity"):
            mk = re.search(rf"{key}[\"']?\s*[:=]\s*(\d)", cleaned, re.I)
            if mk:
                parsed[key] = int(min(5, max(1, int(mk.group(1)))))
    return parsed, ok


def pointwise_judge(
    model, tok, cjk_proc: CJKBlockLogitsProcessor, concept: Dict[str, Any], sess: Dict[str, Any]
) -> Tuple[Dict[str, int], bool]:
    conv_text = "\n".join(
        [f"[Tur {x['turn']}] Öğrenci: {x['student']}\n[Tur {x['turn']}] Eğitmen: {x['tutor']}" for x in sess["transcript"]]
    )
    prompt = (
        "Aşağıdaki öğrenci-eğitmen diyaloğunu değerlendir ve YALNIZCA tek satır JSON döndür.\n"
        f"Kavram: {concept['name']}\n"
        f"Doğru İlke: {concept['gold_principle']}\n"
        f"Başlangıç Yanılgısı: {concept['misconception']}\n\n"
        f"Diyalog:\n{conv_text}\n\n"
        "Kurallar:\n"
        "- misconception_resolved: Öğrenci diyalog sonunda doğru ilkeyi kabul edip kendi ifadesiyle doğruladıysa true, yoksa false.\n"
        "- socratic_score (1-5): Sokratik yönlendirme kalitesi.\n"
        "- prerequisite_score (1-5): Önkoşul kavramlarla bağ kurma.\n"
        "- turkish_purity (1-5): Türkçe akıcılık ve temizlik.\n"
        'Çıktı formatı: {"misconception_resolved": true, "socratic_score": 4, "prerequisite_score": 3, "turkish_purity": 5}'
    )
    msgs = [
        {"role": "system", "content": "Sen nesnel bir eğitim bilimleri hakemisin. Yalnızca geçerli JSON üret."},
        {"role": "user", "content": prompt},
    ]
    raw = generate_chat_nocjk(model, tok, msgs, cjk_processor=cjk_proc, max_new_tokens=200, temperature=0.0)
    return parse_judge_json(raw)


def pairwise_judge(
    model,
    tok,
    cjk_proc: CJKBlockLogitsProcessor,
    concept: Dict[str, Any],
    sess_a: Dict[str, Any],
    sess_b: Dict[str, Any],
    rng: random.Random,
) -> str:
    swap = rng.random() < 0.5
    first, second = (sess_b, sess_a) if swap else (sess_a, sess_b)
    txt1 = "\n".join([f"Tur {x['turn']} Öğrenci: {x['student']}\nTur {x['turn']} Eğitmen: {x['tutor']}" for x in first["transcript"]])
    txt2 = "\n".join([f"Tur {x['turn']} Öğrenci: {x['student']}\nTur {x['turn']} Eğitmen: {x['tutor']}" for x in second["transcript"]])
    prompt = (
        f"Aşağıda aynı İSG kavramı ('{concept['name']}') ve aynı başlangıç yanılgısı ('{concept['misconception']}') için "
        "iki farklı eğitmenin yürüttüğü iki diyalog var.\n"
        f"Doğru ilke: {concept['gold_principle']}\n\n"
        f"=== DİYALOG X ===\n{txt1}\n\n"
        f"=== DİYALOG Y ===\n{txt2}\n\n"
        "Hangi eğitmenin (X mi, Y mi, yoksa EŞİT mi) öğrencinin kavram yanılgısını Sokratik biçimde çözmede daha başarılı olduğunu seç.\n"
        'YALNIZCA tek satır JSON döndür: {"winner": "X" | "Y" | "TIE"}'
    )
    msgs = [
        {"role": "system", "content": "Sen tarafsız bir eğitim değerlendirme hakemisin. Sadece JSON döndür."},
        {"role": "user", "content": prompt},
    ]
    raw = generate_chat_nocjk(model, tok, msgs, cjk_processor=cjk_proc, max_new_tokens=80, temperature=0.0)
    m = re.search(r'"winner"\s*:\s*"?(X|Y|TIE)"?', raw, re.I)
    choice = m.group(1).upper() if m else "TIE"
    if choice == "TIE":
        return "tie"
    if not swap:
        return "A" if choice == "X" else "B"
    return "B" if choice == "X" else "A"


def two_sided_sign_p(wins_a: int, wins_b: int) -> float:
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = min(wins_a, wins_b)
    p1 = sum(math.comb(n, i) * (0.5 ** n) for i in range(0, k + 1))
    return round(min(1.0, 2.0 * p1), 6)


def main() -> None:
    t0 = time.time()
    src_path = Path("results/hpc_cjk_blocked_dag_ablation_report.json")
    data = json.loads(src_path.read_text(encoding="utf-8"))
    sessions = data["sessions"]
    concept_by_id = {c["id"]: c for c in ISG_CONCEPTS}

    # 1. Deterministic metrics
    det_by_cond: Dict[str, List[Dict[str, Any]]] = {}
    for s in sessions:
        dm = compute_deterministic_metrics(s, concept_by_id[s["concept_id"]])
        s["deterministic"] = dm
        det_by_cond.setdefault(s["condition"], []).append(dm)

    det_summary = {}
    for cond, items in det_by_cond.items():
        n = len(items)
        det_summary[cond] = {
            "n": n,
            "t1_question_rate": round(sum(x["t1_has_question"] for x in items) / n, 4),
            "avg_question_turns_per_4": round(sum(x["q_turns"] for x in items) / n, 3),
            "avg_tutor_words": round(sum(x["avg_tutor_words"] for x in items) / n, 1),
            "avg_student_words": round(sum(x["avg_student_words"] for x in items) / n, 1),
            "prereq_lexical_hit_rate": round(sum(x["prereq_lexical_hit"] for x in items) / n, 4),
            "student_prompt_echo_rate": round(sum(x["student_prompt_echo"] for x in items) / n, 4),
        }
    logger.info("Deterministic summary: %s", json.dumps(det_summary, ensure_ascii=False))

    # Organize paired triplets
    by_cond_map: Dict[str, Dict[Tuple[int, str, str], Dict[str, Any]]] = {}
    for s in sessions:
        by_cond_map.setdefault(s["condition"], {})[(s["seed"], s["concept_id"], s["persona_id"])] = s
    keys = sorted(by_cond_map["no_dag_base_14b"].keys())

    pairs = [
        ("task_separated_hybrid", "no_dag_base_14b"),
        ("task_separated_hybrid", "prompt_dag_rag_base_14b"),
        ("prompt_dag_rag_base_14b", "no_dag_base_14b"),
    ]

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Fixed Qwen2.5-7B-Instruct Pointwise + Pairwise
    logger.info("Loading Qwen2.5-7B-Instruct for fixed pointwise + pairwise judging...")
    j7_tok = AutoTokenizer.from_pretrained("/users/ali.gunder/models/Qwen2.5-7B-Instruct")
    j7_model = AutoModelForCausalLM.from_pretrained("/users/ali.gunder/models/Qwen2.5-7B-Instruct", quantization_config=bnb, device_map="auto")
    j7_model.eval()
    j7_proc = CJKBlockLogitsProcessor(build_cjk_token_ids(j7_tok))

    parse_ok_7b = 0
    for idx, s in enumerate(sessions, start=1):
        sc, ok = pointwise_judge(j7_model, j7_tok, j7_proc, concept_by_id[s["concept_id"]], s)
        s["scores_7b_fixed"] = sc
        parse_ok_7b += int(ok)
        if idx % 108 == 0:
            logger.info("7B pointwise progress: %d/%d (parse_ok=%d)", idx, len(sessions), parse_ok_7b)

    rng = random.Random(2026)
    pw_7b: Dict[str, Dict[str, Any]] = {}
    for c_a, c_b in pairs:
        wa, wb, ties = 0, 0, 0
        for k in keys:
            res = pairwise_judge(j7_model, j7_tok, j7_proc, concept_by_id[k[1]], by_cond_map[c_a][k], by_cond_map[c_b][k], rng)
            if res == "A":
                wa += 1
            elif res == "B":
                wb += 1
            else:
                ties += 1
        pw_7b[f"{c_a}__vs__{c_b}"] = {"wins_a": wa, "wins_b": wb, "ties": ties, "sign_p": two_sided_sign_p(wa, wb)}
    logger.info("7B pairwise: %s", json.dumps(pw_7b, ensure_ascii=False))

    del j7_model
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Qwen2.5-14B-Instruct Blinded Pairwise
    logger.info("Loading Qwen2.5-14B-Instruct for blinded pairwise judging...")
    j14_tok = AutoTokenizer.from_pretrained("/users/ali.gunder/models/Qwen2.5-14B-Instruct")
    j14_model = AutoModelForCausalLM.from_pretrained("/users/ali.gunder/models/Qwen2.5-14B-Instruct", quantization_config=bnb, device_map="auto")
    j14_model.eval()
    j14_proc = CJKBlockLogitsProcessor(build_cjk_token_ids(j14_tok))

    pw_14b: Dict[str, Dict[str, Any]] = {}
    for c_a, c_b in pairs:
        wa, wb, ties = 0, 0, 0
        for k in keys:
            res = pairwise_judge(j14_model, j14_tok, j14_proc, concept_by_id[k[1]], by_cond_map[c_a][k], by_cond_map[c_b][k], rng)
            if res == "A":
                wa += 1
            elif res == "B":
                wb += 1
            else:
                ties += 1
        pw_14b[f"{c_a}__vs__{c_b}"] = {"wins_a": wa, "wins_b": wb, "ties": ties, "sign_p": two_sided_sign_p(wa, wb)}
    logger.info("14B pairwise: %s", json.dumps(pw_14b, ensure_ascii=False))

    # Summarize 7B fixed pointwise & agreement with 14B pointwise
    pointwise_7b_summary = {}
    for cond in ("no_dag_base_14b", "prompt_dag_rag_base_14b", "task_separated_hybrid"):
        sub = [s for s in sessions if s["condition"] == cond]
        n = len(sub)
        pointwise_7b_summary[cond] = {
            "n": n,
            "resolution_rate_7b": round(sum(x["scores_7b_fixed"]["misconception_resolved"] for x in sub) / n, 4),
            "socratic_mean_7b": round(sum(x["scores_7b_fixed"]["socratic_score"] for x in sub) / n, 3),
            "prerequisite_mean_7b": round(sum(x["scores_7b_fixed"]["prerequisite_score"] for x in sub) / n, 3),
        }

    y14 = [s["scores"]["misconception_resolved"] for s in sessions]
    y7 = [s["scores_7b_fixed"]["misconception_resolved"] for s in sessions]
    inter_judge = {
        "n_sessions": len(sessions),
        "parse_ok_7b": parse_ok_7b,
        "raw_agreement": round(sum(1 for a, b in zip(y14, y7) if a == b) / len(sessions), 4),
        "cohens_kappa": cohens_kappa(y14, y7),
    }

    out_path = Path("results/hpc_cross_judge_pairwise_report.json")
    out_path.write_text(
        json.dumps(
            {
                "metadata": {"source_report": str(src_path), "elapsed_sec": round(time.time() - t0, 1)},
                "deterministic_summary": det_summary,
                "pointwise_7b_fixed_summary": pointwise_7b_summary,
                "inter_judge_14b_vs_7b_fixed": inter_judge,
                "pairwise_7b": pw_7b,
                "pairwise_14b": pw_14b,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Option 1 Finished in %.1fs -> %s", time.time() - t0, out_path)


if __name__ == "__main__":
    main()
