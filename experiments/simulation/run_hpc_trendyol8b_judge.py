# -*- coding: utf-8 -*-
"""
Experiment 7: Independent Turkish External Judge Verification (Trendyol-LLM-8B-T1)
on Boğaziçi HPC (A40 48GB).

Re-evaluates saved transcripts from:
  - results/hpc_7b_bf16_3seed_nonlatin_report.json (216 sessions: 108 paired 7B-bf16)
  - results/hpc_dpo_r2_nonlatin_report.json (108 sessions: 36 paired Kumru-2B triplets)

Computes blinded pointwise rubric scores, randomized A/B pairwise preference
(onpolicy_dpo_v2 vs task_separated_hybrid), and inter-judge Cohen's kappa vs Qwen2.5-7B.

Outputs:
  results/hpc_trendyol8b_judge_report.json
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_hpc_cjk_blocked_dag_ablation import (
    CJKBlockLogitsProcessor,
    bootstrap_ci,
    exact_mcnemar_p,
)
from run_hpc_dpo_r2_nonlatin import build_nonlatin_token_ids
from run_hpc_live_student_simulation import ISG_CONCEPTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp7Judge14B")


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate_judge_text(model, tok, proc, messages: List[Dict[str, str]], max_new_tokens: int = 240) -> str:
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            logits_processor=[proc] if proc is not None else None,
            pad_token_id=tok.eos_token_id,
        )
    raw = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return strip_think(raw)


def parse_json_obj(text: str) -> Dict[str, Any] | None:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def pointwise_judge(model, tok, proc, concept: Dict[str, Any], session: Dict[str, Any]) -> Tuple[Dict[str, int], bool]:
    conv = "\n".join(
        f"Tur {t['turn']} - Öğrenci: {t['student']}\nTur {t['turn']} - Öğretmen: {t['tutor']}"
        for t in session["transcript"]
    )
    prompt = (
        f"Kavram: {concept['name']}\n"
        f"Hedef Doğru İlke: {concept['gold_principle']}\n"
        f"Başlangıç Yanılgısı: {concept['misconception']}\n\n"
        f"Diyalog:\n{conv}\n\n"
        "Aşağıdaki 4 ölçütü JSON olarak puanla:\n"
        "- misconception_resolved: 0 veya 1 (öğrenci son turlarda doğru ilkeyi kendi cümleleriyle kavradıysa 1)\n"
        "- socratic_score: 1-5 arası tam sayı (doğrudan cevabı söylemek yerine soruyla yönlendirme)\n"
        "- prerequisite_score: 1-5 arası tam sayı (ön koşul/temel kavram bağı)\n"
        "- turkish_purity: 1-5 arası tam sayı (doğal ve akıcı Türkçe)\n"
        'Sadece tek satır JSON döndür: {"misconception_resolved": 1, "socratic_score": 4, "prerequisite_score": 3, "turkish_purity": 5}'
    )
    msgs = [
        {"role": "system", "content": "Sen deneyimli bir Türkçe eğitim bilimleri değerlendiricisisin. Sadece geçerli JSON döndür."},
        {"role": "user", "content": prompt},
    ]
    raw = generate_judge_text(model, tok, proc, msgs, max_new_tokens=160)
    obj = parse_json_obj(raw)
    if not obj:
        return {"misconception_resolved": 0, "socratic_score": 3, "prerequisite_score": 3, "turkish_purity": 3}, False
    try:
        return {
            "misconception_resolved": int(bool(int(obj.get("misconception_resolved", 0)))),
            "socratic_score": max(1, min(5, int(obj.get("socratic_score", 3)))),
            "prerequisite_score": max(1, min(5, int(obj.get("prerequisite_score", 3)))),
            "turkish_purity": max(1, min(5, int(obj.get("turkish_purity", 3)))),
        }, True
    except Exception:
        return {"misconception_resolved": 0, "socratic_score": 3, "prerequisite_score": 3, "turkish_purity": 3}, False


def pairwise_judge(
    model, tok, proc, concept: Dict[str, Any], s_a: Dict[str, Any], s_b: Dict[str, Any], rng: random.Random
) -> str:
    flip = rng.random() < 0.5
    first, second = (s_b, s_a) if flip else (s_a, s_b)
    fmt = lambda s: "\n".join(f"T{t['turn']} Öğr: {t['student']}\nT{t['turn']} Eğt: {t['tutor']}" for t in s["transcript"])
    prompt = (
        f"Kavram: {concept['name']} | İlke: {concept['gold_principle']}\n\n"
        f"--- OTURUM A ---\n{fmt(first)}\n\n"
        f"--- OTURUM B ---\n{fmt(second)}\n\n"
        "Hangi öğretmen öğrencinin kavram yanılgısını daha iyi Sokratik yöntemle ve temiz Türkçeyle çözdü? "
        'Sadece JSON döndür: {"winner": "A" | "B" | "TIE"}'
    )
    msgs = [
        {"role": "system", "content": "Sen tarafsız bir pedagoji hakemisin. Sadece JSON döndür."},
        {"role": "user", "content": prompt},
    ]
    raw = generate_judge_text(model, tok, proc, msgs, max_new_tokens=80)
    obj = parse_json_obj(raw)
    w = str(obj.get("winner", "TIE")).upper() if obj else "TIE"
    if w not in ("A", "B", "TIE"):
        w = "TIE"
    if flip:
        return "B" if w == "A" else ("A" if w == "B" else "TIE")
    return w


def cohens_kappa(y1: List[int], y2: List[int]) -> float:
    n = len(y1)
    if n == 0:
        return 0.0
    po = sum(1 for a, b in zip(y1, y2) if a == b) / n
    p1_1 = sum(y1) / n
    p2_1 = sum(y2) / n
    pe = p1_1 * p2_1 + (1 - p1_1) * (1 - p2_1)
    if abs(1.0 - pe) < 1e-9:
        return 1.0 if po == 1.0 else 0.0
    return round((po - pe) / (1.0 - pe), 4)


def evaluate_report(
    model, tok, proc, report_path: Path, concept_by_id: Dict[str, Dict[str, Any]], rng: random.Random
) -> Dict[str, Any]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    sessions = data.get("sessions", [])
    by_cond: Dict[str, List[Dict[str, Any]]] = {}
    qwen_res: List[int] = []
    trend_res: List[int] = []
    parse_ok = 0

    by_key: Dict[Tuple[int, str, str], Dict[str, Dict[str, Any]]] = {}
    for idx, s in enumerate(sessions, start=1):
        sc, ok = pointwise_judge(model, tok, proc, concept_by_id[s["concept_id"]], s)
        if ok:
            parse_ok += 1
        qwen_res.append(int(s["scores"]["misconception_resolved"]))
        trend_res.append(int(sc["misconception_resolved"]))
        item = {
            "seed": s.get("seed", 42),
            "concept_id": s["concept_id"],
            "persona_id": s["persona_id"],
            "condition": s["condition"],
            "trendyol_scores": sc,
            "qwen_scores": s["scores"],
            "transcript": s["transcript"],
        }
        by_cond.setdefault(s["condition"], []).append(item)
        by_key.setdefault((s.get("seed", 42), s["concept_id"], s["persona_id"]), {})[s["condition"]] = item
        if idx % 36 == 0 or idx == len(sessions):
            logger.info("[%s] Judged %d/%d sessions (parse_ok=%d)", report_path.name, idx, len(sessions), parse_ok)

    summary = {}
    for cond, items in by_cond.items():
        n = len(items)
        res = [x["trendyol_scores"]["misconception_resolved"] for x in items]
        summary[cond] = {
            "total_sessions": n,
            "misconception_resolution_rate": round(sum(res) / n, 4),
            "resolution_bootstrap_95ci": list(bootstrap_ci(res)),
            "socratic_mean": round(sum(x["trendyol_scores"]["socratic_score"] for x in items) / n, 3),
            "prerequisite_mean": round(sum(x["trendyol_scores"]["prerequisite_score"] for x in items) / n, 3),
            "turkish_purity_mean": round(sum(x["trendyol_scores"]["turkish_purity"] for x in items) / n, 3),
        }

    # Pairwise & McNemar: onpolicy_dpo_v2 vs task_separated_hybrid
    wins_dpo, wins_hyb, ties = 0, 0, 0
    a1_b0, a0_b1 = 0, 0
    for key, cmap in by_key.items():
        if "onpolicy_dpo_v2" in cmap and "task_separated_hybrid" in cmap:
            dpo_s = cmap["onpolicy_dpo_v2"]
            hyb_s = cmap["task_separated_hybrid"]
            w = pairwise_judge(model, tok, proc, concept_by_id[key[1]], dpo_s, hyb_s, rng)
            if w == "A":
                wins_dpo += 1
            elif w == "B":
                wins_hyb += 1
            else:
                ties += 1
            r_d = dpo_s["trendyol_scores"]["misconception_resolved"]
            r_h = hyb_s["trendyol_scores"]["misconception_resolved"]
            if r_d == 1 and r_h == 0:
                a1_b0 += 1
            elif r_d == 0 and r_h == 1:
                a0_b1 += 1

    agree = sum(1 for a, b in zip(qwen_res, trend_res) if a == b) / max(len(qwen_res), 1)
    return {
        "total_sessions": len(sessions),
        "parse_success_rate": round(parse_ok / max(len(sessions), 1), 4),
        "inter_judge_vs_qwen7b": {
            "raw_agreement": round(agree, 4),
            "cohens_kappa": cohens_kappa(qwen_res, trend_res),
        },
        "summary": summary,
        "paired_dpo_v2_vs_hybrid": {
            "blinded_pairwise": {"dpo_v2_wins": wins_dpo, "hybrid_wins": wins_hyb, "ties": ties},
            "mcnemar": {"a1_b0": a1_b0, "a0_b1": a0_b1, "p_value": exact_mcnemar_p(a1_b0, a0_b1)},
        },
    }


def main():
    t0 = time.time()
    rng = random.Random(42)
    concept_by_id = {c["id"]: c for c in ISG_CONCEPTS}

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    judge_path = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
    tok = AutoTokenizer.from_pretrained(judge_path)
    model = AutoModelForCausalLM.from_pretrained(judge_path, quantization_config=bnb, device_map="auto")
    model.eval()
    proc = CJKBlockLogitsProcessor(build_nonlatin_token_ids(tok))

    out_report: Dict[str, Any] = {
        "metadata": {"judge_model": "Qwen2.5-14B-Instruct (4-bit NF4, Non-Latin blocked)"},
        "reports": {},
    }

    for name, path in [
        ("exp6_7b_bf16_3seed", Path("results/hpc_7b_bf16_3seed_nonlatin_report.json")),
        ("exp4_dpo_r2_kumru2b", Path("results/hpc_dpo_r2_nonlatin_report.json")),
    ]:
        if path.exists():
            out_report["reports"][name] = evaluate_report(model, tok, proc, path, concept_by_id, rng)
            Path("results/hpc_trendyol8b_judge_partial.json").write_text(
                json.dumps(out_report, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    out_report["metadata"]["elapsed_sec"] = round(time.time() - t0, 1)
    Path("results/hpc_trendyol8b_judge_report.json").write_text(
        json.dumps(out_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Exp 7 Trendyol-8B Judge Report:\n%s", json.dumps(out_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
