# -*- coding: utf-8 -*-
"""
Experiment 5: Alignment Tax Check — CEFR Readability (B1/B2/C1) & Structured JSON İSG Extraction
across Base 14B, CEFR-SFT, Socratic-V2, On-Policy DPO-V1, and On-Policy DPO-V2.

Measures whether multi-turn Socratic DPO alignment causes catastrophic forgetting on EduTerm's
core structured CEFR explanation and İSG terminology extraction tasks.

Outputs:
  results/hpc_alignment_tax_cefr_report.json
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_hpc_cjk_blocked_dag_ablation import CJKBlockLogitsProcessor, generate_chat_nocjk
from run_hpc_dpo_3seed_ablation import resolve_dpo_path
from run_hpc_dpo_r2_nonlatin import build_nonlatin_token_ids, has_nonlatin
from run_hpc_live_student_simulation import ISG_CONCEPTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp5AlignmentTax")


def calculate_atesman_score(text: str) -> float:
    if not text or not text.strip():
        return 60.0
    vowels = "aeıioöuü"
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    if not words:
        return 60.0
    syllables = sum(sum(1 for c in w if c in vowels) for w in words)
    spw = syllables / max(1, len(words))
    wps = len(words) / max(1, len(sentences))
    score = 198.825 - (40.175 * spw) - (2.610 * wps)
    return max(0.0, min(100.0, round(score, 1)))


def evaluate_cefr_compliance(target_level: str, atesman_score: float) -> bool:
    if target_level == "B1":
        return atesman_score >= 50.0
    if target_level == "B2":
        return 35.0 <= atesman_score < 50.0
    if target_level == "C1":
        return atesman_score < 35.0
    return False


def resolve_dpo_v2_path() -> str:
    base = Path("checkpoints/eduterm_14b_onpolicy_dpo_v2")
    if (base / "adapter_config.json").exists():
        return str(base)
    if (base / "dpo_v2" / "adapter_config.json").exists():
        return str(base / "dpo_v2")
    raise FileNotFoundError(f"Could not find adapter_config.json in {base}")


def main() -> None:
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
    )
    base_path = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
    tok = AutoTokenizer.from_pretrained(base_path)
    base = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb, device_map="auto")
    model = PeftModel.from_pretrained(base, "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora", adapter_name="cefr")
    model.load_adapter("/users/ali.gunder/eduterm-hybrid-sim/checkpoints/eduterm_14b_socratic_v2", adapter_name="socratic_v2")
    model.load_adapter(resolve_dpo_path(), adapter_name="dpo_v1")
    model.load_adapter(resolve_dpo_v2_path(), adapter_name="dpo_v2")
    model.eval()

    proc = CJKBlockLogitsProcessor(build_nonlatin_token_ids(tok))
    models = ["base_14b", "cefr", "socratic_v2", "dpo_v1", "dpo_v2"]
    cefr_levels = [
        ("B1", "Sade, günlük örneklerle lise/1. sınıf düzeyinde anlaşılır kısa cümlelerle açıkla."),
        ("B2", "Teknik üniversite düzeyinde, mesleki terminoloji kullanarak yapılandırılmış biçimde açıkla."),
        ("C1", "Akademik, mevzuat ve mühendislik kavramlarıyla yoğun ve ileri düzeyde analiz ederek açıkla."),
    ]

    results: Dict[str, Any] = {}
    for m_name in models:
        cefr_records: List[Dict[str, Any]] = []
        json_records: List[Dict[str, Any]] = []

        for concept in ISG_CONCEPTS:
            # 1. CEFR B1 / B2 / C1 readability probes
            for lvl, lvl_instr in cefr_levels:
                msgs = [
                    {"role": "system", "content": f"Sen EduTerm İSG uzmanısın. Hedef okunabilirlik seviyesi: CEFR-{lvl}. {lvl_instr}"},
                    {"role": "user", "content": f"'{concept['name']}' kavramını CEFR-{lvl} düzeyinde 3-4 cümleyle Türkçe açıkla."},
                ]
                if m_name == "base_14b":
                    with model.disable_adapter():
                        out = generate_chat_nocjk(model, tok, msgs, cjk_processor=proc, max_new_tokens=180, temperature=0.0)
                else:
                    model.set_adapter(m_name)
                    out = generate_chat_nocjk(model, tok, msgs, cjk_processor=proc, max_new_tokens=180, temperature=0.0)
                atesman = calculate_atesman_score(out)
                cefr_records.append({
                    "concept_id": concept["id"],
                    "level": lvl,
                    "atesman": atesman,
                    "compliant": evaluate_cefr_compliance(lvl, atesman),
                    "nonlatin": has_nonlatin(out),
                })

            # 2. Structured JSON İSG extraction probe
            prereq_str = ", ".join(concept["prerequisites"]) if concept["prerequisites"] else "Yok"
            j_msgs = [
                {"role": "system", "content": "Sen EduTerm İSG JSON çıkarım motorusun. SADECE geçerli tek bir JSON nesnesi döndür."},
                {
                    "role": "user",
                    "content": (
                        f"Kavram: {concept['name']}\nÖnkoşullar: {prereq_str}\nİlke: {concept['gold_principle']}\n"
                        'Şu JSON şemasını doldur: {"term": "...", "cefr_level": "B1|B2|C1", "prerequisites": [...], "key_rule": "..."}'
                    ),
                },
            ]
            if m_name == "base_14b":
                with model.disable_adapter():
                    j_out = generate_chat_nocjk(model, tok, j_msgs, cjk_processor=proc, max_new_tokens=180, temperature=0.0)
            else:
                model.set_adapter(m_name)
                j_out = generate_chat_nocjk(model, tok, j_msgs, cjk_processor=proc, max_new_tokens=180, temperature=0.0)

            cleaned = re.sub(r"```(?:json)?", "", j_out).replace("```", "").strip()
            m_json = re.search(r"\{.*\}", cleaned, re.DOTALL)
            valid_json = False
            schema_ok = False
            if m_json:
                try:
                    obj = json.loads(m_json.group(0))
                    valid_json = True
                    schema_ok = all(k in obj for k in ("term", "cefr_level", "prerequisites", "key_rule"))
                except Exception:
                    pass
            json_records.append({"concept_id": concept["id"], "valid_json": valid_json, "schema_ok": schema_ok, "nonlatin": has_nonlatin(j_out)})

        by_lvl = {}
        for lvl, _ in cefr_levels:
            sub = [r for r in cefr_records if r["level"] == lvl]
            by_lvl[lvl] = {
                "mean_atesman": round(sum(r["atesman"] for r in sub) / len(sub), 2),
                "compliance_rate": round(sum(r["compliant"] for r in sub) / len(sub), 4),
            }
        b1_at = by_lvl["B1"]["mean_atesman"]
        c1_at = by_lvl["C1"]["mean_atesman"]
        results[m_name] = {
            "cefr_overall_compliance": round(sum(r["compliant"] for r in cefr_records) / len(cefr_records), 4),
            "cefr_b1_minus_c1_atesman_gap": round(b1_at - c1_at, 2),
            "by_level": by_lvl,
            "json_valid_rate": round(sum(r["valid_json"] for r in json_records) / len(json_records), 4),
            "json_schema_rate": round(sum(r["schema_ok"] for r in json_records) / len(json_records), 4),
            "nonlatin_leakage_rate": round(
                (sum(r["nonlatin"] for r in cefr_records) + sum(r["nonlatin"] for r in json_records))
                / (len(cefr_records) + len(json_records)),
                4,
            ),
        }
        logger.info("Alignment Tax [%s]: %s", m_name, json.dumps(results[m_name], ensure_ascii=False))

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hpc_alignment_tax_cefr_report.json"
    out_path.write_text(
        json.dumps({"metadata": {"elapsed_sec": round(time.time() - t0, 1)}, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Exp 5 Finished in %.1fs -> %s", time.time() - t0, out_path)


if __name__ == "__main__":
    main()
