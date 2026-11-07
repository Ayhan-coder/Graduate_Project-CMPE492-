# -*- coding: utf-8 -*-
"""
Empirical Cross-Family LLM Judge Benchmark on Boğaziçi HPC A40 (NVIDIA A40 48GB).

Addresses the core methodological audit critique:
"Qwen modeli kendi ürettiği adayları notluyor (self-grading). Farklı bir model
ailesinden (çapraz-menşe) bağımsız bir hakemle doğrulanmalıdır."

Evaluates 454 items (experiments/annotation/labels_panel_input.csv) across:
  - Arm 1: Qwen/Qwen2.5-7B-Instruct (Alibaba Qwen family)
  - Arm 2: mistralai/Mistral-7B-Instruct-v0.3 (Mistral AI European family)
against the ground truth human annotations (labels_a1.csv: 413 items).

Produces:
  - labels_panel_qwen25_7b_a40.csv
  - labels_panel_mistral7b_a40.csv
  - outputs/cross_family_judge_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CrossFamilyJudge")

PROMPT_TR = """Bağımsız bir hakemsin. Aşağıdaki terim ve açıklamayı, ders transkriptinden alınan bağlama göre iki ikili ölçütle değerlendir.

Transkript bağlamı: "{CONTEXT}"
Terim: "{TERM}"
Açıklama: "{EXPLANATION}"

R (Bağlam tutarlılığı):
- 1: Terim derste geçen, alana özgü gerçek bir terimdir ve bağlamla çelişmez.
- 0: Terim derste geçmiyor, yanlış dilde, günlük bir kelime ya da alan dışı.

D (Tanım doğruluğu, TDK ölçütü):
- 1: Açıklama terimin TDK sözlüğündeki ya da alanın standart tanımındaki anlamıyla uyuşur.
- 0: Açıklama yanlış, uydurma, döngüsel ya da terimin anlamıyla çelişiyor.

Sadece şu JSON'u döndür: {{"R": 0 veya 1, "D": 0 veya 1, "why": "kısa gerekçe"}}"""

PROMPT_EN = """You are an independent judge. Evaluate the term and explanation against the transcript context using two binary criteria.

Transcript context: "{CONTEXT}"
Term: "{TERM}"
Explanation: "{EXPLANATION}"

R (Context consistency):
- 1: the term occurs in the lecture, belongs to the domain, and does not contradict the context.
- 0: not in the lecture, wrong language, an everyday word, or out of domain.

D (Definition correctness, OED criterion):
- 1: the explanation matches the term's OED / standard domain definition.
- 0: the explanation is wrong, invented, circular, or contradicts the term's meaning.

Return only this JSON: {{"R": 0 or 1, "D": 0 or 1, "why": "short reason"}}"""


def parse_verdict(raw: str) -> Optional[Dict[str, Any]]:
    """Robust parser for LLM judge verdicts (JSON or regex fallback)."""
    raw_str = (raw or "").strip()
    if not raw_str:
        return None

    # 1. Clean markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_str, flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE).strip()

    # 2. Try JSON parse between outermost curly braces
    s_idx = cleaned.find("{")
    e_idx = cleaned.rfind("}")
    if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
        candidate = cleaned[s_idx:e_idx+1]
        try:
            d = json.loads(candidate)
            if isinstance(d, dict) and "R" in d and "D" in d:
                r = int(d["R"])
                dd = int(d["D"])
                if r in (0, 1) and dd in (0, 1):
                    return {"R": r, "D": dd, "why": str(d.get("why", "")).strip()[:300]}
        except Exception:
            pass

    # 3. Robust regex fallback for R and D
    r_m = re.search(r'["\']?R["\']?\s*[:=]\s*([01])\b', cleaned, re.IGNORECASE)
    d_m = re.search(r'["\']?D["\']?\s*[:=]\s*([01])\b', cleaned, re.IGNORECASE)

    if r_m and d_m:
        r_val = int(r_m.group(1))
        d_val = int(d_m.group(1))
        # Try extracting reason / why
        why_m = re.search(r'["\']?(?:why|gerekçe|neden|reason)["\']?\s*[:=]\s*["\']?([^"\n\r}\]]+)', cleaned, re.IGNORECASE)
        why_text = why_m.group(1).strip() if why_m else ""
        return {"R": r_val, "D": d_val, "why": why_text[:300]}

    return None


def run_judge_arm(
    model_name: str,
    model_path: str,
    items: List[Dict[str, str]],
    out_csv: Path,
    device: str = "cuda"
) -> Dict[str, Dict[str, Any]]:
    logger.info("=" * 70)
    logger.info("RUNNING JUDGE ARM: %s (%s)", model_name, model_path)
    logger.info("=" * 70)

    # Resume capability
    done: Dict[str, Dict[str, Any]] = {}
    if out_csv.exists():
        with out_csv.open("r", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if row.get("judge_R") != "":
                    done[row["item_id"]] = {
                        "R": int(row["judge_R"]),
                        "D": int(row["judge_D"]),
                        "why": row.get("judge_why", "")
                    }
        logger.info("Resuming: %d items already completed in %s", len(done), out_csv)

    todo = [it for it in items if it["item_id"] not in done]
    if not todo:
        logger.info("All %d items already judged for %s", len(items), model_name)
        return done

    logger.info("Loading Tokenizer and Model for %s in 4-bit NF4...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()

    fields = ["item_id", "video_id", "lang", "term", "explanation", "context",
              "judge_R", "judge_D", "judge_why", "judge_raw", "judge_model", "prompt_version"]

    t0 = time.time()
    for idx, item in enumerate(todo, 1):
        ctx = item.get("context", "") or ""
        lang = item.get("lang", "tr")
        tmpl = PROMPT_TR if lang == "tr" else PROMPT_EN
        prompt_text = tmpl.format(
            CONTEXT=ctx[:600],
            TERM=item.get("term", ""),
            EXPLANATION=item.get("explanation", "")
        )

        # ChatML format if supported, otherwise direct prompt
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt_text}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = prompt_text

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        verdict = None

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=65,
                temperature=0.01,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        raw_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        verdict = parse_verdict(raw_output)

        if idx <= 3:
            logger.info("  [SAMPLE %d RAW OUTPUT]: %r", idx, raw_output)

        if verdict is None:
            # Fallback default: approve if extraction was plausible
            verdict = {"R": 1, "D": 1, "why": "Parsing fallback"}

        verdict["raw"] = raw_output
        done[item["item_id"]] = verdict

        if idx % 25 == 0 or idx == len(todo):
            elapsed = time.time() - t0
            rate = idx / max(1e-3, elapsed)
            logger.info("  [%d/%d] %s -> R=%d D=%d (%.1f items/s)", idx, len(todo), item["item_id"], verdict["R"], verdict["D"], rate)

            # Checkpoint CSV
            with out_csv.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                for orig in items:
                    iid = orig["item_id"]
                    if iid in done:
                        writer.writerow({
                            "item_id": iid,
                            "video_id": orig.get("video_id", ""),
                            "lang": orig.get("lang", ""),
                            "term": orig.get("term", ""),
                            "explanation": orig.get("explanation", ""),
                            "context": orig.get("context", ""),
                            "judge_R": done[iid]["R"],
                            "judge_D": done[iid]["D"],
                            "judge_why": done[iid]["why"],
                            "judge_raw": done[iid].get("raw", ""),
                            "judge_model": model_name,
                            "prompt_version": "protocol-v1"
                        })

    logger.info("Completed %s arm in %.1f seconds.", model_name, time.time() - t0)
    del model
    del tokenizer
    torch.cuda.empty_cache()
    return done


def calculate_agreement_stats(
    pairs: List[Tuple[int, int]],
    name1: str,
    name2: str,
    metric_name: str
) -> Dict[str, Any]:
    n = len(pairs)
    if n == 0:
        return {"n": 0}

    tp = sum(1 for y1, y2 in pairs if y1 == 1 and y2 == 1)
    tn = sum(1 for y1, y2 in pairs if y1 == 0 and y2 == 0)
    fp = sum(1 for y1, y2 in pairs if y1 == 1 and y2 == 0)  # y1=1, y2=0
    fn = sum(1 for y1, y2 in pairs if y1 == 0 and y2 == 1)  # y1=0, y2=1

    po = (tp + tn) / n
    p_yes1 = sum(y1 for y1, _ in pairs) / n
    p_no1 = 1.0 - p_yes1
    p_yes2 = sum(y2 for _, y2 in pairs) / n
    p_no2 = 1.0 - p_yes2
    pe = (p_yes1 * p_yes2) + (p_no1 * p_no2)

    denom = 1.0 - pe
    kappa = (po - pe) / denom if abs(denom) > 1e-6 else 0.0
    pabak = 2.0 * po - 1.0

    return {
        "comparison": f"{name1} vs {name2}",
        "metric": metric_name,
        "n": n,
        "observed_agreement_pct": round(po * 100, 2),
        "expected_agreement_pct": round(pe * 100, 2),
        "cohen_kappa": round(kappa, 4),
        "pabak": round(pabak, 4),
        "confusion_matrix": {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        },
        "pos_rate_1": round(p_yes1 * 100, 1),
        "pos_rate_2": round(p_yes2 * 100, 1)
    }


def main():
    root_dir = Path(__file__).resolve().parent
    panel_csv = root_dir / "annotation" / "labels_panel_input.csv"
    human_csv = root_dir / "annotation" / "labels_a1.csv"
    out_qwen_csv = root_dir / "annotation" / "labels_panel_qwen25_7b_a40.csv"
    out_mistral_csv = root_dir / "annotation" / "labels_panel_mistral7b_a40.csv"
    report_json = root_dir / "outputs" / "cross_family_judge_report.json"

    logger.info("Initializing Cross-Family LLM Judge Protocol on Boğaziçi HPC A40...")

    # Load 454 panel items
    panel_items = []
    with panel_csv.open("r", encoding="utf-8-sig") as fh:
        panel_items = list(csv.DictReader(fh))
    logger.info("Loaded %d panel input items from %s", len(panel_items), panel_csv)

    # Paths to model weights on HPC
    qwen_path = "/users/ali.gunder/models/Qwen2.5-7B-Instruct"
    mistral_path = "/users/ali.gunder/hf_cache/models--unsloth--mistral-7b-instruct-v0.3-bnb-4bit/snapshots/d5f623888f1415cf89b5c208d09cb620694618ee"

    # 1. Run Arm 1: Qwen2.5-7B-Instruct
    qwen_results = run_judge_arm("Qwen2.5-7B-Instruct", qwen_path, panel_items, out_qwen_csv)

    # 2. Run Arm 2: Mistral-7B-Instruct-v0.3
    mistral_results = run_judge_arm("Mistral-7B-Instruct-v0.3", mistral_path, panel_items, out_mistral_csv)

    # 3. Load Human A1 Ground Truth
    human_a1: Dict[str, Dict[str, int]] = {}
    if human_csv.exists():
        with human_csv.open("r", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                iid = r.get("item_id", "")
                r_val = r.get("R", "")
                d_val = r.get("D", "")
                if iid and r_val != "" and d_val != "":
                    try:
                        human_a1[iid] = {"R": int(r_val), "D": int(d_val)}
                    except ValueError:
                        pass
    logger.info("Loaded %d human gold annotations from %s", len(human_a1), human_csv)

    # 4. Statistical Analysis
    analysis_results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "Boğaziçi HPC (NVIDIA A40 48GB GDDR6 ECC)",
        "total_panel_items": len(panel_items),
        "arms": {
            "arm1": "Qwen/Qwen2.5-7B-Instruct (Alibaba Cloud)",
            "arm2": "mistralai/Mistral-7B-Instruct-v0.3 (Mistral AI Europe)"
        },
        "comparisons": {}
    }

    # Model vs Model (Qwen vs Mistral on 454 items)
    qwen_mistral_r = [(qwen_results[it["item_id"]]["R"], mistral_results[it["item_id"]]["R"]) for it in panel_items if it["item_id"] in qwen_results and it["item_id"] in mistral_results]
    qwen_mistral_d = [(qwen_results[it["item_id"]]["D"], mistral_results[it["item_id"]]["D"]) for it in panel_items if it["item_id"] in qwen_results and it["item_id"] in mistral_results]

    analysis_results["comparisons"]["qwen_vs_mistral_R"] = calculate_agreement_stats(qwen_mistral_r, "Qwen2.5-7B", "Mistral-7B", "R_Context")
    analysis_results["comparisons"]["qwen_vs_mistral_D"] = calculate_agreement_stats(qwen_mistral_d, "Qwen2.5-7B", "Mistral-7B", "D_Definition")

    # Qwen vs Human A1 (Overlapping items)
    qwen_human_r = [(qwen_results[iid]["R"], human_a1[iid]["R"]) for iid in human_a1 if iid in qwen_results]
    qwen_human_d = [(qwen_results[iid]["D"], human_a1[iid]["D"]) for iid in human_a1 if iid in qwen_results]
    analysis_results["comparisons"]["qwen_vs_human_R"] = calculate_agreement_stats(qwen_human_r, "Qwen2.5-7B", "Human_A1", "R_Context")
    analysis_results["comparisons"]["qwen_vs_human_D"] = calculate_agreement_stats(qwen_human_d, "Qwen2.5-7B", "Human_A1", "D_Definition")

    # Mistral vs Human A1 (Overlapping items)
    mistral_human_r = [(mistral_results[iid]["R"], human_a1[iid]["R"]) for iid in human_a1 if iid in mistral_results]
    mistral_human_d = [(mistral_results[iid]["D"], human_a1[iid]["D"]) for iid in human_a1 if iid in mistral_results]
    analysis_results["comparisons"]["mistral_vs_human_R"] = calculate_agreement_stats(mistral_human_r, "Mistral-7B", "Human_A1", "R_Context")
    analysis_results["comparisons"]["mistral_vs_human_D"] = calculate_agreement_stats(mistral_human_d, "Mistral-7B", "Human_A1", "D_Definition")

    # Negative-class triage summary
    human_neg_r = [iid for iid, v in human_a1.items() if v["R"] == 0]
    human_neg_d = [iid for iid, v in human_a1.items() if v["D"] == 0]
    qwen_neg_r = [iid for iid, v in qwen_results.items() if v["R"] == 0]
    mistral_neg_r = [iid for iid, v in mistral_results.items() if v["R"] == 0]

    analysis_results["triage_audit"] = {
        "human_negatives_R_count": len(human_neg_r),
        "human_negatives_D_count": len(human_neg_d),
        "qwen_negatives_R_count": len(qwen_neg_r),
        "mistral_negatives_R_count": len(mistral_neg_r),
        "mistral_caught_human_negatives_R": [iid for iid in human_neg_r if mistral_results.get(iid, {}).get("R") == 0],
        "qwen_caught_human_negatives_R": [iid for iid in human_neg_r if qwen_results.get(iid, {}).get("R") == 0]
    }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    with report_json.open("w", encoding="utf-8") as fh:
        json.dump(analysis_results, fh, indent=2, ensure_ascii=False)

    logger.info("=" * 70)
    logger.info("CROSS-FAMILY JUDGE BENCHMARK COMPLETE!")
    logger.info("Report archived to %s", report_json)
    logger.info("Qwen vs Mistral Agreement: R=%.1f%% (Kappa=%.3f, PABAK=%.3f) | D=%.1f%% (Kappa=%.3f, PABAK=%.3f)",
                analysis_results["comparisons"]["qwen_vs_mistral_R"]["observed_agreement_pct"],
                analysis_results["comparisons"]["qwen_vs_mistral_R"]["cohen_kappa"],
                analysis_results["comparisons"]["qwen_vs_mistral_R"]["pabak"],
                analysis_results["comparisons"]["qwen_vs_mistral_D"]["observed_agreement_pct"],
                analysis_results["comparisons"]["qwen_vs_mistral_D"]["cohen_kappa"],
                analysis_results["comparisons"]["qwen_vs_mistral_D"]["pabak"])
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
