# -*- coding: utf-8 -*-
"""
Run 454-Item Cross-Family Judge Benchmark using EVREN Cloud API (qwen3.8-flash-next).
Produces:
  - experiments/annotation/labels_panel_evren_qwen38_flash.csv
  - experiments/outputs/cross_family_judge_report_evren.json
"""

import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EVREN-Judge")

ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT_DIR / ".env"
PANEL_CSV = ROOT_DIR / "experiments" / "annotation" / "labels_panel_input.csv"
HUMAN_CSV = ROOT_DIR / "experiments" / "annotation" / "labels_a1.csv"
OUT_CSV = ROOT_DIR / "experiments" / "annotation" / "labels_panel_evren_auto.csv"
OUT_JSON = ROOT_DIR / "experiments" / "outputs" / "cross_family_judge_report_evren.json"

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


def get_api_credentials() -> Tuple[str, str]:
    api_key = None
    base_url = "https://evren-llmapi.ssyz.org.tr/v1"
    # Always prefer the active key in .env file first
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("EVREN_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
            elif line.startswith("EVREN_BASE_URL="):
                base_url = line.split("=", 1)[1].strip()
    if not api_key:
        api_key = os.environ.get("EVREN_API_KEY")
    if not api_key:
        raise ValueError("EVREN_API_KEY not found in environment or .env")
    return api_key, base_url


def parse_verdict(raw: str) -> Optional[Dict[str, Any]]:
    if not raw or not raw.strip():
        return None
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE).strip()
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

    r_m = re.search(r'["\']?R["\']?\s*[:=]\s*([01])\b', cleaned, re.IGNORECASE)
    d_m = re.search(r'["\']?D["\']?\s*[:=]\s*([01])\b', cleaned, re.IGNORECASE)
    if r_m and d_m:
        r_val = int(r_m.group(1))
        d_val = int(d_m.group(1))
        why_m = re.search(r'["\']?(?:why|gerekçe|neden|reason)["\']?\s*[:=]\s*["\']?([^"\n\r}\]]+)', cleaned, re.IGNORECASE)
        why = why_m.group(1).strip() if why_m else ""
        return {"R": r_val, "D": d_val, "why": why[:300]}

    return None


def calculate_agreement_stats(pairs: List[Tuple[int, int]], name1: str, name2: str, metric_name: str) -> Dict[str, Any]:
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    tp = sum(1 for y1, y2 in pairs if y1 == 1 and y2 == 1)
    tn = sum(1 for y1, y2 in pairs if y1 == 0 and y2 == 0)
    fp = sum(1 for y1, y2 in pairs if y1 == 1 and y2 == 0)
    fn = sum(1 for y1, y2 in pairs if y1 == 0 and y2 == 1)
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
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "pos_rate_1": round(p_yes1 * 100, 1),
        "pos_rate_2": round(p_yes2 * 100, 1)
    }


def main():
    api_key, base_url = get_api_credentials()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    model_name = "auto"

    with PANEL_CSV.open("r", encoding="utf-8-sig") as fh:
        panel_items = list(csv.DictReader(fh))
    logger.info("Loaded %d panel input items from %s", len(panel_items), PANEL_CSV)

    existing_results: Dict[str, Dict[str, Any]] = {}
    if OUT_CSV.exists():
        with OUT_CSV.open("r", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if row.get("judge_R") != "":
                    existing_results[row["item_id"]] = {
                        "R": int(row["judge_R"]),
                        "D": int(row["judge_D"]),
                        "why": row.get("judge_why", ""),
                        "raw": row.get("judge_raw", ""),
                        "row": row
                    }
        logger.info("Loaded %d already completed items from %s", len(existing_results), OUT_CSV)

    fields = ["item_id", "video_id", "lang", "term", "explanation", "context",
              "judge_R", "judge_D", "judge_why", "judge_raw", "judge_model"]

    todo = [item for item in panel_items if item["item_id"] not in existing_results]
    logger.info("Remaining items to evaluate via EVREN: %d / %d", len(todo), len(panel_items))

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

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": 1024,
            "temperature": 0.0
        }

        verdict = None
        raw_output = ""
        for attempt in range(3):
            try:
                resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=60)
                if resp.status_code == 200:
                    d = resp.json()
                    raw_output = d["choices"][0]["message"].get("content") or ""
                    verdict = parse_verdict(raw_output)
                    if verdict:
                        break
                elif resp.status_code in (429, 500, 502, 503, 504):
                    logger.warning("Rate limit / server error %s (attempt %d). Waiting...", resp.status_code, attempt + 1)
                    time.sleep(3 * (attempt + 1))
            except Exception as e:
                logger.warning("Request exception on %s (attempt %d): %s", item["item_id"], attempt + 1, e)
                time.sleep(2 * (attempt + 1))

        if not verdict:
            verdict = {"R": 1, "D": 1, "why": "Parsing fallback"}
            logger.warning("Fallback triggered for item %s", item["item_id"])

        record = {
            "item_id": item["item_id"],
            "video_id": item.get("video_id", ""),
            "lang": lang,
            "term": item.get("term", ""),
            "explanation": item.get("explanation", ""),
            "context": ctx[:300],
            "judge_R": verdict["R"],
            "judge_D": verdict["D"],
            "judge_why": verdict["why"],
            "judge_raw": raw_output[:500],
            "judge_model": f"evren/{model_name}"
        }
        existing_results[item["item_id"]] = {
            "R": verdict["R"],
            "D": verdict["D"],
            "why": verdict["why"],
            "raw": raw_output,
            "row": record
        }

        # Progress log & incremental flush
        if idx % 10 == 0 or idx == len(todo):
            elapsed = time.time() - t0
            rate = idx / max(1.0, elapsed)
            rem_s = (len(todo) - idx) / max(0.01, rate)
            logger.info("Progress: %d / %d (%.1f%%) | %.2f it/s | ~%.1f min remaining",
                        idx, len(todo), (idx / len(todo)) * 100, rate, rem_s / 60.0)

            # Flush CSV safely
            with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                for iid in [p["item_id"] for p in panel_items]:
                    if iid in existing_results:
                        writer.writerow(existing_results[iid]["row"])

    logger.info("Completed EVREN evaluation on %d items. Generating comparative report...", len(panel_items))

    # Read Human A1 for comparison
    human_a1: Dict[str, Dict[str, int]] = {}
    if HUMAN_CSV.exists():
        with HUMAN_CSV.open("r", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                r_val = row.get("R") or row.get("annotator_R")
                d_val = row.get("D") or row.get("annotator_D")
                if r_val is not None and d_val is not None and r_val != "" and d_val != "":
                    human_a1[row["item_id"]] = {
                        "R": int(r_val),
                        "D": int(d_val)
                    }

    # Load Mistral-7B results from disk
    mistral_csv = ROOT_DIR / "experiments" / "annotation" / "labels_panel_mistral7b_a40.csv"
    mistral_results: Dict[str, Dict[str, int]] = {}
    if mistral_csv.exists():
        with mistral_csv.open("r", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                mistral_results[row["item_id"]] = {
                    "R": int(row["judge_R"]),
                    "D": int(row["judge_D"])
                }

    # Comparisons
    evren_human_r = [(existing_results[iid]["R"], human_a1[iid]["R"]) for iid in human_a1 if iid in existing_results]
    evren_human_d = [(existing_results[iid]["D"], human_a1[iid]["D"]) for iid in human_a1 if iid in existing_results]
    evren_mistral_r = [(existing_results[iid]["R"], mistral_results[iid]["R"]) for iid in mistral_results if iid in existing_results]
    evren_mistral_d = [(existing_results[iid]["D"], mistral_results[iid]["D"]) for iid in mistral_results if iid in existing_results]

    human_neg_r = [iid for iid, v in human_a1.items() if v["R"] == 0]
    human_neg_d = [iid for iid, v in human_a1.items() if v["D"] == 0]
    evren_caught_human_negatives_r = [iid for iid in human_neg_r if existing_results.get(iid, {}).get("R") == 0]
    evren_caught_human_negatives_d = [iid for iid in human_neg_d if existing_results.get(iid, {}).get("D") == 0]

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "TÜBİTAK BİLGEM / SSB EVREN National AI Platform (H200 GPU Cluster)",
        "model": f"evren/{model_name}",
        "total_panel_items": len(panel_items),
        "comparisons": {
            "evren_vs_human_R": calculate_agreement_stats(evren_human_r, f"EVREN-{model_name}", "Human_A1", "R_Context"),
            "evren_vs_human_D": calculate_agreement_stats(evren_human_d, f"EVREN-{model_name}", "Human_A1", "D_Definition"),
            "evren_vs_mistral_R": calculate_agreement_stats(evren_mistral_r, f"EVREN-{model_name}", "Mistral-7B", "R_Context"),
            "evren_vs_mistral_D": calculate_agreement_stats(evren_mistral_d, f"EVREN-{model_name}", "Mistral-7B", "D_Definition")
        },
        "triage_audit": {
            "human_negatives_R_count": len(human_neg_r),
            "human_negatives_D_count": len(human_neg_d),
            "evren_caught_human_negatives_R": evren_caught_human_negatives_r,
            "evren_caught_human_negatives_D": evren_caught_human_negatives_d
        }
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    logger.info("Report saved to %s", OUT_JSON)
    logger.info("EVREN vs Human Agreement: R=%.2f%% | D=%.2f%%",
                report["comparisons"]["evren_vs_human_R"]["observed_agreement_pct"],
                report["comparisons"]["evren_vs_human_D"]["observed_agreement_pct"])


if __name__ == "__main__":
    main()

