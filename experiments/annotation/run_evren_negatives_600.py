# -*- coding: utf-8 -*-
"""
Score the 600-sample controlled negative-class benchmark (`hpc_judge_negatives_600_dataset.json`)
using EVREN `evren/qwen3.8-flash-next` (reasoning_effort="none", temperature=0.0).

Outputs:
  experiments/outputs/evren_qwen38_flash_negatives_600_report.json
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.evren_llm import generate_evren  # noqa: E402

SYS_PROMPT = (
    "Sen titiz ve eleştirel bir Türkçe eğitim hakemisin. Hatalı, çelişkili, döngüsel, "
    "ilgisiz veya aşırı karmaşık açıklamaları kesinlikle reddet (verdict=0). Sadece JSON döndür."
)


def score_one(item: Dict[str, Any], model_id: str) -> Tuple[str, int, bool]:
    prompt = (
        f"Hedef Kavram: {item['term']}\n"
        f"Açıklama: {item['explanation']}\n\n"
        "Bu açıklama başlangıç düzeyindeki bir üniversite öğrencisi için:\n"
        "1) Kavramsal olarak doğru ve çelişkisiz mi?\n"
        "2) Döngüsel tanım veya ilgisiz bağlam içermiyor mu?\n"
        "3) Aşırı karmaşık jargon yerine anlaşılır Türkçe ile mi yazılmış?\n"
        "Tüm 3 koşulu sağlıyorsa verdict=1, herhangi birinde hata varsa verdict=0 ver.\n"
        'Sadece tek satır JSON döndür: {"verdict": 0 veya 1}'
    )
    for attempt in range(3):
        try:
            raw = generate_evren(
                model_id=model_id,
                prompt=prompt,
                max_tokens=512 if "gemma" in model_id.lower() else 64,
                temperature=0.0,
                system_prompt=SYS_PROMPT,
                reasoning_effort="none",
            )
            m = re.search(r"\{[\s\S]*?\}", raw)
            if m:
                obj = json.loads(m.group(0))
                return item["id"], int(bool(int(obj.get("verdict", 0)))), True
            return item["id"], 0, False
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return item["id"], 0, False


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="evren/qwen3.8-flash-next")
    parser.add_argument("--out", default="evren_qwen38_flash_negatives_600_report.json")
    args = parser.parse_args()

    t0 = time.time()
    model_id = args.model
    ds_path = ROOT / "experiments" / "outputs" / "hpc_judge_negatives_600_dataset.json"
    dataset: List[Dict[str, Any]] = json.loads(ds_path.read_text(encoding="utf-8"))

    preds_by_id: Dict[str, Tuple[int, bool]] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(score_one, it, model_id): it["id"] for it in dataset}
        done = 0
        for fut in as_completed(futs):
            item_id, pred, ok = fut.result()
            preds_by_id[item_id] = (pred, ok)
            done += 1
            if done % 100 == 0 or done == len(dataset):
                print(f"[{model_id}] Scored {done}/{len(dataset)} ({time.time() - t0:.1f}s)", flush=True)

    tp = tn = fp = fn = parse_ok = 0
    by_cat: Dict[str, Dict[str, int]] = {}
    for item in dataset:
        pred, ok = preds_by_id[item["id"]]
        if ok:
            parse_ok += 1
        gold = item["gold_label"]
        cat = item["category"]
        by_cat.setdefault(cat, {"n": 0, "correct": 0})
        by_cat[cat]["n"] += 1
        if pred == gold:
            by_cat[cat]["correct"] += 1
        if gold == 1 and pred == 1:
            tp += 1
        elif gold == 0 and pred == 0:
            tn += 1
        elif gold == 0 and pred == 1:
            fp += 1
        else:
            fn += 1

    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    bal_acc = (sens + spec) / 2.0
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-9)

    report = {
        "metadata": {
            "judge_model": model_id,
            "n_total": len(dataset),
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "parse_success_rate": round(parse_ok / max(len(dataset), 1), 4),
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "sensitivity_tpr": round(sens, 4),
        "specificity_tnr": round(spec, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "by_category_accuracy": {k: round(v["correct"] / max(v["n"], 1), 4) for k, v in by_cat.items()},
    }
    out_path = ROOT / "experiments" / "outputs" / args.out
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
