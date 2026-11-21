# -*- coding: utf-8 -*-
"""
Experiment 16 (P0): 600-Sample Controlled Negative-Class LLM-as-a-Judge Benchmark
on Boğaziçi HPC (A40 48GB).

Addresses §8.15 (TN = 0 limitation) by evaluating local judges on 600 balanced items:
  - 100 correct explanations (label = 1)
  - 100 factual_error (label = 0)
  - 100 circular_definition (label = 0)
  - 100 out_of_context (label = 0)
  - 100 conceptual_contradiction (label = 0)
  - 100 overly_complex_jargon (label = 0)

Evaluates both Qwen2.5-7B-Instruct (bfloat16) and Qwen2.5-14B-Instruct (4-bit NF4),
computing Sensitivity, Specificity, Balanced Accuracy, F1, Parse Rate, and per-category
True Negative catch rates. Also saves the 600-item dataset for external EVREN/Mistral scoring.

Outputs:
  results/hpc_judge_negatives_600_dataset.json
  results/hpc_judge_negatives_600_report.json
"""

from __future__ import annotations

import gc
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_hpc_alignment_tax_cefr import calculate_atesman_score
from run_hpc_cjk_blocked_dag_ablation import CJKBlockLogitsProcessor
from run_hpc_dpo_r2_nonlatin import build_nonlatin_token_ids
from run_hpc_live_student_simulation import ISG_CONCEPTS
from run_hpc_ood_tutoring import OOD_CONCEPTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp16JudgeNeg600")


def build_600_dataset(seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    pool = ISG_CONCEPTS + OOD_CONCEPTS  # 24 verified Turkish concepts across ISG, CS, KVKK, Banking, EE
    items: List[Dict[str, Any]] = []

    correct_prefixes = [
        "Temel ilke olarak: ",
        "Kısaca özetlemek gerekirse, ",
        "Ders kapsamındaki doğru tanıma göre ",
        "Başlangıç düzeyinde açıkça ifade edersek, ",
        "Uygulamalı perspektiften bakıldığında ",
    ]

    for i in range(100):
        c = pool[i % len(pool)]
        pref = correct_prefixes[(i // len(pool)) % len(correct_prefixes)]
        items.append({
            "id": f"NEG600_POS_{i+1:03d}",
            "category": "correct",
            "gold_label": 1,
            "term": c["name"],
            "explanation": f"{pref}{c['gold_principle']}",
        })

    for i in range(100):
        c = pool[i % len(pool)]
        items.append({
            "id": f"NEG600_FACT_{i+1:03d}",
            "category": "factual_error",
            "gold_label": 0,
            "term": c["name"],
            "explanation": f"{c['misconception']} Bu nedenle literatürde {c['name']} yalnızca bu sınırlı koşulda geçerli kabul edilir.",
        })

    circ_templates = [
        "{name}, {name} süreçlerinin {name} ilkeleri çerçevesinde yürütülmesi anlamına gelir; yani {name} özünde {name} demektir.",
        "{name} kavramı, bir sistemde {name} oluştuğunda ortaya çıkan {name} durumunu ifade eder ve {name} ile tanımlanır.",
        "Eğitimde {name}, doğrudan {name} niteliği taşıyan her türlü {name} olgusunun genel adıdır.",
    ]
    for i in range(100):
        c = pool[i % len(pool)]
        tmpl = circ_templates[i % len(circ_templates)]
        items.append({
            "id": f"NEG600_CIRC_{i+1:03d}",
            "category": "circular_definition",
            "gold_label": 0,
            "term": c["name"],
            "explanation": tmpl.format(name=c["name"]),
        })

    for i in range(100):
        c = pool[i % len(pool)]
        other = pool[(i + 12) % len(pool)]
        items.append({
            "id": f"NEG600_OOC_{i+1:03d}",
            "category": "out_of_context",
            "gold_label": 0,
            "term": c["name"],
            "explanation": f"{other['gold_principle']} Özellikle {other['name']} uygulamasında bu kural belirleyicidir.",
        })

    for i in range(100):
        c = pool[i % len(pool)]
        items.append({
            "id": f"NEG600_CONTR_{i+1:03d}",
            "category": "conceptual_contradiction",
            "gold_label": 0,
            "term": c["name"],
            "explanation": f"{c['gold_principle']} Ancak bunun tam tersi de doğrudur: {c['misconception']}",
        })

    jargon_templates = [
        "{name}, çok katmanlı epistemolojik paradigmaların deterministik olmayan stokastik yakınsama uzayındaki metakognitif regülarizasyon katsayılarının asimptotik diferansiyel varyasyonlarıyla karakterize edilen, izomorfik olmayan yüksek mertebeli bir normatif hiper-parametrizasyon mekanizmasıdır.",
        "Sistematik ontolojik sınıflandırmada {name}, heterojen alt-sistemler arasındaki entropik dengesizliklerin non-lineer regülasyon matrisleri üzerinden projeksiyonuyla türetilen, konvansiyonel analitik indirgemeciliğe dirençli bir makro-epistemik konfigürasyondur.",
    ]
    for i in range(100):
        c = pool[i % len(pool)]
        tmpl = jargon_templates[i % len(jargon_templates)]
        items.append({
            "id": f"NEG600_JARG_{i+1:03d}",
            "category": "overly_complex_jargon",
            "gold_label": 0,
            "term": c["name"],
            "explanation": tmpl.format(name=c["name"]),
        })

    for it in items:
        it["atesman"] = calculate_atesman_score(it["explanation"])
    rng.shuffle(items)
    return items


def judge_explanation(model, tok, proc, item: Dict[str, Any]) -> Tuple[int, bool]:
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
    msgs = [
        {"role": "system", "content": "Sen titiz ve eleştirel bir Türkçe eğitim hakemisin. Hatalı, çelişkili, döngüsel, ilgisiz veya aşırı karmaşık açıklamaları kesinlikle reddet (verdict=0). Sadece JSON döndür."},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    enc.pop("token_type_ids", None)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=48,
            do_sample=False,
            logits_processor=[proc],
            pad_token_id=tok.eos_token_id,
        )
    raw = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    m = re.search(r"\{[\s\S]*?\}", raw)
    if not m:
        return 0, False
    try:
        obj = json.loads(m.group(0))
        return int(bool(int(obj.get("verdict", 0)))), True
    except Exception:
        return 0, False


def compute_metrics(dataset: List[Dict[str, Any]], preds: List[Tuple[int, bool]]) -> Dict[str, Any]:
    tp = tn = fp = fn = parse_ok = 0
    by_cat: Dict[str, Dict[str, int]] = {}
    for item, (pred, ok) in zip(dataset, preds):
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
    return {
        "n_total": len(dataset),
        "parse_success_rate": round(parse_ok / max(len(dataset), 1), 4),
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "sensitivity_tpr": round(sens, 4),
        "specificity_tnr": round(spec, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "by_category_accuracy": {k: round(v["correct"] / max(v["n"], 1), 4) for k, v in by_cat.items()},
    }


def eval_judge(model_name: str, model_path: str, use_4bit: bool, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(model_path)
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    proc = CJKBlockLogitsProcessor(build_nonlatin_token_ids(tok))

    preds: List[Tuple[int, bool]] = []
    for idx, item in enumerate(dataset, start=1):
        preds.append(judge_explanation(model, tok, proc, item))
        if idx % 100 == 0:
            logger.info("[%s] Scored %d/%d", model_name, idx, len(dataset))

    res = compute_metrics(dataset, preds)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main() -> None:
    t0 = time.time()
    dataset = build_600_dataset(seed=42)
    Path("results/hpc_judge_negatives_600_dataset.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Category verification diagnostics
    cat_atesman: Dict[str, List[float]] = {}
    for it in dataset:
        cat_atesman.setdefault(it["category"], []).append(it["atesman"])
    cat_diag = {k: round(sum(v) / len(v), 2) for k, v in cat_atesman.items()}

    results = {
        "Qwen2.5-7B-Instruct_bf16": eval_judge("Qwen2.5-7B-Instruct", "/users/ali.gunder/models/Qwen2.5-7B-Instruct", False, dataset),
        "Qwen2.5-14B-Instruct_nf4": eval_judge("Qwen2.5-14B-Instruct", "/users/ali.gunder/models/Qwen2.5-14B-Instruct", True, dataset),
    }
    report = {
        "metadata": {"n_items": len(dataset), "category_mean_atesman": cat_diag, "elapsed_sec": round(time.time() - t0, 1)},
        "results": results,
    }
    Path("results/hpc_judge_negatives_600_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Exp 16 Report:\n%s", json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
