# -*- coding: utf-8 -*-
"""
Generative Evaluation of EduTerm-7B-DPO vs EduTerm-7B-TR (SFT) vs Qwen2.5-7B-Instruct (Base)
on Boğaziçi HPC A40 (48 GB).

Produces verified generative Ateşman readability, CEFR B1 band-share %, JSON compliance,
and terminology extraction F1 across the 8 held-out BENCH vertical texts.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BENCH = [
    {
        "id": "BENCH-EE",
        "text": "Üç fazlı asenkron motorlarda kayma frekansı arttıkça rotor endüktif reaktansı yükselir ve güç faktörü düşer.",
        "gold_terms": ["Asenkron Motor", "Kayma Frekansı", "Endüktif Reaktans", "Güç Faktörü"],
    },
    {
        "id": "BENCH-BIO",
        "text": "DNA replikasyonu sırasında helikaz enzimi çift zinciri açar ve okazaki parçaları ligaz enzimi ile birleştirilir.",
        "gold_terms": ["DNA Replikasyonu", "Helikaz Enzimi", "Okazaki Parçaları", "Ligaz Enzimi"],
    },
    {
        "id": "BENCH-CS",
        "text": "İkili arama ağaçlarında silme ve arama işlemleri dengeli ağaç yapısında logaritmik O(log n) karmaşıklık sunar.",
        "gold_terms": ["İkili Arama Ağacı", "Logaritmik Karmaşıklık", "Dengeli Ağaç"],
    },
    {
        "id": "BENCH-TOUR",
        "text": "Dinamik paketleme motorunda uçak bileti ve otel kontenjanı birleştirilirken no-show riski için erken rezervasyon sigortası uygulanır.",
        "gold_terms": ["Dinamik Paketleme", "No-Show Riski", "Erken Rezervasyon Sigortası"],
    },
    {
        "id": "BENCH-ISG",
        "text": "Şantiyede meydana gelen ramak kala olayları kayıt altına alınmalı ve acil durum eylem planı yenilenmelidir.",
        "gold_terms": ["Ramak Kala Olayı", "Acil Durum Eylem Planı"],
    },
    {
        "id": "BENCH-KVKK",
        "text": "Veri sorumlusu aydınlatma yükümlülüğünü yerine getirdikten sonra ilgili kişiden açık rıza temin etmelidir.",
        "gold_terms": ["Veri Sorumlusu", "Aydınlatma Yükümlülüğü", "Açık Rıza"],
    },
    {
        "id": "BENCH-SEC",
        "text": "Sıfırıncı gün açığı (zero-day) tespit edildiğinde derhal yama uygulanmalı ve çok faktörlü kimlik doğrulama zorunlu kılınmalıdır.",
        "gold_terms": ["Sıfırıncı Gün Açığı", "Çok Faktörlü Kimlik Doğrulama"],
    },
    {
        "id": "BENCH-BNK",
        "text": "Bankaların likidite karşılama oranı Basel III standartlarına göre belirlenir ve TCMB zorunlu karşılık oranına tabidir.",
        "gold_terms": ["Likidite Karşılama Oranı", "Basel III", "Zorunlu Karşılık"],
    },
]

SYSTEM_PROMPT = (
    "Sen Türkçe akademik ve kurumsal eğitim içeriklerini analiz eden uzman bir pedagojik asistansın. "
    "Verilen metindeki teknik terimleri çıkar ve başlangıç düzeyindeki (CEFR B1) bir öğrencinin kolayca "
    "anlayabileceği kısa, sade bir pedagojik özet yaz. Çıktıyı geçerli JSON olarak ver:\n"
    '{"kavramlar": ["Terim1", "Terim2"], "pedagojik_ozet": "Kısa ve sade B1 açıklaması...", "bloom_seviyesi": "Anlama"}'
)


def calculate_atesman(text: str) -> float:
    vowels = "aeıioöuü"
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    if not words:
        return 0.0
    syllables = sum(sum(1 for c in w if c in vowels) for w in words)
    spw = syllables / max(1, len(words))
    wps = len(words) / max(1, len(sentences))
    score = 198.825 - (40.175 * spw) - (2.610 * wps)
    return max(0.0, min(100.0, round(score, 2)))


def parse_json_output(raw: str) -> Dict[str, Any]:
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"```$", "", s, flags=re.MULTILINE).strip()
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j > i:
        try:
            obj = json.loads(s[i : j + 1])
            if isinstance(obj, dict):
                return {"ok": True, "obj": obj}
        except Exception:
            pass
    return {"ok": False, "obj": {}}


def eval_model(label: str, base_path: str, adapter_path: str | None) -> Dict[str, Any]:
    print(f"=== Evaluating {label} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    rows: List[Dict[str, Any]] = []
    for item in BENCH:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Ders metni:\n{item['text']}"},
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=220, do_sample=False, pad_token_id=tok.eos_token_id)
        dt = time.time() - t0
        raw = tok.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()
        parsed = parse_json_output(raw)
        obj = parsed["obj"]
        summary_text = str(obj.get("pedagojik_ozet", "")).strip() if parsed["ok"] else raw
        atesman = calculate_atesman(summary_text)
        preds = [str(x).lower() for x in (obj.get("kavramlar") or [])] if parsed["ok"] else []
        golds = [g.lower() for g in item["gold_terms"]]
        tp = sum(1 for g in golds if any(g in p or p in g for p in preds))
        prec = tp / len(preds) if preds else 0.0
        rec = tp / len(golds) if golds else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        rows.append(
            {
                "id": item["id"],
                "json_ok": parsed["ok"],
                "atesman": atesman,
                "in_b1_band_50_69": 50.0 <= atesman <= 69.9,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "latency_s": round(dt, 2),
                "summary_preview": summary_text[:180],
            }
        )
        print(f"  {item['id']}: json={parsed['ok']} atesman={atesman} f1={f1:.3f} ({dt:.2f}s)", flush=True)

    n = len(rows)
    summary = {
        "model": label,
        "adapter_path": adapter_path,
        "n_cases": n,
        "json_compliance_pct": round(100.0 * sum(r["json_ok"] for r in rows) / n, 1),
        "mean_atesman": round(sum(r["atesman"] for r in rows) / n, 2),
        "b1_band_share_pct": round(100.0 * sum(r["in_b1_band_50_69"] for r in rows) / n, 1),
        "mean_precision_pct": round(100.0 * sum(r["precision"] for r in rows) / n, 1),
        "mean_recall_pct": round(100.0 * sum(r["recall"] for r in rows) / n, 1),
        "mean_f1_pct": round(100.0 * sum(r["f1"] for r in rows) / n, 1),
        "mean_latency_s": round(sum(r["latency_s"] for r in rows) / n, 2),
        "per_case": rows,
    }
    del model
    torch.cuda.empty_cache()
    return summary


def main():
    base_path = "/users/ali.gunder/models/Qwen2.5-7B-Instruct"
    sft_path = "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_7b_tr_sft"
    dpo_path = "/users/ali.gunder/eduterm-dpo/checkpoints/eduterm_7b_tr_dpo"

    results = [
        eval_model("Qwen2.5-7B-Instruct (Base)", base_path, None),
        eval_model("EduTerm-7B-TR (SFT)", base_path, sft_path),
        eval_model("EduTerm-7B-DPO (SFT+DPO)", base_path, dpo_path),
    ]
    out_path = Path("/users/ali.gunder/eduterm-dpo/outputs/dpo_7b_generative_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()

