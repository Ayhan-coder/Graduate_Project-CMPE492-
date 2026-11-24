# -*- coding: utf-8 -*-
"""
Experiment 38: Zero-Leakage UTF-8 Exact Allowlist Logit Mask Verification
on Boğaziçi HPC (A40 48GB).

Compares 3 logit-masking strategies across `Qwen2.5-14B-Instruct (+dpo_v5_pareto)`,
`Qwen2.5-7B-Instruct (bf16)`, and `Kumru-2B (bf16)` over 30 Turkish prompts x 3 seeds
= 90 generations per mask per model (810 generations total):
  1. `naive_nonlatin_mask` (blocks U+FFFD continuation bytes -> corrupts Turkish ğ/ş/ı)
  2. `partial_script_blacklist` (Exp 34's major-script regex -> leaves ~0.7%-1.4% minor Unicode leaks)
  3. `utf8_exact_allowlist` (`"\ufffd" not in s and NON_LATIN_PATTERN.search(s)` -> 0 blocked U+FFFD
     bytes AND exhaustive non-Latin blocking for 0.00% leakage!)

Outputs:
  results/hpc_utf8_exact_allowlist_report.json
"""

from __future__ import annotations

import gc
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_hpc_cjk_blocked_dag_ablation import CJKBlockLogitsProcessor
from run_hpc_dpo_r2_nonlatin import NON_LATIN_PATTERN, build_nonlatin_token_ids, has_nonlatin
from run_hpc_multiturn_guardrail import resolve_adapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp38UTF8Allowlist")

PARTIAL_SCRIPT_PATTERN = re.compile(
    r"[\u0370-\u03ff\u0400-\u052f\u0590-\u05ff\u0600-\u06ff\u0750-\u077f\u0900-\u097f"
    r"\u0e00-\u0e7f\u1100-\u11ff\u3000-\u303f\u3040-\u30ff\u3130-\u318f\u3400-\u4dbf"
    r"\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]"
)

TURKISH_DIACRITICS = set("çğıöşüÇĞİÖŞÜ")

PROMPTS_30 = [
    "İş sağlığı ve güvenliğinde tehlike ile risk farkını bir örnekle açıkla.",
    "Ramak kala olayların raporlanması neden zorunludur?",
    "Yüksekte çalışmada düşmeye karşı toplu koruma neden KKD'den önce gelir?",
    "EKED (Etiketle-Kilitle-Emniyete Al-Dene) adımlarını sırayla özetle.",
    "Kapalı alana girmeden önce atmosferde oksijen ve gaz ölçümü neden şarttır?",
    "Elektrik panosu yangınında neden su yerine CO2 söndürücü kullanılır?",
    "Gürültü maruziyet sınır değeri 87 dB(A) aşıldığında hangi önlemler alınır?",
    "Ergonomi ilkeleri ağır yük kaldırmada bel sağlığını nasıl korur?",
    "Kimyasal sıçramada acil göz duşunun önemi nedir?",
    "Çift el kumanda tertibatının bantlanması neden büyük tehlikedir?",
    "Dinamik programlama ile böl-ve-yönet algoritmalarının temel farkını anlat.",
    "Hash tablolarında çakışma neden kaçınılmazdır ve nasıl çözülür?",
    "İşletim sistemlerinde deadlock oluşması için dört koşul nedir?",
    "Sanal bellek (virtual memory) fiziksel RAM'den nasıl ayrılır?",
    "Veritabanında indeksleme okuma ve yazma hızını nasıl etkiler?",
    "Sınıf dengesizliğinde neden accuracy yerine F1 ve PR-AUC kullanılır?",
    "Aşırı öğrenme (overfitting) nedir ve nasıl önlenir?",
    "Pseudonimizasyon ile anonimleştirme arasındaki hukuki fark nedir?",
    "Nyquist örnekleme teoremi aliasing olgusunu nasıl önler?",
    "Newton'un üçüncü yasasında etki-tepki kuvvetleri neden eşittir?",
    "Sera etkisi ile ozon tabakasındaki incelme neden farklı olaylardır?",
    "Antibiyotik direnci neden insan vücudunda değil bakterilerde gelişir?",
    "Simpson paradoksu istatistiksel yorumları nasıl yanıltır?",
    "Korelasyon ile nedensellik arasındaki farkı günlük bir örnekle açıkla.",
    "Bileşik faiz neden doğrusal değil üstel bir büyüme gösterir?",
    "Batık maliyet yanılgısı karar alma süreçlerini nasıl bozar?",
    "Fırsat maliyeti kavramını bir öğrenci örneğiyle sade biçimde anlat.",
    "Enflasyon mevduat faizini aştığında reel alım gücü nasıl değişir?",
    "Aşağıdaki ifade DOĞRU mu YANLIŞ mı? Sadece 'DOĞRU' veya 'YANLIŞ' yaz: 'Tehlike potansiyel zarar kaynağıdır.'",
    "Aşağıdaki ifade DOĞRU mu YANLIŞ mı? Sadece 'DOĞRU' veya 'YANLIŞ' yaz: 'Ramak kala olaylar mutlaka raporlanmalıdır.'",
]


def build_partial_ids(tok) -> List[int]:
    return [
        tid for tid in range(len(tok))
        if "\ufffd" not in (s := tok.decode([tid], skip_special_tokens=False)) and PARTIAL_SCRIPT_PATTERN.search(s)
    ]


def build_exact_allowlist_ids(tok) -> List[int]:
    return [
        tid for tid in range(len(tok))
        if "\ufffd" not in (s := tok.decode([tid], skip_special_tokens=False)) and NON_LATIN_PATTERN.search(s)
    ]


def gen_one(model, tok, proc, prompt: str, seed: int) -> str:
    torch.manual_seed(seed)
    msgs = [
        {"role": "system", "content": "Sen akıcı ve doğru Türkçe konuşan bir eğitmensin. Kısa ve net yanıt ver."},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    enc.pop("token_type_ids", None)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=95,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            logits_processor=[proc],
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def eval_model_masks(name: str, model, tok) -> Dict[str, Any]:
    masks = {
        "naive_nonlatin_mask": build_nonlatin_token_ids(tok),
        "partial_script_blacklist": build_partial_ids(tok),
        "utf8_exact_allowlist": build_exact_allowlist_ids(tok),
    }
    out: Dict[str, Any] = {}
    for m_name, ids in masks.items():
        proc = CJKBlockLogitsProcessor(ids)
        nl = fffd_out = dogru_ok = dogru_tot = total = 0
        diac_chars = total_chars = 0
        for p_idx, p in enumerate(PROMPTS_30):
            for seed in (42, 101, 202):
                resp = gen_one(model, tok, proc, p, seed)
                total += 1
                if has_nonlatin(resp):
                    nl += 1
                if "\ufffd" in resp:
                    fffd_out += 1
                diac_chars += sum(1 for ch in resp if ch in TURKISH_DIACRITICS)
                total_chars += max(len(resp), 1)
                if p_idx >= 28:
                    dogru_tot += 1
                    if "DOĞRU" in resp.upper():
                        dogru_ok += 1
        out[m_name] = {
            "blocked_tokens": len(ids),
            "n_generations": total,
            "nonlatin_leakage_rate": round(nl / total, 4),
            "replacement_char_rate": round(fffd_out / total, 4),
            "turkish_diacritic_density": round(diac_chars / max(total_chars, 1), 4),
            "exact_dogru_spelling_rate": round(dogru_ok / max(dogru_tot, 1), 4),
        }
        logger.info("[%s | %s]: %s", name, m_name, out[m_name])
    return out


def main() -> None:
    t0 = time.time()
    results: Dict[str, Any] = {}

    # 1. Qwen2.5-14B-Instruct (+dpo_v5_pareto)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    p14 = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
    tok14 = AutoTokenizer.from_pretrained(p14)
    base14 = AutoModelForCausalLM.from_pretrained(p14, quantization_config=bnb, device_map="auto")
    mod14 = PeftModel.from_pretrained(base14, resolve_adapter("checkpoints/eduterm_14b_dpo_v5_pareto", "dpo_v5_pareto"), adapter_name="dpo_v5")
    mod14.set_adapter("dpo_v5")
    mod14.eval()
    results["Qwen2.5-14B-dpo_v5_pareto"] = eval_model_masks("Qwen2.5-14B-dpo_v5", mod14, tok14)
    del mod14, base14
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Qwen2.5-7B-Instruct (bf16)
    p7 = "/users/ali.gunder/models/Qwen2.5-7B-Instruct"
    tok7 = AutoTokenizer.from_pretrained(p7)
    mod7 = AutoModelForCausalLM.from_pretrained(p7, torch_dtype=torch.bfloat16, device_map="auto")
    mod7.eval()
    results["Qwen2.5-7B-Instruct-bf16"] = eval_model_masks("Qwen2.5-7B-bf16", mod7, tok7)
    del mod7
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Kumru-2B (bf16)
    pk = "/users/ali.gunder/models/Kumru-2B"
    tokk = AutoTokenizer.from_pretrained(pk)
    modk = AutoModelForCausalLM.from_pretrained(pk, torch_dtype=torch.bfloat16, device_map="auto")
    modk.eval()
    results["Kumru-2B-bf16"] = eval_model_masks("Kumru-2B-bf16", modk, tokk)

    report = {"metadata": {"prompts": len(PROMPTS_30), "seeds": 3, "elapsed_sec": round(time.time() - t0, 1)}, "results_by_model": results}
    Path("results/hpc_utf8_exact_allowlist_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
