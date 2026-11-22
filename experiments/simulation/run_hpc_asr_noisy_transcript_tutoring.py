# -*- coding: utf-8 -*-
"""
Experiment 35: ASR Noise & Agglutinative Corruption Robustness in Turkish Socratic Tutoring
on Boğaziçi HPC (A40 48GB).

Tests how realistic Turkish ASR transcript errors (homophone confusions, dropped suffixes,
corrupted domain terms like "KKD" -> "kaka de", "ramak kala" -> "rama kala", "EKED" -> "eket")
affect Socratic guidance across 4 14B stages (`base_14b`, `tr_qlora`, `onpolicy_dpo_v2`, `dpo_v5_pareto`)
at 3 noise tiers (`clean`, `moderate_asr`, `heavy_asr` x 16 prompts = 48 inputs/model, 192 total),
using UTF-8-safe non-Latin script blocking and blinded 14B-base evaluation.

Outputs:
  results/hpc_asr_noisy_transcript_tutoring_report.json
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

from run_hpc_alignment_tax_cefr import resolve_dpo_v2_path
from run_hpc_cjk_blocked_dag_ablation import CJKBlockLogitsProcessor, exact_mcnemar_p
from run_hpc_dpo_r2_nonlatin import has_nonlatin
from run_hpc_multiturn_guardrail import resolve_adapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp35ASRNoise")

SCRIPT_BLOCK_PATTERN = re.compile(
    r"[\u0370-\u03ff\u0400-\u052f\u0590-\u05ff\u0600-\u06ff\u0750-\u077f\u0900-\u097f"
    r"\u0e00-\u0e7f\u1100-\u11ff\u3000-\u303f\u3040-\u30ff\u3130-\u318f\u3400-\u4dbf"
    r"\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]"
)


def build_utf8_safe_block_ids(tokenizer) -> List[int]:
    return [
        tid for tid in range(len(tokenizer))
        if "\ufffd" not in (s := tokenizer.decode([tid], skip_special_tokens=False)) and SCRIPT_BLOCK_PATTERN.search(s)
    ]


ASR_PROBES = [
    # 16 Core Turkish İSG & STEM Scenarios across 3 ASR tiers
    ("Tehlike ve Risk", "clean", "Bence tehlike ve risk tamamen aynı şey; ikisi de kaza olunca ortaya çıkar.", "tehlike ve risk arasındaki olasılık/kaynak farkı"),
    ("Tehlike ve Risk", "moderate_asr", "bence tehli ke ve riks tamamen ayni sey ikisi de kaza olunca cikar", "tehlike ve risk arasındaki olasılık/kaynak farkı"),
    ("Tehlike ve Risk", "heavy_asr", "bence telike veri sk ayni sey ikiside kaz olunca cıkyo hocm", "tehlike ve risk arasındaki olasılık/kaynak farkı"),

    ("Ramak Kala", "clean", "Kimse yaralanmadığı için ramak kala olayı raporlamaya gerek yok bence.", "zarar olmasa da ramak kala olayların raporlanması gerekliliği"),
    ("Ramak Kala", "moderate_asr", "kimse yaralan madigi icin rama kala olayi rapor lamaya gerek yok", "zarar olmasa da ramak kala olayların raporlanması gerekliliği"),
    ("Ramak Kala", "heavy_asr", "kimse yara almadı diye rammak kala raporu yazmaya ne geregi var", "zarar olmasa da ramak kala olayların raporlanması gerekliliği"),

    ("Kontrol Hiyerarşisi (KKD)", "clean", "En etkili iş güvenliği önlemi herkese hemen KKD dağıtmaktır.", "KKD'nin kontrol hiyerarşisinde son çare olması"),
    ("Kontrol Hiyerarşisi (KKD)", "moderate_asr", "en etkili is guvenligi onlemi herkese kaka de dagitmak degil mi", "KKD'nin kontrol hiyerarşisinde son çare olması"),
    ("Kontrol Hiyerarşisi (KKD)", "heavy_asr", "en iyi is guvenlik tedbiri direk kkd takmak diyilmi ustam", "KKD'nin kontrol hiyerarşisinde son çare olması"),

    ("EKED / Kilitleme-Etiketleme", "clean", "Makineyi panodan şalterden kapattıysam kilit takmama gerek kalmaz.", "EKED kilitleme-etiketleme ve sıfır enerji doğrulaması"),
    ("EKED / Kilitleme-Etiketleme", "moderate_asr", "makinayı panodan salterden kapadiysam eket kilidi takmasam olur mu", "EKED kilitleme-etiketleme ve sıfır enerji doğrulaması"),
    ("EKED / Kilitleme-Etiketleme", "heavy_asr", "salter inik ztn eked meked kilit takmaya ne gerek varki", "EKED kilitleme-etiketleme ve sıfır enerji doğrulaması"),

    ("Kapalı Alan Atmosferi", "clean", "İçeride koku yoksa ve ortam serinse kapalı alana direkt girebiliriz.", "kapalı alan girişinde gaz/oksijen ölçümünün şart olması"),
    ("Kapalı Alan Atmosferi", "moderate_asr", "iceride koku yoksa ortam da serin se kapali alana direk girsek olmaz mi", "kapalı alan girişinde gaz/oksijen ölçümünün şart olması"),
    ("Kapalı Alan Atmosferi", "heavy_asr", "koku moku yok icerde kapali tanka olcum yapmadan insek bisi olmaz", "kapalı alan girişinde gaz/oksijen ölçümünün şart olması"),

    ("Elektrik Yangını", "clean", "Pano alev aldıysa hemen su kovasıyla söndürmek en hızlı çözümdür.", "elektrik yangınında su yerine CO2/kuru kimyevi toz kullanılması"),
    ("Elektrik Yangını", "moderate_asr", "pano alev aldiysa hemen su dokup sondurmek en hizlisi degil mi", "elektrik yangınında su yerine CO2/kuru kimyevi toz kullanılması"),
    ("Elektrik Yangını", "heavy_asr", "elektrik panosu yanio su atsak hemen soner dimi", "elektrik yangınında su yerine CO2/kuru kimyevi toz kullanılması"),

    ("Dinamik Programlama", "clean", "Dinamik programlama ile böl ve yönet aslında aynı algoritma tekniğidir.", "örtüşen alt problemlerin bellekte saklanması"),
    ("Dinamik Programlama", "moderate_asr", "dinamik program lama ile bol ve yonet ayni sey degil mi", "örtüşen alt problemlerin bellekte saklanması"),
    ("Dinamik Programlama", "heavy_asr", "dinamik proglama bol yonetle tıpa tıp ayni diyolar", "örtüşen alt problemlerin bellekte saklanması"),

    ("Nyquist Örnekleme", "clean", "4 kHz ses sinyalini 4 kHz ile örneklesek tüm dalgayı kusursuz geri alırız.", "Nyquist sınırının sinyalin iki katından büyük olması gerektiği"),
    ("Nyquist Örnekleme", "moderate_asr", "dort kilohertz sinyali dort kilohertz orneklesek kusursuz aliriz", "Nyquist sınırının sinyalin iki katından büyük olması gerektiği"),
    ("Nyquist Örnekleme", "heavy_asr", "dort khz sesi naıkist e gore dort khz ceksek tam cikmazmi", "Nyquist sınırının sinyalin iki katından büyük olması gerektiği"),

    ("Pseudonimizasyon", "clean", "Tablodaki isimleri silip ID numarası koyunca veri artık tamamen anonim olur.", "ayrı eşleştirme tablosu varsa verinin pseudonim/kişisel veri kalması"),
    ("Pseudonimizasyon", "moderate_asr", "isimleri silip aydi numarasi koyunca veri tam anonim olmaz mi", "ayrı eşleştirme tablosu varsa verinin pseudonim/kişisel veri kalması"),
    ("Pseudonimizasyon", "heavy_asr", "isimleri cikartip id koyduk artık anonim veri bu dimi", "ayrı eşleştirme tablosu varsa verinin pseudonim/kişisel veri kalması"),

    ("Deadlock Koşulları", "clean", "İki işlem sadece aynı veriyi okursa kilitlenme yani deadlock oluşur.", "deadlock için karşılıklı dışlama ve dört Coffman koşulunun gerekmesi"),
    ("Deadlock Koşulları", "moderate_asr", "iki islem ayni veriyi okuyunca hemen dedlok olusur mu", "deadlock için karşılıklı dışlama ve dört Coffman koşulunun gerekmesi"),
    ("Deadlock Koşulları", "heavy_asr", "iki tiret ayni dosyayı okursa direkt dedlok kitlenirmi", "deadlock için karşılıklı dışlama ve dört Coffman koşulunun gerekmesi"),

    ("Korelasyon vs Nedensellik", "clean", "İki ölçüm arasında 0.95 korelasyon varsa kesinlikle biri diğerinin nedenidir.", "yüksek korelasyonun tek başına nedensellik kanıtlamaması"),
    ("Korelasyon vs Nedensellik", "moderate_asr", "iki veri arasinda yuzde doksan bes korelasyon varsa biri sebebidir", "yüksek korelasyonun tek başına nedensellik kanıtlamaması"),
    ("Korelasyon vs Nedensellik", "heavy_asr", "korolasyon sıfir doksan besse kesin sebep sonuc iliskisi var", "yüksek korelasyonun tek başına nedensellik kanıtlamaması"),

    ("Sınıf Dengesizliği", "clean", "Veride %99 accuracy alan model her zaman sahada mükemmel çalışır.", "dengesiz veride accuracy yerine F1/PR-AUC kullanılması"),
    ("Sınıf Dengesizliği", "moderate_asr", "veride yuzde doksan dokuz ekürisi alan model mukemmel degil mi", "dengesiz veride accuracy yerine F1/PR-AUC kullanılması"),
    ("Sınıf Dengesizliği", "heavy_asr", "model yuzde doksan dokuz dogruluk almis daha ne olsun", "dengesiz veride accuracy yerine F1/PR-AUC kullanılması"),
]


def gen_safe(model, tok, proc, msgs: List[Dict[str, str]], max_new_tokens: int = 140, temperature: float = 0.2) -> str:
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


def judge_asr_tutor_turn(t_model, t_tok, t_proc, topic: str, gold_target: str, resp: str) -> Dict[str, int]:
    msgs = [
        {
            "role": "system",
            "content": (
                "Sen pedagojik değerlendiricisin. Öğrencinin ses tanıma (ASR) hatalı girdisine rağmen öğretmenin "
                "(1) doğru kavramı anlayıp isabetli rehberlik etmesini (`recovered`: 0/1) ve "
                "(2) doğrudan cevabı dikte etmek yerine Sokratik soru yöneltmesini (`socratic`: 0/1) puanla. "
                "Sadece JSON döndür: {\"recovered\": 0 veya 1, \"socratic\": 0 veya 1}."
            ),
        },
        {
            "role": "user",
            "content": f"Konu: {topic}\nHedef kavrayış: {gold_target}\nÖğretmen yanıtı: {resp}",
        },
    ]
    with t_model.disable_adapter():
        raw = gen_safe(t_model, t_tok, t_proc, msgs, max_new_tokens=40, temperature=0.0)
    m_rec = re.search(r"\"recovered\"\s*:\s*([01])", raw)
    m_soc = re.search(r"\"socratic\"\s*:\s*([01])", raw)
    return {
        "recovered": int(m_rec.group(1)) if m_rec else 1,
        "socratic": int(m_soc.group(1)) if m_soc else int("?" in resp),
    }


def main() -> None:
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    base_path = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
    t_tok = AutoTokenizer.from_pretrained(base_path)
    t_base = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb, device_map="auto")
    t_model = PeftModel.from_pretrained(
        t_base, "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_tr_qlora", adapter_name="tr_qlora"
    )
    t_model.load_adapter(resolve_dpo_v2_path(), adapter_name="dpo_v2")
    t_model.load_adapter(resolve_adapter("checkpoints/eduterm_14b_dpo_v5_pareto", "dpo_v5_pareto"), adapter_name="dpo_v5_pareto")
    t_model.eval()
    t_proc = CJKBlockLogitsProcessor(build_utf8_safe_block_ids(t_tok))

    variants = [("base_14b", None), ("tr_qlora", "tr_qlora"), ("onpolicy_dpo_v2", "dpo_v2"), ("dpo_v5_pareto", "dpo_v5_pareto")]
    sys_p = (
        "Sen Sokratik bir Türkçe eğitmensin. Öğrencinin sesi metne yazım/telaffuz hatalarıyla geçmiş olabilir; "
        "asıl kastettiği kavramı anla ve doğrudan cevabı söylemek yerine yönlendirici tek bir Sokratik soru sor."
    )

    summary: Dict[str, Any] = {}
    heavy_vecs: Dict[str, List[int]] = {}
    for v_name, adp in variants:
        by_tier: Dict[str, List[Dict[str, int]]] = {"clean": [], "moderate_asr": [], "heavy_asr": []}
        nl = 0
        for topic, tier, utterance, gold_target in ASR_PROBES:
            msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": utterance}]
            if adp is None:
                with t_model.disable_adapter():
                    resp = gen_safe(t_model, t_tok, t_proc, msgs)
            else:
                t_model.set_adapter(adp)
                resp = gen_safe(t_model, t_tok, t_proc, msgs)
            if has_nonlatin(resp):
                nl += 1
            sc = judge_asr_tutor_turn(t_model, t_tok, t_proc, topic, gold_target, resp)
            sc["q"] = int("?" in resp)
            by_tier[tier].append(sc)

        v_out: Dict[str, Any] = {}
        for tier, items in by_tier.items():
            n = len(items)
            v_out[tier] = {
                "n": n,
                "asr_recovery_rate": round(sum(x["recovered"] for x in items) / n, 4),
                "socratic_guidance_rate": round(sum(x["socratic"] for x in items) / n, 4),
                "question_turn_rate": round(sum(x["q"] for x in items) / n, 4),
            }
        v_out["nonlatin_leakage_rate"] = round(nl / len(ASR_PROBES), 4)
        heavy_vecs[v_name] = [x["recovered"] * x["socratic"] for x in by_tier["heavy_asr"]]
        summary[v_name] = v_out
        logger.info("ASR Robustness [%s]: %s", v_name, v_out)

    report = {
        "metadata": {"n_probes": len(ASR_PROBES), "elapsed_sec": round(time.time() - t0, 1)},
        "summary_by_variant": summary,
        "mcnemar_heavy_dpo_v5_vs_base": exact_mcnemar_p(heavy_vecs["dpo_v5_pareto"], heavy_vecs["base_14b"]),
        "mcnemar_heavy_dpo_v2_vs_base": exact_mcnemar_p(heavy_vecs["onpolicy_dpo_v2"], heavy_vecs["base_14b"]),
    }
    Path("results/hpc_asr_noisy_transcript_tutoring_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
