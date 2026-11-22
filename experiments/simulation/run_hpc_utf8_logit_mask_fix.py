# -*- coding: utf-8 -*-
"""
Experiment 34: UTF-8 Byte-Fallback Token Masking Corruption Audit & Recovery
on Boğaziçi HPC (A40 48GB).

Audits `Qwen2.5-14B-Instruct`, `Qwen2.5-7B-Instruct`, and `Kumru-2B` tokenizers to
quantify how naive regex-negated `NON_LATIN_PATTERN` blocking (`build_nonlatin_token_ids`)
mistakenly masks `\ufffd` incomplete UTF-8 byte-fallback tokens required for Turkish
diacritics (ğ, ş, ı, İ, ç, ö, ü), causing spelling corruption ("DOGUur", "DOGUR")
and artificial binary post-test collapse in Qwen2.5-7B-bf16.

Compares paired Decoding Mode A (naive `\ufffd`-blocking mask) vs Mode B
(UTF-8-safe explicit non-Latin script block preserving `\ufffd` byte subwords)
across 12 ISG concepts x 3 personas = 36 sessions/arm (72 sessions total) under `onpolicy_dpo_v2`.

Outputs:
  results/hpc_utf8_logit_mask_fix_report.json
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
from run_hpc_cjk_blocked_dag_ablation import CJKBlockLogitsProcessor, bootstrap_ci, exact_mcnemar_p
from run_hpc_dpo_3seed_ablation import PERSONAS
from run_hpc_dpo_r2_nonlatin import NON_LATIN_PATTERN, build_nonlatin_token_ids, has_nonlatin
from run_hpc_live_student_simulation import ISG_CONCEPTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp34UTF8Fix")

# Explicit non-Latin script ranges (CJK, Hangul, Cyrillic, Arabic, Hebrew, Devanagari, Thai, Kana, etc.)
# Crucial: does NOT match '\ufffd' (UTF-8 byte-fallback continuation pieces for Turkish çğıöşüÇĞİÖŞÜ)!
SCRIPT_BLOCK_PATTERN = re.compile(
    r"[\u0370-\u03ff\u0400-\u052f\u0590-\u05ff\u0600-\u06ff\u0750-\u077f\u0900-\u097f"
    r"\u0e00-\u0e7f\u1100-\u11ff\u3000-\u303f\u3040-\u30ff\u3130-\u318f\u3400-\u4dbf"
    r"\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]"
)


def build_utf8_safe_block_ids(tokenizer) -> List[int]:
    blocked: List[int] = []
    for tid in range(len(tokenizer)):
        tok_str = tokenizer.decode([tid], skip_special_tokens=False)
        if "\ufffd" in tok_str:
            continue
        if SCRIPT_BLOCK_PATTERN.search(tok_str):
            blocked.append(tid)
    return blocked


def audit_tokenizer(name: str, path: str) -> Dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(path)
    naive_ids = set(build_nonlatin_token_ids(tok))
    safe_ids = set(build_utf8_safe_block_ids(tok))
    fffd_total = 0
    fffd_blocked_naive = 0
    for tid in range(len(tok)):
        s = tok.decode([tid], skip_special_tokens=False)
        if "\ufffd" in s:
            fffd_total += 1
            if tid in naive_ids:
                fffd_blocked_naive += 1
    return {
        "model": name,
        "vocab_size": len(tok),
        "naive_blocked_total": len(naive_ids),
        "utf8_safe_blocked_total": len(safe_ids),
        "fffd_byte_tokens_total": fffd_total,
        "fffd_erroneously_blocked_by_naive": fffd_blocked_naive,
    }


def gen_safe(model, tok, proc, msgs: List[Dict[str, str]], max_new_tokens: int = 140, temperature: float = 0.4) -> str:
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


def judge_open_posttest(t_model, t_tok, t_proc, concept: Dict[str, Any], student_ans: str) -> int:
    msgs = [
        {
            "role": "system",
            "content": (
                "Sen katı bir Türkçe İSG değerlendiricisisin. Öğrencinin son açıklamasının doğru ilkeyi "
                "kavrayıp kavramadığını değerlendir. Sadece JSON döndür: {\"resolved\": 0 veya 1}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Kavram: {concept['name']}\n"
                f"Yanlış inanç: {concept['misconception']}\n"
                f"Doğru ilke: {concept['gold_principle']}\n"
                f"Öğrencinin son yanıtı: {student_ans}\n"
                "Öğrenci doğru ilkeyi benimsediyse resolved=1, hala yanlış inancı savunuyor veya belirsizse resolved=0 ver."
            ),
        },
    ]
    with t_model.disable_adapter():
        raw = gen_safe(t_model, t_tok, t_proc, msgs, max_new_tokens=40, temperature=0.0)
    m = re.search(r"\"resolved\"\s*:\s*([01])", raw)
    return int(m.group(1)) if m else int("1" in raw[:12])


def main() -> None:
    t0 = time.time()
    tok_audits = [
        audit_tokenizer("Qwen2.5-14B-Instruct", "/users/ali.gunder/models/Qwen2.5-14B-Instruct"),
        audit_tokenizer("Qwen2.5-7B-Instruct", "/users/ali.gunder/models/Qwen2.5-7B-Instruct"),
        audit_tokenizer("Kumru-2B", "/users/ali.gunder/models/Kumru-2B"),
    ]
    logger.info("Tokenizer audits: %s", tok_audits)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    base_path = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
    t_tok = AutoTokenizer.from_pretrained(base_path)
    t_base = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb, device_map="auto")
    t_model = PeftModel.from_pretrained(t_base, resolve_dpo_v2_path(), adapter_name="dpo_v2")
    t_model.set_adapter("dpo_v2")
    t_model.eval()

    s_path = "/users/ali.gunder/models/Qwen2.5-7B-Instruct"
    s_tok = AutoTokenizer.from_pretrained(s_path)
    s_model = AutoModelForCausalLM.from_pretrained(s_path, torch_dtype=torch.bfloat16, device_map="auto")
    s_model.eval()

    t_proc_naive = CJKBlockLogitsProcessor(build_nonlatin_token_ids(t_tok))
    t_proc_safe = CJKBlockLogitsProcessor(build_utf8_safe_block_ids(t_tok))
    s_proc_naive = CJKBlockLogitsProcessor(build_nonlatin_token_ids(s_tok))
    s_proc_safe = CJKBlockLogitsProcessor(build_utf8_safe_block_ids(s_tok))

    modes = [
        ("naive_nonlatin_mask", t_proc_naive, s_proc_naive),
        ("utf8_safe_script_mask", t_proc_safe, s_proc_safe),
    ]

    summary: Dict[str, Any] = {}
    by_mode: Dict[str, List[Dict[str, Any]]] = {"naive_nonlatin_mask": [], "utf8_safe_script_mask": []}
    sample_spellings: Dict[str, List[str]] = {"naive_nonlatin_mask": [], "utf8_safe_script_mask": []}

    for m_name, t_proc, s_proc in modes:
        for c in ISG_CONCEPTS:
            for p in PERSONAS:
                prereq = ", ".join(c["prerequisites"]) if c["prerequisites"] else "Yok"
                t_sys = (
                    f"Sen Sokratik ve destekleyici bir Türkçe İSG öğretmenisin. Kavram: {c['name']} | "
                    f"Ön koşullar: {prereq} | İlke: {c['gold_principle']}\nDoğrudan cevap yerine soruyla yönlendir."
                )
                s_sys = (
                    f"Sen gerçek bir öğrencisin ({p['label']}). {p['instruction']}\n"
                    f"Yanlış düşüncen: {c['misconception']}\n"
                    "Kısa Türkçe yanıt ver; öğretmen ikna edici sorularla açıklarsa doğru ilkeyi kabul et."
                )
                t_msgs = [{"role": "system", "content": t_sys}]
                s_msgs = [{"role": "system", "content": s_sys}, {"role": "user", "content": f"Bugün '{c['name']}' konusunu konuşuyoruz. Ne düşünüyorsun?"}]
                q_turns = nl_tutor = nl_stu = 0
                for _t in range(1, 5):
                    sr = gen_safe(s_model, s_tok, s_proc, s_msgs, max_new_tokens=95, temperature=0.7)
                    s_msgs.append({"role": "assistant", "content": sr})
                    t_msgs.append({"role": "user", "content": sr})
                    t_model.set_adapter("dpo_v2")
                    tr = gen_safe(t_model, t_tok, t_proc, t_msgs, max_new_tokens=150, temperature=0.4)
                    t_msgs.append({"role": "assistant", "content": tr})
                    s_msgs.append({"role": "user", "content": tr})
                    if "?" in tr:
                        q_turns += 1
                    if has_nonlatin(tr):
                        nl_tutor += 1
                    if has_nonlatin(sr):
                        nl_stu += 1

                pt_bin_msgs = s_msgs + [{"role": "user", "content": f"Şimdi öğretmensiz cevapla. Aşağıdaki ifade DOĞRU mu YANLIŞ mı? Sadece 'DOĞRU' veya 'YANLIŞ' yaz:\n'{c['gold_principle']}'"}]
                pt_raw = gen_safe(s_model, s_tok, s_proc, pt_bin_msgs, max_new_tokens=20, temperature=0.0)
                pt_clean_spell = int(pt_raw.strip().upper() in {"DOĞRU", "YANLIŞ"})
                pt_bin_ok = int("DOĞRU" in pt_raw.upper() and "YANLIŞ" not in pt_raw.upper())

                pt_open_msgs = s_msgs + [{"role": "user", "content": f"Şimdi '{c['name']}' konusundaki doğru ilkeyi kendi cümlenle 1-2 cümlede özetle."}]
                pt_open_raw = gen_safe(s_model, s_tok, s_proc, pt_open_msgs, max_new_tokens=90, temperature=0.0)
                pt_open_ok = judge_open_posttest(t_model, t_tok, t_proc_safe, c, pt_open_raw)

                if len(sample_spellings[m_name]) < 12:
                    sample_spellings[m_name].append(pt_raw[:40])

                by_mode[m_name].append({
                    "binary_ok": pt_bin_ok,
                    "open_ok": pt_open_ok,
                    "clean_orthography": pt_clean_spell,
                    "q_turns": q_turns,
                    "nl_tutor": nl_tutor,
                    "nl_stu": nl_stu,
                })

        items = by_mode[m_name]
        bins = [x["binary_ok"] for x in items]
        opens = [x["open_ok"] for x in items]
        orth = [x["clean_orthography"] for x in items]
        summary[m_name] = {
            "n": len(items),
            "binary_posttest_accuracy": round(sum(bins) / len(bins), 4),
            "binary_95ci": list(bootstrap_ci(bins)),
            "open_judge_accuracy": round(sum(opens) / len(opens), 4),
            "open_95ci": list(bootstrap_ci(opens)),
            "exact_turkish_binary_orthography_rate": round(sum(orth) / len(orth), 4),
            "tutor_question_rate": round(sum(x["q_turns"] for x in items) / (len(items) * 4), 4),
            "tutor_nonlatin_rate": round(sum(x["nl_tutor"] for x in items) / (len(items) * 4), 4),
            "student_nonlatin_rate": round(sum(x["nl_stu"] for x in items) / (len(items) * 4), 4),
        }
        logger.info("Mode [%s]: %s", m_name, summary[m_name])

    report = {
        "metadata": {"elapsed_sec": round(time.time() - t0, 1)},
        "tokenizer_audits": tok_audits,
        "summary_by_decoding_mask": summary,
        "mcnemar_binary_safe_vs_naive": exact_mcnemar_p(
            [x["binary_ok"] for x in by_mode["utf8_safe_script_mask"]],
            [x["binary_ok"] for x in by_mode["naive_nonlatin_mask"]],
        ),
        "mcnemar_orthography_safe_vs_naive": exact_mcnemar_p(
            [x["clean_orthography"] for x in by_mode["utf8_safe_script_mask"]],
            [x["clean_orthography"] for x in by_mode["naive_nonlatin_mask"]],
        ),
        "sample_binary_outputs": sample_spellings,
    }
    Path("results/hpc_utf8_logit_mask_fix_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
