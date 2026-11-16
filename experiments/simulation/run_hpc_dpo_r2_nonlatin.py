# -*- coding: utf-8 -*-
"""
Experiment 4: Non-Latin Script Blocking + Purity-Filtered On-Policy DPO Round 2
on Boğaziçi HPC (A40 48GB).

1. `NonLatinBlockLogitsProcessor`: blocks ALL non-Latin script tokens (CJK, Cyrillic, Arabic,
   Hangul, Kana, Devanagari, etc.) while preserving ASCII + Turkish/Latin Extended + punctuation.
2. Extracts purity-filtered preference pairs from Job 34740 & Job 35536 transcripts:
   - chosen: misconception_resolved == 1 AND turkish_purity == 5 AND cjk_leakage_turns == 0
   - rejected: misconception_resolved == 0 OR turkish_purity <= 3 OR cjk_leakage_turns > 0
3. Trains `eduterm_14b_onpolicy_dpo_v2` (2 epochs, beta=0.1, lr=5e-5).
4. Evaluates `task_separated_hybrid` vs `onpolicy_dpo_v1` vs `onpolicy_dpo_v2` on Kumru-2B
   across 12 ISG concepts x 3 personas (36 paired sessions/arm = 108 sessions, 432 turns).

Outputs:
  results/hpc_dpo_r2_nonlatin_report.json
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
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_hpc_cjk_blocked_dag_ablation import (
    CJKBlockLogitsProcessor,
    bootstrap_ci,
    exact_mcnemar_p,
    generate_chat_nocjk,
    judge_session_blinded,
)
from run_hpc_dpo_3seed_ablation import generate_cefr_card, resolve_dpo_path
from run_hpc_live_student_simulation import ISG_CONCEPTS, PERSONAS
from run_hpc_onpolicy_dpo_and_eval import encode_pair, sequence_logp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp4DPOR2NonLatin")

# Allows ASCII, Latin-1/Extended-A/B (Turkish çğıöşüÇĞİÖŞÜ), Latin Extended Additional, General Punctuation
NON_LATIN_PATTERN = re.compile(r"[^\x00-\x7f\u00c0-\u024f\u1e00-\u1eff\u2000-\u206f]")


def build_nonlatin_token_ids(tokenizer) -> List[int]:
    blocked: List[int] = []
    for tid in range(len(tokenizer)):
        tok_str = tokenizer.decode([tid], skip_special_tokens=False)
        if NON_LATIN_PATTERN.search(tok_str):
            blocked.append(tid)
    return blocked


def has_nonlatin(text: str) -> bool:
    return bool(NON_LATIN_PATTERN.search(text))


def build_purity_dpo_pairs(report_paths: List[Path], concept_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, str, str], List[Dict[str, Any]]] = {}
    for rp in report_paths:
        if not rp.exists():
            continue
        data = json.loads(rp.read_text(encoding="utf-8"))
        for s in data.get("sessions", []):
            grouped.setdefault((rp.name, s.get("seed", 42), s["concept_id"], s["persona_id"]), []).append(s)

    pairs: List[Dict[str, Any]] = []
    for (_, _, cid, _), group in grouped.items():
        concept = concept_by_id[cid]
        prereq_str = ", ".join(concept["prerequisites"]) if concept["prerequisites"] else "Yok (Temel Kavram)"
        winners = [
            s for s in group
            if s["scores"]["misconception_resolved"] == 1
            and s["scores"]["turkish_purity"] == 5
            and s.get("cjk_leakage_turns", 0) == 0
        ]
        losers = [
            s for s in group
            if s["scores"]["misconception_resolved"] == 0
            or s["scores"]["turkish_purity"] <= 3
            or s.get("cjk_leakage_turns", 0) > 0
        ]
        if not winners or not losers:
            continue
        w = max(winners, key=lambda x: (x["scores"]["socratic_score"], x["scores"]["prerequisite_score"]))
        l = min(losers, key=lambda x: (x["scores"]["misconception_resolved"], x["scores"]["turkish_purity"], x["scores"]["socratic_score"]))

        sys_msg = (
            f"Sen EduTerm Sokratik İSG Eğitmenisin. Hedef kavram: {concept['name']}.\n"
            f"DAG Önkoşulları: {prereq_str}.\n"
            "Kurallar: Cevabı ilk turda doğrudan verme; kısa, yönlendirici Sokratik soru sor ve yalnızca saf Türkçe yaz."
        )
        pairs.append({
            "messages_prefix": [{"role": "system", "content": sys_msg}, {"role": "user", "content": w["transcript"][0]["student"]}],
            "chosen": w["transcript"][0]["tutor"],
            "rejected": l["transcript"][0]["tutor"],
        })
        if len(w["transcript"]) >= 2 and len(l["transcript"]) >= 2:
            pairs.append({
                "messages_prefix": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": w["transcript"][0]["student"]},
                    {"role": "assistant", "content": w["transcript"][0]["tutor"]},
                    {"role": "user", "content": w["transcript"][1]["student"]},
                ],
                "chosen": w["transcript"][1]["tutor"],
                "rejected": l["transcript"][1]["tutor"],
            })
    return pairs


def run_eval_session(
    s_model, s_tok, s_proc, t_model, t_tok, t_proc,
    concept: Dict[str, Any], persona: Dict[str, Any], condition: str, num_turns: int = 4
) -> Dict[str, Any]:
    prereq_str = ", ".join(concept["prerequisites"]) if concept["prerequisites"] else "Yok (Temel Kavram)"
    student_sys = (
        f"Sen gerçek bir öğrencisin ({persona['label']}). {persona['instruction']}\n"
        f"Konu: {concept['name']}.\n"
        f"Başlangıçtaki yanlış inancın: {concept['misconception']}\n"
        "ÖNEMLİ: Sadece kendi ağzından ('Ben...' diyerek) 1-2 cümlelik kısa Türkçe yanıt ver. "
        "Asla 'Öğretmen:' veya 'Öğrenci:' diye senaryo yazma, yabancı dil kullanma."
    )

    cefr_card = ""
    if condition == "task_separated_hybrid":
        cefr_card = generate_cefr_card(t_model, t_tok, t_proc, concept)
        tutor_sys = (
            f"Sen Sokratik bir İSG eğitmenisin. Hedef kavram: {concept['name']}.\n"
            f"DAG Önkoşulları: {prereq_str}.\n"
            f"EduTerm CEFR-B1 Önkoşul Özeti: {cefr_card}\n"
            "Kurallar: Cevabı ilk turda doğrudan verme. Yukarıdaki önkoşul özetinden yararlanarak "
            "öğrenciye kısa, yönlendirici bir soru sor. Yalnızca temiz Türkçe (2-4 cümle) yaz."
        )
    else:
        tutor_sys = (
            f"Sen EduTerm Sokratik İSG Eğitmenisin. Hedef kavram: {concept['name']}.\n"
            f"DAG Önkoşulları: {prereq_str}.\n"
            "Kurallar: Cevabı ilk turda doğrudan verme; kısa, yönlendirici Sokratik soru sor ve yalnızca saf Türkçe yaz."
        )

    student_msgs = [
        {"role": "system", "content": student_sys},
        {"role": "user", "content": f"Hocaya '{concept['name']}' hakkındaki yanlış düşünceni doğrudan 1-2 cümleyle sor."},
    ]
    tutor_msgs = [{"role": "system", "content": tutor_sys}]
    transcript: List[Dict[str, str]] = []

    for t in range(1, num_turns + 1):
        s_reply = generate_chat_nocjk(s_model, s_tok, student_msgs, cjk_processor=s_proc, max_new_tokens=100, temperature=0.7)
        student_msgs.append({"role": "assistant", "content": s_reply})
        tutor_msgs.append({"role": "user", "content": s_reply})

        if condition == "onpolicy_dpo_v1":
            t_model.set_adapter("dpo_v1")
            t_reply = generate_chat_nocjk(t_model, t_tok, tutor_msgs, cjk_processor=t_proc, max_new_tokens=140, temperature=0.4)
        elif condition == "onpolicy_dpo_v2":
            t_model.set_adapter("dpo_v2")
            t_reply = generate_chat_nocjk(t_model, t_tok, tutor_msgs, cjk_processor=t_proc, max_new_tokens=140, temperature=0.4)
        else:
            with t_model.disable_adapter():
                t_reply = generate_chat_nocjk(t_model, t_tok, tutor_msgs, cjk_processor=t_proc, max_new_tokens=140, temperature=0.4)

        tutor_msgs.append({"role": "assistant", "content": t_reply})
        student_msgs.append({"role": "user", "content": f"Hocan şöyle dedi: {t_reply}\nÖğrenci olarak 1-2 cümleyle yanıtla."})
        transcript.append({"turn": t, "student": s_reply, "tutor": t_reply})

    return {
        "concept_id": concept["id"],
        "concept_name": concept["name"],
        "persona_id": persona["id"],
        "condition": condition,
        "cefr_prereq_card": cefr_card,
        "nonlatin_leakage_turns": sum(1 for x in transcript if has_nonlatin(x["tutor"]) or has_nonlatin(x["student"])),
        "tutor_question_turns": sum(1 for x in transcript if "?" in x["tutor"]),
        "transcript": transcript,
    }


def main() -> None:
    t0 = time.time()
    random.seed(42)
    torch.manual_seed(42)
    concept_by_id = {c["id"]: c for c in ISG_CONCEPTS}

    pairs = build_purity_dpo_pairs(
        [Path("results/hpc_cjk_blocked_dag_ablation_report.json"), Path("results/hpc_dpo_3seed_ablation_report.json")],
        concept_by_id,
    )
    random.shuffle(pairs)
    pairs = pairs[:240]
    logger.info("Built %d purity-filtered DPO Round 2 preference pairs.", len(pairs))

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    base_path = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
    tok = AutoTokenizer.from_pretrained(base_path)
    base = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb, device_map="auto")
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base, lora_cfg, adapter_name="dpo_v2")
    model.set_adapter("dpo_v2")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    beta, grad_accum, epochs = 0.1, 8, 2
    model.train()
    epoch_losses: List[float] = []
    for ep in range(1, epochs + 1):
        random.shuffle(pairs)
        ep_loss = 0.0
        optimizer.zero_grad()
        for idx, ex in enumerate(pairs, start=1):
            c_ids, c_lbl = encode_pair(tok, ex["messages_prefix"], ex["chosen"])
            r_ids, r_lbl = encode_pair(tok, ex["messages_prefix"], ex["rejected"])
            c_ids, c_lbl = c_ids.to(model.device), c_lbl.to(model.device)
            r_ids, r_lbl = r_ids.to(model.device), r_lbl.to(model.device)
            with torch.no_grad():
                with model.disable_adapter():
                    ref_c = sequence_logp(model, c_ids, c_lbl)
                    ref_r = sequence_logp(model, r_ids, r_lbl)
            pol_c = sequence_logp(model, c_ids, c_lbl)
            pol_r = sequence_logp(model, r_ids, r_lbl)
            loss = -F.logsigmoid(beta * ((pol_c - ref_c) - (pol_r - ref_r))).mean() / grad_accum
            loss.backward()
            ep_loss += loss.item() * grad_accum
            if idx % grad_accum == 0 or idx == len(pairs):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        avg_ep = round(ep_loss / max(len(pairs), 1), 4)
        epoch_losses.append(avg_ep)
        logger.info("DPO-R2 Epoch %d avg loss: %.4f", ep, avg_ep)

    dpo_v2_ckpt = Path("checkpoints/eduterm_14b_onpolicy_dpo_v2")
    dpo_v2_ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dpo_v2_ckpt))
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    model.load_adapter("/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora", adapter_name="cefr")
    model.load_adapter(resolve_dpo_path(), adapter_name="dpo_v1")
    model.eval()

    t_blocked = build_nonlatin_token_ids(tok)
    t_proc = CJKBlockLogitsProcessor(t_blocked)
    logger.info("Tutor non-Latin blocked vocab size: %d", len(t_blocked))

    s_tok = AutoTokenizer.from_pretrained("/users/ali.gunder/models/Kumru-2B")
    s_model = AutoModelForCausalLM.from_pretrained("/users/ali.gunder/models/Kumru-2B", torch_dtype=torch.bfloat16, device_map="auto")
    s_model.eval()
    s_proc = CJKBlockLogitsProcessor(build_nonlatin_token_ids(s_tok))

    conditions = ("task_separated_hybrid", "onpolicy_dpo_v1", "onpolicy_dpo_v2")
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / "hpc_dpo_r2_nonlatin_partial.json"

    sessions: List[Dict[str, Any]] = []
    total = len(ISG_CONCEPTS) * len(PERSONAS) * len(conditions)
    for c_idx, concept in enumerate(ISG_CONCEPTS):
        for p_idx, persona in enumerate(PERSONAS):
            pair_seed = 42 + c_idx * 100 + p_idx * 10
            for cond in conditions:
                torch.manual_seed(pair_seed)
                random.seed(pair_seed)
                sess = run_eval_session(s_model, s_tok, s_proc, model, tok, t_proc, concept, persona, cond, num_turns=4)
                scores = judge_session_blinded(model, tok, t_proc, concept, sess, disable_adapter=True)
                sess["scores"] = scores
                sessions.append(sess)
                logger.info(
                    "[%d/%d] [%s | %s | %s] res=%d soc=%d pre=%d pur=%d nonlat=%d",
                    len(sessions), total, concept["id"], persona["id"], cond,
                    scores["misconception_resolved"], scores["socratic_score"],
                    scores["prerequisite_score"], scores["turkish_purity"], sess["nonlatin_leakage_turns"],
                )
            if len(sessions) % 18 == 0:
                partial_path.write_text(json.dumps({"completed": len(sessions), "sessions": sessions}, ensure_ascii=False), encoding="utf-8")

    summary: Dict[str, Any] = {}
    cmap = {c: {} for c in conditions}
    for s in sessions:
        cmap[s["condition"]][(s["concept_id"], s["persona_id"])] = s

    for cond in conditions:
        sub = [s for s in sessions if s["condition"] == cond]
        n = len(sub)
        res_vals = [s["scores"]["misconception_resolved"] for s in sub]
        ci_lo, ci_hi = bootstrap_ci([float(v) for v in res_vals])
        summary[cond] = {
            "total_sessions": n,
            "total_turns": n * 4,
            "misconception_resolution_rate": round(sum(res_vals) / n, 4),
            "resolution_bootstrap_95ci": [ci_lo, ci_hi],
            "socratic_mean": round(sum(s["scores"]["socratic_score"] for s in sub) / n, 3),
            "prerequisite_grounding_mean": round(sum(s["scores"]["prerequisite_score"] for s in sub) / n, 3),
            "turkish_purity_mean": round(sum(s["scores"]["turkish_purity"] for s in sub) / n, 3),
            "nonlatin_leakage_turn_rate": round(sum(s["nonlatin_leakage_turns"] for s in sub) / (n * 4), 4),
            "tutor_question_turn_rate": round(sum(s["tutor_question_turns"] for s in sub) / (n * 4), 4),
        }

    keys = sorted(cmap[conditions[0]].keys())
    paired = {}
    for c_a, c_b in [
        ("onpolicy_dpo_v2", "task_separated_hybrid"),
        ("onpolicy_dpo_v2", "onpolicy_dpo_v1"),
        ("onpolicy_dpo_v1", "task_separated_hybrid"),
    ]:
        va = [cmap[c_a][k]["scores"]["misconception_resolved"] for k in keys]
        vb = [cmap[c_b][k]["scores"]["misconception_resolved"] for k in keys]
        diffs = [float(a - b) for a, b in zip(va, vb)]
        d_lo, d_hi = bootstrap_ci(diffs)
        paired[f"{c_a}__vs__{c_b}"] = {
            "mean_diff_pp": round(100.0 * sum(diffs) / len(diffs), 2),
            "diff_pp_95ci": [round(100.0 * d_lo, 2), round(100.0 * d_hi, 2)],
            "mcnemar": exact_mcnemar_p(va, vb),
        }

    out_path = out_dir / "hpc_dpo_r2_nonlatin_report.json"
    out_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "dpo_r2_pairs": len(pairs),
                    "dpo_r2_epoch_losses": epoch_losses,
                    "nonlatin_blocked_vocab_tutor": len(t_blocked),
                    "elapsed_sec": round(time.time() - t0, 1),
                },
                "summary": summary,
                "paired_comparisons": paired,
                "sessions": sessions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Exp 4 Finished in %.1fs -> %s", time.time() - t0, out_path)


if __name__ == "__main__":
    main()
