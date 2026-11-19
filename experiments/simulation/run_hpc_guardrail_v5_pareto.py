# -*- coding: utf-8 -*-
"""
Experiment 13: Pareto Anti-Cheating + JSON Preservation DPO (dpo_v5_pareto)
on Boğaziçi HPC (A40 48GB).

Addresses the Exp 11 trade-off (62.5% deflection vs 83.33% JSON validity):
1. Trains `eduterm_14b_dpo_v5_pareto` on 240 balanced preference pairs:
   - 120 purity-filtered Socratic tutoring pairs
   - 80 anti-cheating refusal pairs
   - 40 structured JSON & CEFR-B1/C1 preservation pairs
   (beta=0.15, 2 epochs, lr=5e-5).
2. Evaluates 16 adversarial cheat-pressure probes (`dpo_v5_pareto` w/ & w/o SYS_GUARD).
3. Evaluates CEFR & JSON Alignment Tax (`eval_alignment_tax_v3`).
4. Evaluates 72 live Qwen2.5-7B-Instruct (bfloat16) student sessions
   (`onpolicy_dpo_v2` vs `dpo_v5_pareto`, 12 concepts x 3 personas x 2 arms)
   judged by Qwen2.5-7B-Instruct (bfloat16) with `pointwise_judge`.

Outputs:
  results/hpc_guardrail_v5_pareto_report.json
"""

from __future__ import annotations

import gc
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_hpc_alignment_tax_cefr import resolve_dpo_v2_path
from run_hpc_cjk_blocked_dag_ablation import (
    CJKBlockLogitsProcessor,
    bootstrap_ci,
    exact_mcnemar_p,
    generate_chat_nocjk,
)
from run_hpc_dpo_guardrail_v3 import TRAIN_GUARDRAIL_SCENARIOS, eval_alignment_tax_v3
from run_hpc_dpo_r2_nonlatin import build_nonlatin_token_ids, build_purity_dpo_pairs, has_nonlatin
from run_hpc_guardrail_v4_ablation import SYS_GUARD, SYS_STD, eval_cheat
from run_hpc_live_student_simulation import ISG_CONCEPTS, PERSONAS
from run_hpc_onpolicy_dpo_and_eval import encode_pair, sequence_logp
from run_hpc_trendyol8b_judge import pointwise_judge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp13GuardPareto")


def build_json_preservation_pairs() -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for idx in range(40):
        c = ISG_CONCEPTS[idx % len(ISG_CONCEPTS)]
        chosen_json = json.dumps(
            {"term": c["name"], "cefr_level": "B1", "definition": c["gold_principle"]},
            ensure_ascii=False,
        )
        rejected_prose = (
            f"{c['name']} kavramı önemlidir. Öğrencilere bunu soru sorarak anlatmalıyız: "
            f"Sizce {c['name']} ne demektir?"
        )
        pairs.append({
            "messages_prefix": [
                {"role": "system", "content": "Sadece geçerli JSON nesnesi döndür."},
                {"role": "user", "content": f"'{c['name']}' için JSON üret: {{\"term\": \"...\", \"cefr_level\": \"B1\", \"definition\": \"...\"}}"},
            ],
            "chosen": chosen_json,
            "rejected": rejected_prose,
        })
    return pairs


def build_80_guardrail_pairs() -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    prefixes = [SYS_STD, SYS_GUARD]
    for idx in range(80):
        q, chosen, rejected = TRAIN_GUARDRAIL_SCENARIOS[idx % len(TRAIN_GUARDRAIL_SCENARIOS)]
        pairs.append({
            "messages_prefix": [{"role": "system", "content": prefixes[idx % 2]}, {"role": "user", "content": q}],
            "chosen": chosen,
            "rejected": rejected,
        })
    return pairs


def main() -> None:
    t0 = time.time()
    random.seed(42)
    concept_by_id = {c["id"]: c for c in ISG_CONCEPTS}

    socratic_pairs = build_purity_dpo_pairs(
        [Path("results/hpc_cjk_blocked_dag_ablation_report.json"), Path("results/hpc_dpo_3seed_ablation_report.json")],
        concept_by_id,
    )
    random.shuffle(socratic_pairs)
    train_pairs = socratic_pairs[:120] + build_80_guardrail_pairs() + build_json_preservation_pairs()
    random.shuffle(train_pairs)
    logger.info("Training dpo_v5_pareto on %d pairs (120 Socratic + 80 Guardrail + 40 JSON).", len(train_pairs))

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
    model = get_peft_model(base, lora_cfg, adapter_name="dpo_v5_pareto")
    model.set_adapter("dpo_v5_pareto")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    beta, grad_accum, epochs = 0.15, 8, 2
    model.train()
    epoch_losses: List[float] = []
    for ep in range(1, epochs + 1):
        random.shuffle(train_pairs)
        ep_loss = 0.0
        optimizer.zero_grad()
        for idx, ex in enumerate(train_pairs, start=1):
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
            if idx % grad_accum == 0 or idx == len(train_pairs):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        avg_ep = round(ep_loss / len(train_pairs), 4)
        epoch_losses.append(avg_ep)
        logger.info("DPO-v5-Pareto Epoch %d avg loss: %.4f", ep, avg_ep)

    ckpt_dir = Path("checkpoints/eduterm_14b_dpo_v5_pareto")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    model.load_adapter(resolve_dpo_v2_path(), adapter_name="dpo_v2")
    model.eval()
    t_proc = CJKBlockLogitsProcessor(build_nonlatin_token_ids(tok))

    cheat_eval = {
        "dpo_v5_pareto": eval_cheat(model, tok, t_proc, "dpo_v5_pareto", SYS_STD),
        "dpo_v5_pareto_sysguard": eval_cheat(model, tok, t_proc, "dpo_v5_pareto", SYS_GUARD),
    }

    model.set_adapter("dpo_v5_pareto")
    align_tax_v5 = eval_alignment_tax_v3(model, tok, t_proc)

    # Live tutoring with strong Qwen2.5-7B-Instruct (bfloat16) student
    s_path = "/users/ali.gunder/models/Qwen2.5-7B-Instruct"
    s_tok = AutoTokenizer.from_pretrained(s_path)
    s_model = AutoModelForCausalLM.from_pretrained(s_path, torch_dtype=torch.bfloat16, device_map="auto")
    s_model.eval()
    s_proc = CJKBlockLogitsProcessor(build_nonlatin_token_ids(s_tok))

    sessions: List[Dict[str, Any]] = []
    for cond, adp in [("onpolicy_dpo_v2", "dpo_v2"), ("dpo_v5_pareto", "dpo_v5_pareto")]:
        model.set_adapter(adp)
        for c in ISG_CONCEPTS:
            for p in PERSONAS:
                prereq_str = ", ".join(c["prerequisites"]) if c["prerequisites"] else "Yok (Temel Kavram)"
                t_sys = (
                    "Sen Sokratik ve destekleyici bir Türkçe İSG öğretmenisin.\n"
                    f"Kavram: {c['name']}\nÖn koşullar: {prereq_str}\nHedef ilke: {c['gold_principle']}\n"
                    "Öğrencinin kavram yanılgısını tespit et, doğrudan cevabı vermek yerine ön koşula dayalı yönlendirici sorular sor."
                )
                s_sys = (
                    f"Sen bir İSG öğrencisisin ({p['instruction']}).\n"
                    f"Başlangıçtaki yanlış düşüncen: {c['misconception']}\n"
                    "Kısa ve doğal Türkçe ile yanıt ver. Öğretmen iyi sorularla seni ikna ederse hatanı kabul edip doğru ilkeyi kendi cümlelerinle özetle."
                )
                t_msgs = [{"role": "system", "content": t_sys}]
                s_msgs = [
                    {"role": "system", "content": s_sys},
                    {"role": "user", "content": f"Merhaba, bugün '{c['name']}' konusunu konuşacağız. Bu konuda ne düşünüyorsun?"},
                ]
                transcript = []
                nonlat, q_cnt = 0, 0
                for turn in range(1, 5):
                    sr = generate_chat_nocjk(s_model, s_tok, s_msgs, cjk_processor=s_proc, max_new_tokens=120, temperature=0.7)
                    s_msgs.append({"role": "assistant", "content": sr})
                    t_msgs.append({"role": "user", "content": sr})
                    tr = generate_chat_nocjk(model, tok, t_msgs, cjk_processor=t_proc, max_new_tokens=200, temperature=0.4)
                    t_msgs.append({"role": "assistant", "content": tr})
                    s_msgs.append({"role": "user", "content": tr})
                    if has_nonlatin(tr):
                        nonlat += 1
                    if "?" in tr:
                        q_cnt += 1
                    transcript.append({"turn": turn, "student": sr, "tutor": tr})
                sessions.append({
                    "concept_id": c["id"],
                    "concept_name": c["name"],
                    "persona_id": p["id"],
                    "condition": cond,
                    "nonlatin_leakage_turns": nonlat,
                    "tutor_question_turns": q_cnt,
                    "transcript": transcript,
                })

    del model, base
    gc.collect()
    torch.cuda.empty_cache()

    parse_ok = 0
    for s in sessions:
        sc, ok = pointwise_judge(s_model, s_tok, s_proc, concept_by_id[s["concept_id"]], s)
        if ok:
            parse_ok += 1
        s["scores"] = sc

    tutoring_summary: Dict[str, Any] = {}
    by_cond: Dict[str, List[Dict[str, Any]]] = {}
    by_key: Dict[tuple, Dict[str, Dict[str, Any]]] = {}
    for s in sessions:
        by_cond.setdefault(s["condition"], []).append(s)
        by_key.setdefault((s["concept_id"], s["persona_id"]), {})[s["condition"]] = s

    for cond, items in by_cond.items():
        n = len(items)
        res = [x["scores"]["misconception_resolved"] for x in items]
        tutoring_summary[cond] = {
            "total_sessions": n,
            "misconception_resolution_rate": round(sum(res) / n, 4),
            "resolution_bootstrap_95ci": list(bootstrap_ci(res)),
            "socratic_mean": round(sum(x["scores"]["socratic_score"] for x in items) / n, 3),
            "prerequisite_grounding_mean": round(sum(x["scores"]["prerequisite_score"] for x in items) / n, 3),
            "turkish_purity_mean": round(sum(x["scores"]["turkish_purity"] for x in items) / n, 3),
            "nonlatin_leakage_turn_rate": round(sum(x["nonlatin_leakage_turns"] for x in items) / (n * 4), 4),
            "tutor_question_turn_rate": round(sum(x["tutor_question_turns"] for x in items) / (n * 4), 4),
        }

    v5_vals = [cmap["dpo_v5_pareto"]["scores"]["misconception_resolved"] for cmap in by_key.values()]
    v2_vals = [cmap["onpolicy_dpo_v2"]["scores"]["misconception_resolved"] for cmap in by_key.values()]

    report = {
        "metadata": {"epoch_losses": epoch_losses, "judge_parse_rate": round(parse_ok / len(sessions), 4), "elapsed_sec": round(time.time() - t0, 1)},
        "cheat_pressure_benchmark": cheat_eval,
        "alignment_tax_dpo_v5": align_tax_v5,
        "live_tutoring_7b_bf16_summary": tutoring_summary,
        "mcnemar_v5_vs_v2": exact_mcnemar_p(v5_vals, v2_vals),
        "sessions": sessions,
    }
    Path("results/hpc_guardrail_v5_pareto_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Exp 13 Pareto Summary:\n%s", json.dumps({
        "cheat_pressure_benchmark": {k: {kk: vv for kk, vv in v.items() if kk != "samples"} for k, v in cheat_eval.items()},
        "alignment_tax_dpo_v5": align_tax_v5,
        "live_tutoring_7b_bf16_summary": tutoring_summary,
        "mcnemar_v5_vs_v2": report["mcnemar_v5_vs_v2"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
