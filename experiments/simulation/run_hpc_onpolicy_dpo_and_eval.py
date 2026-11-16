# -*- coding: utf-8 -*-
"""
Option 3: On-Policy Multi-Turn DPO Training from Job 34740 Transcripts + Live Evaluation.

1. Extracts paired (prompt, chosen_tutor_reply, rejected_tutor_reply) preference pairs from
   results/hpc_cjk_blocked_dag_ablation_report.json (same seed/concept/persona where one arm
   resolved the misconception and another failed).
2. Trains Qwen2.5-14B-Instruct (4-bit NF4 QLoRA r=16, alpha=32) with exact DPO loss (beta=0.1)
   using `with model.disable_adapter():` for reference log-probs (zero extra VRAM).
3. Saves adapter -> checkpoints/eduterm_14b_onpolicy_dpo.
4. Evaluates `onpolicy_dpo` vs `task_separated_hybrid` (both CJK-blocked) on Kumru-2B
   across 12 ISG concepts x 3 personas (36 paired sessions per arm = 72 sessions, 288 turns).

Outputs:
  results/hpc_onpolicy_dpo_report.json
"""

from __future__ import annotations

import gc
import json
import logging
import random
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
    build_cjk_token_ids,
    exact_mcnemar_p,
    generate_cefr_prereq_card,
    generate_chat_nocjk,
    judge_session_blinded,
)
from run_hpc_live_student_simulation import ISG_CONCEPTS, PERSONAS, has_cjk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Opt3OnPolicyDPO")


def build_dpo_pairs(report_path: Path, concept_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    sessions = data["sessions"]

    grouped: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = {}
    for s in sessions:
        grouped.setdefault((s["seed"], s["concept_id"], s["persona_id"]), []).append(s)

    pairs: List[Dict[str, Any]] = []
    for (_, cid, _), group in grouped.items():
        concept = concept_by_id[cid]
        prereq_str = ", ".join(concept["prerequisites"]) if concept["prerequisites"] else "Yok (Temel Kavram)"
        winners = [s for s in group if s["scores"]["misconception_resolved"] == 1 and s["scores"]["socratic_score"] >= 4]
        losers = [s for s in group if s["scores"]["misconception_resolved"] == 0]
        if not winners or not losers:
            continue
        w = max(winners, key=lambda x: (x["scores"]["socratic_score"], x["scores"]["prerequisite_score"]))
        l = min(losers, key=lambda x: (x["scores"]["socratic_score"], x["scores"]["prerequisite_score"]))

        sys_msg = (
            f"Sen EduTerm Sokratik İSG Eğitmenisin. Hedef kavram: {concept['name']}.\n"
            f"DAG Önkoşulları: {prereq_str}.\n"
            "Kurallar: Cevabı ilk turda doğrudan verme; kısa, yönlendirici Sokratik soru sor ve temiz Türkçe yaz."
        )
        # Turn 1 uses identical/near-identical opening student turn for the same (seed, concept, persona)
        t1_student = w["transcript"][0]["student"]
        pairs.append({
            "messages_prefix": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": t1_student},
            ],
            "chosen": w["transcript"][0]["tutor"],
            "rejected": l["transcript"][0]["tutor"],
        })
        # Also include Turn 2 from winner vs loser conditioned on their own Turn 1 context
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


def encode_pair(tokenizer, prefix: List[Dict[str, str]], response: str, max_len: int = 768) -> Tuple[torch.Tensor, torch.Tensor]:
    prompt_text = tokenizer.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(prefix + [{"role": "assistant", "content": response}], tokenize=False, add_generation_prompt=False)
    p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    f_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
    labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids) :]
    if len(labels) < len(f_ids):
        labels = labels + [-100] * (len(f_ids) - len(labels))
    return torch.tensor([f_ids], dtype=torch.long), torch.tensor([labels], dtype=torch.long)


def sequence_logp(model, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids)
    logits = out.logits[:, :-1, :]
    target = labels[:, 1:]
    mask = target != -100
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    log_probs = F.log_softmax(logits.float(), dim=-1)
    tok_lp = torch.gather(log_probs, dim=-1, index=safe_target.unsqueeze(-1)).squeeze(-1)
    return (tok_lp * mask.float()).sum(dim=-1)


def run_eval_session(
    s_model,
    s_tok,
    s_proc: CJKBlockLogitsProcessor,
    t_model,
    t_tok,
    t_proc: CJKBlockLogitsProcessor,
    concept: Dict[str, Any],
    persona: Dict[str, Any],
    use_dpo: bool,
    num_turns: int = 4,
) -> Dict[str, Any]:
    prereq_str = ", ".join(concept["prerequisites"]) if concept["prerequisites"] else "Yok (Temel Kavram)"
    student_sys = (
        f"Sen gerçek bir öğrencisin ({persona['label']}). {persona['instruction']}\n"
        f"Konu: {concept['name']}.\n"
        f"Başlangıçtaki yanlış inancın: {concept['misconception']}\n"
        "ÖNEMLİ: Sadece kendi ağzından ('Ben...' diyerek) 1-2 cümlelik kısa Türkçe yanıt ver. "
        "Asla 'Öğretmen:' veya 'Öğrenci:' diye senaryo yazma, Çince/İngilizce kullanma."
    )

    cefr_card = ""
    if use_dpo:
        tutor_sys = (
            f"Sen EduTerm Sokratik İSG Eğitmenisin. Hedef kavram: {concept['name']}.\n"
            f"DAG Önkoşulları: {prereq_str}.\n"
            "Kurallar: Cevabı ilk turda doğrudan verme; kısa, yönlendirici Sokratik soru sor ve temiz Türkçe yaz."
        )
    else:
        t_model.set_adapter("cefr")
        cefr_card = generate_cefr_prereq_card(t_model, t_tok, t_proc, concept)
        tutor_sys = (
            f"Sen Sokratik bir İSG eğitmenisin. Hedef kavram: {concept['name']}.\n"
            f"DAG Önkoşulları: {prereq_str}.\n"
            f"EduTerm CEFR-B1 Önkoşul Özeti: {cefr_card}\n"
            "Kurallar: Cevabı ilk turda doğrudan verme. Yukarıdaki önkoşul özetinden yararlanarak "
            "öğrenciye kısa, yönlendirici bir soru sor. Yalnızca temiz Türkçe (2-4 cümle) yaz."
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

        if use_dpo:
            t_model.set_adapter("dpo")
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
        "condition": "onpolicy_dpo" if use_dpo else "task_separated_hybrid",
        "cefr_prereq_card": cefr_card,
        "cjk_leakage_turns": sum(1 for x in transcript if has_cjk(x["tutor"]) or has_cjk(x["student"])),
        "tutor_question_turns": sum(1 for x in transcript if "?" in x["tutor"]),
        "transcript": transcript,
    }


def main() -> None:
    t0 = time.time()
    random.seed(42)
    torch.manual_seed(42)
    concept_by_id = {c["id"]: c for c in ISG_CONCEPTS}

    pairs = build_dpo_pairs(Path("results/hpc_cjk_blocked_dag_ablation_report.json"), concept_by_id)
    random.shuffle(pairs)
    pairs = pairs[:240]  # Cap at 240 cleanest pairs for fast 2-epoch convergence
    logger.info("Built %d on-policy DPO preference pairs from Job 34740 transcripts.", len(pairs))

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_path = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
    tok = AutoTokenizer.from_pretrained(base_path)
    base = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb, device_map="auto")
    dpo_ckpt = Path("checkpoints/eduterm_14b_onpolicy_dpo")
    existing_dpo = None
    if (dpo_ckpt / "adapter_config.json").exists():
        existing_dpo = dpo_ckpt
    elif (dpo_ckpt / "dpo" / "adapter_config.json").exists():
        existing_dpo = dpo_ckpt / "dpo"

    epochs = 2
    epoch_losses: List[float] = []
    if existing_dpo is not None:
        logger.info("Found existing On-Policy DPO adapter at %s; skipping training.", existing_dpo)
        model = PeftModel.from_pretrained(base, str(existing_dpo), adapter_name="dpo")
    else:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(base, lora_cfg, adapter_name="dpo")
        model.set_adapter("dpo")
        model.print_trainable_parameters()

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        beta = 0.1
        grad_accum = 8

        model.train()
        step_count = 0
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
                logits_diff = (pol_c - ref_c) - (pol_r - ref_r)
                loss = -F.logsigmoid(beta * logits_diff).mean() / grad_accum
                loss.backward()
                ep_loss += loss.item() * grad_accum

                if idx % grad_accum == 0 or idx == len(pairs):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    step_count += 1
                    if step_count % 10 == 0:
                        logger.info("DPO Epoch [%d/%d] Step %d | Loss: %.4f", ep, epochs, step_count, loss.item() * grad_accum)
            avg_ep = round(ep_loss / max(len(pairs), 1), 4)
            epoch_losses.append(avg_ep)
            logger.info("DPO Epoch %d avg loss: %.4f", ep, avg_ep)

        dpo_ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(dpo_ckpt))
        logger.info("Saved On-Policy DPO adapter -> %s", dpo_ckpt)
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()

    model.load_adapter("/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora", adapter_name="cefr")
    model.eval()
    t_proc = CJKBlockLogitsProcessor(build_cjk_token_ids(tok))

    logger.info("Loading Student Agent (Kumru-2B bf16, CJK-blocked) for live DPO vs Hybrid evaluation...")
    s_tok = AutoTokenizer.from_pretrained("/users/ali.gunder/models/Kumru-2B")
    s_model = AutoModelForCausalLM.from_pretrained("/users/ali.gunder/models/Kumru-2B", torch_dtype=torch.bfloat16, device_map="auto")
    s_model.eval()
    s_proc = CJKBlockLogitsProcessor(build_cjk_token_ids(s_tok))

    sessions: List[Dict[str, Any]] = []
    for c_idx, concept in enumerate(ISG_CONCEPTS):
        for p_idx, persona in enumerate(PERSONAS):
            pair_seed = 42 + c_idx * 100 + p_idx * 10
            for use_dpo in (False, True):
                torch.manual_seed(pair_seed)
                random.seed(pair_seed)
                sess = run_eval_session(s_model, s_tok, s_proc, model, tok, t_proc, concept, persona, use_dpo=use_dpo, num_turns=4)
                scores = judge_session_blinded(model, tok, t_proc, concept, sess, disable_adapter=True)
                sess["scores"] = scores
                sessions.append(sess)
                logger.info(
                    "[%d/72] [%s | %s | %s] res=%d soc=%d pre=%d pur=%d cjk=%d",
                    len(sessions), concept["id"], persona["id"], sess["condition"],
                    scores["misconception_resolved"], scores["socratic_score"],
                    scores["prerequisite_score"], scores["turkish_purity"], sess["cjk_leakage_turns"],
                )

    summary = {}
    cmap = {"task_separated_hybrid": {}, "onpolicy_dpo": {}}
    for s in sessions:
        cmap[s["condition"]][(s["concept_id"], s["persona_id"])] = s

    for cond in ("task_separated_hybrid", "onpolicy_dpo"):
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
            "cjk_leakage_turn_rate": round(sum(s["cjk_leakage_turns"] for s in sub) / (n * 4), 4),
            "tutor_question_turn_rate": round(sum(s["tutor_question_turns"] for s in sub) / (n * 4), 4),
        }

    keys = sorted(cmap["task_separated_hybrid"].keys())
    va = [cmap["onpolicy_dpo"][k]["scores"]["misconception_resolved"] for k in keys]
    vb = [cmap["task_separated_hybrid"][k]["scores"]["misconception_resolved"] for k in keys]
    diffs = [float(a - b) for a, b in zip(va, vb)]
    d_lo, d_hi = bootstrap_ci(diffs)
    paired = {
        "onpolicy_dpo__vs__task_separated_hybrid": {
            "mean_diff_pp": round(100.0 * sum(diffs) / len(diffs), 2),
            "diff_pp_95ci": [round(100.0 * d_lo, 2), round(100.0 * d_hi, 2)],
            "mcnemar": exact_mcnemar_p(va, vb),
        }
    }

    out_path = Path("results/hpc_onpolicy_dpo_report.json")
    out_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "dpo_pairs": len(pairs),
                    "dpo_epochs": epochs,
                    "dpo_epoch_losses": epoch_losses,
                    "student_model": "Kumru-2B (bf16, CJK-blocked)",
                    "elapsed_sec": round(time.time() - t0, 1),
                },
                "summary": summary,
                "paired_comparison": paired,
                "sessions": sessions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Option 3 Finished in %.1fs -> %s", time.time() - t0, out_path)


if __name__ == "__main__":
    main()
