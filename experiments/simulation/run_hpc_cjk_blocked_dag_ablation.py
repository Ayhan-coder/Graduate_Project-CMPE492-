# -*- coding: utf-8 -*-
"""
CJK-Blocked Multi-Seed Paired DAG-RAG vs. Task-Separated Hybrid Ablation on Boğaziçi HPC (A40 48GB).

Addresses three methodological gaps identified in Jobs 34699, 34707, and 34721:
1. Systematic CJK Leakage Elimination: `CJKBlockLogitsProcessor` masks all CJK tokens at decoding
   time for both Student (Kumru-2B) and Tutor (Qwen2.5-14B-Instruct).
2. Causal Isolation (3 Paired Arms):
   - `no_dag_base_14b`: Base Qwen2.5-14B-Instruct without DAG prerequisite context.
   - `prompt_dag_rag_base_14b`: Base Qwen2.5-14B-Instruct with structured DAG prerequisite context
     and Socratic prompting (no fine-tuned adapter).
   - `task_separated_hybrid`: Specialized EduTerm-14B-CEFR adapter generates CEFR-B1 prerequisite
     card; Base Qwen2.5-14B-Instruct conducts multi-turn Socratic guidance.
3. Statistical Power & Cross-Judge Validation:
   - 3 seeds (42, 1337, 2026) x 12 concepts x 3 personas = 108 paired sessions per arm (324 total, 1,296 turns).
   - 10,000-resample paired bootstrap 95% CIs + exact McNemar's test on discordant pairs.
   - Blinded dual-judge cross-check (Qwen2.5-14B-Instruct vs. Qwen2.5-7B-Instruct) with Cohen's kappa.

Outputs:
  results/hpc_cjk_blocked_dag_ablation_report.json
"""

from __future__ import annotations

import gc
import json
import logging
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
)

from run_hpc_live_student_simulation import ISG_CONCEPTS, PERSONAS, has_cjk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CJKBlockedDAGAblation")

CJK_PATTERN = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
SEEDS = [42, 1337, 2026]


def build_cjk_token_ids(tokenizer) -> List[int]:
    """Find all vocabulary token IDs whose decoded text contains CJK characters."""
    blocked: List[int] = []
    vocab_size = len(tokenizer)
    for tid in range(vocab_size):
        tok_str = tokenizer.decode([tid], skip_special_tokens=False)
        if CJK_PATTERN.search(tok_str):
            blocked.append(tid)
    return blocked


class CJKBlockLogitsProcessor(LogitsProcessor):
    """Masks CJK token logits to -inf during generation."""

    def __init__(self, blocked_ids: List[int]) -> None:
        self.blocked_ids = blocked_ids
        self._cached_tensors: Dict[torch.device, torch.Tensor] = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.blocked_ids:
            return scores
        dev = scores.device
        idx = self._cached_tensors.get(dev)
        if idx is None:
            idx = torch.tensor(self.blocked_ids, dtype=torch.long, device=dev)
            self._cached_tensors[dev] = idx
        scores[:, idx] = -float("inf")
        return scores


def generate_chat_nocjk(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    cjk_processor: Optional[CJKBlockLogitsProcessor] = None,
    max_new_tokens: int = 160,
    temperature: float = 0.6,
) -> str:
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        merged = []
        sys_buf = ""
        for m in messages:
            if m["role"] == "system":
                sys_buf += m["content"] + "\n\n"
            elif m["role"] == "user" and sys_buf:
                merged.append({"role": "user", "content": sys_buf + m["content"]})
                sys_buf = ""
            else:
                merged.append(m)
        text = tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    inputs.pop("token_type_ids", None)
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "renormalize_logits": True,
    }
    if cjk_processor is not None:
        gen_kwargs["logits_processor"] = LogitsProcessorList([cjk_processor])
    if temperature > 0:
        gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.9})
    else:
        gen_kwargs.update({"do_sample": False})

    with torch.no_grad():
        try:
            out = model.generate(**inputs, **gen_kwargs)
        except RuntimeError:
            gen_kwargs["do_sample"] = False
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)
            out = model.generate(**inputs, **gen_kwargs)

    gen_ids = out[0][inputs["input_ids"].shape[1] :]
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    decoded = re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL).strip()
    return decoded


def generate_cefr_prereq_card(
    tutor_model,
    tutor_tok,
    tutor_cjk_proc: CJKBlockLogitsProcessor,
    concept: Dict[str, Any],
) -> str:
    """Use specialized EduTerm-14B-CEFR adapter to generate a CEFR-B1 prerequisite card."""
    prereqs = concept["prerequisites"]
    prereq_str = ", ".join(prereqs) if prereqs else concept["name"]
    msgs = [
        {
            "role": "system",
            "content": "Sen EduTerm CEFR-B1 eğitmenisin. Verilen önkoşul kavramı öğrenciye 2 kısa, sade Türkçe cümleyle özetle.",
        },
        {
            "role": "user",
            "content": (
                f"Hedef kavram: {concept['name']}\n"
                f"Önkoşul kavram(lar): {prereq_str}\n"
                "Bu önkoşul kavramı CEFR-B1 düzeyinde, günlük bir örnekle 2 cümlede açıkla."
            ),
        },
    ]
    return generate_chat_nocjk(
        tutor_model, tutor_tok, msgs, cjk_processor=tutor_cjk_proc, max_new_tokens=95, temperature=0.3
    )


def run_ablation_session(
    student_model,
    student_tok,
    student_cjk_proc: CJKBlockLogitsProcessor,
    tutor_model,
    tutor_tok,
    tutor_cjk_proc: CJKBlockLogitsProcessor,
    concept: Dict[str, Any],
    persona: Dict[str, Any],
    condition: str,
    seed: int,
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
    if condition == "task_separated_hybrid":
        cefr_card = generate_cefr_prereq_card(tutor_model, tutor_tok, tutor_cjk_proc, concept)
        tutor_sys = (
            f"Sen Sokratik bir İSG eğitmenisin. Hedef kavram: {concept['name']}.\n"
            f"DAG Önkoşulları: {prereq_str}.\n"
            f"EduTerm CEFR-B1 Önkoşul Özeti: {cefr_card}\n"
            "Kurallar: Cevabı ilk turda doğrudan verme. Yukarıdaki önkoşul özetinden yararlanarak "
            "öğrenciye kısa, yönlendirici bir soru sor ve kavram yanılgısını kendisinin keşfetmesini sağla. "
            "Yalnızca temiz Türkçe (2-4 cümle) yaz."
        )
    elif condition == "prompt_dag_rag_base_14b":
        tutor_sys = (
            f"Sen Sokratik bir İSG eğitmenisin. Hedef kavram: {concept['name']}.\n"
            f"DAG Önkoşulları: {prereq_str}.\n"
            "Kurallar: Cevabı ilk turda doğrudan verme. Önce yukarıdaki önkoşul kavramlarla bağ kur, "
            "günlük hayattan kısa bir örnek ve yönlendirici bir soru sorarak öğrencinin yanılgısını "
            "kendisinin keşfetmesini sağla. Yalnızca temiz Türkçe (2-4 cümle) yaz."
        )
    elif condition == "no_dag_base_14b":
        tutor_sys = (
            f"Sen bir İSG öğretmenisin. Konu: {concept['name']}. "
            "Öğrencinin sorusuna sadece Türkçe olarak kısa ve anlaşılır yanıt ver."
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")

    student_msgs = [{"role": "system", "content": student_sys}]
    tutor_msgs = [{"role": "system", "content": tutor_sys}]
    transcript: List[Dict[str, str]] = []

    student_msgs.append(
        {
            "role": "user",
            "content": f"Hocaya '{concept['name']}' hakkındaki yanlış düşünceni doğrudan 1-2 cümleyle sor.",
        }
    )

    for t in range(1, num_turns + 1):
        s_reply = generate_chat_nocjk(
            student_model,
            student_tok,
            student_msgs,
            cjk_processor=student_cjk_proc,
            max_new_tokens=100,
            temperature=0.7,
        )
        student_msgs.append({"role": "assistant", "content": s_reply})
        tutor_msgs.append({"role": "user", "content": s_reply})

        # All 3 conditions use Base 14B (adapter disabled) for multi-turn dialogue
        with tutor_model.disable_adapter():
            t_reply = generate_chat_nocjk(
                tutor_model,
                tutor_tok,
                tutor_msgs,
                cjk_processor=tutor_cjk_proc,
                max_new_tokens=140,
                temperature=0.4,
            )

        tutor_msgs.append({"role": "assistant", "content": t_reply})
        student_msgs.append(
            {"role": "user", "content": f"Hocan şöyle dedi: {t_reply}\nÖğrenci olarak 1-2 cümleyle yanıtla."}
        )
        transcript.append({"turn": t, "student": s_reply, "tutor": t_reply})

    return {
        "seed": seed,
        "concept_id": concept["id"],
        "concept_name": concept["name"],
        "persona_id": persona["id"],
        "condition": condition,
        "cefr_prereq_card": cefr_card,
        "cjk_leakage_turns": sum(1 for x in transcript if has_cjk(x["tutor"]) or has_cjk(x["student"])),
        "tutor_question_turns": sum(1 for x in transcript if "?" in x["tutor"]),
        "transcript": transcript,
    }


def judge_session_blinded(
    judge_model,
    judge_tok,
    judge_cjk_proc: Optional[CJKBlockLogitsProcessor],
    concept: Dict[str, Any],
    session: Dict[str, Any],
    disable_adapter: bool = False,
) -> Dict[str, Any]:
    conv_text = "\n".join(
        [f"[Tur {x['turn']}] Öğrenci: {x['student']}\n[Tur {x['turn']}] Eğitmen: {x['tutor']}" for x in session["transcript"]]
    )
    prompt = (
        "Aşağıdaki öğrenci-eğitmen diyaloğunu nesnel biçimde puanla ve SADECE geçerli bir JSON nesnesi döndür.\n"
        f"Kavram: {concept['name']}\n"
        f"Doğru İlke: {concept['gold_principle']}\n"
        f"Başlangıç Yanılgısı: {concept['misconception']}\n\n"
        f"Diyalog:\n{conv_text}\n\n"
        "Puanlama kuralları:\n"
        "- misconception_resolved (boolean): Öğrenci son turlarda yanlış inancını gerçekten bırakıp doğru ilkeyi kendi sözleriyle ifade ettiyse true, aksi halde false.\n"
        "- socratic_score (tam sayı 1-5): Eğitmen cevabı ilk turda doğrudan vermek yerine soru ve ipuçlarıyla düşündürdüyse yüksek puan (4-5), doğrudan cevabı verdiyse düşük puan (1-2).\n"
        "- prerequisite_score (tam sayı 1-5): Eğitmen önkoşul kavramlarla bağ kurduysa yüksek puan (4-5), hiç bağ kurmadıysa düşük puan (1-2).\n"
        "- turkish_purity (tam sayı 1-5): Diyalogda Çince/yabancı dil bozulması yoksa 5, hafif bozulma varsa 3, başka dile kayma varsa 1.\n"
        "Yanıt şeması: "
        '{"misconception_resolved": <true|false>, "socratic_score": <1-5>, "prerequisite_score": <1-5>, "turkish_purity": <1-5>}'
    )
    msgs = [
        {"role": "system", "content": "Sen titiz ve tarafsız bir eğitim bilimleri hakemisin. Sadece JSON çıktı ver."},
        {"role": "user", "content": prompt},
    ]
    if disable_adapter:
        with judge_model.disable_adapter():
            raw = generate_chat_nocjk(
                judge_model, judge_tok, msgs, cjk_processor=judge_cjk_proc, max_new_tokens=110, temperature=0.0
            )
    else:
        raw = generate_chat_nocjk(
            judge_model, judge_tok, msgs, cjk_processor=judge_cjk_proc, max_new_tokens=110, temperature=0.0
        )

    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    parsed = {"misconception_resolved": 0, "socratic_score": 3, "prerequisite_score": 3, "turkish_purity": 5}
    if m:
        try:
            obj = json.loads(m.group(0))
            parsed["misconception_resolved"] = int(bool(obj.get("misconception_resolved", False)))
            parsed["socratic_score"] = int(min(5, max(1, int(obj.get("socratic_score", 3)))))
            parsed["prerequisite_score"] = int(min(5, max(1, int(obj.get("prerequisite_score", 3)))))
            parsed["turkish_purity"] = int(min(5, max(1, int(obj.get("turkish_purity", 5)))))
        except Exception:
            pass
    if session.get("cjk_leakage_turns", 0) > 0:
        parsed["turkish_purity"] = min(parsed["turkish_purity"], 2)
    return parsed


def bootstrap_ci(values: List[float], n_boot: int = 10000, seed: int = 42) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot)], 4)


def exact_mcnemar_p(a_vals: List[int], b_vals: List[int]) -> Dict[str, Any]:
    b_only = sum(1 for a, b in zip(a_vals, b_vals) if a == 1 and b == 0)  # A=1, B=0
    c_only = sum(1 for a, b in zip(a_vals, b_vals) if a == 0 and b == 1)  # A=0, B=1
    n_disc = b_only + c_only
    if n_disc == 0:
        return {"a1_b0": 0, "a0_b1": 0, "p_value": 1.0}
    k = min(b_only, c_only)
    p_one_tail = sum(math.comb(n_disc, i) * (0.5 ** n_disc) for i in range(0, k + 1))
    p_two_tail = min(1.0, 2.0 * p_one_tail)
    return {"a1_b0": b_only, "a0_b1": c_only, "n_discordant": n_disc, "p_value": round(p_two_tail, 6)}


def cohens_kappa(y1: List[int], y2: List[int]) -> float:
    n = len(y1)
    if n == 0:
        return 0.0
    p_o = sum(1 for a, b in zip(y1, y2) if a == b) / n
    p1_1 = sum(y1) / n
    p2_1 = sum(y2) / n
    p_e = (p1_1 * p2_1) + ((1 - p1_1) * (1 - p2_1))
    if abs(1.0 - p_e) < 1e-12:
        return 1.0
    return round((p_o - p_e) / (1.0 - p_e), 4)


def main() -> None:
    t0 = time.time()
    student_path = "/users/ali.gunder/models/Kumru-2B"
    tutor_base_path = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
    cefr_adapter_path = "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora"
    judge_7b_path = "/users/ali.gunder/models/Qwen2.5-7B-Instruct"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading Student Agent (Kumru-2B bf16)...")
    s_tok = AutoTokenizer.from_pretrained(student_path)
    s_model = AutoModelForCausalLM.from_pretrained(student_path, torch_dtype=torch.bfloat16, device_map="auto")
    s_model.eval()
    s_cjk_ids = build_cjk_token_ids(s_tok)
    s_cjk_proc = CJKBlockLogitsProcessor(s_cjk_ids)
    logger.info("Student CJK blocked token count: %d", len(s_cjk_ids))

    logger.info("Loading Tutor/Primary-Judge (Qwen2.5-14B-Instruct 4-bit + eduterm_14b_cefr_qlora)...")
    t_tok = AutoTokenizer.from_pretrained(tutor_base_path)
    t_base = AutoModelForCausalLM.from_pretrained(tutor_base_path, quantization_config=bnb, device_map="auto")
    t_model = PeftModel.from_pretrained(t_base, cefr_adapter_path)
    t_model.eval()
    t_cjk_ids = build_cjk_token_ids(t_tok)
    t_cjk_proc = CJKBlockLogitsProcessor(t_cjk_ids)
    logger.info("Tutor CJK blocked token count: %d", len(t_cjk_ids))

    conditions = ("no_dag_base_14b", "prompt_dag_rag_base_14b", "task_separated_hybrid")
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / "hpc_cjk_blocked_dag_ablation_partial.json"

    sessions: List[Dict[str, Any]] = []
    done_keys = set()
    if partial_path.exists():
        try:
            prev = json.loads(partial_path.read_text(encoding="utf-8"))
            sessions = prev.get("sessions", [])
            done_keys = {(s["seed"], s["concept_id"], s["persona_id"], s["condition"]) for s in sessions}
            logger.info("Resuming from %s: %d sessions already completed.", partial_path, len(sessions))
        except Exception as exc:
            logger.warning("Could not load partial checkpoint %s (%s); starting fresh.", partial_path, exc)

    concept_by_id = {c["id"]: c for c in ISG_CONCEPTS}

    total_planned = len(SEEDS) * len(ISG_CONCEPTS) * len(PERSONAS) * len(conditions)
    logger.info("Starting %d paired sessions across %d seeds (%d remaining)...", total_planned, len(SEEDS), total_planned - len(done_keys))

    for seed in SEEDS:
        for c_idx, concept in enumerate(ISG_CONCEPTS):
            for p_idx, persona in enumerate(PERSONAS):
                pair_seed = seed + c_idx * 100 + p_idx * 10
                ran_any = False
                for cond in conditions:
                    key = (seed, concept["id"], persona["id"], cond)
                    if key in done_keys:
                        continue
                    ran_any = True
                    torch.manual_seed(pair_seed)
                    random.seed(pair_seed)
                    sess = run_ablation_session(
                        s_model,
                        s_tok,
                        s_cjk_proc,
                        t_model,
                        t_tok,
                        t_cjk_proc,
                        concept,
                        persona,
                        condition=cond,
                        seed=seed,
                        num_turns=4,
                    )
                    scores_14b = judge_session_blinded(
                        t_model, t_tok, t_cjk_proc, concept, sess, disable_adapter=True
                    )
                    sess["scores"] = scores_14b
                    sessions.append(sess)
                    done_keys.add(key)
                    logger.info(
                        "[%d/%d] seed=%d [%s | %s | %s] res=%d soc=%d pre=%d pur=%d cjk=%d",
                        len(sessions),
                        total_planned,
                        seed,
                        concept["id"],
                        persona["id"],
                        cond,
                        scores_14b["misconception_resolved"],
                        scores_14b["socratic_score"],
                        scores_14b["prerequisite_score"],
                        scores_14b["turkish_purity"],
                        sess["cjk_leakage_turns"],
                    )
                if ran_any and len(sessions) % 18 == 0:
                    partial_path.write_text(
                        json.dumps({"completed": len(sessions), "sessions": sessions}, ensure_ascii=False),
                        encoding="utf-8",
                    )

    # Free Student + Tutor 14B and load independent-scale Judge (Qwen2.5-7B-Instruct)
    logger.info("Unloading Student & 14B Tutor; loading Qwen2.5-7B-Instruct for cross-judge validation...")
    del s_model, t_model, t_base
    gc.collect()
    torch.cuda.empty_cache()

    j7_tok = AutoTokenizer.from_pretrained(judge_7b_path)
    j7_model = AutoModelForCausalLM.from_pretrained(judge_7b_path, quantization_config=bnb, device_map="auto")
    j7_model.eval()
    j7_cjk_proc = CJKBlockLogitsProcessor(build_cjk_token_ids(j7_tok))

    for idx, sess in enumerate(sessions, start=1):
        concept = concept_by_id[sess["concept_id"]]
        sess["scores_7b"] = judge_session_blinded(
            j7_model, j7_tok, j7_cjk_proc, concept, sess, disable_adapter=False
        )
        if idx % 54 == 0 or idx == len(sessions):
            logger.info("7B cross-judge progress: %d/%d", idx, len(sessions))

    # Build summary statistics
    summary: Dict[str, Any] = {}
    by_cond_map: Dict[str, Dict[Tuple[int, str, str], Dict[str, Any]]] = {c: {} for c in conditions}
    for s in sessions:
        by_cond_map[s["condition"]][(s["seed"], s["concept_id"], s["persona_id"])] = s

    for cond in conditions:
        subset = [s for s in sessions if s["condition"] == cond]
        n = len(subset)
        res_vals = [s["scores"]["misconception_resolved"] for s in subset]
        res_7b_vals = [s["scores_7b"]["misconception_resolved"] for s in subset]
        ci_low, ci_high = bootstrap_ci([float(v) for v in res_vals])
        by_persona = {}
        for p in PERSONAS:
            ps = [s for s in subset if s["persona_id"] == p["id"]]
            by_persona[p["id"]] = {
                "n": len(ps),
                "empirical_transition_rate": round(sum(x["scores"]["misconception_resolved"] for x in ps) / max(len(ps), 1), 4),
                "socratic_mean": round(sum(x["scores"]["socratic_score"] for x in ps) / max(len(ps), 1), 3),
                "prerequisite_mean": round(sum(x["scores"]["prerequisite_score"] for x in ps) / max(len(ps), 1), 3),
            }
        summary[cond] = {
            "total_sessions": n,
            "total_turns": n * 4,
            "misconception_resolution_rate": round(sum(res_vals) / max(n, 1), 4),
            "resolution_bootstrap_95ci": [ci_low, ci_high],
            "resolution_rate_7b_judge": round(sum(res_7b_vals) / max(n, 1), 4),
            "socratic_mean": round(sum(s["scores"]["socratic_score"] for s in subset) / max(n, 1), 3),
            "prerequisite_grounding_mean": round(sum(s["scores"]["prerequisite_score"] for s in subset) / max(n, 1), 3),
            "turkish_purity_mean": round(sum(s["scores"]["turkish_purity"] for s in subset) / max(n, 1), 3),
            "cjk_leakage_turn_rate": round(sum(s["cjk_leakage_turns"] for s in subset) / max(n * 4, 1), 4),
            "tutor_question_turn_rate": round(sum(s["tutor_question_turns"] for s in subset) / max(n * 4, 1), 4),
            "by_persona": by_persona,
        }

    # Paired comparisons
    keys = sorted(by_cond_map[conditions[0]].keys())
    paired_tests: Dict[str, Any] = {}
    for c_a, c_b in [
        ("prompt_dag_rag_base_14b", "no_dag_base_14b"),
        ("task_separated_hybrid", "no_dag_base_14b"),
        ("task_separated_hybrid", "prompt_dag_rag_base_14b"),
    ]:
        vals_a = [by_cond_map[c_a][k]["scores"]["misconception_resolved"] for k in keys]
        vals_b = [by_cond_map[c_b][k]["scores"]["misconception_resolved"] for k in keys]
        diffs = [float(a - b) for a, b in zip(vals_a, vals_b)]
        d_low, d_high = bootstrap_ci(diffs)
        paired_tests[f"{c_a}__vs__{c_b}"] = {
            "mean_diff_pp": round(100.0 * sum(diffs) / len(diffs), 2),
            "diff_pp_95ci": [round(100.0 * d_low, 2), round(100.0 * d_high, 2)],
            "mcnemar": exact_mcnemar_p(vals_a, vals_b),
        }

    # Inter-judge agreement (14B vs 7B)
    y14 = [s["scores"]["misconception_resolved"] for s in sessions]
    y7 = [s["scores_7b"]["misconception_resolved"] for s in sessions]
    inter_judge = {
        "n_sessions": len(sessions),
        "raw_agreement_misconception_resolved": round(sum(1 for a, b in zip(y14, y7) if a == b) / len(sessions), 4),
        "cohens_kappa_misconception_resolved": cohens_kappa(y14, y7),
    }

    out_path = out_dir / "hpc_cjk_blocked_dag_ablation_report.json"
    out_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "student_model": "Kumru-2B (bf16, CJK-blocked)",
                    "tutor_base_model": "Qwen2.5-14B-Instruct (4-bit NF4, CJK-blocked)",
                    "primary_judge": "Qwen2.5-14B-Instruct (blinded, disable_adapter)",
                    "secondary_judge": "Qwen2.5-7B-Instruct (blinded, 4-bit NF4)",
                    "seeds": SEEDS,
                    "num_concepts": len(ISG_CONCEPTS),
                    "num_personas": len(PERSONAS),
                    "turns_per_session": 4,
                    "cjk_blocked_vocab_student": len(s_cjk_ids),
                    "cjk_blocked_vocab_tutor": len(t_cjk_ids),
                    "elapsed_sec": round(time.time() - t0, 1),
                },
                "summary": summary,
                "paired_comparisons": paired_tests,
                "inter_judge_agreement": inter_judge,
                "sessions": sessions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Finished in %.1fs -> %s", time.time() - t0, out_path)
    logger.info("Summary: %s", json.dumps(summary, ensure_ascii=False))
    logger.info("Paired Tests: %s", json.dumps(paired_tests, ensure_ascii=False))
    logger.info("Inter-Judge: %s", json.dumps(inter_judge, ensure_ascii=False))


if __name__ == "__main__":
    main()
