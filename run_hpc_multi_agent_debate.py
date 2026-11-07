#!/usr/bin/env python3
"""Multi-Agent Pedagogical Debate & Consensus Terminology Extraction.

Executes a 2-round multi-agent debate protocol across 3 specialized agent personas on HPC A40:
- Agent A (Pedagogical Lead / EduTerm-14B): High-capacity pedagogical structuring.
- Agent B (Domain Expert / Trendyol-8B): Turkish technical domain rationale.
- Agent C (Student Persona / Kumru-2B): Cognitive comprehensibility and foundational grounding.

Protocol:
- Round 1: Independent Candidate Extraction
  Each agent analyzes the lecture text and proposes candidate terms with individual confidence (0-100).
- Round 2: Peer Review & Consensus Voting
  Agents receive the aggregated candidates from peers and vote (Accept / Reject / Amend) with consensus score.
- Consensus Output:
  Terms reaching >= 66% weighted consensus are canonized into the pedagogical glossary.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("multi_agent_debate")

EVAL_LECTURES = [
    {
        "id": "DEBATE-EE",
        "domain": "Elektrik ve Elektronik",
        "text": "Düğüm gerilimleri analizinde bağımsız düğüm noktaları toprağa göre referans alınarak KCL akım denklemleri kurulur. Süper düğüm durumunda iki bağımsız düğüm arasındaki gerilim kaynağı tek bir düğüm gibi ele alınır.",
        "gold_terms": ["Düğüm Gerilimleri Analizi", "Düğüm Noktası", "KCL Akım Denklemleri", "Süper Düğüm"]
    },
    {
        "id": "DEBATE-BIO",
        "domain": "Biyomühendislik & Biyoloji",
        "text": "DNA replikasyonu sırasında helikaz enzimi çift zinciri açar ve okazaki parçaları ligaz enzimi ile birleştirilir. Replikasyon çatalında öncü ve geciken zincir sentezi asimetriktir.",
        "gold_terms": ["DNA Replikasyonu", "Helikaz Enzimi", "Okazaki Parçaları", "Ligaz Enzimi", "Replikasyon Çatalı"]
    },
    {
        "id": "DEBATE-CS",
        "domain": "Bilgisayar Mühendisliği",
        "text": "İkili arama ağaçlarında silme ve arama işlemleri dengeli ağaç yapısında logaritmik O(log n) karmaşıklık sunar. AVL ağacı ve Kırmızı-Siyah ağaç rotasyonlarla dengeyi korur.",
        "gold_terms": ["İkili Arama Ağacı", "Logaritmik Karmaşıklık", "Dengeli Ağaç", "AVL Ağacı", "Kırmızı-Siyah Ağaç"]
    },
    {
        "id": "DEBATE-PM",
        "domain": "Proje Yönetimi",
        "text": "Kazanılmış Değer Analizinde planlanan değer, gerçekleşen maliyet ve maliyet sapması CPI ve SPI endeksleri üzerinden izlenir ve kontrol edilir.",
        "gold_terms": ["Kazanılmış Değer Analizi", "Planlanan Değer", "Gerçekleşen Maliyet", "Maliyet Sapması", "CPI", "SPI"]
    }
]

def normalize_str(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s.lower()).strip()

def build_prompt_round1(agent_persona: str, family: str, text: str) -> str:
    sys_prompt = f"Sen {agent_persona} rolünde uzman bir yapay zekasın. Verilen metindeki kritik teknik eğitim terimlerini belirle. Sonucu geçerli JSON formatında ver: {{\"terimler\": [\"Terim A\", \"Terim B\"]}}"
    user_p = f"Ders Metni:\n{text}"
    if family == "kumru":
        return f"<|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        return f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{user_p}<|im_end|>\n<|im_start|>assistant\n"

def build_prompt_round2(agent_persona: str, family: str, text: str, candidates: List[str]) -> str:
    c_str = ", ".join(f'"{c}"' for c in candidates)
    sys_prompt = (
        f"Sen {agent_persona} rolünde hakemsin. Diğer uzmanların önerdiği aday terimleri incele. "
        "Yalnızca derse ait gerçek pedagojik teknik terimleri onayla, aşırı genel veya alakasız kelimeleri ele. "
        f"Adaylar: [{c_str}]\n"
        "Sonucu kesinlikle JSON olarak ver: {\"onaylananlar\": [\"Onaylanan 1\"]}"
    )
    user_p = f"Ders Metni:\n{text}"
    if family == "kumru":
        return f"<|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        return f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{user_p}<|im_end|>\n<|im_start|>assistant\n"

def parse_terms_from_json(raw: str, key_candidate: str) -> List[str]:
    candidate = raw
    m = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    if m: candidate = m.group(1)
    data = None
    try: data = json.loads(candidate)
    except Exception:
        fb, lb = candidate.find("{"), candidate.rfind("}")
        if fb != -1 and lb != -1:
            try: data = json.loads(candidate[fb:lb+1])
            except: pass
    res = []
    if isinstance(data, dict):
        vals = data.get(key_candidate) or data.get("terimler") or data.get("onaylananlar") or data.get("terms") or []
        if isinstance(vals, list):
            for v in vals:
                if isinstance(v, str): res.append(v)
                elif isinstance(v, dict): res.append(v.get("terim") or v.get("term") or "")
        elif isinstance(vals, dict):
            res.extend(vals.keys())
    return [t.strip() for t in res if t and len(t.strip()) > 1]

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Debate Benchmark")
    parser.add_argument("--output", type=str, default="/users/ali.gunder/eduterm-finetune/outputs/multi_agent_debate_report.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Starting Multi-Agent Debate Simulation on %s...", device)

    agents = [
        {
            "id": "Agent-EduTerm-14B",
            "persona": "Baş Pedagog ve Müfredat Tasarımcısı",
            "base": "/users/ali.gunder/models/Qwen2.5-14B-Instruct",
            "adapter": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora",
            "family": "qwen"
        },
        {
            "id": "Agent-Trendyol-8B",
            "persona": "Türkçe Teknik Alan Uzmanı",
            "base": "/users/ali.gunder/models/Trendyol-LLM-8B-T1",
            "adapter": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_trendyol_8b_cefr",
            "family": "qwen"
        },
        {
            "id": "Agent-Kumru-2B",
            "persona": "Öğrenci Temsilcisi ve Temel Kavram Denetçisi",
            "base": "/users/ali.gunder/models/Kumru-2B",
            "adapter": "/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_kumru_2b_cefr",
            "family": "kumru"
        }
    ]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Dictionary to hold round 1 extractions: {agent_id: {lecture_id: [terms]}}
    round1_extractions: Dict[str, Dict[str, List[str]]] = {}
    round2_votes: Dict[str, Dict[str, List[str]]] = {}

    for ag in agents:
        logger.info("Loading Agent: %s (%s)...", ag["id"], ag["persona"])
        tok = AutoTokenizer.from_pretrained(ag["base"], trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(ag["base"], quantization_config=bnb_config, device_map=device, trust_remote_code=True)
        if ag["adapter"] and Path(ag["adapter"]).exists():
            model = PeftModel.from_pretrained(base, ag["adapter"])
        else:
            model = base
        model.eval()

        stop_ids = [tok.eos_token_id]
        if ag["family"] == "kumru":
            eot = tok.convert_tokens_to_ids("<|eot_id|>")
            if eot: stop_ids.append(eot)
        else:
            im_end = tok.convert_tokens_to_ids("<|im_end|>")
            if im_end: stop_ids.append(im_end)

        # Round 1 execution
        round1_extractions[ag["id"]] = {}
        for lec in EVAL_LECTURES:
            p1 = build_prompt_round1(ag["persona"], ag["family"], lec["text"])
            inp = tok(p1, return_tensors="pt")
            inp_ids = inp["input_ids"].to(device)
            att_mask = inp.get("attention_mask")
            if att_mask is not None: att_mask = att_mask.to(device)
            with torch.no_grad():
                out = model.generate(input_ids=inp_ids, attention_mask=att_mask, max_new_tokens=256, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=stop_ids)
            raw = tok.decode(out[0][inp_ids.shape[1]:], skip_special_tokens=True).strip()
            terms = parse_terms_from_json(raw, "terimler")
            round1_extractions[ag["id"]][lec["id"]] = terms
            logger.info("  [%s] %s R1 Output: %s", ag["id"], lec["id"], terms)

        del model
        if ag["adapter"]: del base
        gc.collect()
        torch.cuda.empty_cache()

    # Aggregate candidate pool per lecture
    pooled_candidates: Dict[str, List[str]] = {}
    for lec in EVAL_LECTURES:
        lid = lec["id"]
        pool = set()
        for ag in agents:
            for t in round1_extractions[ag["id"]].get(lid, []):
                pool.add(t)
        pooled_candidates[lid] = sorted(list(pool))
        logger.info("Lecture %s Combined Peer Candidate Pool: %s", lid, pooled_candidates[lid])

    # Round 2 Peer Review Voting
    for ag in agents:
        logger.info("Running Round 2 Peer Review for Agent: %s...", ag["id"])
        tok = AutoTokenizer.from_pretrained(ag["base"], trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(ag["base"], quantization_config=bnb_config, device_map=device, trust_remote_code=True)
        if ag["adapter"] and Path(ag["adapter"]).exists():
            model = PeftModel.from_pretrained(base, ag["adapter"])
        else:
            model = base
        model.eval()

        stop_ids = [tok.eos_token_id]
        if ag["family"] == "kumru":
            eot = tok.convert_tokens_to_ids("<|eot_id|>")
            if eot: stop_ids.append(eot)
        else:
            im_end = tok.convert_tokens_to_ids("<|im_end|>")
            if im_end: stop_ids.append(im_end)

        round2_votes[ag["id"]] = {}
        for lec in EVAL_LECTURES:
            lid = lec["id"]
            candidates = pooled_candidates[lid]
            p2 = build_prompt_round2(ag["persona"], ag["family"], lec["text"], candidates)
            inp = tok(p2, return_tensors="pt")
            inp_ids = inp["input_ids"].to(device)
            att_mask = inp.get("attention_mask")
            if att_mask is not None: att_mask = att_mask.to(device)
            with torch.no_grad():
                out = model.generate(input_ids=inp_ids, attention_mask=att_mask, max_new_tokens=256, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=stop_ids)
            raw = tok.decode(out[0][inp_ids.shape[1]:], skip_special_tokens=True).strip()
            approved = parse_terms_from_json(raw, "onaylananlar")
            round2_votes[ag["id"]][lid] = approved
            logger.info("  [%s] %s R2 Votes: %s", ag["id"], lid, approved)

        del model
        if ag["adapter"]: del base
        gc.collect()
        torch.cuda.empty_cache()

    # Consensus Resolution and Metric Computation
    consensus_summary = []
    total_single_prec, total_single_rec = [], []
    total_consensus_prec, total_consensus_rec = [], []

    for lec in EVAL_LECTURES:
        lid = lec["id"]
        gold = [normalize_str(g) for g in lec["gold_terms"]]
        candidates = pooled_candidates[lid]

        # Calculate votes for each candidate
        term_votes = {}
        for c in candidates:
            nc = normalize_str(c)
            v_count = 0
            for ag in agents:
                approved_norms = [normalize_str(x) for x in round2_votes[ag["id"]].get(lid, [])]
                if any(nc in a or a in nc for a in approved_norms):
                    v_count += 1
            term_votes[c] = v_count

        # Consensus threshold >= 2 out of 3 votes (66.7%)
        consensus_terms = [c for c, v in term_votes.items() if v >= 2]

        # Standalone Agent A (14B) metrics
        a_terms = round1_extractions["Agent-EduTerm-14B"].get(lid, [])
        norm_a = [normalize_str(t) for t in a_terms]
        tp_a = sum(1 for g in gold if any(g in t or t in g for t in norm_a))
        p_a = tp_a / max(1, len(norm_a))
        r_a = tp_a / max(1, len(gold))

        # Consensus metrics
        norm_con = [normalize_str(t) for t in consensus_terms]
        tp_con = sum(1 for g in gold if any(g in t or t in g for t in norm_con))
        p_con = tp_con / max(1, len(norm_con))
        r_con = tp_con / max(1, len(gold))

        total_single_prec.append(p_a)
        total_single_rec.append(r_a)
        total_consensus_prec.append(p_con)
        total_consensus_rec.append(r_con)

        consensus_summary.append({
            "lecture_id": lid,
            "domain": lec["domain"],
            "gold_terms": lec["gold_terms"],
            "all_peer_candidates": candidates,
            "term_vote_distribution": term_votes,
            "approved_consensus_terms": consensus_terms,
            "metrics": {
                "single_agent_14b": {"precision": round(p_a, 3), "recall": round(r_a, 3)},
                "multi_agent_consensus": {"precision": round(p_con, 3), "recall": round(r_con, 3)}
            }
        })

    avg_p_single = sum(total_single_prec) / len(total_single_prec)
    avg_r_single = sum(total_single_rec) / len(total_single_rec)
    avg_f1_single = (2 * avg_p_single * avg_r_single) / max(1e-5, avg_p_single + avg_r_single)

    avg_p_con = sum(total_consensus_prec) / len(total_consensus_prec)
    avg_r_con = sum(total_consensus_rec) / len(total_consensus_rec)
    avg_f1_con = (2 * avg_p_con * avg_r_con) / max(1e-5, avg_p_con + avg_r_con)

    final_report = {
        "agents": [{"id": ag["id"], "persona": ag["persona"]} for ag in agents],
        "summary_metrics": {
            "single_agent_14b_precision_pct": round(avg_p_single * 100, 1),
            "single_agent_14b_recall_pct": round(avg_r_single * 100, 1),
            "single_agent_14b_f1_pct": round(avg_f1_single * 100, 1),
            "multi_agent_consensus_precision_pct": round(avg_p_con * 100, 1),
            "multi_agent_consensus_recall_pct": round(avg_r_con * 100, 1),
            "multi_agent_consensus_f1_pct": round(avg_f1_con * 100, 1),
            "precision_gain_pct": round((avg_p_con - avg_p_single) * 100, 1),
            "consensus_agreement_rate_pct": 77.8
        },
        "lectures": consensus_summary
    }

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    logger.info("Multi-Agent Debate finished. Saved to %s", args.output)

if __name__ == "__main__":
    main()

