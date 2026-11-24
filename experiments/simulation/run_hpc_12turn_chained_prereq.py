# -*- coding: utf-8 -*-
"""
Experiment 39: 12-Turn Extended Chained-Prerequisite Live Tutoring Drift & Memory Stress Test
on Boğaziçi HPC (A40 48GB).

Evaluates 4 14B variants (`base_14b`, `onpolicy_dpo_v2`, `dpo_v5_pareto`, `dpo_v6_multidomain`)
over 4 chained 3-concept prerequisite trajectories x 2 personas = 8 extended 12-turn sessions
per model (32 sessions = 384 tutor turns total, using Kumru-2B student + utf8_exact_allowlist).

Measures Block 1 (Turns 1-4) vs Block 2 (Turns 5-8) vs Block 3 (Turns 9-12):
  - Socratic question-turn drift across long 12-turn context
  - 4-gram self-repetition rate across tutor turns
  - Cumulative 3-concept independent post-test mastery (0/3 to 3/3)
  - Non-Latin leakage rate

Outputs:
  results/hpc_12turn_chained_prereq_report.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_hpc_alignment_tax_cefr import resolve_dpo_v2_path
from run_hpc_cjk_blocked_dag_ablation import CJKBlockLogitsProcessor, bootstrap_ci
from run_hpc_dpo_3seed_ablation import PERSONAS
from run_hpc_dpo_r2_nonlatin import NON_LATIN_PATTERN, has_nonlatin
from run_hpc_multiturn_guardrail import resolve_adapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp39Long12Turn")

PREREQ_CHAINS = [
    {
        "id": "CH01_ISG_Core",
        "concepts": [
            {"name": "Tehlike", "misc": "Tehlike sadece kaza gerçekleştiğinde ortaya çıkar.", "gold": "Tehlike, kaza olmadan önce de zarar verme potansiyeli taşıyan kaynak veya durumdur."},
            {"name": "Risk", "misc": "Tehlike ile risk tamamen aynı kavramdır.", "gold": "Risk, bir tehlikeden kaynaklanan zararın gerçekleşme olasılığı ile şiddetinin bileşimidir."},
            {"name": "Kontrol Hiyerarşisi (KKD)", "misc": "En etkili ilk önlem herkese hemen KKD dağıtmaktır.", "gold": "Bertaraf ve mühendislik kontrolleri önceliklidir; KKD kontrol hiyerarşisinde son çaredir."},
        ],
    },
    {
        "id": "CH02_ISG_Field",
        "concepts": [
            {"name": "Ramak Kala Olay", "misc": "Kimse yaralanmadıysa ramak kala olayı raporlamaya gerek yoktur.", "gold": "Ramak kala olaylar gelecekteki ağır kazaların erken uyarısıdır ve mutlaka raporlanır."},
            {"name": "Kapalı Alan Atmosferi", "misc": "İçeride koku yoksa kapalı alana ölçüm yapmadan girilebilir.", "gold": "Kokusuz gazlar ve oksijen eksikliği ölümcül olabilir; giriş öncesi gaz ölçümü şarttır."},
            {"name": "EKED (Kilitleme-Etiketleme)", "misc": "Makineyi şalterden kapatmak bakım için yeterlidir, kilide gerek yoktur.", "gold": "Bakımda enerji kesildikten sonra kişisel kilit-etiket takılmalı ve sıfır enerji doğrulanmalıdır."},
        ],
    },
    {
        "id": "CH03_CS_Algo",
        "concepts": [
            {"name": "Dinamik Programlama", "misc": "Dinamik programlama ile böl-ve-yönet tamamen aynı yöntemdir.", "gold": "Dinamik programlama örtüşen alt problemleri bellekte saklar, böl-ve-yönet ayrık alt problemler çözer."},
            {"name": "Hash Çakışması", "misc": "İyi bir hash fonksiyonu iki farklı anahtarın çakışmasını imkansız kılar.", "gold": "Güvercin yuvası ilkesine göre sınırsız anahtarda hash çakışması kaçınılmazdır."},
            {"name": "Deadlock Koşulları", "misc": "İki thread aynı veriyi okursa deadlock oluşur.", "gold": "Deadlock için karşılıklı dışlama dahil dört Coffman koşulunun tamamı gerekir."},
        ],
    },
    {
        "id": "CH04_Data_Stats",
        "concepts": [
            {"name": "Korelasyon vs Nedensellik", "misc": "r=0.95 korelasyon kesinlikle sebep-sonuç kanıtlar.", "gold": "Yüksek korelasyon tek başına nedensellik kanıtlamaz; karıştırıcı değişken bulunabilir."},
            {"name": "Simpson Paradoksu", "misc": "Alt grupların hepsinde A iyiyse toplamda da A mutlaka iyidir.", "gold": "Farklı grup ağırlıkları ve karıştırıcı değişkenler toplam eğilimi tersine çevirebilir."},
            {"name": "Sınıf Dengesizliğinde Accuracy", "misc": "%99 negatif veride %99 accuracy alan model mükemmeldir.", "gold": "Sabit çoğunluk tahmini de %99 accuracy verir; dengesiz veride F1 veya PR-AUC gerekir."},
        ],
    },
]


def build_exact_allowlist_ids(tok) -> List[int]:
    return [
        tid for tid in range(len(tok))
        if "\ufffd" not in (s := tok.decode([tid], skip_special_tokens=False)) and NON_LATIN_PATTERN.search(s)
    ]


def gen_safe(model, tok, proc, msgs: List[Dict[str, str]], max_new_tokens: int = 135, temperature: float = 0.4) -> str:
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


def fourgram_repetition(turns: List[str]) -> float:
    seen: Set[tuple] = set()
    total_ng = repeated_ng = 0
    for t in turns:
        words = t.lower().split()
        ngs = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
        for ng in ngs:
            total_ng += 1
            if ng in seen:
                repeated_ng += 1
            else:
                seen.add(ng)
    return repeated_ng / max(total_ng, 1)


def main() -> None:
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    base_path = "/users/ali.gunder/models/Qwen2.5-14B-Instruct"
    t_tok = AutoTokenizer.from_pretrained(base_path)
    t_base = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb, device_map="auto")
    t_model = PeftModel.from_pretrained(t_base, resolve_dpo_v2_path(), adapter_name="dpo_v2")
    t_model.load_adapter(resolve_adapter("checkpoints/eduterm_14b_dpo_v5_pareto", "dpo_v5_pareto"), adapter_name="dpo_v5_pareto")

    variants: List[tuple] = [("base_14b", None), ("onpolicy_dpo_v2", "dpo_v2"), ("dpo_v5_pareto", "dpo_v5_pareto")]
    v6_root = Path("checkpoints/eduterm_14b_dpo_v6_multidomain")
    if v6_root.exists():
        t_model.load_adapter(resolve_adapter(str(v6_root), "dpo_v6_multidomain"), adapter_name="dpo_v6_multidomain")
        variants.append(("dpo_v6_multidomain", "dpo_v6_multidomain"))
    t_model.eval()
    t_proc = CJKBlockLogitsProcessor(build_exact_allowlist_ids(t_tok))

    s_path = "/users/ali.gunder/models/Kumru-2B"
    s_tok = AutoTokenizer.from_pretrained(s_path)
    s_model = AutoModelForCausalLM.from_pretrained(s_path, torch_dtype=torch.bfloat16, device_map="auto")
    s_model.eval()
    s_proc = CJKBlockLogitsProcessor(build_exact_allowlist_ids(s_tok))

    eval_personas = PERSONAS[:2]
    summary: Dict[str, Any] = {}

    for v_name, adp in variants:
        b_q = [0, 0, 0]
        b_tot = [0, 0, 0]
        reps: List[float] = []
        mastery_scores: List[float] = []
        per_block_mastery = [[], [], []]
        nl = tot_turns = 0

        for ch in PREREQ_CHAINS:
            c_list = ch["concepts"]
            chain_desc = " -> ".join(c["name"] for c in c_list)
            for p in eval_personas:
                t_sys = (
                    f"Sen uzun oturumlu Sokratik bir Türkçe eğitmensin. Önkoşul zinciri: {chain_desc}.\n"
                    "Her kavramı sırayla doğrudan cevap vermeden sorularla kavrat, kendini tekrar etme ve yeni kavrama geçince öncekine bağla."
                )
                s_sys = (
                    f"Sen gerçek bir öğrencisin ({p['label']}). {p['instruction']}\n"
                    "Kısa Türkçe yanıtlar ver; öğretmen ikna edici sorularla açıklarsa doğru ilkeleri kabul et."
                )
                t_msgs = [{"role": "system", "content": t_sys}]
                s_msgs = [{"role": "system", "content": s_sys}]
                tutor_turns: List[str] = []

                for b_idx, c in enumerate(c_list):
                    intro = f"Şimdi {b_idx + 1}. aşamadayız — kavram: '{c['name']}'. Ben şunu düşünüyorum: {c['misc']}"
                    s_msgs.append({"role": "user", "content": f"Öğretmene şu başlangıç düşünceni söyle: '{c['misc']}'"})
                    for _sub in range(4):
                        sr = gen_safe(s_model, s_tok, s_proc, s_msgs, max_new_tokens=85, temperature=0.7) if _sub > 0 else intro
                        s_msgs.append({"role": "assistant", "content": sr})
                        t_msgs.append({"role": "user", "content": sr})
                        if adp is None:
                            with t_model.disable_adapter():
                                tr = gen_safe(t_model, t_tok, t_proc, t_msgs, max_new_tokens=135, temperature=0.4)
                        else:
                            t_model.set_adapter(adp)
                            tr = gen_safe(t_model, t_tok, t_proc, t_msgs, max_new_tokens=135, temperature=0.4)
                        t_msgs.append({"role": "assistant", "content": tr})
                        s_msgs.append({"role": "user", "content": tr})
                        tutor_turns.append(tr)
                        tot_turns += 1
                        b_tot[b_idx] += 1
                        if "?" in tr:
                            b_q[b_idx] += 1
                        if has_nonlatin(tr):
                            nl += 1

                reps.append(fourgram_repetition(tutor_turns))
                # Cumulative 3-concept post-test at end of 12-turn session
                got = 0
                for b_idx, c in enumerate(c_list):
                    pt_msgs = s_msgs + [{"role": "user", "content": f"Öğretmensiz cevapla: '{c['gold']}' ifadesi DOĞRU mu YANLIŞ mı? Sadece 'DOĞRU' veya 'YANLIŞ' yaz."}]
                    pt_raw = gen_safe(s_model, s_tok, s_proc, pt_msgs, max_new_tokens=15, temperature=0.0)
                    ok = int("DOĞRU" in pt_raw.upper() and "YANLIŞ" not in pt_raw.upper())
                    got += ok
                    per_block_mastery[b_idx].append(ok)
                mastery_scores.append(got / 3.0)

        summary[v_name] = {
            "n_12turn_sessions": len(mastery_scores),
            "block1_t1_4_q_rate": round(b_q[0] / b_tot[0], 4),
            "block2_t5_8_q_rate": round(b_q[1] / b_tot[1], 4),
            "block3_t9_12_q_rate": round(b_q[2] / b_tot[2], 4),
            "q_rate_drift_b3_minus_b1": round((b_q[2] / b_tot[2]) - (b_q[0] / b_tot[0]), 4),
            "mean_4gram_repetition": round(sum(reps) / len(reps), 4),
            "cumulative_3concept_mastery": round(sum(mastery_scores) / len(mastery_scores), 4),
            "block1_mastery": round(sum(per_block_mastery[0]) / len(per_block_mastery[0]), 4),
            "block2_mastery": round(sum(per_block_mastery[1]) / len(per_block_mastery[1]), 4),
            "block3_mastery": round(sum(per_block_mastery[2]) / len(per_block_mastery[2]), 4),
            "nonlatin_leakage_rate": round(nl / max(tot_turns, 1), 4),
        }
        logger.info("Exp39 12-Turn [%s]: %s", v_name, summary[v_name])

    report = {"metadata": {"chains": len(PREREQ_CHAINS), "elapsed_sec": round(time.time() - t0, 1)}, "summary_by_variant": summary}
    Path("results/hpc_12turn_chained_prereq_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
