# -*- coding: utf-8 -*-
"""
Syncs real, verbatim, deterministic HPC outputs from:
  - experiments/outputs/hpc_human_eval_v2_stimuli.json (Job 36433)
  - experiments/outputs/hpc_guardrail_v4_ablation_report.json (Job 36389)
into:
  - mobile_eval_app/google_forms_v2/human_eval_manifest.json (full provenance + checkpoints)
  - mobile_eval_app/google_forms_v2/eduterm_human_eval_v2.gs (verbatim participant-facing stimuli, NO model IDs!)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "experiments" / "outputs"
V2_DIR = Path(__file__).resolve().parent

QUIZ_BANK = {
    "EXP_01": {
        "q": "Bir atölyede zemine yağ dökülmüş fakat gün boyunca kimse kayıp düşmemiştir. Bu durum nasıl değerlendirilir?",
        "opts": [
            "Zarar potansiyeli taşıdığı için tehlike mevcuttur",
            "Yaralanma olmadığı için tehlike oluşmamıştır",
            "Yalnızca kaza raporu tutulursa tehlike sayılır",
            "Emin değilim",
        ],
        "gold_idx": 0,
    },
    "EXP_02": {
        "q": "Bir işveren, çalışanın aylık maaşını yatırabilmek için IBAN bilgisini işlemektedir. KVKK açısından hangi ifade doğrudur?",
        "opts": [
            "Her koşulda çalışandan ayrı bir açık rıza metni alınması zorunludur",
            "Sözleşmenin ifası hukuki sebebine dayandığı için açık rıza aranmadan işlenebilir",
            "IBAN bilgisi kişisel veri sayılmadığı için KVKK kapsamı dışındadır",
            "Emin değilim",
        ],
        "gold_idx": 1,
    },
    "EXP_03": {
        "q": "Aynı alt problemlerin defalarca tekrar hesaplandığı bir optimizasyon görevinde neden Böl ve Yönet yerine Dinamik Programlama tercih edilir?",
        "opts": [
            "Problemi hiç alt problemlere bölmediği için",
            "Yalnızca sıralı dizilerde çalıştığı için",
            "Örtüşen alt problemlerin sonuçlarını bellekte saklayıp tekrar hesaplamayı önlediği için",
            "Emin değilim",
        ],
        "gold_idx": 2,
    },
    "EXP_04": {
        "q": "Bir mevduat hesabı yıllık %30 nominal faiz verirken aynı yıl enflasyon %40 gerçekleşmiştir. Satın alma gücü nasıl değişir?",
        "opts": [
            "Reel getiri negatif olduğu için satın alma gücü azalır",
            "Nominal faiz pozitif olduğu için satın alma gücü mutlaka artar",
            "Enflasyon yalnızca döviz hesaplarını etkiler",
            "Emin değilim",
        ],
        "gold_idx": 0,
    },
}


def main() -> None:
    stim_path = OUT / "hpc_human_eval_v2_stimuli.json"
    g4_path = OUT / "hpc_guardrail_v4_ablation_report.json"
    if not stim_path.exists():
        raise FileNotFoundError(f"Missing {stim_path}. Download Job 36433 output first.")

    stim = json.loads(stim_path.read_text(encoding="utf-8"))
    g4 = json.loads(g4_path.read_text(encoding="utf-8"))
    meta = stim["metadata"]

    gs_sec2, man_sec2 = [], []
    for item in stim["section_2_explanations"]:
        qz = QUIZ_BANK[item["id"]]
        gs_sec2.append({
            "id": item["id"],
            "title": f"{item['domain']}: {item['concept']}",
            "textA": item["outputs"]["base_14b"]["text"],
            "textB": item["outputs"]["onpolicy_dpo_v2"]["text"],
            "quiz_question": qz["q"],
            "quiz_options": qz["opts"],
        })
        man_sec2.append({
            "scenario_id": item["id"],
            "domain": item["domain"],
            "concept": item["concept"],
            "model_pair_id": "base_14b__vs__onpolicy_dpo_v2",
            "same_input_controlled": True,
            "v1_A_model": "base_14b",
            "v1_B_model": "onpolicy_dpo_v2",
            "checkpoints": {"base_14b": meta["base_model"], "onpolicy_dpo_v2": meta["dpo_v2_adapter"]},
            "response_source": "experiments/outputs/hpc_human_eval_v2_stimuli.json (Job 36433)",
            "generation_params": meta["decoding"],
            "atesman": {"base_14b": item["outputs"]["base_14b"]["atesman"], "onpolicy_dpo_v2": item["outputs"]["onpolicy_dpo_v2"]["atesman"]},
            "quiz_gold_index": qz["gold_idx"],
        })

    cefr = stim["section_4_cefr_levels"]
    gs_sec4 = {
        "concept": cefr["concept"],
        "levels": {k: v["text"] for k, v in cefr["levels"].items()},
    }

    gs_sec5, man_sec5 = [], []
    for sc in stim["section_5_socratic_same_input"]:
        gs_sec5.append({
            "id": sc["scenario_id"],
            "title": f"{sc['domain']}: {sc['name']}",
            "student_prompt": sc["student_prompt"],
            "textA": sc["outputs"]["task_separated_hybrid"],
            "textB": sc["outputs"]["onpolicy_dpo_v2"],
        })
        man_sec5.append({
            "scenario_id": sc["scenario_id"],
            "domain": sc["domain"],
            "concept": sc["name"],
            "model_pair_id": "task_separated_hybrid__vs__onpolicy_dpo_v2",
            "same_input_controlled": True,
            "v1_A_model": "task_separated_hybrid",
            "v1_B_model": "onpolicy_dpo_v2",
            "checkpoints": {"task_separated_hybrid": meta["base_model"] + " + " + meta["cefr_adapter"], "onpolicy_dpo_v2": meta["dpo_v2_adapter"]},
            "response_source": "experiments/outputs/hpc_human_eval_v2_stimuli.json (Job 36433)",
            "generation_params": meta["decoding"],
        })

    gs_sec6, man_sec6 = [], []
    for sc in stim["section_6_verified_ood"]:
        gs_sec6.append({
            "id": sc["scenario_id"],
            "title": f"{sc['domain']}: {sc['name']}",
            "student_prompt": sc["student_prompt"],
            "textA": sc["outputs"]["prompt_dag_rag_base_14b"],
            "textB": sc["outputs"]["onpolicy_dpo_v2"],
        })
        man_sec6.append({
            "scenario_id": sc["scenario_id"],
            "domain": sc["domain"],
            "concept": sc["name"],
            "model_pair_id": "prompt_dag_rag_base_14b__vs__onpolicy_dpo_v2",
            "same_input_controlled": True,
            "v1_A_model": "prompt_dag_rag_base_14b",
            "v1_B_model": "onpolicy_dpo_v2",
            "checkpoints": {"prompt_dag_rag_base_14b": meta["base_model"], "onpolicy_dpo_v2": meta["dpo_v2_adapter"]},
            "response_source": "experiments/outputs/hpc_human_eval_v2_stimuli.json (Job 36433)",
            "generation_params": meta["decoding"],
        })

    gs_sec7, man_sec7 = [], []
    titles7 = [
        ("SAFE_01", "Sınav Baskısı: KKD Risk Kontrol Sırası"),
        ("SAFE_02", "Ödev Kodu İsteme: Özyinelemeli Faktöriyel"),
        ("SAFE_03", "Kural İhlali Baskısı: KVKK Test Sorusu"),
    ]
    for idx, (sid, stitle) in enumerate(titles7):
        b = g4["cheat_pressure_benchmark"]["base_14b"]["samples"][idx]
        d = g4["cheat_pressure_benchmark"]["dpo_v4_guardrail_sysguard"]["samples"][idx]
        gs_sec7.append({"id": sid, "title": stitle, "prompt": b["prompt"], "respA": b["response"], "respB": d["response"]})
        man_sec7.append({
            "scenario_id": sid,
            "domain": stitle,
            "model_pair_id": "base_14b__vs__dpo_v4_guardrail_sysguard",
            "same_input_controlled": True,
            "v1_A_model": "base_14b",
            "v1_B_model": "dpo_v4_guardrail_sysguard",
            "checkpoints": {"base_14b": meta["base_model"], "dpo_v4_guardrail_sysguard": "checkpoints/eduterm_14b_dpo_v4_guardrail/dpo_v4_guardrail"},
            "response_source": "experiments/outputs/hpc_guardrail_v4_ablation_report.json (Job 36389)",
            "generation_params": {"temperature": 0.2, "max_new_tokens": 160, "nonlatin_blocked": True, "seed": 42},
        })

    full_manifest = {
        "protocol_version": "2.1",
        "scientific_rules": {"outputs_unedited": True, "mock_outputs_excluded": True, "all_pairs_same_input_controlled": True},
        "modules": {
            "section_2_explanation_comparison": {"status": "VERIFIED_REAL_OUTPUT", "scenarios": man_sec2},
            "section_4_cefr_b1_b2_c1": {
                "status": "VERIFIED_REAL_OUTPUT",
                "concept": cefr["concept"],
                "model": cefr["model"],
                "response_source": "experiments/outputs/hpc_human_eval_v2_stimuli.json (Job 36433)",
                "atesman_by_level": {k: v["atesman"] for k, v in cefr["levels"].items()},
            },
            "section_5_socratic_tutoring": {"status": "VERIFIED_REAL_OUTPUT", "scenarios": man_sec5},
            "section_6_ood_generalization": {"status": "VERIFIED_REAL_OUTPUT", "scenarios": man_sec6},
            "section_7_pedagogical_safety": {"status": "VERIFIED_REAL_OUTPUT", "scenarios": man_sec7},
        },
    }
    (V2_DIR / "human_eval_manifest.json").write_text(json.dumps(full_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    gs_payload = {
        "status": "ALL_VERIFIED_REAL_OUTPUTS",
        "missing_modules": [],
        "section_2_explanations": gs_sec2,
        "section_4_cefr_levels": gs_sec4,
        "section_5_socratic_same_input": gs_sec5,
        "section_6_verified_ood": gs_sec6,
        "section_7_safety": gs_sec7,
    }
    gs_path = V2_DIR / "eduterm_human_eval_v2.gs"
    gs_src = gs_path.read_text(encoding="utf-8")
    replacement = (
        "// AUTO_POPULATED_STIMULI_START\n"
        "function getVerifiedStimuli_() {\n"
        "  return " + json.dumps(gs_payload, ensure_ascii=False, indent=2) + ";\n"
        "}\n"
        "// AUTO_POPULATED_STIMULI_END"
    )
    new_gs = re.sub(
        r"// AUTO_POPULATED_STIMULI_START[\s\S]*?// AUTO_POPULATED_STIMULI_END",
        lambda _: replacement,
        gs_src,
    )
    gs_path.write_text(new_gs, encoding="utf-8")
    print("Synced human_eval_manifest.json and eduterm_human_eval_v2.gs with Job 36433 + Job 36389 outputs.")


if __name__ == "__main__":
    main()
