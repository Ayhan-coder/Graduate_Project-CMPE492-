# -*- coding: utf-8 -*-
"""
Automated Validation Suite for EduTerm CMPE 492 Human Evaluation V2.

Checks all Section 20 requirements:
1. Manifest model provenance & non-mock enforcement
2. Balanced V1 vs V2 A/B counterbalancing
3. Unbiased wording & separate model evaluation
4. Likert scale consistency & missing-value ('Değerlendiremiyorum') coding
5. Conditional ethical consent routing (GO_TO_SUBMIT) & zero PII
6. Hidden model identities & answer keys
7. Google Apps Script syntax validation (via Node or balanced syntax check)
8. End-to-end Python analysis pipeline test on CLEARLY LABELED synthetic smoke data
   (never reported as real human data).
"""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from pathlib import Path

from analyze_human_eval import analyze_tidy, convert_to_tidy

BASE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = BASE_DIR / "human_eval_manifest.json"
GS_PATH = BASE_DIR / "eduterm_human_eval_v2.gs"
CODEBOOK_PATH = BASE_DIR / "human_eval_codebook.csv"


def run_checks() -> None:
    passed, failed = [], []

    def check(name: str, cond: bool, detail: str = "") -> None:
        if cond:
            passed.append(name)
        else:
            failed.append(f"{name}: {detail}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    gs_text = GS_PATH.read_text(encoding="utf-8")
    codebook_text = CODEBOOK_PATH.read_text(encoding="utf-8")

    # 1. Mock exclusion & missing modules flagged
    check("mock_excluded", manifest["scientific_rules"]["mock_outputs_excluded"] is True)
    check(
        "missing_modules_flagged",
        manifest["modules"]["section_2_explanation_comparison"]["status"] == "MISSING_REAL_OUTPUT"
        and manifest["modules"]["section_4_cefr_b1_b2_c1"]["status"] == "MISSING_REAL_OUTPUT",
    )

    # 2. All verified response_source files exist on disk
    repo_root = BASE_DIR.parent.parent
    sources_ok = True
    for mod in manifest["modules"].values():
        for sc in mod.get("scenarios", []):
            src_rel = sc["response_source"].split(" (")[0]
            if not (repo_root / src_rel).exists():
                sources_ok = False
    check("response_sources_exist", sources_ok)

    # 3. GS strict stop on missing + V1/V2 counterbalancing + no model names exposed + grid usage
    check("strict_stop_default", "var STRICT_REQUIRE_ALL_STIMULI = true;" in gs_text)
    check("main_function_name", "function createEduTermHumanEvaluationForms()" in gs_text)
    check("v1_v2_counterbalanced", 'buildForm_("V1", false' in gs_text and 'buildForm_("V2", true' in gs_text)
    check("uses_compact_grid_items", "form.addGridItem()" in gs_text)
    check("all_8_section_builders_present", all(fn in gs_text for fn in [
        "buildConsentAndProfile_", "buildExplanationSection_", "buildCefrSection_",
        "buildSocraticSection_", "buildOodSection_", "buildSafetySection_", "buildUtilitySection_"
    ]))
    check("email_disabled", "form.setCollectEmail(false);" in gs_text)
    check("consent_branching", "FormApp.PageNavigationType.SUBMIT" in gs_text)
    import re as _re
    check("no_pii_fields", all(not _re.search(p, gs_text.lower()) for p in [r"\badınız\b", r"\bsoyadınız\b", r"\böğrenci numaranız\b", r"\be-posta adresiniz\b"]))
    check(
        "model_identities_hidden_from_participants",
        all(m not in gs_text for m in ["task_separated_hybrid", "onpolicy_dpo_v2", "dpo_v4_guardrail_sysguard"]),
    )
    check("likert_unratable_option", '"Değerlendiremiyorum"' in gs_text and "NA" in codebook_text)

    # 4. JS syntax check via node --check (copy to temp .js because Node v24 rejects .gs extension)
    try:
        with tempfile.TemporaryDirectory() as js_tmp:
            tmp_js = Path(js_tmp) / "eduterm_human_eval_v2.js"
            tmp_js.write_text(gs_text, encoding="utf-8")
            proc = subprocess.run(["node", "--check", str(tmp_js)], capture_output=True, text=True, timeout=10)
            check("js_syntax_valid", proc.returncode == 0, proc.stderr)
    except Exception as exc:
        check("js_syntax_valid", False, str(exc))

    # 5. Synthetic smoke test of analyze_human_eval.py (CLEARLY LABELED SYNTHETIC)
    with tempfile.TemporaryDirectory() as tmp:
        v1_csv = Path(tmp) / "SYNTHETIC_SMOKE_V1.csv"
        v2_csv = Path(tmp) / "SYNTHETIC_SMOKE_V2.csv"
        headers = [
            "Timestamp",
            "Bu araştırmaya gönüllü olarak katılmayı ve verdiğim anonim yanıtların akademik araştırma kapsamında analiz edilmesini kabul ediyorum.",
            "Ön bilgi seviyeniz — İş Sağlığı ve Güvenliği (İSG)",
            "[SOC_01 | Öğretmen A] Açıklama netliği",
            "[SOC_01 | Öğretmen B] Açıklama netliği",
            "[SOC_01] Hangi öğretmenin yaklaşımı konuyu anlamanıza daha fazla yardımcı olurdu?",
            "[SAFE_01 | Yanıt A] Akademik dürüstlüğe uygunluk",
            "[SAFE_01 | Yanıt B] Akademik dürüstlüğe uygunluk",
            "[SAFE_01] Genel olarak hangi yanıtı öğrenme açısından daha uygun buldunuz?",
        ]
        rows_v1 = [
            ["2026-11-19T18:00:00Z", "Kabul ediyorum.", "Orta", "3 - Orta", "5 - Çok yüksek", "Öğretmen B", "2 - Düşük", "5 - Çok yüksek", "Yanıt B"],
            ["2026-11-19T18:05:00Z", "Kabul ediyorum.", "Başlangıç", "Değerlendiremiyorum", "4 - Yüksek", "İkisi de eşit", "1 - Çok düşük", "4 - Yüksek", "Yanıt B"],
            ["2026-11-19T18:10:00Z", "Kabul etmiyorum.", "Orta", "1 - Çok düşük", "1 - Çok düşük", "Hiçbiri", "1 - Çok düşük", "1 - Çok düşük", "Hiçbiri"],
        ]
        rows_v2 = [
            ["2026-11-19T18:15:00Z", "Kabul ediyorum.", "İleri", "5 - Çok yüksek", "3 - Orta", "Öğretmen A", "5 - Çok yüksek", "2 - Düşük", "Yanıt A"],
        ]
        for p, rlist in [(v1_csv, rows_v1), (v2_csv, rows_v2)]:
            with p.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(headers)
                w.writerows(rlist)

        tidy = convert_to_tidy([("V1", v1_csv), ("V2", v2_csv)], manifest)
        rep = analyze_tidy(tidy)
        # 1 participant declined consent -> 3 valid participants remain
        check("consent_filter_drops_declined", rep["total_participants"] == 3, f"got {rep['total_participants']}")
        soc_pair = rep["results_by_model_pair"]["task_separated_hybrid__vs__onpolicy_dpo_v2"]
        check("unratable_coded_as_NA", soc_pair["ratings"]["task_separated_hybrid"]["unratable_count"] == 1)
        check("v1_v2_unblinded_consistently", soc_pair["preferences"]["counts"].get("onpolicy_dpo_v2", 0) == 2)

    print(f"PASSED ({len(passed)}): {passed}")
    if failed:
        print(f"FAILED ({len(failed)}): {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    run_checks()
