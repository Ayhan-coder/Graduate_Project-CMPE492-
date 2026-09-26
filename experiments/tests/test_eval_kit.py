"""Unit tests for metrics, IAA, and report aggregation. No model downloads."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from export_labels import flatten_terms  # noqa: E402
from fetch_models import CACHE_DIR, default_hub_cache, load_registry, resolve_hub_cache  # noqa: E402
from iaa import cohen_kappa, load_labels, score_pair  # noqa: E402
from metrics import normalize, pair_metrics  # noqa: E402
from prepare_audio import caption_text  # noqa: E402
from report import attach_labels, paper_blockers, summarize_ablation, summarize_asr, write_clip_markdown  # noqa: E402
from run_asr import completed_pairs  # noqa: E402
from run_llm import _strip_think, parse_terms  # noqa: E402
from eval_gold_terms import evaluate_clip, normalize_term, turkish_lower  # noqa: E402


class MetricsTests(unittest.TestCase):
    def test_identical(self):
        m = pair_metrics("Hello world", "Hello world")
        self.assertEqual(m["wer_raw"], 0.0)
        self.assertEqual(m["cer_norm"], 0.0)

    def test_case_and_punct_only_in_raw(self):
        m = pair_metrics("Hello, World!", "hello world")
        self.assertGreater(m["wer_raw"], 0.0)
        self.assertEqual(m["wer_norm"], 0.0)

    def test_empty_reference_does_not_divide_by_zero(self):
        m = pair_metrics("", "hello")
        self.assertGreater(m["wer_raw"], 0.0)
        self.assertGreater(m["cer_raw"], 0.0)

    def test_insertion_and_deletion(self):
        ins = pair_metrics("one two", "one extra two")
        dele = pair_metrics("one extra two", "one two")
        self.assertGreater(ins["wer_raw"], 0.0)
        self.assertGreater(dele["wer_raw"], 0.0)

    def test_turkish_dotted_i_after_locale_map(self):
        self.assertEqual(normalize("İstanbul", "tr"), "istanbul")
        self.assertEqual(normalize("ISINMA", "tr"), "ısınma")
        same = pair_metrics("İstanbul ısınma", "istanbul ısınma", lang="tr")
        self.assertEqual(same["wer_norm"], 0.0)
        # English must not map capital I to dotless ı
        self.assertEqual(normalize("Introduction", "en"), "introduction")


class IaaTests(unittest.TestCase):
    def test_perfect_kappa(self):
        y = [1, 1, 0, 0, 1]
        k = cohen_kappa(y, list(y))
        self.assertEqual(k["n"], 5)
        self.assertEqual(k["agreement"], 1.0)
        self.assertEqual(k["kappa"], 1.0)

    def test_csv_roundtrip(self):
        text = (
            "item_id,video_id,model,lang,term,explanation,R,D\n"
            "1,EN-1,gpt-4o-mini,en,sprint,A short work cycle,1,1\n"
            "2,EN-1,gpt-4o-mini,en,backlog,A list of work,R0,D1\n"
        )
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "a1.csv"
            p2 = Path(td) / "a2.csv"
            p1.write_text(text, encoding="utf-8")
            p2.write_text(text.replace("R0,D1", "0,1"), encoding="utf-8")
            report = score_pair(load_labels(p1), load_labels(p2))
        self.assertEqual(report["n_common"], 2)
        self.assertEqual(report["R"]["agreement"], 1.0)
        self.assertEqual(report["D"]["agreement"], 1.0)
        self.assertEqual(report["R"]["kappa"], 1.0)

    def test_empty_kappa(self):
        k = cohen_kappa([], [])
        self.assertEqual(k["n"], 0)
        self.assertIsNone(k["kappa"])

    def test_kappa_undefined_when_pe_is_one(self):
        k = cohen_kappa([1, 1, 1], [1, 1, 1])
        self.assertEqual(k["agreement"], 1.0)
        self.assertIsNone(k["kappa"])

    def test_invalid_label_raises(self):
        text = "item_id,R,D\n1,maybe,1\n"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.csv"
            p.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_labels(p)


class ReportTests(unittest.TestCase):
    def test_asr_by_language(self):
        rows = [
            {"id": "TR-1", "lang": "tr", "model": "openai/whisper-medium", "wer_norm": 0.02, "cer_norm": 0.01, "wer_raw": 0.05, "cer_raw": 0.02, "latency_s": 100, "rtf": 0.1},
            {"id": "TR-2", "lang": "tr", "model": "openai/whisper-medium", "wer_norm": 0.04, "cer_norm": 0.03, "wer_raw": 0.06, "cer_raw": 0.03, "latency_s": 120, "rtf": 0.2},
            {"id": "EN-1", "lang": "en", "model": "openai/whisper-medium", "wer_norm": 0.10, "cer_norm": 0.08, "wer_raw": 0.12, "cer_raw": 0.09, "latency_s": 80, "rtf": 0.1},
        ]
        summary = summarize_asr(rows)
        tr = summary["models"]["openai/whisper-medium"]["langs"]["tr"]
        en = summary["models"]["openai/whisper-medium"]["langs"]["en"]
        self.assertEqual(tr["n"], 2)
        self.assertAlmostEqual(tr["wer_norm"], 0.03)
        self.assertAlmostEqual(en["wer_norm"], 0.10)
        self.assertEqual(summary["models"]["openai/whisper-medium"]["pooled"]["n"], 3)

    def test_bootstrap_ci_and_micro_wer(self):
        from report import bootstrap_ci
        ci = bootstrap_ci([0.02, 0.04, 0.08], n_boot=1000)
        self.assertIsNotNone(ci)
        self.assertLess(ci[0], 0.05)
        self.assertGreater(ci[1], 0.03)

        rows = [
            {"id": "TR-1", "lang": "tr", "model": "m", "wer_norm": 0.02},
            {"id": "TR-2", "lang": "tr", "model": "m", "wer_norm": 0.04},
        ]
        info = {
            "TR-1": {"words": 100, "duration": 10.0, "lang": "tr"},
            "TR-2": {"words": 300, "duration": 30.0, "lang": "tr"},
        }
        s = summarize_asr(rows, clip_info=info, n_boot=100)
        tr = s["models"]["m"]["langs"]["tr"]
        self.assertAlmostEqual(tr["wer_macro"], 0.03)
        # Word-weighted: (0.02*100 + 0.04*300) / 400 = (2 + 12) / 400 = 0.035
        self.assertAlmostEqual(tr["wer_micro"], 0.035)

    def test_ablation_term_counts(self):
        rows = [
            {"id": "TR-1", "lang": "tr", "asr_model": "openai/whisper-medium", "model": "gpt-4o-mini", "n_terms": 10, "R1": 9, "R0": 1, "D1": 10, "D0": 0},
            {"id": "TR-2", "lang": "tr", "asr_model": "openai/whisper-medium", "model": "gpt-4o-mini", "n_terms": 8, "R1": 7, "R0": 1, "D1": 7, "D0": 1},
        ]
        summary = summarize_ablation(rows)
        block = summary["models"]["openai/whisper-medium"]["langs"]["tr"]
        self.assertEqual(block["n_terms"], 18)
        self.assertAlmostEqual(block["R1_pct"], 100.0 * 16 / 18)
        self.assertEqual(summary["llm"], "gpt-4o-mini")

    def test_paper_blockers_need_confirmed_gold(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.jsonl"
            path.write_text(
                json.dumps({"id": "TR-1", "confirmed": False, "reference_source": "caption_draft"})
                + "\n"
                + json.dumps({"id": "EN-1", "confirmed": True, "reference_source": "gold"})
                + "\n",
                encoding="utf-8",
            )
            blockers = paper_blockers(path)
        self.assertTrue(any("TR-1: confirmed=false" in b for b in blockers))
        self.assertTrue(any("TR-1: reference_source=caption_draft" in b for b in blockers))
        self.assertFalse(any(b.startswith("EN-1:") for b in blockers))

    def test_attach_labels(self):
        rows = [
            {
                "id": "EN-1",
                "lang": "en",
                "asr_model": "openai/whisper-medium",
                "model": "gpt-4o-mini",
                "n_terms": 2,
            }
        ]
        csv_text = (
            "item_id,video_id,model,lang,term,explanation,R,D\n"
            "a,EN-1,openai/whisper-medium|gpt-4o-mini,en,sprint,x,1,1\n"
            "b,EN-1,openai/whisper-medium|gpt-4o-mini,en,noise,y,0,1\n"
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "l.csv"
            p.write_text(csv_text, encoding="utf-8")
            joined = attach_labels(rows, p)
        self.assertEqual(joined[0]["R1"], 1)
        self.assertEqual(joined[0]["R0"], 1)
        self.assertEqual(joined[0]["D1"], 2)

    def test_summarize_asr_skips_error_rows(self):
        rows = [
            {"id": "TR-1", "lang": "tr", "model": "openai/whisper-medium", "wer_norm": 0.02, "cer_norm": 0.01, "wer_raw": 0.05, "cer_raw": 0.02, "latency_s": 10, "rtf": 0.1},
            {"id": "TR-2", "lang": "tr", "model": "openai/whisper-medium", "error": "OOM"},
        ]
        summary = summarize_asr(rows)
        tr = summary["models"]["openai/whisper-medium"]["langs"]["tr"]
        self.assertEqual(tr["n"], 1)
        self.assertEqual(summary["models"]["openai/whisper-medium"]["pooled"]["n"], 1)

    def test_clip_markdown_marks_missing_and_errors(self):
        md = write_clip_markdown(
            [
                {
                    "id": "TR-1",
                    "lang": "tr",
                    "model": "openai/whisper-medium",
                    "wer_norm": 0.04,
                    "cer_norm": 0.01,
                    "rtf": 0.2,
                    "reference_source": "caption_draft",
                    "confirmed": False,
                },
                {
                    "id": "TR-3",
                    "lang": "tr",
                    "model": "openai/whisper-medium",
                    "wer_norm": None,
                    "reference_source": "missing",
                    "confirmed": False,
                    "rtf": 0.3,
                },
                {"id": "EN-1", "lang": "en", "model": "openai/whisper-medium", "error": "OOM"},
            ]
        )
        self.assertIn("caption_draft", md)
        self.assertIn("| TR-3 |", md)
        self.assertIn("| --- |", md)
        self.assertIn("| ERR | ERR | ERR |", md)

    def test_summarize_asr_skips_null_wer(self):
        rows = [
            {"id": "TR-1", "lang": "tr", "model": "openai/whisper-medium", "wer_norm": 0.02, "cer_norm": 0.01, "wer_raw": 0.05, "cer_raw": 0.02, "latency_s": 10, "rtf": 0.1},
            {"id": "TR-3", "lang": "tr", "model": "openai/whisper-medium", "wer_norm": None, "reference_source": "missing", "rtf": 0.3},
        ]
        summary = summarize_asr(rows)
        tr = summary["models"]["openai/whisper-medium"]["langs"]["tr"]
        self.assertEqual(tr["n"], 1)
        self.assertAlmostEqual(tr["wer_norm"], 0.02)
        self.assertEqual(tr["ids"], ["TR-1"])

    def test_paper_blockers_missing_manifest(self):
        blockers = paper_blockers(Path("/no/such/manifest.jsonl"))
        self.assertTrue(any("missing" in b for b in blockers))

    def test_real_manifest_has_no_gold_blockers(self):
        blockers = paper_blockers(ROOT / "data" / "manifest.jsonl")
        self.assertEqual(blockers, [])

    def test_gold_rescored_whisper_language_means(self):
        from report import load_jsonl

        path = ROOT / "outputs" / "asr-whisper-combined.rescored.jsonl"
        if not path.exists():
            self.skipTest("rescored whisper file missing")
        summary = summarize_asr(load_jsonl(path))
        med = summary["models"]["openai/whisper-medium"]["langs"]
        large = summary["models"]["openai/whisper-large-v3"]["langs"]
        turbo = summary["models"]["openai/whisper-large-v3-turbo"]["langs"]
        self.assertEqual(med["tr"]["n"], 3)
        self.assertEqual(med["en"]["n"], 4)
        self.assertAlmostEqual(med["tr"]["wer_norm"], 0.049, places=3)
        self.assertAlmostEqual(med["en"]["wer_norm"], 0.011, places=3)
        self.assertAlmostEqual(large["tr"]["wer_norm"], 0.021, places=3)
        self.assertAlmostEqual(large["en"]["wer_norm"], 0.010, places=3)
        self.assertAlmostEqual(turbo["tr"]["wer_norm"], 0.037, places=3)
        self.assertAlmostEqual(turbo["en"]["wer_norm"], 0.011, places=3)
        self.assertLess(large["tr"]["wer_norm"], med["tr"]["wer_norm"])

    def test_real_iaa_kappa(self):
        a1 = ROOT / "annotation" / "labels_a1.csv"
        a2 = ROOT / "annotation" / "labels_a2.csv"
        if not a1.exists() or not a2.exists():
            self.skipTest("annotation CSVs missing")
        report = score_pair(load_labels(a1), load_labels(a2))
        self.assertEqual(report["n_common"], 115)
        self.assertAlmostEqual(report["R"]["kappa"], 0.782, places=3)
        self.assertEqual(report["D"]["kappa"], 1.0)

    def test_resume_skips_ok_keeps_errors(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "asr.jsonl"
            p.write_text(
                json.dumps({"id": "TR-1", "model": "openai/whisper-medium", "wer_norm": 0.1})
                + "\n"
                + json.dumps({"id": "TR-2", "model": "openai/whisper-medium", "error": "OOM"})
                + "\n"
                + json.dumps({"id": "TR-3", "model": "openai/whisper-medium", "skipped": True})
                + "\n",
                encoding="utf-8",
            )
            done = completed_pairs(p)
        self.assertIn(("TR-1", "openai/whisper-medium"), done)
        self.assertNotIn(("TR-2", "openai/whisper-medium"), done)
        self.assertNotIn(("TR-3", "openai/whisper-medium"), done)

    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "asr.jsonl"
            rec = {"id": "EN-1", "lang": "en", "model": "openai/whisper-medium", "wer_norm": 0.2, "cer_norm": 0.1, "wer_raw": 0.3, "cer_raw": 0.15}
            p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
            rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(summarize_asr(rows)["models"]["openai/whisper-medium"]["langs"]["en"]["n"], 1)


class ParseAndExportTests(unittest.TestCase):
    def test_parse_terms_from_fenced_json(self):
        text = (
            "Sure.\n```json\n"
            '[{"term": "sprint", "explanation": "A short work cycle"}]\n'
            "```\n"
        )
        terms = parse_terms(text)
        self.assertEqual(len(terms), 1)
        self.assertEqual(terms[0]["term"], "sprint")

    def test_parse_terms_rejects_garbage(self):
        self.assertEqual(parse_terms("no json here"), [])
        self.assertEqual(parse_terms('{"term": "x"}'), [])

    def test_strip_think_then_parse(self):
        raw = (
            "<think>plan the json</think>\n"
            '[{"term": "sprint", "explanation": "A short work cycle"}]'
        )
        terms = parse_terms(_strip_think(raw))
        self.assertEqual(len(terms), 1)
        self.assertEqual(terms[0]["term"], "sprint")

    def test_flatten_terms_uses_asr_and_llm_tag(self):
        rows = [
            {
                "id": "EN-1",
                "lang": "en",
                "asr_model": "openai/whisper-medium",
                "model": "gpt-4o-mini",
                "terms": [
                    {"term": "sprint", "explanation": "A short cycle"},
                    {"term": "backlog", "explanation": "A list of work"},
                ],
            }
        ]
        items = flatten_terms(rows)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["video_id"], "EN-1")
        self.assertEqual(items[0]["model"], "openai/whisper-medium|gpt-4o-mini")
        self.assertEqual(items[0]["R"], "")
        self.assertTrue(items[0]["item_id"].startswith("EN-1-"))


class CaptionTests(unittest.TestCase):
    def test_caption_text_strips_timestamps(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            (folder / "TR-1.tr.vtt").write_text(
                "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n"
                "Proje yönetimi.\n\n00:00:02.000 --> 00:00:04.000\n"
                "Proje yönetimi.\n\n00:00:04.000 --> 00:00:06.000\n"
                "Kontrol işlemleri.\n",
                encoding="utf-8",
            )
            text = caption_text(folder, "TR-1")
        self.assertIn("Proje yönetimi.", text)
        self.assertIn("Kontrol işlemleri.", text)
        self.assertNotIn("-->", text)
        self.assertEqual(text.count("Proje yönetimi."), 1)


class RegistryTests(unittest.TestCase):
    def test_selected_asr_have_backend_and_revision(self):
        reg = load_registry()
        self.assertTrue(reg["asr_selected"])
        for m in reg["asr_selected"]:
            self.assertIn("backend", m, m["id"])
            self.assertIn("revision", m, m["id"])
            self.assertIn("subset", m, m["id"])
        ids = {m["id"] for m in reg["asr_selected"]}
        self.assertIn("nvidia/nemotron-3.5-asr-streaming-0.6b", ids)
        llm_ids = {m["id"] for m in reg["llm_selected"]}
        self.assertNotIn("gpt-5.6-sol", llm_ids)
        self.assertNotIn("gpt-5.6-luna", llm_ids)
        self.assertIn("Qwen/Qwen3.5-9B", llm_ids)
        self.assertIn("Qwen/Qwen3.5-4B", llm_ids)
        self.assertIn("meta-llama/Llama-3.1-8B-Instruct", llm_ids)
        self.assertNotIn("Qwen/Qwen3-8B", llm_ids)

    def test_pilot_ids_are_in_report_display(self):
        from report import DISPLAY, PILOT_ASR, REFRESH_ASR

        for mid in list(PILOT_ASR) + list(REFRESH_ASR):
            self.assertIn(mid, DISPLAY)

    def test_completed_pairs_missing_file(self):
        self.assertEqual(completed_pairs(Path("/no/such/asr.jsonl")), set())

    def test_resolve_hub_cache_prefers_existing_snapshot(self):
        mid = "openai/whisper-medium"
        slug = "models--" + mid.replace("/", "--")
        default = default_hub_cache()
        if (default / slug).exists() and not (CACHE_DIR / slug).exists():
            self.assertEqual(resolve_hub_cache(mid), default)
        self.assertEqual(resolve_hub_cache("nobody/does-not-exist-xyz"), CACHE_DIR)


class GoldTermEvalTests(unittest.TestCase):
    def test_turkish_normalization(self):
        self.assertEqual(normalize_term("İzleme"), "izleme")
        self.assertEqual(normalize_term("izleme"), "izleme")
        self.assertEqual(normalize_term("ISINMA"), "ısınma")
        self.assertEqual(normalize_term("the Daily Scrum"), "daily scrum")
        self.assertEqual(normalize_term("çizelge sapması"), "çizelge sapması")

    def test_exact_and_boundary_matching(self):
        spec = {
            "terms": [
                {"canonical": "Maliyet Kontrolü", "aliases": ["maliyet kontrolü", "cost control"]},
                {"canonical": "Kontrol", "aliases": ["kontrol", "proje kontrolü"]},
                {"canonical": "Düzeltici Eylemler", "aliases": ["düzeltici eylemler", "düzeltici eylem"]},
            ]
        }
        # Exact alias match
        res_exact = evaluate_clip(["Maliyet Kontrolü"], spec)
        self.assertEqual(res_exact["tp"], 1)
        self.assertEqual(res_exact["recalled_gold"], ["Maliyet Kontrolü"])

        # Token-boundary match in longer extracted phrase
        res_boundary = evaluate_clip(["kapsamlı maliyet kontrolü sistemi"], spec)
        self.assertEqual(res_boundary["tp"], 1)
        self.assertEqual(res_boundary["recalled_gold"], ["Maliyet Kontrolü"])

    def test_rejects_loose_substring_inflation(self):
        spec = {
            "terms": [
                {"canonical": "Maliyet Kontrolü", "aliases": ["maliyet kontrolü", "cost control"]},
                {"canonical": "Düzeltici Eylemler", "aliases": ["düzeltici eylemler", "düzeltici eylem"]},
            ]
        }
        # "kontrol" alone must NOT match "Maliyet Kontrolü"
        res1 = evaluate_clip(["kontrol"], spec)
        self.assertEqual(res1["tp"], 0)
        self.assertIn("kontrol", res1["false_positives"])

        # "eylem" alone must NOT match "Düzeltici Eylemler"
        res2 = evaluate_clip(["eylem"], spec)
        self.assertEqual(res2["tp"], 0)
        self.assertIn("eylem", res2["false_positives"])

    def test_precision_recall_f1_calculation(self):
        spec = {
            "terms": [
                {"canonical": "A", "aliases": ["a"]},
                {"canonical": "B", "aliases": ["b"]},
                {"canonical": "C", "aliases": ["c"]},
                {"canonical": "D", "aliases": ["d"]},
            ]
        }
        # Emitted: A, B, X (2 true positives, 1 false positive, 2 false negatives)
        res = evaluate_clip(["A", "B", "X"], spec)
        self.assertEqual(res["gold_count"], 4)
        self.assertEqual(res["emitted_count"], 3)
        self.assertEqual(res["tp"], 2)
        self.assertEqual(res["fp"], 1)
        self.assertEqual(res["fn"], 2)
        self.assertAlmostEqual(res["precision"], 2 / 3)
        self.assertAlmostEqual(res["recall"], 2 / 4)
        expected_f1 = 2 * (2 / 3 * 2 / 4) / (2 / 3 + 2 / 4)
        self.assertAlmostEqual(res["f1"], expected_f1)


class ReadabilityTests(unittest.TestCase):
    def test_atesman_score_basic(self):
        from metrics import atesman_score, count_turkish_syllables

        self.assertEqual(count_turkish_syllables("proje"), 2)
        self.assertEqual(count_turkish_syllables("yönetimi"), 4)
        self.assertEqual(count_turkish_syllables("değer"), 2)

        sample = (
            "Proje yönetimi belirli hedeflere ulaşmak için kaynakların planlanmasıdır. "
            "Kazanılmış değer analizi bütçe kontrolünü sağlar."
        )
        res = atesman_score(sample)
        self.assertGreater(res["score"], 0.0)
        self.assertLessEqual(res["score"], 100.0)
        self.assertIn(res["level"], ["Çok Kolay", "Kolay", "Orta Güçlükte", "Zor", "Çok Zor"])
        self.assertGreater(res["syllables"], 10)
        self.assertGreater(res["words"], 5)

    def test_flesch_score_basic(self):
        from metrics import flesch_score

        sample = "Agile software development focuses on short sprints and regular customer feedback."
        res = flesch_score(sample)
        self.assertGreater(res["reading_ease"], 0.0)
        self.assertGreater(res["grade_level"], 0.0)
        self.assertIn(res["level"], ["Very Easy", "Easy", "Fairly Easy", "Standard", "Fairly Difficult", "Difficult", "Very Confusing"])

    def test_unified_readability_routing(self):
        from metrics import readability_score

        tr_res = readability_score("Bu bir test metnidir.", lang="tr")
        self.assertEqual(tr_res["lang"], "tr")
        self.assertEqual(tr_res["metric"], "atesman")

        en_res = readability_score("This is a simple test sentence.", lang="en")
        self.assertEqual(en_res["lang"], "en")
        self.assertEqual(en_res["metric"], "flesch_reading_ease")


class SemanticSimilarityTests(unittest.TestCase):
    def test_bert_score_identical_pairs(self):
        from eval_semantic import compute_bert_score

        cands = ["Proje yönetimi hedeflere zamanında ulaşma sürecidir."]
        refs = ["Proje yönetimi hedeflere zamanında ulaşma sürecidir."]
        P, R, F1 = compute_bert_score(cands, refs, lang="tr")
        self.assertAlmostEqual(P[0], 1.0, places=2)
        self.assertAlmostEqual(R[0], 1.0, places=2)
        self.assertAlmostEqual(F1[0], 1.0, places=2)

    def test_bert_score_dissimilar_pairs(self):
        from eval_semantic import compute_bert_score

        cands = ["Deniz kenarında güneşli bir günde yüzmek çok eğlencelidir."]
        refs = ["Kazanılmış değer yönetimi projenin maliyet sapmasını hesaplar."]
        P, R, F1 = compute_bert_score(cands, refs, lang="tr")
        self.assertLess(F1[0], 0.35)


class NoiseInjectionTests(unittest.TestCase):
    def test_calibrate_noise_levels(self):
        from inject_noise import calibrate_noise

        text = (
            "Proje yöneticisi ekibin faaliyetlerini izlemek ve kontrol etmekle yükümlüdür. "
            "Kazanılmış değer analizi projenin takvim ve bütçe durumunu ortaya koyar."
        )
        for target in [0.05, 0.15, 0.25]:
            hyp, actual, details = calibrate_noise(text, target, lang="tr", seed=42)
            self.assertAlmostEqual(actual, target, delta=0.06)
            self.assertGreaterEqual(details["S"] + details["D"] + details["I"], 1)

    def test_ladder_generation(self):
        from inject_noise import generate_synthetic_noise_ladder

        text = "Agile methodologies promote cross-functional teamwork and iterative software delivery."
        ladder = generate_synthetic_noise_ladder(text, lang="en", wer_levels=[0.10, 0.20])
        self.assertEqual(len(ladder), 3)  # 0.0 + 2 levels
        self.assertEqual(ladder[0]["actual_wer"], 0.0)
        self.assertGreater(ladder[1]["actual_wer"], 0.0)


class AsrImprovementTests(unittest.TestCase):
    def test_vad_speech_segmentation(self):
        import numpy as np
        from vad_chunker import segment_audio_vad

        sr = 16000
        # 30 seconds of tone with a 2-second silence in middle (14-16s)
        t = np.linspace(0, 30, 30 * sr, dtype=np.float32)
        audio = np.sin(2 * np.pi * 400 * t) * 0.2
        audio[int(14 * sr) : int(16 * sr)] = 0.00001

        chunks = segment_audio_vad(audio, sr=sr, min_chunk_s=10.0, max_chunk_s=25.0)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertAlmostEqual(chunks[0][0], 0.0)
        self.assertAlmostEqual(chunks[-1][1], 30.0)

    def test_glossary_prompt_builder(self):
        from run_asr import _get_glossary_prompt

        prompt_tr = _get_glossary_prompt("TR-1", "tr")
        self.assertIn("Proje Yönetimi", prompt_tr)

        custom = _get_glossary_prompt("TR-1", "tr", custom_text="Özel anahtar terimler")
        self.assertEqual(custom, "Özel anahtar terimler")

    def test_post_correction_delta(self):
        from asr_post_correction import compute_correction_delta

        ref = "Kazanılmış değer yönetimi projenin bütçe ve takvim performansını ölçer."
        raw = "Kazanılmış dere yönetimi projenin büçe ve takvim performansını ölçer."
        corr = "Kazanılmış değer yönetimi projenin bütçe ve takvim performansını ölçer."

        res = compute_correction_delta(ref, raw, corr, lang="tr")
        self.assertTrue(res["wer_improved"])
        self.assertGreater(res["delta_wer"], 0.0)
        self.assertEqual(res["corr_wer_norm"], 0.0)


class PromptStrategyTests(unittest.TestCase):
    def test_build_strategy_prompt(self):
        from prompt_strategies import build_strategy_prompt

        txt = "Proje yönetimi ve kazanılmış değer analizi."
        p_zero = build_strategy_prompt("zero_shot", txt, lang="tr")
        self.assertIn("JSON array", p_zero)
        self.assertIn(txt, p_zero)

        p_cot = build_strategy_prompt("cot", txt, lang="tr")
        self.assertIn("Chain-of-Thought", p_cot)
        self.assertIn(txt, p_cot)

        p_rag = build_strategy_prompt("rag", txt, lang="tr", glossary_terms=["Proje Yönetimi"])
        self.assertIn("RAG Anchors", p_rag)
        self.assertIn("Proje Yönetimi", p_rag)


class LlmJudgeTests(unittest.TestCase):
    def test_parse_judge_json(self):
        from llm_judge import parse_judge_json

        raw_valid = '{"R": 1, "D": 1, "reasoning": "Clear explanation of a core domain concept"}'
        r, d, reason = parse_judge_json(raw_valid)
        self.assertEqual(r, 1)
        self.assertEqual(d, 1)

        raw_generic = '{"R": 0, "D": 1, "reasoning": "Generic noun"}'
        r, d, reason = parse_judge_json(raw_generic)
        self.assertEqual(r, 0)
        self.assertEqual(d, 1)

    def test_judge_term_explanation_mock(self):
        from llm_judge import judge_term_explanation

        r1, d1, _ = judge_term_explanation("Kazanılmış Değer", "Proje bütçe ve ilerleme ölçütü.", use_mock=True)
        self.assertEqual(r1, 1)
        self.assertEqual(d1, 1)

        r2, d2, _ = judge_term_explanation("şey", "Belirsiz kavram.", use_mock=True)
        self.assertEqual(r2, 0)


class EnrichedSchemaTests(unittest.TestCase):
    def test_concept_dag_building(self):
        from enriched_schema import build_concept_dag, build_enriched_output

        raw = [
            {"term": "Proje Yönetimi", "explanation": "Hedeflere ulaşma planlamasıdır."},
            {"term": "Proje Yönetim Planı", "explanation": "Ana rehber belgedir."},
            {"term": "Kazanılmış Değer Yönetimi", "explanation": "Performans ölçüm yöntemidir."},
        ]
        out = build_enriched_output("TR-1", "Proje Yönetimi", "tr", raw)
        self.assertEqual(out["schema_version"], "2.0.0")
        self.assertEqual(out["total_terms"], 3)
        self.assertIn("concept_graph", out)
        self.assertGreaterEqual(out["concept_graph"]["n_nodes"], 3)
        self.assertGreaterEqual(out["concept_graph"]["n_edges"], 1)

        # Check explanation variants
        t1 = out["terms"][0]
        self.assertIn("b1_analogy", t1["explanations"])
        self.assertIn("formal_definition", t1["explanations"])
        self.assertIn("elevator_pitch", t1["explanations"])
        self.assertIn("exam_tip", t1["explanations"])


class Dataset30PlusTests(unittest.TestCase):
    def test_sources_30plus_structure(self):
        p = ROOT / "data" / "sources_30plus.json"
        self.assertTrue(p.exists())
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["total_clips"], 32)
        self.assertEqual(data["language_split"]["tr"], 16)
        self.assertEqual(data["language_split"]["en"], 16)
        self.assertEqual(len(data["clips"]), 32)

    def test_manifest_30plus_records(self):
        p = ROOT / "data" / "manifest_30plus.jsonl"
        self.assertTrue(p.exists())
        records = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        self.assertEqual(len(records), 32)
        tr_count = sum(1 for r in records if r["lang"] == "tr")
        en_count = sum(1 for r in records if r["lang"] == "en")
        self.assertEqual(tr_count, 16)
        self.assertEqual(en_count, 16)
        for r in records:
            self.assertTrue(r["confirmed"])
            self.assertIn(r["reference_source"], {"gold", "verified_curated"})
            self.assertGreater(len(r["reference"]), 20)

    def test_annotation_template_30plus(self):
        p = ROOT.parent / "annotation" / "annotation_template_30plus.jsonl"
        self.assertTrue(p.exists())
        with open(p, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
        self.assertGreaterEqual(len(items), 32)
        first = items[0]
        self.assertIn("video_id", first)
        self.assertIn("term", first)
        self.assertIn("start_time", first)
        self.assertIn("end_time", first)
        self.assertIn("annotator_1_R", first)
        self.assertIn("annotator_2_R", first)


class EvalAgreementTests(unittest.TestCase):
    def test_cohen_kappa_and_pabak(self):
        from eval_agreement import cohen_kappa

        y1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        y2 = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        res = cohen_kappa(y1, y2)
        self.assertEqual(res["n"], 10)
        self.assertEqual(res["po"], 0.9)
        self.assertAlmostEqual(res["pabak"], 0.8, places=3)
        self.assertGreater(res["kappa"], 0.0)

    def test_temporal_span_iou(self):
        from eval_agreement import temporal_span_iou

        # identical
        self.assertEqual(temporal_span_iou((10.0, 20.0), (10.0, 20.0)), 1.0)
        # disjoint
        self.assertEqual(temporal_span_iou((0.0, 5.0), (10.0, 15.0)), 0.0)
        # partial overlap
        self.assertAlmostEqual(temporal_span_iou((10.0, 20.0), (15.0, 25.0)), 0.3333, places=3)

    def test_term_f1_with_timestamps(self):
        from eval_agreement import evaluate_term_f1_with_timestamps

        hyps = [
            {"canonical_term": "EVM", "span": (70.0, 85.0)},
            {"canonical_term": "Sprint", "span": (120.0, 130.0)},
        ]
        refs = [
            {"canonical_term": "EVM", "span": (72.0, 88.0)},
            {"canonical_term": "Kanban", "span": (200.0, 210.0)},
        ]
        res = evaluate_term_f1_with_timestamps(hyps, refs, min_iou=0.3)
        self.assertEqual(res["true_positives"], 1)
        self.assertEqual(res["false_positives"], 1)
        self.assertEqual(res["false_negatives"], 1)
        self.assertEqual(res["precision"], 0.5)
        self.assertEqual(res["recall"], 0.5)
        self.assertEqual(res["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()




