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


if __name__ == "__main__":
    unittest.main()
