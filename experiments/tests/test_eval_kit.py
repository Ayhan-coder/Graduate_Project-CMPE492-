"""Unit tests for metrics, IAA, and report aggregation. No model downloads."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iaa import cohen_kappa, load_labels, score_pair  # noqa: E402
from metrics import pair_metrics  # noqa: E402
from report import attach_labels, summarize_ablation, summarize_asr  # noqa: E402


class MetricsTests(unittest.TestCase):
    def test_identical(self):
        m = pair_metrics("Hello world", "Hello world")
        self.assertEqual(m["wer_raw"], 0.0)
        self.assertEqual(m["cer_norm"], 0.0)

    def test_case_and_punct_only_in_raw(self):
        m = pair_metrics("Hello, World!", "hello world")
        self.assertGreater(m["wer_raw"], 0.0)
        self.assertEqual(m["wer_norm"], 0.0)


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

    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "asr.jsonl"
            rec = {"id": "EN-1", "lang": "en", "model": "openai/whisper-medium", "wer_norm": 0.2, "cer_norm": 0.1, "wer_raw": 0.3, "cer_raw": 0.15}
            p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
            rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(summarize_asr(rows)["models"]["openai/whisper-medium"]["langs"]["en"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
