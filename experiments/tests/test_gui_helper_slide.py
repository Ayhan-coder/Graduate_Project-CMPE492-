"""Unit tests for the Smart Helper Slide feature and /api/helper-slide endpoint."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from gui.pipeline import build_slide_specification
from gui.server import JOBS, app

client = TestClient(app)


class BuildSlideSpecificationTests(unittest.TestCase):
    def test_slide_spec_default_heuristic_and_cards(self):
        terms = [
            {"term": "Maliyet Yönetimi", "explanation": "Bütçe planlama süreci.", "timestamp": 12.5},
            {"term": "Sprint Süresi", "explanation": "2 haftalık geliştirme dönemi.", "timestamp": 25.0},
            {"term": "Scrum Takımı", "explanation": "Geliştiriciler ve ürün sahibi.", "timestamp": 30.0},
        ]
        spec = build_slide_specification(
            topic="Proje Yönetimi",
            recent_text="Maliyet ve sprint süreleri ele alınıyor.",
            active_terms=terms,
            lang="tr",
            use_llm=False,
        )

        self.assertIn("core_takeaway", spec)
        self.assertIn("cards", spec)
        self.assertIn("mermaid_diagram", spec)

        # Check keyword-based icon assignments
        cards = spec["cards"]
        self.assertEqual(len(cards), 3)
        self.assertEqual(cards[0]["icon"], "💰")  # Maliyet
        self.assertEqual(cards[1]["icon"], "⏱️")  # Sprint / süre
        self.assertEqual(cards[2]["icon"], "👥")  # Takım / scrum

    def test_slide_spec_domain_analogies_and_tips(self):
        # 1. EVM / Kazanılmış Değer
        spec_evm = build_slide_specification("Kazanılmış Değer Analizi (EVM)", "", [], lang="tr")
        self.assertIn("tuğla", spec_evm["analogy"])
        self.assertIn("CV = EV - AC", spec_evm["exam_tip"])

        # 2. Turizm / UKTKG
        spec_turizm = build_slide_specification("Turizm Sektörü ve UKTKG", "", [], lang="tr")
        self.assertIn("restoran", spec_turizm["analogy"])
        self.assertIn("UKTKG", spec_turizm["exam_tip"])

        # 3. Python / Veri
        spec_veri = build_slide_specification("Python ile Veri Ön İşleme", "", [], lang="tr")
        self.assertIn("rafineri", spec_veri["analogy"])
        self.assertIn("Data Preprocessing", spec_veri["exam_tip"])

        # 4. Agile / Scrum
        spec_agile = build_slide_specification("Agile ve Scrum Çerçevesi", "", [], lang="tr")
        self.assertIn("kısa bir sahne", spec_agile["analogy"])

    def test_slide_spec_mermaid_syntax(self):
        spec = build_slide_specification("Genel Kavram", "Girdi ve çıktılar", [], lang="tr")
        chart = spec.get("mermaid_diagram", "")
        self.assertTrue(chart.startswith("graph LR") or chart.startswith("graph TD"))
        self.assertIn("Girdi", chart)
        self.assertIn("Surec", chart)
        self.assertIn("Cikti", chart)

    def test_slide_spec_normalizes_string_terms(self):
        spec = build_slide_specification(
            topic="ML",
            recent_text="Neural network basics",
            active_terms=["Gradient Descent", "Loss Function"],
            lang="en",
        )
        self.assertEqual(len(spec["cards"]), 2)
        self.assertEqual(spec["cards"][0]["title"], "Gradient Descent")

    @patch("run_llm.get_or_load_llm")
    def test_slide_spec_with_mocked_deep_llm(self, mock_get_llm):
        custom_payload = {
            "title": "Deep Learning",
            "subtitle": "Visualized Neural Concepts",
            "core_takeaway": "AI Takeaway",
            "analogy": "Deep AI Analogy",
            "exam_tip": "High-yield exam tip",
            "cards": [{"icon": "🤖", "title": "Model", "desc": "Trained weights"}],
            "mermaid_diagram": "graph LR\nA-->B",
        }

        mock_tokenizer = MagicMock()
        mock_input_ids = MagicMock()
        mock_input_ids.shape = [1, 10]
        mock_tokenizer.return_value.to.return_value = MagicMock(input_ids=mock_input_ids)
        mock_tokenizer.decode.return_value = json.dumps(custom_payload)

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = [[0] * 20]

        mock_get_llm.return_value = (mock_model, mock_tokenizer)

        spec = build_slide_specification(
            topic="Deep Learning",
            recent_text="Lecture transcript...",
            active_terms=[{"term": "Transformer", "explanation": "Self-attention architecture"}],
            use_llm=True,
        )
        self.assertEqual(spec["core_takeaway"], "AI Takeaway")
        self.assertEqual(spec["analogy"], "Deep AI Analogy")
        self.assertEqual(spec["mermaid_diagram"], "graph LR\nA-->B")

    @patch("run_llm.get_or_load_llm", side_effect=RuntimeError("GPU OOM error"))
    def test_slide_spec_with_failing_llm_falls_back_gracefully(self, mock_get_llm):
        spec = build_slide_specification(
            topic="Kazanılmış Değer",
            recent_text="EVM lecture text",
            active_terms=[{"term": "EVM", "explanation": "Earned Value"}],
            use_llm=True,
        )
        # Should gracefully fall back to preset heuristic without raising NameError on logger
        self.assertIn("tuğla", spec["analogy"])
        self.assertIn("Kazan", spec["title"])


class HelperSlideApiEndpointTests(unittest.TestCase):
    def setUp(self):
        self.orig_jobs = dict(JOBS)
        JOBS.clear()

        self.job_id = "test-helper-slide-job"
        JOBS[self.job_id] = {
            "status": "completed",
            "result": {
                "job_id": self.job_id,
                "lang": "tr",
                "asr": {
                    "text": "Bu bölümde maliyet yönetimi ve kazanılmış değer analizi ele alınmaktadır.",
                    "segments": [
                        {"start": 0.0, "end": 15.0, "text": "Giriş yapıyoruz ve hedefleri belirliyoruz."},
                        {"start": 15.0, "end": 45.0, "text": "Maliyet yönetimi bütçenin planlanmasıdır."},
                        {"start": 45.0, "end": 80.0, "text": "Kazanılmış değer analizi projenin sağlığını ölçer."},
                    ],
                },
                "llm": {
                    "model": "Qwen/Qwen3.5-4B",
                    "terms": [
                        {"term": "Maliyet Yönetimi", "explanation": "Bütçe planlama süreci.", "start": 20.0},
                        {"term": "Kazanılmış Değer", "explanation": "EVM performans metriği.", "start": 60.0},
                    ],
                },
            },
        }

    def tearDown(self):
        JOBS.clear()
        JOBS.update(self.orig_jobs)

    @patch("gui.server.fetch_instant_transcript", side_effect=RuntimeError("no transcript"))
    def test_helper_slide_404_when_job_not_found(self, mock_fetch):
        JOBS.clear()
        res = client.post("/api/helper-slide", json={"job_id": "nonexistent-job-999", "current_time_s": 30.0})
        self.assertEqual(res.status_code, 404)

    def test_helper_slide_window_and_terms_matching(self):
        # At 70 seconds: window is [20.0, 75.0]
        # Should match segment 2 & 3 and terms 'Maliyet Yönetimi' (start 20.0) & 'Kazanılmış Değer' (start 60.0)
        res = client.post(
            "/api/helper-slide",
            json={
                "job_id": self.job_id,
                "current_time_s": 70.0,
                "generate_deep_slide": False,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertIn("topic", data)
        self.assertIn("time_range", data)
        self.assertEqual(data["current_time_s"], 70.0)
        self.assertEqual(data["rewind_time_s"], 60.0)  # max(0, t - 10)
        self.assertIn("core_summary", data)
        self.assertIn("slide_spec", data)
        self.assertGreaterEqual(len(data["terms"]), 1)

    def test_helper_slide_preview_job_fallback(self):
        # Even if job_id has 'preview_' prefix or vice versa, fallback works
        res = client.post(
            "/api/helper-slide",
            json={
                "job_id": f"preview_{self.job_id}",
                "current_time_s": 25.0,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["current_time_s"], 25.0)


if __name__ == "__main__":
    unittest.main()
