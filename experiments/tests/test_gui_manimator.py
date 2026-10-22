"""Unit tests for the Manimator AI Narrated Video Generator module and API endpoints."""

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
from gui.manimator import (
    _clean_json_output,
    _heal_manim_code,
    generate_default_manim_code,
    generate_explainer_script,
    generate_fallback_explainer_spec,
    produce_student_explainer_video,
)
from gui.server import JOBS, app

client = TestClient(app)


class ManimatorScriptGenerationTests(unittest.TestCase):
    def test_clean_json_output_with_thinking_and_markdown(self):
        raw = """
        <think>Bu bir akıl yürütme bölümüdür.</think>
        ```json
        {
          "title": "Test Title",
          "target_concept": "Test Concept",
          "pedagogical_hook": "Bir benzetme",
          "terms": [{"term": "T1", "explanation": "E1"}],
          "narration_script": "Seslendirme metni",
          "scenes": [{"id": 1, "title": "Giris", "narration": "S1", "duration_s": 10.0}],
          "manim_code": "from manim import *\\nclass ExplainerScene(Scene): pass"
        }
        ```
        """
        parsed = _clean_json_output(raw)
        self.assertEqual(parsed["title"], "Test Title")
        self.assertEqual(len(parsed["terms"]), 1)
        self.assertIn("ExplainerScene", parsed["manim_code"])

    def test_fallback_explainer_spec_evm(self):
        spec = generate_fallback_explainer_spec("Kazanılmış Değer Analizi (EVM)", "EVM sapma formülleri", "tr")
        self.assertIn("Kazanılmış Değer", spec["title"])
        self.assertIn("tuğla", spec["pedagogical_hook"])
        self.assertTrue(len(spec["terms"]) >= 3)
        self.assertIn("CV eşittir EV eksi AC", spec["narration_script"])
        self.assertTrue(len(spec["scenes"]) >= 3)
        self.assertIn("class ExplainerScene(Scene):", spec["manim_code"])

    def test_fallback_explainer_spec_turizm(self):
        spec = generate_fallback_explainer_spec("Turizm ve UKTKG Modeli", "", "tr")
        self.assertIn("UKTKG", spec["title"])
        self.assertIn("restoran", spec["pedagogical_hook"])
        self.assertIn("stoklanamaz", spec["narration_script"])

    def test_default_manim_code_valid_python_syntax(self):
        terms = [
            {"term": "Planlanan Değer (PV)", "explanation": "Hedef bütçe."},
            {"term": "Gerçekleşen Maliyet (AC)", "explanation": "Fiili harcama."},
        ]
        code = generate_default_manim_code(
            topic="Kazanılmış Değer Analizi",
            analogy="Duvar ustası örneği",
            terms=terms,
            lang="tr",
        )
        self.assertIn("class ExplainerScene(Scene):", code)
        self.assertIn("self.camera.background_color", code)
        self.assertIn("Text", code)
        # Verify valid Python syntax by compiling
        compiled = compile(code, "<string>", "exec")
        self.assertIsNotNone(compiled)

    def test_manim_code_self_healing(self):
        broken_code = 'Text("Unbalanced quote)'
        healed = _heal_manim_code(broken_code, "SyntaxError: EOL while scanning string literal")
        self.assertIn("from manim import *", healed)
        self.assertTrue(healed.endswith('"'))

        latex_code = 'MathTex(r"\\frac{a}{b}")'
        healed_latex = _heal_manim_code(latex_code, "LaTeX error: xdvipdfmx failed")
        self.assertIn("Text(", healed_latex)

    def test_evren_backend_authors_script(self):
        """A successful EVREN call is used verbatim and attributed to the evren backend."""
        payload = json.dumps({
            "title": "Kazanılmış Değer",
            "target_concept": "EVM",
            "pedagogical_hook": "Duvar ustası benzetmesi",
            "terms": [{"term": "PV", "explanation": "Hedef bütçe"}],
            "narration_script": "EVREN'in yazdığı anlatım metni",
            "scenes": [{"id": 1, "title": "Giriş", "narration": "Giriş", "duration_s": 12.0}],
            "manim_code": "from manim import *\n\n\nclass ExplainerScene(Scene):\n    def construct(self):\n        pass\n",
        }, ensure_ascii=False)
        with patch("evren_llm.generate_evren", return_value=payload) as mock_evren:
            spec = generate_explainer_script("Kazanılmış Değer", "EV ile AC farkı", "tr", 50, backend="evren")
        self.assertTrue(mock_evren.called)
        self.assertEqual(spec["script_backend"], "evren")
        self.assertEqual(spec["narration_script"], "EVREN'in yazdığı anlatım metni")
        self.assertIn("ExplainerScene", spec["manim_code"])

    def test_evren_failure_degrades_to_template_not_another_model(self):
        """EVREN being down must not silently relabel a local run as the chosen engine."""
        with patch("evren_llm.generate_evren", side_effect=RuntimeError("platform unreachable")):
            spec = generate_explainer_script("Turizm ve UKTKG Modeli", "", "tr", 50, backend="evren")
        self.assertEqual(spec["script_backend"], "heuristic")
        self.assertIn("UKTKG", spec["title"])


class ManimatorApiEndpointsTests(unittest.TestCase):
    def setUp(self):
        JOBS.clear()
        JOBS["test_job_1"] = {
            "status": "completed",
            "result": {
                "job_id": "test_job_1",
                "lang": "tr",
                "asr": {
                    "text": "Bugün projelerde maliyet sapması ve kazanılmış değer analizi formüllerini ele alıyoruz.",
                    "segments": [
                        {"start": 0.0, "end": 15.0, "text": "Bugün projelerde maliyet sapması ve kazanılmış değer analizi formüllerini ele alıyoruz."},
                        {"start": 15.0, "end": 35.0, "text": "Planlanan değer PV ile gerçek maliyet AC arasındaki fark hesaplanır."},
                    ],
                },
                "llm": {
                    "terms": [
                        {"term": "Maliyet Sapması (CV)", "explanation": "EV ile AC arasındaki fark.", "start": 5.0},
                        {"term": "Planlanan Değer (PV)", "explanation": "Hedef bütçe.", "start": 18.0},
                    ],
                },
            },
        }

    @patch("gui.server.produce_student_explainer_video")
    def test_generate_manimator_video_api(self, mock_produce):
        mock_produce.return_value = {
            "video_id": "explainer_mock_123",
            "video_url": "/static/generated_videos/explainer_mock_123.mp4",
            "title": "Maliyet Sapması (CV)",
            "target_concept": "Maliyet Sapması (CV)",
            "pedagogical_hook": "Duvar ustası benzetmesi",
            "terms": [{"term": "Maliyet Sapması (CV)", "explanation": "Açıklama"}],
            "narration_script": "Seslendirme metni...",
            "scenes": [{"id": 1, "title": "Giriş", "narration": "Giriş anlatımı", "duration_s": 10.0}],
            "subtitles": [{"start": 0.0, "end": 10.0, "title": "Giriş", "text": "Giriş metni"}],
            "manim_code": "class ExplainerScene(Scene): pass",
            "duration_s": 25.0,
            "generation_time_s": 5.2,
            "video_style": "manim_animation",
        }

        resp = client.post(
            "/api/manimator/generate",
            json={
                "job_id": "test_job_1",
                "current_time_s": 10.0,
                "video_style": "manim_animation",
                "duration_s": 35,
                "llm_backend": "heuristic",
                "video_quality": "high",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["video_id"], "explainer_mock_123")
        self.assertEqual(data["title"], "Maliyet Sapması (CV)")
        self.assertIn(".mp4", data["video_url"])

    def test_project_management_explainer_spec_alignment(self):
        """Verify that topic 'Proje Yönetimi' produces a dedicated tailored explainer matching the lecture."""
        spec = generate_fallback_explainer_spec("Proje Yönetimi", context="ortaya çıkan problemlerin çözümü için önerilen alternatifler")
        self.assertEqual(spec["title"], "Proje Yönetimi ve Karar Verme")
        term_names = [t["term"] for t in spec["terms"]]
        self.assertIn("Proje Yöneticisi", term_names)
        self.assertIn("Alternatiflerin Değerlendirilmesi", term_names)
        self.assertIn("alternatif", spec["narration_script"].lower())

    @patch("gui.server.produce_student_explainer_video")
    def test_generate_manimator_video_honors_client_topic_and_context(self, mock_produce):
        """Verify that API honors topic and context passed by client without drift."""
        mock_produce.return_value = {
            "video_id": "explainer_pm_1",
            "video_url": "/static/generated_videos/explainer_pm_1.mp4",
            "title": "Proje Yönetimi ve Karar Verme",
            "target_concept": "Proje Yönetimi",
            "pedagogical_hook": "Orkestra şefi benzetmesi",
            "terms": [{"term": "Proje Yöneticisi", "explanation": "Açıklama"}],
            "narration_script": "Seslendirme...",
            "scenes": [{"id": 1, "title": "Giriş", "narration": "Giriş", "duration_s": 10.0}],
            "subtitles": [{"start": 0.0, "end": 10.0, "title": "Giriş", "text": "Giriş"}],
            "manim_code": "class ExplainerScene(Scene): pass",
            "duration_s": 35.0,
            "generation_time_s": 2.1,
            "video_style": "manim_animation",
        }

        resp = client.post(
            "/api/manimator/generate",
            json={
                "job_id": "test_job_1",
                "current_time_s": 85.0,
                "topic": "Proje Yönetimi",
                "context": "ortaya çıkan problemlerin çözümü için önerilen alternatiflerden hangisinin proje amaçlarına daha uygun olduğuna karar vermek durumundadır.",
                "video_style": "manim_animation",
                "duration_s": 35,
                "llm_backend": "heuristic",
                "video_quality": "high",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["target_concept"], "Proje Yönetimi")
        self.assertIn("Proje", data["title"])
        mock_produce.assert_called_once()
        call_kwargs = mock_produce.call_args[1]
        self.assertEqual(call_kwargs["topic"], "Proje Yönetimi")
        self.assertIn("problemlerin", call_kwargs["context"])
        self.assertEqual(call_kwargs["backend"], "heuristic")
        self.assertEqual(call_kwargs["video_quality"], "high")


class NarrationEngineTests(unittest.TestCase):
    """The reported voice must be the one that actually produced the audio."""

    def _run(self, edge_error, google_ok, sapi_voice):
        import gui.manimator as manimator

        with patch.object(manimator, "_EDGE_RETRY_WAS_S", (0.0,)), \
             patch.object(manimator, "_synthesize_edge_tts_async",
                          side_effect=edge_error or (lambda *a, **k: None)), \
             patch.object(manimator, "_synthesize_with_google_tts", return_value=google_ok), \
             patch.object(manimator, "_synthesize_with_windows_sapi", return_value=sapi_voice):
            return manimator.synthesize_narration_audio(
                "Kazanılmış Değer Analizi bir pusuladır.", lang="tr",
                output_filename="narration_test.mp3",
            )

    def test_edge_voice_is_preferred(self):
        async def fake(text, voice, out_path):
            Path(out_path).write_bytes(b"x" * 2000)

        import gui.manimator as manimator

        with patch.object(manimator, "_EDGE_RETRY_WAS_S", (0.0,)), \
             patch.object(manimator, "_synthesize_edge_tts_async", fake), \
             patch.object(manimator, "_synthesize_with_google_tts", return_value=False), \
             patch.object(manimator, "_synthesize_with_windows_sapi",
                          return_value="Microsoft Zira Desktop (en-US)"):
            path, info = manimator.synthesize_narration_audio(
                "Kazanılmış Değer Analizi bir pusuladır.", lang="tr",
                output_filename="narration_test.mp3",
            )
        self.assertEqual(info["narration_engine"], "edge-tts")
        self.assertEqual(info["narration_voice"], "tr-TR-AhmetNeural")

    def test_google_tts_before_mismatched_system_voice(self):
        path, info = self._run(RuntimeError("throttled"), True, "Microsoft Zira Desktop (en-US)")
        self.assertEqual(info["narration_engine"], "google-tts")

    def test_system_voice_only_when_no_online_voice_works(self):
        path, info = self._run(RuntimeError("throttled"), False, "Microsoft Zira Desktop (en-US)")
        self.assertEqual(info["narration_engine"], "sapi5-offline")
        self.assertEqual(info["narration_voice"], "Microsoft Zira Desktop (en-US)")

    def test_silent_is_reported_as_silent(self):
        path, info = self._run(RuntimeError("throttled"), False, None)
        self.assertEqual(info["narration_engine"], "silent")
        self.assertEqual(info["narration_voice"], "")


if __name__ == "__main__":
    unittest.main()
