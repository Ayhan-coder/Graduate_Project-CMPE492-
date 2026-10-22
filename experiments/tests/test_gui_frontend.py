"""Unit tests for GUI frontend static assets and DOM contract integrity."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from gui.server import STATIC_DIR, app

client = TestClient(app)


class StaticFilesServingTests(unittest.TestCase):
    def test_serve_homepage(self):
        res = client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers.get("content-type", ""))
        self.assertIn("<title>EduTerm AI — Speech-to-Terminology Studio</title>", res.text)

    def test_serve_app_js(self):
        res = client.get("/static/app.js")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(any(t in res.headers.get("content-type", "") for t in ("javascript", "plain")))
        self.assertIn("DOMContentLoaded", res.text)

    def test_serve_style_css(self):
        res = client.get("/static/style.css")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/css", res.headers.get("content-type", ""))
        self.assertIn(":root", res.text)

    def test_serve_poster(self):
        res = client.get("/poster")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers.get("content-type", ""))
        self.assertIn("A3 Akademik Poster Sunumu", res.text)

    def test_download_poster_pdf(self):
        res = client.get("/api/download/poster-pdf")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("content-type", ""), "application/pdf")
        self.assertGreater(len(res.content), 10000)


class HtmlStructureTests(unittest.TestCase):
    def setUp(self):
        self.html_file = STATIC_DIR / "index.html"
        self.assertTrue(self.html_file.exists(), "index.html must exist in static directory")
        self.html_content = self.html_file.read_text(encoding="utf-8")

    def test_meta_and_fonts(self):
        self.assertIn('<meta name="viewport"', self.html_content)
        self.assertIn('<meta name="description"', self.html_content)
        self.assertIn('/static/style.css', self.html_content)
        self.assertIn('/static/app.js', self.html_content)

    def test_required_core_ids(self):
        essential_ids = [
            "tabYoutube",
            "tabUpload",
            "ytUrlInput",
            "btnFetchMeta",
            "videoPreviewCard",
            "previewThumb",
            "previewTitle",
            "previewAuthor",
            "previewDuration",
            "dropZone",
            "fileInput",
            "selectAsrModel",
            "selectLang",
            "selectLlmModel",
            "btnRunPipeline",
            "btnCancelJob",
            "pipelineProgress",
            "progressBarFill",
            "statusMessage",
            "logStream",
            "emptyState",
            "resultsView",
            "ytPlayerContainer",
            "htmlAudioContainer",
            "audioPlayer",
            "diagRtf",
            "diagAsrLat",
            "diagLlmLat",
            "diagTermsCount",
            "diagLang",
            "btnExportCsv",
            "btnExportJson",
            "btnCopyMd",
            "termsSearchInput",
            "termsGrid",
            "transcriptScroll",
        ]
        for el_id in essential_ids:
            self.assertIn(f'id="{el_id}"', self.html_content, f"Element ID '{el_id}' missing in index.html")

    def test_audio_scope_presets(self):
        presets = ['data-range="full"', 'data-range="120"', 'data-range="300"', 'data-range="600"', 'data-range="custom"']
        for preset in presets:
            self.assertIn(preset, self.html_content, f"Preset '{preset}' missing in index.html")

    def test_quick_demo_chips(self):
        self.assertIn("TR-1: Project Management", self.html_content)
        self.assertIn("EN-1: Agile Methodologies", self.html_content)
        self.assertIn("EN-3: Generative AI", self.html_content)


class JavascriptContractTests(unittest.TestCase):
    def test_dom_ids_referenced_in_js_exist_in_html(self):
        """Ensure every document.getElementById('...') call in app.js has a corresponding element in index.html."""
        html_path = STATIC_DIR / "index.html"
        js_path = STATIC_DIR / "app.js"

        html_content = html_path.read_text(encoding="utf-8")
        js_content = js_path.read_text(encoding="utf-8")

        # Find all document.getElementById('...') or document.getElementById("...")
        id_pattern = r'document\.getElementById\([\'"]([a-zA-Z0-9_-]+)[\'"]\)'
        referenced_ids = set(re.findall(id_pattern, js_content))

        # IDs created dynamically in JS (element.id = "...") exist only at runtime
        dynamic_id_pattern = r'\.id\s*=\s*[\'"]([a-zA-Z0-9_-]+)[\'"]'
        dynamic_ids = set(re.findall(dynamic_id_pattern, js_content))

        self.assertGreater(len(referenced_ids), 10, "Should find at least 10 getElementById calls")

        missing_ids = []
        for elem_id in referenced_ids:
            if f'id="{elem_id}"' not in html_content and elem_id not in dynamic_ids:
                missing_ids.append(elem_id)

        self.assertEqual(
            missing_ids,
            [],
            f"The following element IDs are referenced in app.js but do not exist in index.html: {missing_ids}",
        )

    def test_xss_protection_in_term_timestamps(self):
        """Ensure term timestamps and badges are HTML-escaped to prevent XSS injection."""
        js_path = STATIC_DIR / "app.js"
        js_content = js_path.read_text(encoding="utf-8")

        # Must have escapeHtml helper
        self.assertIn("function escapeHtml(", js_content)
        self.assertIn(".replace(/</g, \"&lt;\")", js_content)
        self.assertIn(".replace(/>/g, \"&gt;\")", js_content)

        # In renderTermsGrid, timestamp string must be passed through escapeHtml
        self.assertIn("escapeHtml(timeStr", js_content)

    def test_sse_defensive_parsing_and_client_timeout(self):
        """Ensure SSE listeners parse defensively with try/catch and enforce client timeout."""
        js_path = STATIC_DIR / "app.js"
        js_content = js_path.read_text(encoding="utf-8")

        # 15-minute client timeout
        self.assertIn("CLIENT_TIMEOUT_MS", js_content)
        self.assertIn("15 * 60 * 1000", js_content)

        # SSE onerror counts non-OK responses
        self.assertIn("reconnectFailures += 1", js_content)

        # renderResults has defensive try/catch and finally re-enables Run button
        self.assertIn("try {", js_content)
        self.assertIn("btnRun.disabled = false", js_content)

    def test_attention_popquiz_dom_contract(self):
        """Ensure attention pop-quiz controls and modals exist in index.html and app.js."""
        html_content = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        popquiz_ids = [
            "btnTestPopQuiz",
            "btnOpenResearchModal",
            "selectPopQuizMode",
            "popQuizModal",
            "popquizMilestoneBadge",
            "popquizTermName",
            "popquizPrompt",
            "popquizOptionsList",
            "popquizFeedbackBox",
            "btnResumeFromPopQuiz",
            "btnSkipPopQuiz",
            "attentionResearchModal",
            "btnCloseResearchModal",
        ]
        for el_id in popquiz_ids:
            self.assertIn(f'id="{el_id}"', html_content, f"Element ID '{el_id}' missing in index.html")

        # Functions in app.js
        js_path = STATIC_DIR / "app.js"
        js_content = js_path.read_text(encoding="utf-8")
        self.assertIn("function setupAttentionPopQuiz(", js_content)
        self.assertIn("function checkAttentionPopQuizTrigger(", js_content)
        self.assertIn("function triggerAttentionPopQuiz(", js_content)
        self.assertIn("setupAttentionPopQuiz()", js_content)

    def test_attention_research_api_endpoint(self):
        """Verify the research endpoint returns citations and default drop-off seconds."""
        res = client.get("/api/exam/attention-research")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("sources", data)
        self.assertIn("default_thresholds_s", data)
        self.assertEqual(data["default_thresholds_s"], [180, 300, 360, 600])

    def test_quick_answer_api_endpoint(self):
        """Verify quick answer updates learner profile."""
        res = client.post(
            "/api/exam/quick-answer",
            json={"term": "Kanban", "correct": True, "video_id": "test_vid"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["term"], "Kanban")
        self.assertGreaterEqual(data["mastery"], 0.5)


    def test_tab_and_watch_distraction_pause_dom_contract(self):
        """Ensure tab visibility and watch-pause controls exist in index.html and app.js."""
        html_content = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        required_ids = [
            "chkTabPause",
            "chkWatchPause",
            "chkAutoResume",
            "attentionPauseOverlay",
            "attentionAlertEmoji",
            "attentionAlertTitle",
            "attentionPauseReason",
            "attentionLiveMsg",
            "btnResumeAttentionVideo",
            "btnDismissAttention",
            "etTabSwitchCount",
        ]
        for el_id in required_ids:
            self.assertIn(f'id="{el_id}"', html_content, f"Element ID '{el_id}' missing in index.html")

        # Functions and listeners in app.js
        js_path = STATIC_DIR / "app.js"
        js_content = js_path.read_text(encoding="utf-8")
        self.assertIn("function setupTabVisibilityTracker(", js_content)
        self.assertIn("visibilitychange", js_content)
        self.assertIn("setupTabVisibilityTracker()", js_content)
        self.assertIn("showAttentionPauseOverlay(", js_content)
        self.assertIn("tabSwitchPausedMedia", js_content)


if __name__ == "__main__":
    unittest.main()

