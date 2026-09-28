"""Unit tests for graduation thesis defense presentation package."""

from __future__ import annotations

import re
import socket
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PRESENTATION_DIR = ROOT / "presentation"


class TestPresentationPackage(unittest.TestCase):
    """Test suite ensuring defense presentation files are complete and consistent."""

    def test_presentation_files_exist(self):
        """Ensure all required presentation files exist."""
        self.assertTrue((PRESENTATION_DIR / "index.html").exists(), "index.html must exist")
        self.assertTrue((PRESENTATION_DIR / "slides.md").exists(), "slides.md must exist")
        self.assertTrue((PRESENTATION_DIR / "DEFENSE_QA_GUIDE.md").exists(), "DEFENSE_QA_GUIDE.md must exist")
        self.assertTrue((PRESENTATION_DIR / "serve_presentation.py").exists(), "serve_presentation.py must exist")
        self.assertTrue((ROOT / "run_presentation.bat").exists(), "run_presentation.bat must exist")

    def test_index_html_has_all_16_slides(self):
        """Ensure index.html contains exactly 16 slides."""
        html_content = (PRESENTATION_DIR / "index.html").read_text(encoding="utf-8")
        slide_matches = re.findall(r'<section class="slide', html_content)
        self.assertEqual(len(slide_matches), 16, f"Expected 16 slides in index.html, found {len(slide_matches)}")

    def test_index_html_has_scientific_guardrails(self):
        """Ensure critical scientific guardrails are present in index.html."""
        html_content = (PRESENTATION_DIR / "index.html").read_text(encoding="utf-8")
        self.assertIn("Guardrail 1", html_content)
        self.assertIn("Guardrail 2", html_content)
        self.assertIn("Guardrail 3", html_content)
        self.assertIn("Guardrail 4", html_content)
        self.assertIn("Guardrail 5", html_content)
        self.assertIn("Guardrail 6", html_content)
        self.assertIn("Prevalence Paradox", html_content)

    def test_index_html_has_exact_benchmark_numbers(self):
        """Verify key benchmark figures align with IEEE paper in index.html."""
        html_content = (PRESENTATION_DIR / "index.html").read_text(encoding="utf-8")
        # Whisper Large-v3 TR & EN WER
        self.assertIn("0.021", html_content)
        self.assertIn("0.010", html_content)
        # Whisper Turbo RTF
        self.assertIn("0.044", html_content)
        self.assertIn("0.025", html_content)
        # Cohen's Kappa
        self.assertIn("0.782", html_content)
        self.assertIn("1.000", html_content)
        # PABAK
        self.assertIn("0.948", html_content)
        # 213 unit tests
        self.assertIn("213", html_content)

    def test_slides_md_marp_frontmatter_and_slides(self):
        """Verify slides.md has valid Marp structure and 16 slide sections."""
        md_content = (PRESENTATION_DIR / "slides.md").read_text(encoding="utf-8")
        self.assertTrue(md_content.startswith("---"), "slides.md must start with frontmatter")
        self.assertIn("marp: true", md_content)
        self.assertIn("EduTerm AI", md_content)

        # Count slides separated by '---'
        # The first '---' is frontmatter start, second is frontmatter end.
        sections = md_content.split("\n---\n")
        # sections[0] is empty/frontmatter, subsequent items are slides
        self.assertGreaterEqual(len(sections), 16, f"Expected at least 16 slides, got {len(sections)}")

    def test_defense_qa_guide_content(self):
        """Ensure DEFENSE_QA_GUIDE.md covers 15 critical questions and 7 guardrails."""
        guide_content = (PRESENTATION_DIR / "DEFENSE_QA_GUIDE.md").read_text(encoding="utf-8")
        for i in range(1, 16):
            self.assertIn(f"Soru {i}:", guide_content, f"Missing Soru {i} in DEFENSE_QA_GUIDE.md")

        for g in range(1, 8):
            self.assertIn(f"Guardrail {g}", guide_content, f"Missing Guardrail {g} in DEFENSE_QA_GUIDE.md")

    def test_serve_presentation_port_finder(self):
        """Test that find_free_port returns a valid bindable port."""
        from presentation.serve_presentation import find_free_port

        port = find_free_port(8900)
        self.assertIsInstance(port, int)
        self.assertGreaterEqual(port, 8900)

        # Check that we can indeed bind to it
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))


if __name__ == "__main__":
    unittest.main()
