"""Unit tests for the exam generation, scoring, and persistence in gui/exam.py."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gui.exam as exam
from gui.learner import empty_profile


class ExamPathAndPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.orig_exams_dir = os.environ.get("EDUTERM_EXAMS_DIR")
        os.environ["EDUTERM_EXAMS_DIR"] = self.temp_dir

    def tearDown(self):
        if self.orig_exams_dir is not None:
            os.environ["EDUTERM_EXAMS_DIR"] = self.orig_exams_dir
        else:
            os.environ.pop("EDUTERM_EXAMS_DIR", None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_exam_path_sanitization(self):
        path = exam.exam_path("valid_exam-123")
        self.assertEqual(path.name, "valid_exam-123.json")
        self.assertEqual(path.parent, Path(self.temp_dir))

        # Strips dangerous path traversal / special characters
        path_dirty = exam.exam_path("../../etc/passwd!@#")
        self.assertEqual(path_dirty.name, "etcpasswd.json")

    def test_exam_path_empty_raises_value_error(self):
        with self.assertRaises(ValueError):
            exam.exam_path("")
        with self.assertRaises(ValueError):
            exam.exam_path("   ")
        with self.assertRaises(ValueError):
            exam.exam_path("!@#$%^&*()")

    def test_save_and_load_exam_roundtrip(self):
        data = {
            "exam_id": "test-exam-001",
            "source": "fallback",
            "lang": "en",
            "questions": [
                {
                    "id": "q1",
                    "type": "flashcard",
                    "term": "Agile",
                    "front": "Agile",
                    "back": "Iterative software development approach.",
                }
            ],
            "submitted": False,
        }
        exam.save_exam(data)

        loaded = exam.load_exam("test-exam-001")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["exam_id"], "test-exam-001")
        self.assertEqual(len(loaded["questions"]), 1)
        self.assertEqual(loaded["questions"][0]["term"], "Agile")

    def test_load_exam_nonexistent_or_corrupt(self):
        self.assertIsNone(exam.load_exam("nonexistent-id"))
        self.assertIsNone(exam.load_exam("!invalid!"))

        # Corrupt file
        bad_file = Path(self.temp_dir) / "corrupt.json"
        bad_file.write_text("NOT A VALID JSON", encoding="utf-8")
        self.assertIsNone(exam.load_exam("corrupt"))


class ExamParsingAndNormalizationTests(unittest.TestCase):
    def test_parse_exam_questions_valid_json(self):
        raw_json = json.dumps([
            {
                "id": "q1",
                "type": "flashcard",
                "term": "Sprint",
                "front": "What is a Sprint?",
                "back": "A timeboxed iteration.",
            },
            {
                "id": "q2",
                "type": "mcq",
                "term": "Scrum Master",
                "prompt": "What is the role of a Scrum Master?",
                "options": ["Facilitator", "CEO", "Programmer", "Investor"],
                "answer_index": 0,
            },
        ])
        questions = exam.parse_exam_questions(raw_json)
        self.assertEqual(len(questions), 2)
        self.assertEqual(questions[0]["type"], "flashcard")
        self.assertEqual(questions[1]["type"], "mcq")

    def test_parse_exam_questions_embedded_in_markdown(self):
        text = (
            "Here is the generated exam questions:\n"
            "```json\n"
            "[\n"
            '  {"id": "q1", "type": "flashcard", "term": "CI/CD", "front": "CI/CD", "back": "Continuous Integration"}\n'
            "]\n"
            "```\n"
            "Good luck with your exam!"
        )
        questions = exam.parse_exam_questions(text)
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["term"], "CI/CD")

    def test_parse_exam_questions_invalid(self):
        self.assertEqual(exam.parse_exam_questions(""), [])
        self.assertEqual(exam.parse_exam_questions("No brackets here"), [])
        self.assertEqual(exam.parse_exam_questions("[Not valid json]"), [])
        self.assertEqual(exam.parse_exam_questions('{"not": "a list"}'), [])

    def test_normalize_question_mcq(self):
        # Valid MCQ
        raw = {
            "type": "mcq",
            "term": "Kanban",
            "prompt": "What is Kanban?",
            "options": ["Visual workflow board", "Database engine"],
            "answer_index": 0,
        }
        q = exam._normalize_question(raw, 1)
        self.assertIsNotNone(q)
        self.assertEqual(q["type"], "mcq")
        self.assertEqual(len(q["options"]), 4)  # Padded to 4
        self.assertEqual(q["options"][-1], "Database engine")
        self.assertEqual(q["answer_index"], 0)

        # MCQ invalid answer_index
        bad_idx = dict(raw, answer_index=99)
        self.assertIsNone(exam._normalize_question(bad_idx, 1))

        # MCQ missing options
        no_opts = dict(raw, options=[])
        self.assertIsNone(exam._normalize_question(no_opts, 1))

    def test_normalize_question_flashcard(self):
        raw = {
            "type": "flashcard",
            "term": "WBS",
            "front": "What does WBS stand for?",
            "back": "Work Breakdown Structure",
        }
        q = exam._normalize_question(raw, 1)
        self.assertIsNotNone(q)
        self.assertEqual(q["type"], "flashcard")
        self.assertEqual(q["back"], "Work Breakdown Structure")

        # Missing back explanation
        self.assertIsNone(exam._normalize_question({"type": "flashcard", "front": "Test", "back": ""}, 1))

        # Unknown question type
        self.assertIsNone(exam._normalize_question({"type": "essay", "prompt": "Write essay"}, 1))


class PrioritizeTermsAndFallbackExamTests(unittest.TestCase):
    def test_prioritize_terms_orders_by_mastery_and_focus(self):
        profile = empty_profile()
        profile["terms"] = {
            "known term": {"term": "Known Term", "mastery": 0.9, "seen": 5},
            "weak term": {"term": "Weak Term", "mastery": 0.2, "seen": 3},
            "learning term": {"term": "Learning Term", "mastery": 0.6, "seen": 2},
        }

        terms = [
            {"term": "Known Term", "explanation": "Already well known."},
            {"term": "New Term", "explanation": "Never seen before."},
            {"term": "Weak Term", "explanation": "Needs urgent review."},
            {"term": "Learning Term", "explanation": "Currently in progress."},
        ]

        prioritized = exam.prioritize_terms(terms, profile, limit=4)
        term_names = [t["term"] for t in prioritized]

        # Weak should come first, then learning, then new, then known
        self.assertEqual(term_names[0], "Weak Term")
        self.assertEqual(term_names[1], "Learning Term")
        self.assertEqual(term_names[2], "New Term")
        self.assertEqual(term_names[3], "Known Term")

    def test_prioritize_terms_deduplicates(self):
        terms = [
            {"term": "Agile", "explanation": "Expl 1"},
            {"term": "agile", "explanation": "Expl 2"},
            {"term": "AGILE", "explanation": "Expl 3"},
        ]
        res = exam.prioritize_terms(terms, None, limit=5)
        self.assertEqual(len(res), 1)

    def test_fallback_exam_from_terms_en_and_tr(self):
        terms_en = [
            {"term": f"Term {i}", "explanation": f"Explanation for term {i}"}
            for i in range(1, 7)
        ]
        exam_en = exam.fallback_exam_from_terms(terms_en, lang="en", limit=6)
        self.assertEqual(len(exam_en), 6)

        # Flashcards predominate; every 3rd question is MCQ
        types = [q["type"] for q in exam_en]
        self.assertEqual(types[0], "flashcard")
        self.assertEqual(types[1], "flashcard")
        self.assertEqual(types[2], "mcq")
        self.assertEqual(types[3], "flashcard")
        self.assertEqual(types[4], "flashcard")
        self.assertEqual(types[5], "mcq")

        # Turkish prompt check
        terms_tr = [
            {"term": f"Terim {i}", "explanation": f"Açıklama {i}"}
            for i in range(1, 5)
        ]
        exam_tr = exam.fallback_exam_from_terms(terms_tr, lang="tr", limit=4)
        mcq = [q for q in exam_tr if q["type"] == "mcq"][0]
        self.assertIn("ne anlama gelir?", mcq["prompt"])

    def test_fallback_exam_empty_terms(self):
        self.assertEqual(exam.fallback_exam_from_terms([], lang="en"), [])


class GeneratePracticeExamLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.orig_exams_dir = os.environ.get("EDUTERM_EXAMS_DIR")
        os.environ["EDUTERM_EXAMS_DIR"] = self.temp_dir

    def tearDown(self):
        if self.orig_exams_dir is not None:
            os.environ["EDUTERM_EXAMS_DIR"] = self.orig_exams_dir
        else:
            os.environ.pop("EDUTERM_EXAMS_DIR", None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_practice_exam_empty_raises(self):
        with self.assertRaises(ValueError):
            exam.generate_practice_exam([], lang="en")
        with self.assertRaises(ValueError):
            exam.generate_practice_exam([{"term": "", "explanation": ""}], lang="en")

    def test_generate_practice_exam_deterministic_fallback(self):
        terms = [
            {"term": "EVM", "explanation": "Earned Value Management."},
            {"term": "WBS", "explanation": "Work Breakdown Structure."},
            {"term": "CPM", "explanation": "Critical Path Method."},
        ]
        cb_msgs = []
        result = exam.generate_practice_exam(
            terms,
            transcript="We discuss EVM and WBS in project management.",
            lang="en",
            use_llm=False,
            progress_cb=lambda msg: cb_msgs.append(msg),
        )

        self.assertIn("exam_id", result)
        self.assertEqual(result["source"], "fallback")
        self.assertIsNone(result["model_id"])
        self.assertFalse(result["submitted"])
        self.assertEqual(len(result["questions"]), 3)
        self.assertTrue(any("flashcard" in msg for msg in cb_msgs))

    @patch("run_llm.generate_hf")
    def test_generate_practice_exam_with_mocked_llm(self, mock_generate_hf):
        mock_generate_hf.return_value = json.dumps([
            {
                "id": "q1",
                "type": "flashcard",
                "term": "EVM",
                "front": "What is EVM?",
                "back": "Earned Value Management",
            }
        ])
        terms = [{"term": "EVM", "explanation": "Earned Value Management."}]
        result = exam.generate_practice_exam(
            terms,
            transcript="Lecture excerpt...",
            lang="en",
            model_id="Qwen/Qwen3.5-4B",
            use_llm=True,
        )
        self.assertEqual(result["source"], "llm")
        self.assertEqual(result["model_id"], "Qwen/Qwen3.5-4B")
        self.assertEqual(len(result["questions"]), 1)

    @patch("run_llm.generate_hf", side_effect=RuntimeError("GPU OOM"))
    def test_generate_practice_exam_falls_back_when_llm_fails(self, mock_generate_hf):
        terms = [
            {"term": "Sprint", "explanation": "Short cycle."},
            {"term": "Scrum", "explanation": "Agile framework."},
        ]
        result = exam.generate_practice_exam(
            terms,
            transcript="Agile talk...",
            lang="en",
            use_llm=True,
        )
        # Should gracefully fall back
        self.assertEqual(result["source"], "fallback")
        self.assertEqual(len(result["questions"]), 2)


class PublicQuestionsAndScoringTests(unittest.TestCase):
    def test_public_questions_strips_scoring_secrets(self):
        questions = [
            {
                "id": "q1",
                "type": "mcq",
                "term": "Agile",
                "prompt": "What is Agile?",
                "options": ["A methodology", "A drink", "A car", "A building"],
                "answer_index": 0,
            },
            {
                "id": "q2",
                "type": "flashcard",
                "term": "Waterfall",
                "front": "Waterfall",
                "back": "Sequential development model",
            },
        ]
        public_qs = exam.public_questions(questions)
        self.assertEqual(len(public_qs), 2)

        # MCQ question has options but MUST NOT leak answer_index
        self.assertNotIn("answer_index", public_qs[0])
        self.assertEqual(public_qs[0]["options"], ["A methodology", "A drink", "A car", "A building"])

        # Flashcard maintains front and back for client-side flip
        self.assertEqual(public_qs[1]["front"], "Waterfall")
        self.assertEqual(public_qs[1]["back"], "Sequential development model")

    def test_score_exam_mcq_and_flashcards(self):
        questions = [
            {
                "id": "q1",
                "type": "mcq",
                "term": "EVM",
                "prompt": "What is EVM?",
                "options": ["Wrong 1", "Correct EVM", "Wrong 2", "Wrong 3"],
                "answer_index": 1,
            },
            {
                "id": "q2",
                "type": "mcq",
                "term": "WBS",
                "prompt": "What is WBS?",
                "options": ["Wrong 1", "Wrong 2", "Correct WBS", "Wrong 3"],
                "answer_index": 2,
            },
            {
                "id": "q3",
                "type": "flashcard",
                "term": "Scrum",
                "front": "Scrum",
                "back": "Agile framework",
            },
            {
                "id": "q4",
                "type": "flashcard",
                "term": "Kanban",
                "front": "Kanban",
                "back": "Visual signal board",
            },
        ]

        answers = [
            {"question_id": "q1", "selected_index": 1},  # Correct MCQ
            {"question_id": "q2", "selected_index": 0},  # Wrong MCQ
            {"question_id": "q3", "self_grade": "know"},  # Correct Flashcard
            {"question_id": "q4", "self_grade": "review"},  # Missed Flashcard
        ]

        scored = exam.score_exam(questions, answers)
        self.assertEqual(scored["total"], 4)
        self.assertEqual(scored["correct"], 2)
        self.assertEqual(scored["score"], 0.5)

        self.assertIn("WBS", scored["weak_terms"])
        self.assertIn("Kanban", scored["weak_terms"])
        self.assertIn("EVM", scored["strong_terms"])
        self.assertIn("Scrum", scored["strong_terms"])

    def test_analysis_after_exam_feedback_messages(self):
        # Empty
        self.assertEqual(exam.analysis_after_exam({"total": 0}, {}), "No answers were recorded.")

        # Perfect score
        scored_perfect = {
            "total": 3,
            "correct": 3,
            "score": 1.0,
            "weak_terms": [],
            "strong_terms": ["A", "B", "C"],
        }
        text_perfect = exam.analysis_after_exam(scored_perfect, {"exams_taken": 2, "avg_score": 0.95})
        self.assertIn("100%", text_perfect)
        self.assertIn("All items were marked correct", text_perfect)
        self.assertIn("2 exam(s)", text_perfect)

        # With missed terms
        scored_weak = {
            "total": 4,
            "correct": 2,
            "score": 0.5,
            "weak_terms": ["Machine Learning", "Backprop"],
            "strong_terms": ["Perceptron", "Epoch"],
        }
        text_weak = exam.analysis_after_exam(scored_weak, {})
        self.assertIn("50%", text_weak)
        self.assertIn("Missed terms", text_weak)
        self.assertIn("Machine Learning", text_weak)


if __name__ == "__main__":
    unittest.main()
