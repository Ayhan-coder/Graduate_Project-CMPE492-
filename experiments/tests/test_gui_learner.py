"""Unit tests for learner profile tracking, mastery computation, and adaptive weights in gui/learner.py."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gui.learner as learner


class LearnerPersistenceAndNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.profile_path = Path(self.temp_dir) / "profile.json"
        self.orig_env = os.environ.get("EDUTERM_LEARNER_PATH")
        os.environ["EDUTERM_LEARNER_PATH"] = str(self.profile_path)

    def tearDown(self):
        if self.orig_env is not None:
            os.environ["EDUTERM_LEARNER_PATH"] = self.orig_env
        else:
            os.environ.pop("EDUTERM_LEARNER_PATH", None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_normalize_term(self):
        self.assertEqual(learner.normalize_term("  Sprint   Planning  "), "sprint planning")
        self.assertEqual(learner.normalize_term("KANBAN"), "kanban")
        self.assertEqual(learner.normalize_term(""), "")
        self.assertEqual(learner.normalize_term(None), "")

    def test_empty_profile_structure(self):
        prof = learner.empty_profile()
        self.assertEqual(prof["learner_id"], "local")
        self.assertIsNone(prof["updated_at"])
        self.assertEqual(prof["terms"], {})
        self.assertEqual(prof["exams"], [])
        self.assertEqual(prof["stats"]["exams_taken"], 0)
        self.assertEqual(prof["stats"]["avg_score"], 0.0)
        self.assertIsNone(prof["stats"]["last_score"])
        self.assertEqual(prof["stats"]["weak_count"], 0)

    def test_load_profile_when_missing_or_corrupt(self):
        # Missing file returns empty profile
        prof = learner.load_profile()
        self.assertEqual(prof["stats"]["exams_taken"], 0)

        # Corrupt JSON file returns empty profile
        self.profile_path.write_text("CORRUPTED NOT JSON", encoding="utf-8")
        prof = learner.load_profile()
        self.assertEqual(prof["stats"]["exams_taken"], 0)

        # Non-dict JSON returns empty profile
        self.profile_path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
        prof = learner.load_profile()
        self.assertEqual(prof["stats"]["exams_taken"], 0)

    def test_save_and_load_profile_roundtrip(self):
        prof = learner.empty_profile()
        prof["learner_id"] = "student-42"
        prof["terms"]["scrum"] = {
            "term": "Scrum",
            "seen": 3,
            "correct": 3,
            "incorrect": 0,
            "mastery": 0.85,
        }
        learner.save_profile(prof)

        loaded = learner.load_profile()
        self.assertEqual(loaded["learner_id"], "student-42")
        self.assertIn("scrum", loaded["terms"])
        self.assertIsNotNone(loaded["updated_at"])


class MasteryClassificationAndUpdateTests(unittest.TestCase):
    def test_classify_mastery_thresholds(self):
        # Seen <= 0 or mastery is None -> "new"
        self.assertEqual(learner.classify_mastery(None, 0), "new")
        self.assertEqual(learner.classify_mastery(0.9, 0), "new")

        # Mastery < 0.45 -> "review"
        self.assertEqual(learner.classify_mastery(0.44, 1), "review")
        self.assertEqual(learner.classify_mastery(0.10, 5), "review")

        # Mastery >= 0.75 -> "known"
        self.assertEqual(learner.classify_mastery(0.75, 2), "known")
        self.assertEqual(learner.classify_mastery(0.95, 3), "known")

        # In-between -> "learning"
        self.assertEqual(learner.classify_mastery(0.45, 1), "learning")
        self.assertEqual(learner.classify_mastery(0.60, 2), "learning")
        self.assertEqual(learner.classify_mastery(0.74, 3), "learning")

    def test_update_mastery_correct_math(self):
        prof = learner.empty_profile()
        # Initial mastery = 0.5. Gain = 0.30 of (1.0 - mastery).
        # Expected new = 0.5 + 0.5 * 0.30 = 0.65
        rec = learner.update_mastery(prof, "Backpropagation", correct=True, video_id="vid1")
        self.assertEqual(rec["seen"], 1)
        self.assertEqual(rec["correct"], 1)
        self.assertEqual(rec["incorrect"], 0)
        self.assertTrue(rec["last_correct"])
        self.assertAlmostEqual(rec["mastery"], 0.65, places=3)
        self.assertEqual(rec["focus"], "learning")
        self.assertIn("vid1", rec["videos"])

        # Second correct answer: 0.65 + (1 - 0.65)*0.30 = 0.65 + 0.105 = 0.755
        rec2 = learner.update_mastery(prof, "Backpropagation", correct=True)
        self.assertEqual(rec2["seen"], 2)
        self.assertEqual(rec2["correct"], 2)
        self.assertAlmostEqual(rec2["mastery"], 0.755, places=3)
        self.assertEqual(rec2["focus"], "known")

    def test_update_mastery_incorrect_math(self):
        prof = learner.empty_profile()
        # Initial mastery = 0.5. Decay = 0.42.
        # Expected new = 0.5 * (1.0 - 0.42) = 0.5 * 0.58 = 0.29
        rec = learner.update_mastery(prof, "Gradient Descent", correct=False, video_id="vid2")
        self.assertEqual(rec["seen"], 1)
        self.assertEqual(rec["correct"], 0)
        self.assertEqual(rec["incorrect"], 1)
        self.assertFalse(rec["last_correct"])
        self.assertAlmostEqual(rec["mastery"], 0.29, places=3)
        self.assertEqual(rec["focus"], "review")

    def test_update_mastery_empty_or_whitespace(self):
        prof = learner.empty_profile()
        self.assertEqual(learner.update_mastery(prof, "", correct=True), {})
        self.assertEqual(learner.update_mastery(prof, "   ", correct=True), {})

    def test_update_mastery_video_list_limit(self):
        prof = learner.empty_profile()
        for i in range(20):
            learner.update_mastery(prof, "Neural Net", correct=True, video_id=f"vid_{i}")
        rec = prof["terms"]["neural net"]
        # Capped at last 12 videos
        self.assertEqual(len(rec["videos"]), 12)
        self.assertEqual(rec["videos"][-1], "vid_19")


class ExamRecordingAndProfileStatsTests(unittest.TestCase):
    def test_record_exam_updates_profile_and_stats(self):
        prof = learner.empty_profile()
        exam_result = {
            "exam_id": "ex-101",
            "job_id": "job-101",
            "video_id": "vid-abc",
            "video_title": "Agile Basics",
            "score": 0.5,
            "correct": 1,
            "total": 2,
            "weak_terms": ["Kanban"],
            "answers": [
                {"term": "Scrum", "correct": True},
                {"term": "Kanban", "correct": False},
            ],
        }

        updated = learner.record_exam(prof, exam_result)
        self.assertEqual(len(updated["exams"]), 1)
        stats = updated["stats"]
        self.assertEqual(stats["exams_taken"], 1)
        self.assertEqual(stats["avg_score"], 0.5)
        self.assertEqual(stats["last_score"], 0.5)
        self.assertEqual(stats["weak_count"], 1)  # Kanban is review

        # Second exam with score 1.0
        exam_result_2 = {
            "exam_id": "ex-102",
            "score": 1.0,
            "correct": 1,
            "total": 1,
            "weak_terms": [],
            "answers": [
                {"term": "Kanban", "correct": True},
            ],
        }
        learner.record_exam(prof, exam_result_2)
        self.assertEqual(prof["stats"]["exams_taken"], 2)
        self.assertAlmostEqual(prof["stats"]["avg_score"], 0.75, places=3)
        self.assertEqual(prof["stats"]["last_score"], 1.0)

    def test_record_exam_caps_history(self):
        prof = learner.empty_profile()
        for i in range(50):
            learner.record_exam(prof, {
                "exam_id": f"ex-{i}",
                "score": 1.0,
                "correct": 1,
                "total": 1,
                "answers": [],
            })
        self.assertEqual(len(prof["exams"]), learner.MAX_EXAM_SUMMARIES)
        self.assertEqual(prof["exams"][-1]["exam_id"], "ex-49")

    def test_get_term_record_case_insensitive(self):
        prof = learner.empty_profile()
        prof["terms"]["convolutional neural network"] = {
            "term": "Convolutional Neural Network",
            "seen": 2,
            "mastery": 0.8,
        }
        rec = learner.get_term_record(prof, "  CONVOLUTIONAL   NEURAL  NETWORK ")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["term"], "Convolutional Neural Network")
        self.assertIsNone(learner.get_term_record(prof, "Recurrent Neural Network"))

    def test_annotate_terms(self):
        prof = learner.empty_profile()
        prof["terms"]["known term"] = {"mastery": 0.9, "seen": 5}
        prof["terms"]["weak term"] = {"mastery": 0.2, "seen": 3}

        input_terms = [
            {"term": "Known Term"},
            {"term": "Weak Term"},
            {"term": "Unseen Term"},
        ]
        annotated = learner.annotate_terms(input_terms, prof)
        self.assertEqual(annotated[0]["focus"], "known")
        self.assertEqual(annotated[1]["focus"], "review")
        self.assertEqual(annotated[2]["focus"], "new")
        self.assertIsNone(annotated[2]["mastery"])


class LearnerPromptBlockAndAdaptiveAnalysisTests(unittest.TestCase):
    def test_build_learner_prompt_block_empty_when_no_data(self):
        prof = learner.empty_profile()
        self.assertEqual(learner.build_learner_prompt_block(prof), "")

    def test_build_learner_prompt_block_with_review_and_overlap(self):
        prof = learner.empty_profile()
        prof["stats"]["exams_taken"] = 2
        prof["stats"]["avg_score"] = 0.50
        prof["stats"]["last_score"] = 0.60
        prof["terms"] = {
            "overfitting": {
                "term": "Overfitting",
                "mastery": 0.20,
                "seen": 3,
                "incorrect": 2,
            },
            "regularization": {
                "term": "Regularization",
                "mastery": 0.80,
                "seen": 4,
                "incorrect": 0,
            },
        }

        # Current lecture contains Overfitting
        current_terms = [{"term": "Overfitting"}, {"term": "Dropout"}]
        block = learner.build_learner_prompt_block(prof, current_terms)

        self.assertIn("Practice exams taken: 2", block)
        self.assertIn("Average score: 50%", block)
        self.assertIn("Most recent exam score: 60%", block)
        self.assertIn("Review-priority terms", block)
        self.assertIn("Overfitting", block)
        self.assertIn("[in this lecture]", block)
        self.assertIn("Already-known terms", block)
        self.assertIn("Regularization", block)

    def test_build_adaptive_analysis_scenarios(self):
        # 1. No exams taken
        prof_empty = learner.empty_profile()
        an_empty = learner.build_adaptive_analysis(prof_empty)
        self.assertFalse(an_empty["has_history"])
        self.assertFalse(an_empty["weight_applied"])
        self.assertIn("No practice exams yet", an_empty["summary"])

        # 2. History with lecture overlap
        prof = learner.empty_profile()
        prof["stats"]["exams_taken"] = 1
        prof["stats"]["avg_score"] = 0.50
        prof["terms"]["attention mechanism"] = {
            "term": "Attention Mechanism",
            "mastery": 0.25,
            "seen": 2,
            "incorrect": 1,
        }
        current_terms = [{"term": "Attention Mechanism", "focus": "review", "mastery": 0.25}]
        an_overlap = learner.build_adaptive_analysis(prof, current_terms)
        self.assertTrue(an_overlap["has_history"])
        self.assertTrue(an_overlap["weight_applied"])
        self.assertIn("overlaps review terms", an_overlap["summary"])
        self.assertIn("Attention Mechanism", an_overlap["summary"])

    def test_public_profile(self):
        prof = learner.empty_profile()
        prof["learner_id"] = "test-learner"
        prof["exams"] = [
            {"exam_id": "e1", "score": 0.8},
            {"exam_id": "e2", "score": 1.0},
        ]
        pub = learner.public_profile(prof)
        self.assertEqual(pub["learner_id"], "test-learner")
        self.assertEqual(len(pub["recent_exams"]), 2)
        # Recent exams should be in reverse chronological order
        self.assertEqual(pub["recent_exams"][0]["exam_id"], "e2")
        self.assertIn("analysis", pub)


if __name__ == "__main__":
    unittest.main()
