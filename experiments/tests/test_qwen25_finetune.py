"""Unit tests for Qwen 2.5 Lecture Notes Dataset Builder, Fine-Tuner, and Evaluator."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.finetune.eval_qwen25_finetune import (
    calculate_atesman_score,
    evaluate_lecture_performance,
    run_live_inference_on_text,
)
from experiments.finetune.finetune_qwen25 import format_chatml_prompt, run_finetune
from experiments.finetune.prepare_lecture_notes_dataset import build_lecture_dataset


class TestQwen25FineTune(unittest.TestCase):
    def test_build_lecture_dataset(self):
        train_file, val_file = build_lecture_dataset()
        self.assertTrue(train_file.exists())
        self.assertTrue(val_file.exists())

        with open(train_file, "r", encoding="utf-8") as f:
            first_line = json.loads(f.readline())
            self.assertIn("instruction", first_line)
            self.assertIn("input", first_line)
            self.assertIn("output", first_line)

    def test_chatml_prompt_format(self):
        prompt = format_chatml_prompt("Terimleri çıkar", "Örnek ders metni", "Çıktı")
        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("<|im_start|>user", prompt)
        self.assertIn("<|im_start|>assistant", prompt)

    def test_finetune_dry_run(self):
        meta = run_finetune(model_id="Qwen/Qwen2.5-0.5B-Instruct", dry_run=True)
        self.assertEqual(meta["status"], "DRY_RUN_PASSED")
        self.assertEqual(meta["lora_r"], 16)
        self.assertIn("sample_prompt_preview", meta)

    def test_atesman_score(self):
        simple_text = "Bugün güzel bir gün. Okula gidiyoruz."
        score = calculate_atesman_score(simple_text)
        self.assertGreater(score, 40.0)
        self.assertLessEqual(score, 100.0)

    def test_evaluate_lecture_performance(self):
        res = evaluate_lecture_performance()
        self.assertIn("metrics", res)
        self.assertFalse(res["is_measured"])
        self.assertEqual(res["benchmark_type"], "architectural_target_projection")
        self.assertIn("note", res)

    def test_run_live_inference_on_text(self):
        sample_text = "Kirchhoff Gerilim Yasası kapalı devrelerde çevre boyunca toplam potansiyelin sıfır olduğunu söyler."
        out = run_live_inference_on_text(sample_text)
        self.assertGreaterEqual(len(out["extracted_terms"]), 1)
        self.assertTrue(out["target_b1_compatible"])
        self.assertTrue(out.get("is_mock_inference", False))


if __name__ == "__main__":
    unittest.main()
