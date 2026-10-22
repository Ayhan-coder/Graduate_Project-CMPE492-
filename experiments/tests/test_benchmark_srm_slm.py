"""Unit test for SRM and SLM benchmark and efficiency suite."""

from __future__ import annotations

import unittest
from pathlib import Path

from experiments.benchmark_srm_slm import (
    SLM_CATALOG,
    SRM_CATALOG,
    render_ascii_table,
    render_markdown,
    run_benchmark,
)


class TestBenchmarkSRMSLM(unittest.TestCase):
    def test_catalogs_non_empty(self):
        self.assertGreaterEqual(len(SRM_CATALOG), 4)
        self.assertGreaterEqual(len(SLM_CATALOG), 4)

    def test_run_benchmark_modes(self):
        res_all = run_benchmark(mode="all", quant="int4")
        self.assertEqual(res_all["total_models"], len(SRM_CATALOG) + len(SLM_CATALOG))
        self.assertEqual(res_all["quantization"], "int4")

        res_srm = run_benchmark(mode="srm", quant="fp16")
        self.assertEqual(res_srm["total_models"], len(SRM_CATALOG))

        res_slm = run_benchmark(mode="slm", quant="int8")
        self.assertEqual(res_slm["total_models"], len(SLM_CATALOG))

    def test_renders(self):
        tbl = render_ascii_table(SRM_CATALOG, quant="int4")
        self.assertIn("Whisper Tiny", tbl)
        self.assertIn("Params", tbl)

        md = render_markdown(SLM_CATALOG, quant="fp16")
        self.assertIn("Qwen 2.5 0.5B", md)
        self.assertIn("VRAM Tüketimi", md)


if __name__ == "__main__":
    unittest.main()
