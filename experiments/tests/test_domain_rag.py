"""Unit tests for Enhanced Domain RAG Extractor and Prerequisite DAG Verifier."""

from __future__ import annotations

import unittest

from experiments.rag.domain_rag_extractor import (
    DEFAULT_KB,
    HybridRetriever,
    PrerequisiteDAGVerifier,
    evaluate_rag_ablation,
    extract_with_rag,
)


class TestDomainRAG(unittest.TestCase):
    def setUp(self):
        self.retriever = HybridRetriever(DEFAULT_KB)
        self.verifier = PrerequisiteDAGVerifier(DEFAULT_KB)

    def test_kb_has_expected_domains(self):
        domains = set(item["domain"] for item in DEFAULT_KB)
        expected = {"tourism", "isg", "kvkk", "banking", "cybersecurity", "sustainability", "circuits"}
        self.assertTrue(expected.issubset(domains))

    def test_dag_is_acyclic(self):
        self.assertTrue(self.verifier.is_valid_dag())

    def test_hybrid_retrieval_tourism(self):
        query = "Dinamik fiyatlandırma ve no show durumlarını inceliyoruz."
        results = self.retriever.retrieve(query, top_k=2)
        self.assertGreaterEqual(len(results), 1)
        found_names = [r["canonical_name"] for r in results]
        self.assertTrue(any("Dinamik" in n or "No-Show" in n for n in found_names))

    def test_hybrid_retrieval_banking(self):
        query = "Kredi verirken Findeks skoru ve MASAK bildirimlerine dikkat edilir."
        results = self.retriever.retrieve(query, top_k=2)
        found_ids = [r["id"] for r in results]
        self.assertTrue("BANK-01" in found_ids or "BANK-02" in found_ids)

    def test_prerequisite_gap_detection(self):
        # Risk Değerlendirmesi requires Ramak Kala (ISG-01). If we only provide ISG-02, it should flag a gap.
        analysis = self.verifier.verify_prerequisites(["ISG-02"])
        self.assertTrue(analysis["has_pedagogical_gaps"])
        self.assertIn("Risk Değerlendirmesi", analysis["missing_prerequisites"])

        # When prerequisite ISG-01 is present, gap should be resolved
        analysis_ok = self.verifier.verify_prerequisites(["ISG-01", "ISG-02"])
        self.assertFalse(analysis_ok["has_pedagogical_gaps"])

    def test_extract_with_rag(self):
        query = "Kişisel verilerin korunmasında veri sorumlusu açık rıza almalı ve aydınlatma metni sunmalıdır."
        out = extract_with_rag(query, self.retriever, self.verifier)
        self.assertEqual(out["hallucination_rate"], 0.0)
        self.assertGreaterEqual(len(out["verified_terms"]), 2)
        term_names = [t["term"] for t in out["verified_terms"]]
        self.assertIn("Veri Sorumlusu", term_names)

    def test_rag_ablation(self):
        samples = [
            {"text": "Otel no-show durumunda voucher işlemi.", "gold_ids": ["TOUR-02", "TOUR-03"]},
            {"text": "İşyerinde ramak kala ve KKD kullanımı.", "gold_ids": ["ISG-01", "ISG-03"]}
        ]
        report = evaluate_rag_ablation(samples)
        self.assertEqual(report["rag_augmented"]["hallucination_rate"], 0.0)
        self.assertGreaterEqual(report["rag_augmented"]["f1"], 0.70)


if __name__ == "__main__":
    unittest.main()
