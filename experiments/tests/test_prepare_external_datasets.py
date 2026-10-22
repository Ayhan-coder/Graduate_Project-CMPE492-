"""Unit tests for the external dataset converter (offline: no Hugging Face access)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.finetune.prepare_external_datasets import (
    LeakageGuard,
    _bloom_from,
    _fold,
    _is_definitional,
    _lead_sentence,
)

GOLD_SENTENCE = (
    "Kazanılmış değer yönetimi projenin kapsam maliyet ve zaman performansını bütünleştirir"
)


def _guard_with(tmp: Path) -> LeakageGuard:
    gold_dir = tmp / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    (gold_dir / "TR-1.txt").write_text(GOLD_SENTENCE + ".", encoding="utf-8")
    (gold_dir / "README.md").write_text("not a transcript", encoding="utf-8")
    (tmp / "gold_terms.json").write_text(json.dumps({
        "TR-1": {
            "domain": "Project Management",
            "terms": [{"canonical": "Kazanılmış Değer", "aliases": ["kazanılmış değer", "EV"]}],
        }
    }, ensure_ascii=False), encoding="utf-8")
    return LeakageGuard(gold_dir=gold_dir)


class FoldAndLeadTests(unittest.TestCase):
    def test_fold_ignores_case_and_turkish_diacritics(self):
        self.assertEqual(_fold("İzleme"), _fold("İZLEME"))
        self.assertEqual(_fold("İzleme"), "izleme")

    def test_lead_strips_repeated_article_heading(self):
        text = "Avustralya kazı Avustralya kazı, ördekgiller familyasından bir kaz türü. İkinci cümle."
        self.assertEqual(
            _lead_sentence(text, "Avustralya kazı"),
            "Avustralya kazı, ördekgiller familyasından bir kaz türü.",
        )

    def test_is_definitional_accepts_gloss_and_rejects_recap(self):
        self.assertTrue(_is_definitional("Big-O gösterimi, bir algoritmanın üst sınırını ifade eden gösterimdir."))
        # appositional gloss: no copula, but a "türü" cue -- matched after folding
        self.assertTrue(_is_definitional("Avustralya kazı, ördekgiller familyasından bir kaz türü."))
        self.assertFalse(_is_definitional("Bu tartışmalar sonraki yıllarda birçok kez yeniden gündeme geldi."))

    def test_bloom_heuristic_prefers_the_strongest_cue(self):
        self.assertEqual(_bloom_from("İki yöntem arasındaki farkı ayrıştıran bir inceleme."), "Analiz")
        self.assertEqual(_bloom_from("Bu hesaplanır."), "Uygulama")
        self.assertEqual(_bloom_from("Bir kişi."), "Hatırlama")


class LeakageGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.guard = _guard_with(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_eval_answer_term_is_rejected_case_insensitively(self):
        self.assertTrue(self.guard.term_is_eval_answer("kazanılmış DEĞER"))
        self.assertFalse(self.guard.term_is_eval_answer("Ohm Kanunu"))

    def test_gold_sentence_is_rejected_but_unrelated_text_passes(self):
        self.assertTrue(self.guard.text_is_eval(GOLD_SENTENCE + " ve devamı"))
        self.assertFalse(self.guard.text_is_eval("Bugün hava çok güzel, dışarıda yürüyüş yaptık."))

    def test_short_text_cannot_reach_the_ngram_threshold(self):
        self.assertFalse(self.guard.text_is_eval("kısa cümle"))


if __name__ == "__main__":
    unittest.main()
