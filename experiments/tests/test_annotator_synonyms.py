import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gui import annotator  # noqa: E402


class TestAnnotatorAutoSynonyms(unittest.TestCase):
    def setUp(self):
        with annotator._SYNONYM_LOCK:
            annotator._SYNONYM_CACHE.clear()

    def test_offline_gold_and_rag_lookup(self):
        res = annotator.lookup_term_synonyms("Kazanılmış Değer Yönetimi", lang="tr", use_online=False)
        self.assertIn("evm", [a.lower() for a in res["aliases"]])
        self.assertIn("earned value management", [a.lower() for a in res["aliases"]])
        self.assertTrue(res["definition_hint"])
        self.assertGreaterEqual(len(res["sources"]), 1)

    def test_offline_rag_ontology_lookup(self):
        res = annotator.lookup_term_synonyms("Dinamik Fiyatlandırma", lang="tr", use_online=False)
        aliases_lower = [a.lower() for a in res["aliases"]]
        self.assertIn("yield management", aliases_lower)
        self.assertIn("dynamic pricing", aliases_lower)
        self.assertIn("RAG Ontolojisi", res["sources"])

    def test_parenthetical_acronym_extraction(self):
        res = annotator.lookup_term_synonyms("Kişisel Verileri Koruma Kurulu (KVKK)", lang="tr", use_online=False)
        aliases_lower = [a.lower() for a in res["aliases"]]
        self.assertIn("kvkk", aliases_lower)
        self.assertIn("kişisel verileri koruma kurulu", aliases_lower)

    def test_online_tdk_and_wikipedia_merged(self):
        def fake_get_json(url: str, timeout: float = 1.5):
            if "sozluk.gov.tr" in url:
                return [{"anlamlarListe": [{"anlam": "Bir amaca erişmek için izlenen yol; yordam, usul (II), metot"}]}]
            if "tr.wikipedia.org" in url:
                return {
                    "query": {
                        "pages": {
                            "1": {
                                "title": "Bilimsel yöntem",
                                "langlinks": [{"lang": "en", "*": "Scientific method"}],
                                "redirects": [{"title": "Araştırma yöntemi"}],
                            }
                        }
                    }
                }
            return None

        with patch.object(annotator, "_http_get_json", side_effect=fake_get_json):
            res = annotator.lookup_term_synonyms("yöntem", lang="tr", use_online=True)
        aliases_lower = [a.lower() for a in res["aliases"]]
        self.assertIn("yordam", aliases_lower)
        self.assertIn("usul", aliases_lower)
        self.assertIn("metot", aliases_lower)
        self.assertIn("TDK Sözlük", res["sources"])

    def test_online_english_datamuse_and_wiki(self):
        def fake_get_json(url: str, timeout: float = 1.5):
            if "api.datamuse.com" in url:
                return [{"word": "replication", "score": 900}, {"word": "copying", "score": 500}]
            if "api.dictionaryapi.dev" in url:
                return [{"meanings": [{"definitions": [{"definition": "Copying genetic material.", "synonyms": ["duplication"]}]}]}]
            if "en.wikipedia.org" in url:
                return {"query": {"pages": {"2": {"title": "DNA replication", "langlinks": [{"lang": "tr", "*": "DNA replikasyonu"}], "redirects": []}}}}
            return None

        with patch.object(annotator, "_http_get_json", side_effect=fake_get_json):
            res = annotator.lookup_term_synonyms("DNA replication", lang="en", use_online=True)
        aliases_lower = [a.lower() for a in res["aliases"]]
        self.assertIn("replication", aliases_lower)
        self.assertIn("duplication", aliases_lower)
        self.assertIn("dna replikasyonu", aliases_lower)


if __name__ == "__main__":
    unittest.main()

