"""Retrieval-Augmented Generation (RAG) Terminology & Prerequisite DAG Extractor.

Addresses meeting note items:
- "RAG denenebilir"
- "Dikey alan terim sözlüğü ve ontolojisi chunk'lara bölünür ve vektörleştirilir (MiniLM / BGE-M3 / BM25)"
- "Transkript akarken geçen cümleler ile sözlük eşleştirilir; ilgili kanonik tanımlar prompt bağlamına enjekte edilir"
- "LLM sadece 'tanımla' yerine 'bağlamdaki ontolojiye göre doğrula ve Bloom düzeyini ata' görevini icra eder"
- "Metrik: Halüsinasyon oranı (%0 hedefi) ve kanonik tanım tamlığı (F1)"

Key Capabilities:
1. Multi-Domain Canonical Knowledge Base (Tourism, OHS/İSG, KVKK, Banking, Cyber, Sustainability, STEM).
2. Hybrid Retrieval: BM25-style term matching + Vectorized TF-IDF n-gram cosine similarity with Reciprocal Rank Fusion (RRF).
3. NetworkX Prerequisite DAG: Validates topological order, acyclicity, and discovers prerequisite learning gaps in transcripts.
4. Timestamped Segment Alignment: Ingests streaming ASR segments [start_s, end_s, text] and associates them with ontology terms.
5. RAG vs Non-RAG Ablation Evaluator: Measures Hallucination Rate (%0 target) and F1 gain.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DomainRAG")

_TURKISH_FOLD = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


def _fold(text: str) -> str:
    """Case- and diacritic-insensitive Turkish/English folding."""
    out = unicodedata.normalize("NFKD", text)
    out = "".join(c for c in out if not unicodedata.combining(c))
    return out.translate(_TURKISH_FOLD).lower()

ROOT = Path(__file__).resolve().parents[2]
KB_DIR = ROOT / "experiments" / "rag" / "kb"
KB_DIR.mkdir(parents=True, exist_ok=True)


# Expanded Multi-Domain Canonical Knowledge Base
DEFAULT_KB: List[Dict[str, Any]] = [
    # --- TURİZM & TATİLSEPETİ AR-GE ---
    {
        "id": "TOUR-01",
        "domain": "tourism",
        "canonical_name": "Dinamik Fiyatlandırma",
        "aliases": ["yield management", "dynamic pricing", "esnek fiyatlandırma", "gelir yönetimi"],
        "definition": "Otel doluluk oranı, talep esnekliği ve pazar koşullarına göre oda fiyatlarının anlık ve kural bazlı optimize edildiği gelir yönetimi sistemi.",
        "bloom_level": "Uygulama",
        "prerequisites": ["TOUR-04"]
    },
    {
        "id": "TOUR-02",
        "domain": "tourism",
        "canonical_name": "No-Show",
        "aliases": ["gelmeyen misafir", "no show", "giriş yapmama", "rezervasyon iptalsiz gelmeme"],
        "definition": "Rezervasyonu bulunan konuğun geçerli bir iptal bildiriminde bulunmaksızın konaklama tesisine giriş yapmaması durumu.",
        "bloom_level": "Hatırlama",
        "prerequisites": []
    },
    {
        "id": "TOUR-03",
        "domain": "tourism",
        "canonical_name": "Voucher",
        "aliases": ["rezervasyon belgesi", "hizmet kuponu", "otel kuponu", "konaklama belgesi"],
        "definition": "Müşterinin acente üzerinden satın aldığı otel, transfer ve tur hizmetlerinin ödendiğini ve hak sahipliğini teyit eden resmi belge.",
        "bloom_level": "Anlama",
        "prerequisites": ["TOUR-02"]
    },
    {
        "id": "TOUR-04",
        "domain": "tourism",
        "canonical_name": "Doluluk Oranı (Occupancy Rate)",
        "aliases": ["doluluk", "occupancy rate", "kapasite kullanım oranı"],
        "definition": "Tesisin belirli bir tarihte satılan oda sayısının satışa hazır toplam oda sayısına oranı.",
        "bloom_level": "Anlama",
        "prerequisites": []
    },

    # --- İŞ SAĞLIĞI VE GÜVENLİĞİ (İSG / OSHA) ---
    {
        "id": "ISG-01",
        "domain": "isg",
        "canonical_name": "Ramak Kala Olayı",
        "aliases": ["near miss", "ramak kala", "ramak kaldı", "ramak kaldi", "ramakalı", "ramakali", "ramak kalma", "ramak", "ucuz atlatılan olay", "tehlikeli vaka", "close call"],
        "definition": "İşyerinde meydana gelen, fakat yaralanma veya mala zarara yol açmaksızın atlatılan tehlikeli durum veya olay.",
        "bloom_level": "Anlama",
        "prerequisites": []
    },
    {
        "id": "ISG-02",
        "domain": "isg",
        "canonical_name": "Risk Değerlendirmesi",
        "aliases": ["risk analizi", "risk analizleri", "tehlike değerlendirmesi", "fine kinney analizi", "risk matrisi", "risk assessment", "hazard identification", "risk evaluation", "risk değerlendirme", "risk değerlendirmeleri"],
        "definition": "İşyerinde mevcut tehlikelerin olasılık ve şiddetini sistematik biçimde puanlayarak öncelikli koruyucu tedbirleri belirleme süreci.",
        "bloom_level": "Analiz",
        "prerequisites": ["ISG-01"]
    },
    {
        "id": "ISG-03",
        "domain": "isg",
        "canonical_name": "Kişisel Koruyucu Donanım (KKD)",
        "aliases": ["PPE", "koruyucu donanım", "baret", "emniyet kemeri", "koruyucu gözlük", "personal protective equipment", "safety helmet", "protective gear", "kişisel koruyucu", "kişisel koruyucular"],
        "definition": "Çalışanı iş kazası ve meslek hastalığı risklerine karşı doğrudan koruyan baret, çelik burunlu ayakkabı, maske ve emniyet kemeri gibi teçhizat.",
        "bloom_level": "Hatırlama",
        "prerequisites": ["ISG-01"]
    },
    {
        "id": "ISG-04",
        "domain": "isg",
        "canonical_name": "Acil Durum Eylem Planı",
        "aliases": ["emergency plan", "emergency plans", "acil durum planı", "acil durum planları", "acil durum plani", "acil durum planlari", "acil durum eylem planı", "acil durum eylem planları", "acil durum planlaması", "tahliye ve yangın planı", "kriz eylem planı", "emergency action plan", "evacuation plan"],
        "definition": "İşyerinde meydana gelebilecek yangın, patlama, deprem ve kimyasal sızıntı gibi acil hallerde personelin tahliyesi ve ilk müdahale organizasyonu.",
        "bloom_level": "Uygulama",
        "prerequisites": ["ISG-02"]
    },

    # --- KİŞİSEL VERİLERİN KORUNMASI (KVKK / GDPR) ---
    {
        "id": "KVKK-01",
        "domain": "kvkk",
        "canonical_name": "Veri Sorumlusu",
        "aliases": ["data controller", "veri yöneticisi", "sorumlu veri işleyen", "controller", "veri sorumluları", "veri sorumlusunun", "veri sorumlusuna"],
        "definition": "Kişisel verilerin işleme amaçlarını ve vasıtalarını belirleyen, veri kayıt sisteminin kurulmasından ve yönetilmesinden sorumlu olan gerçek veya tüzel kişi.",
        "bloom_level": "Anlama",
        "prerequisites": []
    },
    {
        "id": "KVKK-02",
        "domain": "kvkk",
        "canonical_name": "Açık Rıza",
        "aliases": ["explicit consent", "özgür irade onayı", "açık rıza beyanı", "consent", "açık rıza", "açık rızası", "açık rızanın", "açık rızasıyla"],
        "definition": "Belirli bir konuya ilişkin, bilgilendirilmeye dayanan ve kişinin özgür iradesiyle açıklanan tereddütsüz rıza beyanı.",
        "bloom_level": "Uygulama",
        "prerequisites": ["KVKK-01"]
    },
    {
        "id": "KVKK-03",
        "domain": "kvkk",
        "canonical_name": "Özel Nitelikli Kişisel Veri",
        "aliases": ["sensitive personal data", "hassas veri", "biyometrik veri", "sağlık verisi", "special category data", "özel nitelikli kişisel veri", "özel nitelikli", "özel nitelikli veri", "özel nitelikli veriler"],
        "definition": "Öğrenilmesi halinde ilgili kişi hakkında ayrımcılık yapılmasına yol açabilecek sağlık, biyometrik, genetik, ceza mahkumiyeti ve sendika üyeliği verileri.",
        "bloom_level": "Hatırlama",
        "prerequisites": ["KVKK-01"]
    },
    {
        "id": "KVKK-04",
        "domain": "kvkk",
        "canonical_name": "Aydınlatma Yükümlülüğü",
        "aliases": ["privacy notice", "bilgilendirme yükümlülüğü", "aydınlatma metni", "aydınlatma yükümlülüğü", "transparency", "right to be informed", "aydınlatma yükümlülüğünün", "aydınlatma metinleri", "aydınlatma"],
        "definition": "Veri sorumlusunun kişisel verilerin hangi amaçla, kimlere aktarılabileceğini ve ilgili kişinin haklarını veri sahibine bildirme zorunluluğu.",
        "bloom_level": "Anlama",
        "prerequisites": ["KVKK-01"]
    },

    # --- BANKACILIK VE FİNANS ---
    {
        "id": "BANK-01",
        "domain": "banking",
        "canonical_name": "Kredi Kayıt Bürosu (KKB) / Findeks",
        "aliases": ["kkb", "findeks skoru", "kredi notu", "kredi bürosu"],
        "definition": "Birey ve kurumların finansal geçmişini, kredi kartı ve kredi geri ödeme performanslarını skorlayan merkezi kredi sicil kurumu.",
        "bloom_level": "Anlama",
        "prerequisites": []
    },
    {
        "id": "BANK-02",
        "domain": "banking",
        "canonical_name": "MASAK Uyum (AML)",
        "aliases": ["masak", "kara para aklama", "aml", "anti money laundering", "şüpheli işlem"],
        "definition": "Mali Suçları Araştırma Kurulu mevzuatına göre suç gelirlerinin aklanmasının önlenmesi ve şüpheli işlem bildirimi yükümlülükleri bütünü.",
        "bloom_level": "Uygulama",
        "prerequisites": ["BANK-01"]
    },
    {
        "id": "BANK-03",
        "domain": "banking",
        "canonical_name": "Likidite Karşılama Oranı (LCR)",
        "aliases": ["likidite oranı", "lcr", "liquidity coverage ratio"],
        "definition": "Bankanın 30 günlük net nakit çıkışlarını karşılayabilecek yüksek kaliteli likit varlık stoğuna sahip olma yeterlilik rasyosu.",
        "bloom_level": "Analiz",
        "prerequisites": ["BANK-01"]
    },

    # --- SİBER GÜVENLİK ---
    {
        "id": "CYBER-01",
        "domain": "cybersecurity",
        "canonical_name": "Oltalama (Phishing)",
        "aliases": ["phishing", "yemleme", "oltalama saldırısı", "aldatıcı e-posta"],
        "definition": "Güvenilir bir kurum izlenimi verilerek sahte bağlantılar ve e-postalar aracılığıyla kimlik ve parola bilgilerini çalmayı amaçlayan sosyal mühendislik saldırısı.",
        "bloom_level": "Anlama",
        "prerequisites": []
    },
    {
        "id": "CYBER-02",
        "domain": "cybersecurity",
        "canonical_name": "Çok Faktörlü Kimlik Doğrulama (MFA)",
        "aliases": ["mfa", "2fa", "iki faktörlü doğrulama", "multi factor authentication"],
        "definition": "Kullanıcının sisteme erişimini sağlamak için bildiği (şifre), sahip olduğu (telefon/token) ve biyometrik verisi olmak üzere en az iki bağımsız kanıt talep eden güvenlik mekanizması.",
        "bloom_level": "Uygulama",
        "prerequisites": ["CYBER-01"]
    },
    {
        "id": "CYBER-03",
        "domain": "cybersecurity",
        "canonical_name": "Sıfır Güven Mimarisi (Zero Trust)",
        "aliases": ["zero trust", "sıfır güven", "asla güvenme daima doğrula"],
        "definition": "Ağ içi veya ağ dışı hiçbir kullanıcıya veya aygıta varsayılan güven tanımayan, her istekte sürekli doğrulama ve en az yetki prensibini uygulayan güvenlik modeli.",
        "bloom_level": "Analiz",
        "prerequisites": ["CYBER-02"]
    },

    # --- SÜRDÜRÜLEBİLİRLİK (ESG) ---
    {
        "id": "ESG-01",
        "domain": "sustainability",
        "canonical_name": "Karbon Ayak İzi",
        "aliases": ["carbon footprint", "sera gazı emisyonu", "karbon salımı"],
        "definition": "Bir kurumun veya bireyin faaliyetleri sonucu doğrudan veya dolaylı olarak atmosfere salınan toplam sera gazı miktarının karbondioksit eşdeğeri.",
        "bloom_level": "Anlama",
        "prerequisites": []
    },
    {
        "id": "ESG-02",
        "domain": "sustainability",
        "canonical_name": "Kapsam 1-2-3 Emisyonları",
        "aliases": ["scope 1 2 3", "kapsam emisyonları", "doğrudan ve dolaylı emisyon"],
        "definition": "Doğrudan tesis salımları (Kapsam 1), satın alınan elektrik/enerji (Kapsam 2) ve tüm tedarik zinciri dolaylı salımlarını (Kapsam 3) kategorize eden sera gazı protokolü.",
        "bloom_level": "Analiz",
        "prerequisites": ["ESG-01"]
    },

    # --- AKADEMİK STEM & DEVRELER ---
    {
        "id": "EE-01",
        "domain": "circuits",
        "canonical_name": "Kirchhoff Gerilim Yasası",
        "aliases": ["KVL", "kirchhoff loop law", "çevre gerilimleri toplamı", "kirchhoff voltaj yasası"],
        "definition": "Kapalı bir elektrik devresi çevresi (loop) boyunca tüm potansiyel farklarının cebirsel toplamı sıfırdır.",
        "bloom_level": "Uygulama",
        "prerequisites": ["EE-02"]
    },
    {
        "id": "EE-02",
        "domain": "circuits",
        "canonical_name": "Düğüm Gerilimleri Analizi",
        "aliases": ["nodal analysis", "düğüm analizi", "düğüm denklemleri"],
        "definition": "Devredeki bağımsız düğüm noktalarının gerilimlerini temel değişken alarak Kirchhoff Akım Yasası yardımıyla devre denklemlerini çözme yöntemi.",
        "bloom_level": "Analiz",
        "prerequisites": []
    }
]


class HybridRetriever:
    """Hybrid Retriever combining BM25-style lexical matching and TF-IDF Cosine Similarity with RRF."""

    def __init__(self, kb: List[Dict[str, Any]]):
        self.kb = kb
        self._corpus_texts: List[str] = []
        for item in self.kb:
            # Combine canonical name, aliases, and definition for dense representation
            text = f"{item['canonical_name']} {' '.join(item.get('aliases', []))} {item['definition']}"
            self._corpus_texts.append(text)

        # Fit TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(self._corpus_texts)

    def retrieve_lexical(self, query_text: str) -> List[Tuple[int, float]]:
        """Exact and partial token matching scores."""
        query_norm = query_text.lower()
        q_tokens = set(re.findall(r"\w+", query_norm))
        scores: List[Tuple[int, float]] = []

        for idx, item in enumerate(self.kb):
            score = 0.0
            canon = item["canonical_name"].lower()
            if canon in query_norm:
                score += 5.0

            for alias in item.get("aliases", []):
                if alias.lower() in query_norm:
                    score += 3.5

            canon_tokens = set(re.findall(r"\w+", canon))
            overlap = len(canon_tokens.intersection(q_tokens))
            score += overlap * 1.0

            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def retrieve_semantic(self, query_text: str) -> List[Tuple[int, float]]:
        """TF-IDF cosine similarity scores."""
        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        scored = [(idx, float(score)) for idx, score in enumerate(sims) if score > 0.05]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def retrieve(
        self, query_text: str, top_k: int = 4, rrf_k: int = 60, domain_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion (RRF) between lexical and semantic retrievers with optional domain filter."""
        lex_ranks = self.retrieve_lexical(query_text)
        sem_ranks = self.retrieve_semantic(query_text)

        rrf_scores: Dict[int, float] = {}

        for rank, (idx, _) in enumerate(lex_ranks):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1)) * 1.5

        for rank, (idx, _) in enumerate(sem_ranks):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1)) * 1.0

        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for idx, _ in sorted_indices:
            item = self.kb[idx]
            if domain_filter and item.get("domain") != domain_filter:
                continue
            results.append(item)
            if len(results) >= top_k:
                break
        return results


class PrerequisiteDAGVerifier:
    """Constructs and verifies educational prerequisite DAGs from extracted terms."""

    def __init__(self, kb: List[Dict[str, Any]]):
        self.kb_map = {item["id"]: item for item in kb}
        self.graph = nx.DiGraph()
        for item in kb:
            self.graph.add_node(item["id"], name=item["canonical_name"], bloom=item["bloom_level"])
            for prereq in item.get("prerequisites", []):
                if prereq in self.kb_map:
                    self.graph.add_edge(prereq, item["id"])

    def is_valid_dag(self) -> bool:
        return nx.is_directed_acyclic_graph(self.graph)

    def verify_prerequisites(self, extracted_ids: List[str]) -> Dict[str, Any]:
        """Check if prerequisite concepts are met or missing in the extracted set."""
        extracted_set = set(extracted_ids)
        missing_prereqs: Dict[str, List[str]] = {}

        for term_id in extracted_set:
            item = self.kb_map.get(term_id)
            if not item:
                continue
            prereqs = item.get("prerequisites", [])
            unmet = [p for p in prereqs if p not in extracted_set]
            if unmet:
                missing_prereqs[item["canonical_name"]] = [
                    self.kb_map[p]["canonical_name"] for p in unmet if p in self.kb_map
                ]

        return {
            "is_acyclic": self.is_valid_dag(),
            "extracted_count": len(extracted_set),
            "missing_prerequisites": missing_prereqs,
            "has_pedagogical_gaps": len(missing_prereqs) > 0,
        }


_TR_SUFFIXES = (
    "larinin", "lerinin", "larinda", "lerinde", "larindan", "lerinden",
    "larina", "lerine", "lari", "leri", "lar", "ler",
    "sindan", "sinden", "sinda", "sinde", "sina", "sine", "sini", "sinin",
    "dan", "den", "tan", "ten",
    "nda", "nde", "nin", "nin", "nun", "nun",
    "ya", "ye", "na", "ne",
    "si", "su", "su", "si",
    "in", "in", "un", "un",
    "i", "i", "u", "u",
)


def _turkish_stem(w: str) -> str:
    """Lightweight Turkish inflection stripper for technical noun phrase matching."""
    s = _fold(w)
    for suff in _TR_SUFFIXES:
        if s.endswith(suff) and len(s) - len(suff) >= 3:
            return s[:-len(suff)]
    return s


def check_surface_match_detailed(
    transcript_text: str, canonical_name: str, aliases: List[str]
) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """Check if the canonical name or any alias appears in transcript text using Turkish folding & stemming.
    
    Returns:
        (matched, matched_form, match_type, match_target)
        where match_type in {'exact_canonical', 'exact_alias', 'stemmed_canonical', 'stemmed_alias', None}
        and match_target in {'canonical', 'alias', None}
    
    NOTE: The morphology matcher uses a lightweight single-pass suffix stripper and sliding
    window (max 4 filler words) rather than a full morphological analyzer (e.g. Zemberek).
    """
    folded_t = _fold(transcript_text)

    # 1. Exact canonical name match
    canon_clean = re.sub(r"\s*\(.*?\)\s*", "", canonical_name).strip()
    if canon_clean and _fold(canon_clean) in folded_t:
        return True, canon_clean, "exact_canonical", "canonical"
    if _fold(canonical_name) in folded_t:
        return True, canonical_name, "exact_canonical", "canonical"

    # 2. Exact alias match (acronym boundary check)
    for alias in aliases:
        alias_norm = alias.strip()
        if not alias_norm:
            continue
        folded_alias = _fold(alias_norm)
        if len(folded_alias) <= 4:
            if re.search(r"\b" + re.escape(folded_alias) + r"\b", folded_t):
                return True, alias, "exact_alias", "alias"
        else:
            if folded_alias in folded_t:
                return True, alias, "exact_alias", "alias"

    # 3. Morphological & multi-word lemma/stem matching
    tokens = re.findall(r"\w+", folded_t)
    token_stems = [_turkish_stem(t) for t in tokens]

    # Try canonical first, then aliases
    if canon_clean:
        cand_words = re.findall(r"\w+", _fold(canon_clean))
        if cand_words:
            c_stems = [_turkish_stem(w) for w in cand_words if len(w) >= 3]
            if c_stems:
                if len(c_stems) == 1:
                    if c_stems[0] in token_stems:
                        idx = token_stems.index(c_stems[0])
                        return True, tokens[idx], "stemmed_canonical", "canonical"
                else:
                    n = len(c_stems)
                    for i in range(len(token_stems) - n + 1):
                        if token_stems[i] == c_stems[0]:
                            curr_pos = i
                            matched = True
                            for s in c_stems[1:]:
                                found = False
                                for step in range(1, 5):
                                    if curr_pos + step < len(token_stems) and token_stems[curr_pos + step] == s:
                                        curr_pos = curr_pos + step
                                        found = True
                                        break
                                if not found:
                                    matched = False
                                    break
                            if matched:
                                return True, " ".join(tokens[i:curr_pos + 1]), "stemmed_canonical", "canonical"

    for alias in aliases:
        cand_words = re.findall(r"\w+", _fold(alias))
        if not cand_words:
            continue
        c_stems = [_turkish_stem(w) for w in cand_words if len(w) >= 3]
        if not c_stems:
            continue
        if len(c_stems) == 1:
            if c_stems[0] in token_stems:
                idx = token_stems.index(c_stems[0])
                return True, tokens[idx], "stemmed_alias", "alias"
        else:
            n = len(c_stems)
            for i in range(len(token_stems) - n + 1):
                if token_stems[i] == c_stems[0]:
                    curr_pos = i
                    matched = True
                    for s in c_stems[1:]:
                        found = False
                        for step in range(1, 5):
                            if curr_pos + step < len(token_stems) and token_stems[curr_pos + step] == s:
                                curr_pos = curr_pos + step
                                found = True
                                break
                        if not found:
                            matched = False
                            break
                    if matched:
                        return True, " ".join(tokens[i:curr_pos + 1]), "stemmed_alias", "alias"

    return False, None, None, None


def check_surface_match(
    transcript_text: str, canonical_name: str, aliases: List[str]
) -> Tuple[bool, Optional[str]]:
    """Check if the canonical name or any alias appears in transcript text using Turkish folding & stemming."""
    m, mf, _, _ = check_surface_match_detailed(transcript_text, canonical_name, aliases)
    return m, mf


def extract_with_rag(
    transcript_text: str,
    retriever: HybridRetriever,
    dag_verifier: Optional[PrerequisiteDAGVerifier] = None,
    top_k: int = 4,
    domain_filter: Optional[str] = None,
    require_surface_match: bool = True,
) -> Dict[str, Any]:
    """Retrieve canonical context, enforce transcript surface-form grounding, and verify DAG."""
    retrieved_items = retriever.retrieve(transcript_text, top_k=top_k, domain_filter=domain_filter)

    kb_snippets = "\n".join(
        [
            f"- [{item['id']}] {item['canonical_name']} (Bloom: {item['bloom_level']}): {item['definition']}"
            for item in retrieved_items
        ]
    )

    augmented_prompt = (
        f"<|im_start|>system\n"
        f"Sen kanonik bir ontolojiye sıkı sıkıya bağlı akademik terminoloji çıkarım asistanısın.\n"
        f"ASLA bilgi tabanı dışından hayali (halüsinasyon) terim üretme.\n"
        f"Yalnızca verilen kanonik listedeki terimleri transkriptle eşleştir ve Bloom taksonomi seviyesini ata.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Kanonik Ontoloji Bilgi Tabanı:\n{kb_snippets}\n\n"
        f"Ders Transkripti:\n{transcript_text}\n\n"
        f"Transkriptte geçen doğrulanmış terimleri JSON formatında listele.<|im_end|>"
    )

    verified_terms: List[Dict[str, Any]] = []
    misgrounded_candidates: List[Dict[str, Any]] = []

    for item in retrieved_items:
        matched, matched_form, match_type, match_target = check_surface_match_detailed(
            transcript_text, item["canonical_name"], item.get("aliases", [])
        )
        term_payload = {
            "term": item["canonical_name"],
            "canonical_id": item["id"],
            "domain": item["domain"],
            "definition": item["definition"],
            "bloom_level": item["bloom_level"],
            "prerequisites": item["prerequisites"],
            "matched_surface_form": matched_form,
            "match_type": match_type,
            "match_target": match_target,
        }

        if require_surface_match:
            if matched:
                term_payload["rag_verified"] = True
                verified_terms.append(term_payload)
            else:
                term_payload["rag_verified"] = False
                term_payload["failure_reason"] = "no_surface_mention_in_transcript"
                misgrounded_candidates.append(term_payload)
        else:
            term_payload["rag_verified"] = True
            verified_terms.append(term_payload)

    # DAG Verification on genuinely verified terms
    extracted_ids = [t["canonical_id"] for t in verified_terms]
    dag_analysis = dag_verifier.verify_prerequisites(extracted_ids) if dag_verifier else {}

    retrieval_prec = len(verified_terms) / max(1, len(retrieved_items))

    return {
        "transcript_length_chars": len(transcript_text),
        "retrieved_kb_items": len(retrieved_items),
        "verified_terms": verified_terms,
        "misgrounded_candidates": misgrounded_candidates,
        "candidate_grounding_precision": round(retrieval_prec, 3),
        "hallucination_rate": 0.0 if require_surface_match else round(1.0 - retrieval_prec, 3),
        "prerequisite_dag": dag_analysis,
        "rag_augmented_prompt_preview": augmented_prompt[:350],
    }


def evaluate_rag_ablation(
    test_queries: List[Dict[str, Any]], top_k: int = 4
) -> Dict[str, Any]:
    """Empirical ablation study comparing Baseline lexical matching vs RAG-Augmented Extraction.

    NOTE: As stated in paper (main.tex:198) and report (§8.4), evaluating glossary-RAG against
    synthetic queries drawn from the same KB is inherently circular without a held-out external
    glossary. This function performs real empirical scoring against the provided queries with fixed
    top_k and surface-form checking rather than hardcoded arithmetic constants.
    """
    retriever = HybridRetriever(DEFAULT_KB)
    verifier = PrerequisiteDAGVerifier(DEFAULT_KB)

    rag_tp = 0
    rag_fp = 0
    rag_fn = 0

    base_tp = 0
    base_fp = 0
    base_fn = 0

    for sample in test_queries:
        text = sample["text"]
        gold_ids = set(sample["gold_ids"])

        # 1. RAG Arm: Hybrid retrieval + strict surface match verification
        rag_out = extract_with_rag(text, retriever, verifier, top_k=top_k, require_surface_match=True)
        rag_extracted_ids = set(t["canonical_id"] for t in rag_out["verified_terms"])

        tp_r = len(rag_extracted_ids.intersection(gold_ids))
        fp_r = len(rag_extracted_ids - gold_ids)
        fn_r = len(gold_ids - rag_extracted_ids)

        rag_tp += tp_r
        rag_fp += fp_r
        rag_fn += fn_r

        # 2. Baseline Arm: Unconstrained lexical retrieval without surface-match grounding
        raw_retrieved = retriever.retrieve(text, top_k=top_k)
        base_extracted_ids = set(item["id"] for item in raw_retrieved)

        tp_b = len(base_extracted_ids.intersection(gold_ids))
        fp_b = len(base_extracted_ids - gold_ids)
        fn_b = len(gold_ids - base_extracted_ids)

        base_tp += tp_b
        base_fp += fp_b
        base_fn += fn_b

    rag_prec = rag_tp / max(1, (rag_tp + rag_fp))
    rag_rec = rag_tp / max(1, (rag_tp + rag_fn))
    rag_f1 = 2 * (rag_prec * rag_rec) / max(1e-6, (rag_prec + rag_rec))
    rag_hallucination = rag_fp / max(1, (rag_tp + rag_fp))

    base_prec = base_tp / max(1, (base_tp + base_fp))
    base_rec = base_tp / max(1, (base_tp + base_fn))
    base_f1 = 2 * (base_prec * base_rec) / max(1e-6, (base_prec + base_rec))
    base_hallucination = base_fp / max(1, (base_tp + base_fp))

    reduction_pct = (
        ((base_hallucination - rag_hallucination) / max(1e-6, base_hallucination) * 100.0)
        if base_hallucination > 0
        else 0.0
    )

    return {
        "evaluation_type": "empirical_grounding_eval",
        "citable": False,
        "status": "non-citable-synthetic-ablation",
        "methodology_note": "Scored empirically by comparing retrieved canonical IDs against query gold IDs with fixed top_k.",
        "circularity_acknowledgement": "Synthetic test queries drawn from KB domains are circular without an external held-out ontology (see main.tex:198).",
        "num_test_queries": len(test_queries),
        "fixed_top_k": top_k,
        "baseline_unconstrained_lexical": {
            "precision": round(base_prec, 3),
            "recall": round(base_rec, 3),
            "f1": round(base_f1, 3),
            "hallucination_rate": round(base_hallucination, 3),
        },
        "rag_augmented": {
            "precision": round(rag_prec, 3),
            "recall": round(rag_rec, 3),
            "f1": round(rag_f1, 3),
            "hallucination_rate": round(rag_hallucination, 3),
        },
        "hallucination_reduction_pct": round(reduction_pct, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Enhanced Domain RAG Extractor.")
    parser.add_argument("--query", type=str, default=None, help="Input lecture transcript text.")
    parser.add_argument("--eval-ablation", action="store_true", help="Run empirical RAG vs Baseline ablation.")
    parser.add_argument("--out", type=str, default=None, help="Save result to JSON path.")
    args = parser.parse_args()

    retriever = HybridRetriever(DEFAULT_KB)
    verifier = PrerequisiteDAGVerifier(DEFAULT_KB)

    if args.eval_ablation:
        benchmark_samples = [
            {
                "text": "Otelcilikte no-show durumunda voucher iptali ve dinamik fiyatlandırma revizyonu yapılır.",
                "gold_ids": ["TOUR-01", "TOUR-02", "TOUR-03"]
            },
            {
                "text": "İşyerinde ramak kala olayları kayıt altına alınmalı, risk değerlendirmesi güncellenmeli ve personele uygun KKD verilmelidir.",
                "gold_ids": ["ISG-01", "ISG-02", "ISG-03"]
            },
            {
                "text": "Veri sorumlusu aydınlatma yükümlülüğü kapsamında açık rıza almalı ve özel nitelikli kişisel verileri korumalıdır.",
                "gold_ids": ["KVKK-01", "KVKK-02", "KVKK-03", "KVKK-04"]
            },
            {
                "text": "Bankalarda kredi tahsisi öncesi KKB ve Findeks skoru incelenir, MASAK uyum kuralları işletilir.",
                "gold_ids": ["BANK-01", "BANK-02"]
            },
            {
                "text": "Şirket çalışanlarına oltalama ve phishing saldırılarına karşı çok faktörlü kimlik doğrulama zorunlu tutulmalıdır.",
                "gold_ids": ["CYBER-01", "CYBER-02"]
            }
        ]
        logger.info("Evaluating RAG vs Baseline Ablation on %d domain samples...", len(benchmark_samples))
        report = evaluate_rag_ablation(benchmark_samples)
        print("\n=== RAG ABLATION BENCHMARK REPORT ===")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        return

    sample_query = (
        args.query or
        "İşyerimizde yaşanan ramak kala olaylarının ardından yeni bir risk değerlendirmesi yaptık. "
        "Tüm çalışanlara baret ve koruyucu donanım (KKD) dağıtılarak acil durum eylem planı güncellendi."
    )

    logger.info("Processing Query with Hybrid RAG & DAG Verifier...")
    result = extract_with_rag(sample_query, retriever, verifier)

    print("\n--- RAG EXTRACTION & DAG VERIFICATION RESULT ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
