"""Relevance gate for the 100-clip candidate pool (rule-based, recorded on disk).

Discovery (discover_100plus.py) trusts YouTube search ranking, which returns personal
development / sales / interview content for Turkish domain queries. Keeping that material
would silently change what "100 ders videosu" means, so every candidate is scored here
against an explicit, auditable rule set and the decision is written to the artifact.

Decision per candidate:
  keep   - title matches a lecture marker, or the channel is on the allowlist
  drop   - title matches a non-lecture marker (marketing, self-help, interview, vlog)
  review - neither; a human decides (kept out of the corpus until then)

Usage:
    python experiments/data/filter_candidates_100plus.py
    python experiments/data/filter_candidates_100plus.py --pool experiments/data/candidates_100plus.json

Output: experiments/data/candidates_100plus_reviewed.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments" / "data"

LECTURE = [
    "ders", "ders anlat", "konu an", "ünite", "bölüm ", "eğitim", "akademi", "istanbul",
    "açık öğretim", "aöf", "kurs", "problem çöz", "örnek çö", "slayt", "sunum",
    "lecture", "tutorial", "course", "lesson", "chapter", "explained", "explanation",
    "example", "solved", "walkthrough", "introduction to", "crash course", "part 1",
    "part 2", "prof", "professor", "dr.", "university", "college", "school",
]
NON_LECTURE = [
    "pazarlama", "satış", "satış stratej", "ikna", "sır", "gizli", "motivasyon",
    "kişisel gelişim", "koçluk", "podcast", "röportaj", "mülakat", "vlog", "canlı yayın",
    "canlı ders" , "reaksiyon", "reaction", "shorts", "ilan", "kampanya", "indirim",
    "üye ol", "katıl", "sponsor", "yaptım", "deneyim", "hikayem", "my story",
    "marketing", "sales", "affiliate", "dropship", "make money", "earn money",
    "interview", "qa with", "tier list", "hot take", "drama", "storytime",
]
CHANNEL_ALLOW = [
    "anadolu üniversitesi", "açıköğretim", "khanacademy", "khan academy", "ocw",
    "mit", "boğaziçi", "itu", "odtü", "metu", "istinye", "termodinamikakademi",
    "engineer4free", "prof.", "professor", "dr.", "university", "college", "academy",
    "akademi", "académie", "fakülte", "hocam", "hoca", "ders", "bi'zaman", "tonguc",
    "crashcourse", "crash course", "amoeba", "ted-ed", "lab", "institute",
]

# per-domain topical markers: a candidate also counts as lecture-like when its title
# carries a term from the domain it was searched under
DOMAIN_TERMS: dict[str, list[str]] = {
    "Software Engineering & Project Management": ["proje", "yönetim", "izleme", "kontrol", "kazanılmış", "earned value", "schedule", "agile", "scrum", "evm"],
    "Computer Science & Data Science": ["veri", "veribilim", "normaliz", "algoritma", "yapılandırma", "data", "algorithm", "database", "machine", "python"],
    "Artificial Intelligence & Machine Learning": ["makine öğren", "öğrenme", "yapay zeka", "gradient", "regresyon", "regulariz", "ağ", "neural", "deep learning", "ml", "sınıfland"],
    "Mechanical & Thermal Engineering": ["termodinamik", "ısı", "entropi", "makina", "mekanik", "fluid", "thermodynamic", "heat", "entropy", "engine"],
    "Electrical Engineering & Circuit Theory": ["devre", "elektrik", "akkor", "thevenin", "norton", "circuit", "voltage", "current", "ohm", "kirc"],
    "Mathematics (Linear Algebra, Calculus)": ["lineer", "matris", "özdeğer", "vektör", "türev", "integral", "linear algebra", "matrix", "eigen", "calculus", "limit"],
    "Physics (Mechanics, Electromagnetism)": ["fizik", "kuvvet", "enerji", "momentum", "indüksiyon", "manyetik", "physics", "force", "motion", "electromagnet", "faraday"],
    "Economics & Finance": ["ekonomi", "arz", "talep", "fiyat", "maliyet", "pahalılık", "economics", "supply", "demand", "elasticity", "finance", "budget"],
    "Cognitive Psychology & Neuroscience": ["bilişsel", "psikoloji", "hafıza", "bellek", "dikkat", "nöron", "learning techniques", "cognitive", "psychology", "memory", "attention", "neuro"],
    "History & Cultural Heritage": ["tarih", "medeniyet", "miras", "arkeoloji", "cumhuriyet", "osmanlı", "history", "heritage", "archaeolog", "civilization"],
    "Biology & Genetics": ["biyoloji", "hücre", "mitoz", "mayoz", "kalıtım", "gen", "canlı", "biology", "cell", "mitosis", "meiosis", "genetics", "dna"],
    "Tourism & Sustainable Hospitality": ["turizm", "turist", "sürdürülebilir", "otel", "konaklama", "tourism", "hospitality", "sustainable", "hotel", "travel"],
    "Occupational Health and Safety (ISG / OHS)": ["iş sağlığı", "güvenlik", "iş kazası", "risk", "tehlike", "meslek", "yangın", "occupational", "safety", "hazard", "risk assessment", "osha"],
    "Corporate Compliance & Personal Data Protection (KVKK/GDPR)": ["kvkk", "kişisel veri", "veri güven", "gizlilik", "aydınlatma", "gdpr", "data protection", "privacy", "compliance", "personal data"],
}



def fold(s: str) -> str:
    s = (s or "").casefold()
    for a, b in (("ı", "i"), ("İ", "i"), ("ğ", "g"), ("ş", "s"), ("ö", "o"), ("ü", "u"), ("ç", "c")):
        s = s.replace(a, b)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def hits(text: str, patterns: list[str]) -> list[str]:
    t = fold(text)
    return [p for p in patterns if fold(p) in t]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=DATA / "candidates_100plus.json")
    ap.add_argument("--out", type=Path, default=DATA / "candidates_100plus_reviewed.json")
    args = ap.parse_args()

    pool = json.loads(args.pool.read_text(encoding="utf-8"))["candidates"]
    bucketed: dict[str, list[dict]] = defaultdict(list)
    for c in pool:
        title_hits = hits(c["title"], NON_LECTURE)
        ch_hits = hits(c["channel"], CHANNEL_ALLOW)
        domain_hits = hits(c["title"], DOMAIN_TERMS.get(c["domain"], []))
        lecture_hits = hits(c["title"], LECTURE)
        # a non-lecture marker wins unless the title is also an obvious lecture AND the
        # channel is academic; that combination is rare and belongs in review, not keep
        if title_hits and not (lecture_hits and ch_hits):
            decision, reason = "drop", "non-lecture isaret: " + "|".join(title_hits[:3])
        elif lecture_hits:
            decision, reason = "keep", "lecture isaret: " + "|".join(lecture_hits[:3])
        elif ch_hits:
            decision, reason = "keep", "akademik kanal: " + "|".join(ch_hits[:2])
        elif domain_hits:
            decision, reason = "keep", "alan terimi: " + "|".join(domain_hits[:3])
        else:
            decision, reason = "review", "isaret yok"
        c2 = dict(c)
        c2["decision"] = decision
        c2["decision_reason"] = reason
        bucketed[decision].append(c2)

    for name, items in bucketed.items():
        items.sort(key=lambda c: (c["domain"], c["lang"], -c["duration_s"]))

    by_cell = defaultdict(Counter)
    for c in bucketed["keep"]:
        by_cell[f"{c['domain']}|{c['lang']}"]["keep"] += 1

    out = {
        "note": "Rule-filtered candidate pool for the 100-clip expansion. Rules and the "
                "per-candidate decision are stored so the selection is auditable; nothing "
                "here is gold and audio duration is still the DECLARED youtube duration "
                "(measured duration must be verified after download).",
        "rules": {"lecture_markers": LECTURE, "non_lecture_markers": NON_LECTURE,
                  "channel_allowlist": CHANNEL_ALLOW, "domain_terms": DOMAIN_TERMS},
        "pool_size": len(pool),
        "counts": {k: len(v) for k, v in bucketed.items()},
        "kept_by_lang": dict(Counter(c["lang"] for c in bucketed["keep"])),
        "kept_by_domain": dict(Counter(c["domain"] for c in bucketed["keep"])),
        "kept_by_cell": {k: dict(v) for k, v in sorted(by_cell.items())},
        "keep": bucketed["keep"],
        "review": bucketed["review"],
        "drop": bucketed["drop"],
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    hours = sum(c["duration_s"] for c in bucketed["keep"]) / 3600
    print(f"havuz {len(pool)} -> keep {len(bucketed['keep'])} ({hours:.1f} ilan edilmis saat), "
          f"review {len(bucketed['review'])}, drop {len(bucketed['drop'])}")
    print(f"keep lang: {out['kept_by_lang']}")
    thin = {k: v["keep"] for k, v in out["kept_by_cell"].items() if v["keep"] < 3}
    print("3'ten az kalan hucreler:", thin if thin else "yok")
    print("yazildi:", args.out)


if __name__ == "__main__":
    main()
