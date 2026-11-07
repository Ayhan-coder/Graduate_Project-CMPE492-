"""Build the 15-clip annotation item sheet (R/D rubric from PROTOCOL.md).

Scope (user decision, 2026-10-29): 7 pilot clips (TR-1..3, EN-1..4) whose items
already exist in annotation/labels_todo.csv, plus 8 corporate clips
(TR-17..20, EN-17..20) whose transcripts exist but have no items yet.

The 19 remaining "template" clips are excluded: their source URLs are placeholder
slugs and the audio does not match the declared durations, so annotating them
would produce labels for content that was never actually recorded.

Outputs
-------
experiments/annotation/labels_todo_15clip.csv
    item_id, video_id, model, lang, term, explanation, context, R, D, comment
experiments/outputs/corporate_terms_cache.json
    raw extraction provenance (chunk text, prompt version, latency)

Usage
-----
python experiments/data/build_item_sheet_15clip.py              # full build
python experiments/data/build_item_sheet_15clip.py --pilot-only # no LLM
python experiments/data/build_item_sheet_15clip.py --force      # ignore cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

TRANSCRIPT_DIR = ROOT / "experiments" / "outputs" / "transcripts" / "openai_whisper-large-v3-turbo"
PILOT_LABELS = ROOT / "experiments" / "annotation" / "labels_todo.csv"
GOLD_DIR = ROOT / "experiments" / "data" / "gold"
ANNOTATION_DIR = ROOT / "experiments" / "annotation"
OUT_CSV = ANNOTATION_DIR / "labels_todo_15clip.csv"
CACHE_JSON = ROOT / "experiments" / "outputs" / "corporate_terms_cache.json"

ASR_MODEL = "openai/whisper-large-v3-turbo"
LLM_MODEL = "Qwen/Qwen3.5-4B"
MODEL_TAG = f"{ASR_MODEL}|{LLM_MODEL}"
PROMPT_VERSION = "extract-v2"

CORPORATE = ["TR-17", "TR-18", "TR-19", "TR-20", "EN-17", "EN-18", "EN-19", "EN-20"]
PILOT = ["TR-1", "TR-2", "TR-3", "EN-1", "EN-2", "EN-3", "EN-4"]

CHUNK_CHARS = 1700
MAX_TERMS_PER_CHUNK = 4


def clean_text(text: str) -> str:
    """Drop ASR repetition loops (e.g. 148x 'İNTRO') and normalise whitespace."""
    text = re.sub(r"(\b[\wÇĞİÖŞÜçğıöşü]{2,}\b)(?:\s+\1\b){2,}", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 3]


def chunk_sentences(sents: list[str], size: int = CHUNK_CHARS) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    for s in sents:
        if buf and sum(len(x) + 1 for x in buf) + len(s) > size:
            chunks.append(" ".join(buf))
            buf = [s]
        else:
            buf.append(s)
    if buf:
        chunks.append(" ".join(buf))
    return chunks


EXTRACT_PROMPT_TR = """Bir eğitim transkriptinden ders terimleri çıkarıyorsun. Aşağıdaki metin parçasında geçen, konuya özgü TEKNİK KAVRAMLARI seç.

Metin:
{CHUNK}

Kurallar:
- Terim, metinde birebir (aynı harflerle) geçmek zorundadır. Uydurma terim üretme.
- Öncelik: kısaltmalar (KVKK, KKD, İSG), hukuki/teknik adlar ve tanımlı süreçler.
- Genel iş kelimelerini seçme: eğitim, şirket, çalışan, süreç, politika, uyum, kültür, durum, adım, eylem, yöntem.
- En fazla {MAX} terim seç. Aynı terimi tekrar etme.
- "explanation" alanına, terimi yeni başlayan birinin anlayacağı sade Türkçeyle en fazla 30 kelimede tanımla. Tanım döngüsel olmasın (terimin kendisini tekrar etmesin).
- "context" alanına, terimin geçtiği cümleyi metinden birebir kopyala.

Sadece şu JSON dizisini döndür:
[{{"term": "...", "explanation": "...", "context": "..."}}]"""

EXTRACT_PROMPT_EN = """You extract domain terminology from an educational transcript. Select topic-specific TECHNICAL CONCEPTS that occur in the passage below.

Passage:
{CHUNK}

Rules:
- Every term must literally occur in the passage, letter for letter. Never invent a term.
- Prefer acronyms (GDPR, PII, DPO), legal/technical nouns and formally defined procedures.
- Do not select generic business words: training, company, employee, process, policy, compliance, culture, thing, step, action, method.
- Select at most {MAX} terms. No duplicates.
- "explanation" must define the term in plain language a beginner understands, at most 30 words, and must not be circular.
- "context" must copy the sentence in which the term occurs, verbatim from the passage.

Return only this JSON array:
[{{"term": "...", "explanation": "...", "context": "..."}}]"""


def _norm(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower().strip()


# Post-filter guard: the prompt asks the model to skip these, this enforces it.
GENERIC_BLOCKLIST = {
    "eğitim", "şirket", "çalışan", "süreç", "politika", "uyum", "kültür", "durum",
    "adım", "eylem", "yöntem", "konu", "bilgi", "sistem", "kişi", "veri",
    "training", "company", "employee", "employees", "process", "policy",
    "compliance", "culture", "thing", "step", "action", "method", "topic",
    "information", "system", "person", "data", "business", "organization",
}


def parse_json_array(raw: str) -> list[dict]:
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        data = []
        for m in re.finditer(r"\{[^{}]*\}", raw[start : end + 1], flags=re.DOTALL):
            try:
                data.append(json.loads(m.group(0)))
            except json.JSONDecodeError:
                continue
    out = []
    for d in data if isinstance(data, list) else []:
        if not isinstance(d, dict):
            continue
        term = str(d.get("term", "")).strip()
        expl = str(d.get("explanation", "")).strip()
        ctx = str(d.get("context", "")).strip()
        if term and expl:
            out.append({"term": term, "explanation": expl, "context": ctx})
    return out


def extract_clip(video_id: str, lang: str, text: str, max_terms: int, force: bool, cache: dict) -> list[dict]:
    from run_llm import generate_hf

    digest = hashlib.sha1(f"{PROMPT_VERSION}|{lang}|{text}|{max_terms}".encode()).hexdigest()[:16]
    cached = cache.get(video_id)
    if cached and not force and cached.get("digest") == digest:
        print(f"{video_id}: cache hit, {len(cached['items'])} items")
        return cached["items"]

    tmpl = EXTRACT_PROMPT_TR if lang == "tr" else EXTRACT_PROMPT_EN
    chunks = chunk_sentences(sentences(text))
    seen: set[str] = set()
    items: list[dict] = []
    t0 = time.time()

    for idx, chunk in enumerate(chunks):
        if len(items) >= max_terms:
            break
        prompt = tmpl.format(CHUNK=chunk, MAX=MAX_TERMS_PER_CHUNK)
        raw = generate_hf(LLM_MODEL, prompt, max_new_tokens=384)
        got = parse_json_array(raw)
        kept = 0
        for cand in got:
            key = _norm(cand["term"])
            if key in seen or key in GENERIC_BLOCKLIST:
                continue
            # hallucination guard: the term must occur somewhere in the clip
            if key not in _norm(text):
                continue
            seen.add(key)
            if key not in _norm(cand.get("context", "")):
                cand["context"] = find_context(cand["term"], chunk) or find_context(cand["term"], text)
            items.append(cand)
            kept += 1
            if len(items) >= max_terms:
                break
        print(f"{video_id}: chunk {idx + 1}/{len(chunks)} -> +{kept} (total {len(items)})", flush=True)

    cache[video_id] = {
        "digest": digest,
        "prompt_version": PROMPT_VERSION,
        "model": LLM_MODEL,
        "asr_model": ASR_MODEL,
        "chunks": len(chunks),
        "max_terms": max_terms,
        "seconds": round(time.time() - t0, 1),
        "items": items,
    }
    return items


def find_context(term: str, text: str) -> str:
    key = _norm(term)
    if key not in _norm(text):
        return ""
    for s in sentences(text):
        if key in _norm(s):
            return s
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-only", action="store_true", help="skip LLM extraction")
    ap.add_argument("--force", action="store_true", help="ignore extraction cache")
    args = ap.parse_args()

    import csv

    rows: list[dict] = []

    # --- pilot clips: reuse existing items, add a context column from gold transcripts
    gold = {}
    for vid in PILOT:
        p = GOLD_DIR / f"{vid}.txt"
        if p.exists():
            gold[vid] = clean_text(p.read_text(encoding="utf-8", errors="replace"))

    with PILOT_LABELS.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            vid = r["video_id"]
            rows.append(
                {
                    "item_id": r["item_id"],
                    "video_id": vid,
                    "model": r["model"],
                    "lang": r["lang"],
                    "term": r["term"],
                    "explanation": r["explanation"],
                    "context": find_context(r["term"], gold.get(vid, "")),
                    "R": "",
                    "D": "",
                    "comment": "pilot; already labeled in labels_a1.csv",
                }
            )
    print(f"pilot items: {len(rows)}")

    # --- corporate clips: fresh extraction
    if not args.pilot_only:
        cache = json.loads(CACHE_JSON.read_text(encoding="utf-8")) if CACHE_JSON.exists() else {}
        for vid in CORPORATE:
            src = TRANSCRIPT_DIR / f"{vid}.json"
            if not src.exists():
                print(f"{vid}: transcript missing, skipped")
                continue
            d = json.loads(src.read_text(encoding="utf-8"))
            text = clean_text(d["full_text"])
            lang = "tr" if vid.startswith("TR") else "en"
            words = len(text.split())
            budget = max(4, min(24, words // 260))
            print(f"{vid}: {words} words after cleanup, target {budget} terms", flush=True)
            items = extract_clip(vid, lang, text, budget, args.force, cache)
            for i, it in enumerate(items, 1):
                rows.append(
                    {
                        "item_id": f"{vid}-{MODEL_TAG}-{i:03d}",
                        "video_id": vid,
                        "model": MODEL_TAG,
                        "lang": lang,
                        "term": it["term"],
                        "explanation": it["explanation"],
                        "context": it.get("context", ""),
                        "R": "",
                        "D": "",
                        "comment": f"auto-extracted {PROMPT_VERSION}",
                    }
                )
            CACHE_JSON.parent.mkdir(parents=True, exist_ok=True)
            CACHE_JSON.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["item_id", "video_id", "model", "lang", "term", "explanation", "context", "R", "D", "comment"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV} ({len(rows)} items, {len({r['video_id'] for r in rows})} clips)")


if __name__ == "__main__":
    main()
