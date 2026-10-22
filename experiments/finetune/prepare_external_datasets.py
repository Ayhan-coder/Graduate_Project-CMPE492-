"""External Turkish corpus converter for the two trainable layers.

Katman 2 (terminology LLM) gets Alpaca/ChatML rows in the same schema as
``train_lecture_notes.jsonl``: 70% general Turkish instruction pairs and 30%
entity-grounded term rows. Katman 1 (lecture ASR) gets 16 kHz mono wav files
plus a manifest, because an ASR adapter cannot be trained from JSONL.

Two honesty rules are enforced by construction, not by comment:
* Nothing from ``experiments/data/gold`` (the 7 held-out evaluation clips) may
  reach a training row -- see ``LeakageGuard``.
* Every row carries the dataset it came from and that dataset's licence, and the
  totals land in ``provenance_manifest.json`` next to the data.

Usage:
    python -m experiments.finetune.prepare_external_datasets --general-rows 1400 \
        --term-rows 600 --asr-hours 1.5
    python -m experiments.finetune.prepare_external_datasets --skip-audio   # text only
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExternalDatasets")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "experiments" / "finetune" / "data"
OUT_DIR = DATA_DIR / "external"
AUDIO_DIR = OUT_DIR / "asr_audio"
GOLD_DIR = ROOT / "experiments" / "data" / "gold"

# Resolved with the HF API on 2026-10-17; recorded per row so a reviewer can audit it.
SOURCES: Dict[str, Dict[str, str]] = {
    "turkish_llm_v10": {
        "dataset": "ogulcanaydogan/Turkish-LLM-v10-Training",
        "licence": "apache-2.0",
        "role": "Katman 2 general instruction pairs",
    },
    "turklink": {
        "dataset": "yakdas/turklink-corpus",
        "licence": "cc-by-sa-4.0",
        "role": "Katman 2 Wikidata-grounded term rows",
    },
    "commonvoice": {
        "dataset": "ysdede/commonvoice_17_tr_fixed",
        "licence": "cc0-1.0",
        "role": "Katman 1 read speech",
    },
    "khanacademy": {
        "dataset": "ysdede/khanacademy-turkish",
        "licence": "cc-by-sa-3.0",
        "role": "Katman 1 lecture-domain speech",
    },
}

GENERAL_SYSTEM = (
    "Sen Türkçe yardımcı bir asistansın. Kullanıcının sorusunu Türkçe yanıtla."
)
TERM_SYSTEM = (
    "Sen ders notlarını ve eğitim konuşmalarını analiz eden, teknik terimleri tespit edip "
    "CEFR B1 düzeyinde sade açıklamalar üreten ve Bloom taksonomi seviyesini atayan uzman bir asistansın."
)
TERM_INSTRUCTION = (
    "Aşağıdaki ders notu ve transkript metninden teknik terimleri tespit et. "
    "Her terim için kanonik ID, CEFR B1 düzeyinde sadeleştirilmiş anlaşılır bir açıklama ve "
    "Bloom taksonomi seviyesini ata. Sonucu yapılandırılmış geçerli JSON olarak ver."
)

# Bloom level is a documented heuristic over the definition wording, not an annotation.
_BLOOM_CUES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Analiz", ("fark", "neden", "ilişki", "karşılaştır", "ayrıştır", "etki")),
    ("Uygulama", ("yöntem", "kullanılır", "hesaplanır", "uygulanır", "aracı", "yordam")),
    ("Anlama", ("türü", "tür", "biçimi", "şeklinde", "sayesinde", "amacı")),
    ("Hatırlama", ("adı", "kimdir", "nedir", "olan", "ifade")),
)

_TURKISH_FOLD = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")

# Turkish Wikipedia leads follow "X, ... tanımlayıcı yüklemi." -- anything that cannot be
# read as a definition (event recaps, award lists) is dropped rather than trained on.
_DEF_COPULA_ENDINGS = ("dır", "dir", "dur", "tur", "tır", "tir", "lardır", "lerdir", "lığı", "liği")
_DEF_CUES = ("türü", "bilimi", "alanı", "kavram", "yöntem", "süreç", "birim",
             "sistem", "kurum", "dalı", "yasası", "kanunu", "kuralı", "teşkilatı", "hastalığı")


def _fold(text: str) -> str:
    """Case- and diacritic-insensitive form used for every overlap test."""
    out = unicodedata.normalize("NFKD", text)
    out = "".join(c for c in out if not unicodedata.combining(c))
    return out.translate(_TURKISH_FOLD).lower()


def _tokens(text: str) -> List[str]:
    return re.findall(r"[^\s]+", _fold(text))


# Cues and copulas are folded once here: matching an unfolded "türü" against folded text
# would silently never fire.
_DEF_COPULA = re.compile("(" + "|".join(_fold(e) for e in _DEF_COPULA_ENDINGS) + ")$")
_DEF_CUES_FOLDED = tuple(_fold(c) for c in _DEF_CUES)


class LeakageGuard:
    """Blocks the held-out evaluation material from becoming training signal.

    Tier A rejects a term row when its term is one of the gold answer-key aliases.
    Tier B rejects any candidate whose wording shares an 8-gram with a gold
    transcript, which catches verbatim eval sentences without penalising ordinary
    Turkish text that merely mentions a common word such as "kontrol".
    """

    def __init__(self, gold_dir: Path = GOLD_DIR, ngram: int = 8) -> None:
        self.ngram = ngram
        self.eval_terms: set[str] = set()
        self.eval_grams: set[Tuple[str, ...]] = set()
        self.gold_clips: List[str] = []

        gt = gold_dir.parent / "gold_terms.json"
        if gt.exists():
            for clip, payload in json.loads(gt.read_text(encoding="utf-8")).items():
                self.gold_clips.append(clip)
                for t in payload.get("terms", []):
                    for alias in [t.get("canonical", "")] + list(t.get("aliases", [])):
                        if alias:
                            self.eval_terms.add(_fold(alias))
        for txt in sorted(gold_dir.glob("*.txt")):
            grams = self._grams(txt.read_text(encoding="utf-8", errors="ignore"))
            self.eval_grams.update(grams)

        logger.info(
            "Leakage guard loaded: %d clips, %d eval term aliases, %d gold %d-grams",
            len(self.gold_clips), len(self.eval_terms), len(self.eval_grams), ngram,
        )

    def _grams(self, text: str) -> Iterable[Tuple[str, ...]]:
        toks = _tokens(text)
        return (tuple(toks[i:i + self.ngram]) for i in range(0, max(0, len(toks) - self.ngram + 1)))

    def term_is_eval_answer(self, term: str) -> bool:
        return _fold(term) in self.eval_terms

    def text_is_eval(self, text: str) -> bool:
        return any(g in self.eval_grams for g in self._grams(text))

    def stats(self) -> Dict[str, Any]:
        return {
            "gold_clips": self.gold_clips,
            "eval_term_aliases": len(self.eval_terms),
            "gold_ngrams": len(self.eval_grams),
            "ngram_size": self.ngram,
        }


# --------------------------------------------------------------------------- #
# Katman 2: general instruction pairs (70%)
# --------------------------------------------------------------------------- #

def build_general_rows(limit: int, guard: LeakageGuard, seed: int,
                       dropped: Dict[str, int]) -> List[Dict[str, Any]]:
    """Sample general Turkish prompt/completion pairs out of the full split."""
    from datasets import load_dataset

    meta = SOURCES["turkish_llm_v10"]
    logger.info("Streaming %s (%s) for %d general rows", meta["dataset"], meta["licence"], limit)
    stream = load_dataset(meta["dataset"], split="train", streaming=True).shuffle(seed=seed, buffer_size=4096)

    rows: List[Dict[str, Any]] = []
    for item in stream:
        if len(rows) >= limit:
            break
        prompt = (item.get("prompt") or "").strip()
        completion = (item.get("completion") or "").strip()
        if len(prompt) < 12 or len(completion) < 16 or len(completion) > 900:
            dropped["too_short"] += 1
            continue
        blob = f"{prompt}\n{completion}"
        if guard.text_is_eval(blob):
            dropped["general_leak"] += 1
            continue
        rows.append({
            "system": GENERAL_SYSTEM,
            "instruction": prompt,
            "input": "",
            "output": completion,
            "source": "turkish_llm_v10",
            "licence": meta["licence"],
        })
    logger.info("General rows kept: %d", len(rows))
    return rows


# --------------------------------------------------------------------------- #
# Katman 2: entity-grounded term rows (30%)
# --------------------------------------------------------------------------- #

def _lead_sentence(text: str, title: str) -> str:
    """Take the definitional first sentence of a Wikipedia lead, heading echo removed.

    Turkish Wikipedia stores the article title as a heading, so the raw text opens with
    the title twice before the real sentence. Only that exact doubled prefix is stripped;
    a single occurrence is the sentence's own subject and has to stay.
    """
    body = re.sub(r"\s+", " ", text or "").strip()
    t = title.strip()
    if t and _fold(body).startswith(_fold(f"{t} {t}")):
        body = body[len(t):].strip()
    parts = re.split(r"(?<=[.!?])\s+", body)
    return parts[0].strip() if parts else ""


def _is_definitional(lead: str) -> bool:
    folded = _fold(lead).strip().rstrip(".!?;: ")
    return bool(_DEF_COPULA.search(folded)) or any(cue in folded for cue in _DEF_CUES_FOLDED)


def _bloom_from(definition: str) -> str:
    folded = _fold(definition)
    for level, cues in _BLOOM_CUES:
        if any(_fold(c) in folded for c in cues):
            return level
    return "Hatırlama"

def _resolve_qids(titles: List[str], timeout: int = 25) -> Dict[str, str]:
    """Map article titles to Wikidata QIDs through the MediaWiki API, in batches."""
    import urllib.parse
    import urllib.request

    resolved: Dict[str, str] = {}
    url = "https://tr.wikipedia.org/w/api.php"
    headers = {"User-Agent": "EduTerm-AI-dataset-builder/0.1 (course project)"}
    for start in range(0, len(titles), 50):
        batch = titles[start:start + 50]
        query = urllib.parse.urlencode({
            "action": "query", "prop": "pageprops", "ppprop": "wikibase_item",
            "redirects": 1, "titles": "|".join(batch), "format": "json",
        })
        try:
            req = urllib.request.Request(f"{url}?{query}", headers=headers)
            payload = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            for page in payload.get("query", {}).get("pages", {}).values():
                qid = page.get("pageprops", {}).get("wikibase_item")
                if qid and "missing" not in page:
                    resolved[page.get("title", "")] = qid
        except Exception as exc:  # offline or rate limited: curid fallback still identifies the term
            logger.warning("Wikidata batch failed (%s); falling back to trwiki curid", str(exc)[:120])
            break
    logger.info("Wikidata resolved %d/%d term titles", len(resolved), len(titles))
    return resolved


def build_term_rows(limit: int, guard: LeakageGuard, seed: int, dropped: Dict[str, int],
                    scan_cap: int, resolve_wikidata: bool) -> List[Dict[str, Any]]:
    """Mine term/definition rows from the Wikidata-aligned Turkish Wikipedia corpus.

    Streaming reads arrive in corpus order, so rows are taken on a fixed stride to
    sample across the head of the file instead of only its first pages.
    """
    from datasets import load_dataset

    meta = SOURCES["turklink"]
    stride = max(1, scan_cap // max(1, limit * 6))
    logger.info("Streaming %s (%s): stride %d, scanning up to %d rows",
                meta["dataset"], meta["licence"], stride, scan_cap)

    candidates: List[Dict[str, Any]] = []
    seen_terms: set[str] = set()
    for idx, row in enumerate(load_dataset(meta["dataset"], split="train", streaming=True)):
        if idx >= scan_cap:
            break
        if idx % stride:
            continue
        title = (row.get("title") or "").strip()
        cats = row.get("categories") or []
        lead = _lead_sentence(row.get("text", ""), title)
        if not (3 <= len(title) <= 60):
            dropped["term_title"] += 1
            continue
        if not (70 <= len(lead) <= 320):
            dropped["term_lead"] += 1
            continue
        if re.search(r"\d{4}|Kategori:", lead):
            dropped["term_lead"] += 1
            continue
        if not _is_definitional(lead):
            dropped["term_not_definitional"] += 1
            continue
        folded = _fold(title)
        if folded in seen_terms:
            continue
        if guard.term_is_eval_answer(title):
            dropped["term_is_eval_answer"] += 1
            continue
        if guard.text_is_eval(lead):
            dropped["term_leak"] += 1
            continue
        seen_terms.add(folded)
        candidates.append({
            "title": title,
            "domain": cats[0] if cats else "Genel",
            "lead": lead,
            "curid": str(row.get("id", "")).strip(),
        })
        if len(candidates) >= limit:
            break

    qids = _resolve_qids([c["title"] for c in candidates]) if (resolve_wikidata and candidates) else {}

    rows: List[Dict[str, Any]] = []
    for c in candidates:
        qid = qids.get(c["title"])
        canonical = qid or (f"trwiki:curid:{c['curid']}" if c["curid"] else "")
        if not canonical:
            dropped["term_no_id"] += 1
            continue
        payload = {
            "domain": c["domain"],
            "terms": [{
                "term": c["title"],
                "canonical_id": canonical,
                "definition": c["lead"],
                "bloom_level": _bloom_from(c["lead"]),
                "cefr_target": "B1",
                "definition_source": "tr.wikipedia.org lead sentence",
            }],
        }
        rows.append({
            "system": TERM_SYSTEM,
            "instruction": TERM_INSTRUCTION,
            "input": f"{c['title']}. {c['lead']}",
            "output": json.dumps(payload, ensure_ascii=False),
            "source": "turklink",
            "licence": meta["licence"],
            "canonical_id_kind": "wikidata_qid" if qid else "trwiki_curid",
        })
    logger.info("Term rows kept: %d (%d with a Wikidata QID)",
                len(rows), sum(1 for r in rows if r["canonical_id_kind"] == "wikidata_qid"))
    return rows


# --------------------------------------------------------------------------- #
# Katman 1: adaptation speech
# --------------------------------------------------------------------------- #

def _write_wave(raw_bytes: bytes, suffix: str, out_path: Path, budget: Dict[str, float]) -> Optional[float]:
    """Decode one HF audio blob to 16 kHz mono wav and return its length in seconds."""
    import io

    import numpy as np
    import soundfile as sf

    try:
        data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    except Exception as exc:
        logger.debug("Undecodable audio (%s): %s", suffix, str(exc)[:80])
        return None
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        sr = 16000
    dur = float(len(data)) / sr
    if dur < 1.0 or dur > 30.0:
        budget["skipped_length"] += 1
        return None
    sf.write(str(out_path), data, sr, subtype="PCM_16")
    return dur


def build_asr_corpus(hours: float, guard: LeakageGuard, seed: int,
                     per_source: Dict[str, float]) -> Dict[str, Any]:
    """Pull a bounded, eval-clean speech set for the Whisper adapter."""
    from datasets import Audio, load_dataset

    budget_s = hours * 3600.0
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    per_source_stats: Dict[str, Any] = {}

    for key, share in per_source.items():
        meta = SOURCES[key]
        target_s = budget_s * share
        accrued = 0.0
        budget = {"skipped_length": 0.0}
        dropped = {"audio_leak": 0, "undecodable": 0}
        logger.info("Streaming %s (%s) for %.2f h of Katman 1 audio",
                    meta["dataset"], meta["licence"], target_s / 3600.0)
        stream = (load_dataset(meta["dataset"], split="train", streaming=True)
                  .cast_column("audio", Audio(decode=False)))
        out_dir = AUDIO_DIR / key
        out_dir.mkdir(parents=True, exist_ok=True)

        n = 0
        t0 = time.time()
        for row in stream:
            if accrued >= target_s:
                break
            audio = row.get("audio") or {}
            raw = audio.get("bytes")
            if not raw:
                continue
            text_key = "text" if "text" in row else "transcription"
            text = re.sub(r"\s+", " ", str(row.get(text_key, ""))).strip()
            if len(text) < 12 or guard.text_is_eval(text):
                dropped["audio_leak" if text else "undecodable"] += 1
                continue
            suffix = Path(str(audio.get("path") or "clip")).suffix or ".wav"
            clip_id = f"{key}-{n:06d}"
            dur = _write_wave(raw, suffix, out_dir / f"{clip_id}.wav", budget)
            if dur is None:
                dropped["undecodable"] += 1
                continue
            accrued += dur
            n += 1
            manifest.append({
                "audio_path": str((out_dir / f"{clip_id}.wav").relative_to(OUT_DIR)),
                "target_text": text,
                "duration_s": round(dur, 3),
                "sampling_rate": 16000,
                "source": key,
                "licence": meta["licence"],
                "split": "train",
            })
            if n % 200 == 0:
                logger.info("  %s: %d clips / %.1f s of %.1f s", key, n, accrued, target_s)

        per_source_stats[key] = {
            "clips": n,
            "hours": round(accrued / 3600.0, 4),
            "seconds": round(accrued, 1),
            "wallclock_s": round(time.time() - t0, 1),
            "dropped": {**dropped, "length_out_of_range": int(budget["skipped_length"])},
        }
        logger.info("  %s done: %d clips, %.2f h", key, n, accrued / 3600.0)

    # Validation split is taken from the pooled clips; ASR needs no speaker split here
    # because CommonVoice's own validated/test splits were never read.
    random.Random(seed).shuffle(manifest)
    n_val = max(1, int(len(manifest) * 0.05)) if manifest else 0
    for row in manifest[:n_val]:
        row["split"] = "validation"

    return {"clips": manifest, "per_source": per_source_stats, "validation_clips": n_val}


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prepare(general_rows: int, term_rows: int, asr_hours: float, seed: int,
            skip_audio: bool, skip_wikidata: bool, term_scan_cap: int,
            asr_share: Dict[str, float]) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dropped = {k: 0 for k in (
        "too_short", "general_leak", "term_title", "term_lead", "term_not_definitional",
        "term_is_eval_answer", "term_leak", "term_no_id", "audio_leak", "undecodable")}
    guard = LeakageGuard()
    t0 = time.time()

    general = build_general_rows(general_rows, guard, seed, dropped)
    terms = build_term_rows(term_rows, guard, seed, dropped, term_scan_cap, not skip_wikidata)

    pool = general + terms
    random.Random(seed).shuffle(pool)
    n_val = max(1, int(len(pool) * 0.1)) if pool else 0
    val_rows, train_rows = pool[:n_val], pool[n_val:]

    train_file = OUT_DIR / "train_external_mixed.jsonl"
    val_file = OUT_DIR / "val_external_mixed.jsonl"
    _write_jsonl(train_file, train_rows)
    _write_jsonl(val_file, val_rows)

    counts = {
        "general_requested": general_rows,
        "general_produced": len(general),
        "term_requested": term_rows,
        "term_produced": len(terms),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "general_share_of_kept": round(len(general) / max(1, len(pool)), 4),
        "term_share_of_kept": round(len(terms) / max(1, len(pool)), 4),
        "dropped": dropped,
    }
    logger.info("Katman 2 mix: %d train / %d val, general %.1f%% vs terms %.1f%%",
                len(train_rows), len(val_rows),
                100 * counts["general_share_of_kept"], 100 * counts["term_share_of_kept"])

    asr_stats: Dict[str, Any] = {"skipped": True}
    if not skip_audio and asr_hours > 0:
        result = build_asr_corpus(asr_hours, guard, seed, asr_share)
        _write_jsonl(OUT_DIR / "asr_manifest.jsonl", result["clips"])
        asr_stats = {k: v for k, v in result.items() if k != "clips"}
        asr_stats["skipped"] = False
        asr_stats["manifest"] = str((OUT_DIR / "asr_manifest.jsonl").relative_to(ROOT))
    elif (OUT_DIR / "provenance_manifest.json").exists():
        # Re-running only the text mix must not erase the measured speech stats.
        previous = json.loads((OUT_DIR / "provenance_manifest.json").read_text(encoding="utf-8"))
        if previous.get("katman1", {}).get("clips") or previous.get("katman1", {}).get("per_source"):
            asr_stats = dict(previous["katman1"])
            asr_stats["preserved_from_previous_build"] = previous.get("generated_utc")

    manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": "experiments/finetune/prepare_external_datasets.py",
        "seed": seed,
        "wallclock_s": round(time.time() - t0, 1),
        "intended_mix": {"general": 0.7, "term": 0.3},
        "sources": SOURCES,
        "katman2": counts,
        "katman1": asr_stats,
        "leakage_guard": guard.stats(),
        "files": {
            "train": str(train_file.relative_to(ROOT)),
            "validation": str(val_file.relative_to(ROOT)),
        },
        "note": (
            "Term definitions are Turkish Wikipedia lead sentences, not hand-checked B1 "
            "glosses; they teach schema and Wikidata grounding. B1 simplification targets "
            "still come only from the locally distilled set."
        ),
    }
    (OUT_DIR / "provenance_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", OUT_DIR / "provenance_manifest.json")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Convert external Turkish HF datasets into trainable data.")
    p.add_argument("--general-rows", type=int, default=1400)
    p.add_argument("--term-rows", type=int, default=600)
    p.add_argument("--asr-hours", type=float, default=1.5)
    p.add_argument("--asr-share", type=str, default="commonvoice=0.5,khanacademy=0.5")
    p.add_argument("--term-scan-cap", type=int, default=6000)
    p.add_argument("--seed", type=int, default=20261017)
    p.add_argument("--skip-audio", action="store_true", help="Build Katman 2 text only.")
    p.add_argument("--skip-wikidata", action="store_true", help="Use trwiki curid instead of QID lookups.")
    args = p.parse_args()

    share = {}
    for pair in args.asr_share.split(","):
        k, _, v = pair.partition("=")
        if k.strip() in SOURCES:
            share[k.strip()] = float(v or 0)
    total = sum(share.values()) or 1.0
    share = {k: v / total for k, v in share.items()}

    manifest = prepare(
        general_rows=args.general_rows,
        term_rows=args.term_rows,
        asr_hours=args.asr_hours,
        seed=args.seed,
        skip_audio=args.skip_audio,
        skip_wikidata=args.skip_wikidata,
        term_scan_cap=args.term_scan_cap,
        asr_share=share,
    )
    print(json.dumps(manifest["katman2"], ensure_ascii=False, indent=2))
    print(json.dumps(manifest["katman1"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
