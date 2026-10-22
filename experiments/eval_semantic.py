"""eval_semantic.py: Semantic Similarity (BERTScore / Embedding Sim) and Readability Evaluator.

Replaces / supplements surface-level BLEU with deep semantic similarity (P, R, F1)
and evaluates pedagogical readability symmetrically across Turkish (Ateşman)
and English (Flesch-Kincaid).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics import atesman_score, flesch_score, readability_score


def normalize_term(t: str) -> str:
    """Normalize a term name for dictionary lookup."""
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", str(t))
    t = t.strip().replace("**", "").replace("’", "'").replace("`", "'")
    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"\s*\([^)]*\)\s*$", "", t).strip()
    t = re.sub(r"^[\s\.\,\:\;\-\–\—]+|[\s\.\,\:\;\!\?\-\–\—]+$", "", t)
    # Turkish-safe lowercase
    t = t.replace("İ", "i").replace("I", "ı").lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_references(ref_path: Path) -> dict[str, str]:
    """Load reference dictionary mapping normalized term -> reference definition."""
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")
    raw = json.loads(ref_path.read_text(encoding="utf-8"))
    return {normalize_term(k): str(v).strip() for k, v in raw.items()}


LANG_ALIASES = {
    "tr": "tr", "turk": "tr", "turkish": "tr", "türkçe": "tr", "turkce": "tr",
    "en": "en", "engl": "en", "english": "en", "ingilizce": "en",
}


def lang_code(value: str | None, clip_id: str = "") -> str:
    """Map 'tr' / 'Turkish' / 'English' / 'en' to a two-letter code.

    A plain startswith("tr") test silently classifies "Turkish" as English, which
    merges the two language groups, so full names are matched on explicit aliases.
    """
    v = str(value or "").strip().lower()
    for ch, ascii_ in (("ü", "u"), ("ş", "s"), ("ı", "i"), ("ğ", "g"), ("ö", "o"), ("ç", "c")):
        v = v.replace(ch, ascii_)
    for key, code in LANG_ALIASES.items():
        if v.startswith(key):
            return code
    return ("tr" if str(clip_id).upper().startswith("TR") else "en") if clip_id else "en"


def _ngram_vector(text: str, n: int = 3) -> Counter:
    """Character n-gram frequency counter for robust fallback similarity."""
    cleaned = re.sub(r"\s+", " ", text.lower().strip())
    if len(cleaned) < n:
        return Counter([cleaned])
    return Counter([cleaned[i : i + n] for i in range(len(cleaned) - n + 1)])


def _cosine_sim(vec1: Counter, vec2: Counter) -> float:
    """Cosine similarity between two frequency vectors."""
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[x] * vec2[x] for x in intersection)
    sum1 = sum(val**2 for val in vec1.values())
    sum2 = sum(val**2 for val in vec2.values())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    return float(numerator) / denominator


# Cached transformers pipeline / model if available
_EMBED_MODEL = None
_EMBED_TOKENIZER = None


def compute_bert_score(
    candidates: Sequence[str],
    references: Sequence[str],
    lang: str = "en",
    model_name: str | None = None,
) -> tuple[list[float], list[float], list[float], str]:
    """Compute semantic similarity (P, R, F1) plus the backend that produced them.

    Strategy 1 (true BERTScore) only runs when the optional ``bert_score`` package is
    importable; otherwise Strategy 2's character-3-gram cosine + word-overlap proxy is
    used, and callers must report it as a lexical proxy rather than a transformer metric.
    """
    if len(candidates) != len(references):
        raise ValueError("Candidates and references must have the same length")
    if not candidates:
        return [], [], [], "none"

    # Strategy 1: bert_score official package if requested or available
    if model_name or "bert_score" in sys.modules:
        try:
            import bert_score
            m_id = model_name or ("bert-base-multilingual-cased" if lang.startswith("tr") else "distilbert-base-uncased")
            P, R, F1 = bert_score.score(
                list(candidates),
                list(references),
                model_type=m_id,
                lang=lang,
                rescale_with_baseline=False,
                verbose=False,
            )
            return (
                [round(float(v), 4) for v in P],
                [round(float(v), 4) for v in R],
                [round(float(v), 4) for v in F1],
                f"bert_score::{m_id}",
            )
        except Exception:
            pass

    # Strategy 2: Fast deterministic subword & character n-gram cosine matching
    # Approximates soft precision, recall, and F1 by asymmetric token overlap
    p_list, r_list, f1_list = [], [], []
    for cand, ref in zip(candidates, references):
        c_vec = _ngram_vector(cand, n=3)
        r_vec = _ngram_vector(ref, n=3)

        # Token sets for word-level coverage
        c_words = set(re.findall(r"\w+", cand.lower()))
        r_words = set(re.findall(r"\w+", ref.lower()))

        # Cosine similarity on 3-grams
        cos_sim = _cosine_sim(c_vec, r_vec)

        # Precision: proportion of candidate content matched in reference
        word_p = len(c_words & r_words) / len(c_words) if c_words else 0.0
        # Recall: proportion of reference content covered by candidate
        word_r = len(c_words & r_words) / len(r_words) if r_words else 0.0

        # Weighted combination of n-gram cos_sim and word overlap
        p_val = 0.6 * cos_sim + 0.4 * word_p
        r_val = 0.6 * cos_sim + 0.4 * word_r
        f1_val = (2 * p_val * r_val / (p_val + r_val)) if (p_val + r_val) > 0 else 0.0

        p_list.append(round(min(1.0, max(0.0, p_val)), 4))
        r_list.append(round(min(1.0, max(0.0, r_val)), 4))
        f1_list.append(round(min(1.0, max(0.0, f1_val)), 4))

    return p_list, r_list, f1_list, "lexical-proxy::char3gram-cosine+word-overlap"


def evaluate_explanations(
    jsonl_path: Path,
    tr_refs: dict[str, str],
    en_refs: dict[str, str],
) -> list[dict]:
    """Evaluate candidate explanations from a JSONL file against reference definitions."""
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = row.get("id", "unknown")
            mid = row.get("model", "unknown").replace("Qwen/", "")
            asr_m = row.get("asr_model", "gold").replace("openai/", "")
            lang = lang_code(row.get("lang"), cid)
            refs = tr_refs if lang == "tr" else en_refs

            terms = row.get("terms", [])
            for item in terms:
                t_raw = item.get("term", "")
                t_norm = normalize_term(t_raw)
                expl = (item.get("explanation") or "").strip()
                if not expl:
                    continue

                ref_def = refs.get(t_norm)
                if not ref_def:
                    # Token-subset match in reference keys
                    for k_norm, v_def in refs.items():
                        if t_norm in k_norm or k_norm in t_norm:
                            ref_def = v_def
                            break

                readability = readability_score(expl, lang=lang)

                records.append({
                    "clip": cid,
                    "model": mid,
                    "asr": asr_m,
                    "lang": "TR" if lang == "tr" else "EN",
                    "term": t_raw,
                    "term_norm": t_norm,
                    "candidate_expl": expl,
                    "reference_def": ref_def or "",
                    "has_reference": bool(ref_def),
                    "readability_score": readability["score"],
                    "readability_level": readability["level"],
                })

    # Batch compute semantic similarity per language group, so an aggregate run over
    # mixed Turkish/English files cannot score one language's pairs with the other's model.
    matched_indices = [i for i, r in enumerate(records) if r["has_reference"]]
    for group in ("TR", "EN"):
        idxs = [i for i in matched_indices if records[i]["lang"] == group]
        if not idxs:
            continue
        cands = [records[i]["candidate_expl"] for i in idxs]
        refs = [records[i]["reference_def"] for i in idxs]
        p_scores, r_scores, f1_scores, backend = compute_bert_score(
            cands, refs, lang="tr" if group == "TR" else "en"
        )
        for idx, p_val, r_val, f1_val in zip(idxs, p_scores, r_scores, f1_scores):
            records[idx]["bert_precision"] = p_val
            records[idx]["bert_recall"] = r_val
            records[idx]["bert_f1"] = f1_val
            records[idx]["similarity_backend"] = backend

    return records


def summarize_evaluation(records: list[dict]) -> list[dict]:
    """Group records by (Model, ASR, Lang) and compute aggregate metrics."""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        key = (r["model"], r["asr"], r["lang"])
        groups.setdefault(key, []).append(r)

    summary = []
    for (model, asr_model, lang), items in sorted(groups.items()):
        n_total = len(items)
        with_ref = [it for it in items if it["has_reference"] and "bert_f1" in it]
        n_matched = len(with_ref)

        avg_bert_f1 = (sum(it["bert_f1"] for it in with_ref) / n_matched) if n_matched > 0 else 0.0
        avg_bert_p = (sum(it["bert_precision"] for it in with_ref) / n_matched) if n_matched > 0 else 0.0
        avg_bert_r = (sum(it["bert_recall"] for it in with_ref) / n_matched) if n_matched > 0 else 0.0

        avg_readability = sum(it["readability_score"] for it in items) / n_total if n_total > 0 else 0.0
        # Target pedagogical band: Ateşman 50-69 (TR) / Flesch 60-70 (EN) ~ CEFR B1.
        in_target = [it for it in items if 50.0 <= it["readability_score"] <= 69.0] if lang == "TR" \
            else [it for it in items if 60.0 <= it["readability_score"] <= 70.0]
        backends = sorted({it.get("similarity_backend", "none") for it in with_ref})

        summary.append({
            "model": model,
            "asr_model": asr_model,
            "lang": lang,
            "total_terms": n_total,
            "ref_matched_terms": n_matched,
            "bert_precision": round(avg_bert_p * 100, 2),
            "bert_recall": round(avg_bert_r * 100, 2),
            "bert_f1": round(avg_bert_f1 * 100, 2),
            "readability_avg": round(avg_readability, 2),
            "readability_metric": "Ateşman" if lang == "TR" else "Flesch",
            "readability_pct_in_target": round(100.0 * len(in_target) / n_total, 2) if n_total else 0.0,
            "similarity_backend": backends[0] if len(backends) == 1 else "+".join(backends),
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate semantic similarity (BERTScore) & readability")
    parser.add_argument("--jsonl", type=Path, nargs="+", required=True, help="Input JSONL file(s)")
    parser.add_argument("--tr-refs", type=Path, default=ROOT / "data" / "reference_defs_tr.json")
    parser.add_argument("--en-refs", type=Path, default=ROOT / "data" / "oxford_refs_en.json")
    parser.add_argument("--format", choices=["table", "markdown", "csv", "json"], default="table")
    parser.add_argument("--out", type=Path, default=None, help="Optional output path")
    args = parser.parse_args()

    tr_refs = load_references(args.tr_refs) if args.tr_refs.exists() else {}
    en_refs = load_references(args.en_refs) if args.en_refs.exists() else {}

    all_records: list[dict] = []
    for p in args.jsonl:
        if not p.exists():
            print(f"Warning: {p} not found", file=sys.stderr)
            continue
        all_records.extend(evaluate_explanations(p, tr_refs, en_refs))

    summary = summarize_evaluation(all_records)

    if args.format == "table":
        header = f"{'Model':<16} {'ASR System':<22} {'Lang':<5} {'Terms':<6} {'Matched':<8} {'BERT-P(%)':<10} {'BERT-R(%)':<10} {'BERT-F1(%)':<11} {'Readability':<12}"
        print("\n" + "=" * 105)
        print(" SEMANTIC QUALITY & SYMMETRIC READABILITY BENCHMARK (Atesman / Flesch / BERTScore)")
        print("=" * 105)
        print(header)
        print("-" * 105)
        for s in summary:
            print(
                f"{s['model']:<16} {s['asr_model']:<22} {s['lang']:<5} {s['total_terms']:<6} {s['ref_matched_terms']:<8} "
                f"{s['bert_precision']:<10.1f} {s['bert_recall']:<10.1f} {s['bert_f1']:<11.1f} "
                f"{s['readability_avg']:<5.1f} ({s['readability_metric']})"
            )
        print("=" * 105)

    elif args.format == "markdown":
        print("| Model | ASR System | Lang | Terms | Matched | BERT-P (%) | BERT-R (%) | BERT-F1 (%) | Readability |")
        print("|---|---|---|---:|---:|---:|---:|---:|---|")
        for s in summary:
            print(
                f"| {s['model']} | {s['asr_model']} | {s['lang']} | {s['total_terms']} | {s['ref_matched_terms']} | "
                f"{s['bert_precision']:.1f} | {s['bert_recall']:.1f} | {s['bert_f1']:.1f} | "
                f"{s['readability_avg']:.1f} ({s['readability_metric']}) |"
            )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.suffix == ".json":
            args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            with args.out.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else [])
                writer.writeheader()
                writer.writerows(summary)
        print(f"\nSaved evaluation summary to: {args.out}")


if __name__ == "__main__":
    main()
