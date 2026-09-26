"""Evaluate model-extracted terms against closed gold domain standard for TR-1 and EN-1.

Computes exact Precision, Recall, and F1 to answer the coverage question.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GOLD_PATH = ROOT / "data" / "gold_terms.json"


def turkish_lower(text: str) -> str:
    """Lowercase text with proper Turkish dotted/dotless I handling."""
    return text.replace("İ", "i").replace("I", "ı").lower()


def normalize_term(term: str) -> str:
    """Normalize term for robust semantic matching."""
    t = turkish_lower(term.strip())
    # Strip leading articles
    for prefix in ("the ", "a ", "an "):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
    # Normalize unicode to NFC
    t = unicodedata.normalize("NFC", t)
    # Remove punctuation
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def evaluate_clip(
    extracted_terms: list[str | dict],
    gold_spec: dict,
) -> dict:
    """Evaluate extracted terms against gold specification for one clip."""
    gold_terms = gold_spec["terms"]
    # Build alias to canonical mapping
    alias_to_canonical: dict[str, str] = {}
    for g in gold_terms:
        canon = g["canonical"]
        alias_to_canonical[normalize_term(canon)] = canon
        for alias in g.get("aliases", []):
            alias_to_canonical[normalize_term(alias)] = canon

    # Sort aliases by length descending so longer/more specific phrases match first
    sorted_aliases = sorted(alias_to_canonical.keys(), key=len, reverse=True)

    recalled_gold = set()
    matched_extracted = []
    unmatched_extracted = []

    for item in extracted_terms:
        term_text = item["term"] if isinstance(item, dict) else str(item)
        norm = normalize_term(term_text)

        matched_canon = None
        # 1. Exact alias match
        if norm in alias_to_canonical:
            matched_canon = alias_to_canonical[norm]
        else:
            # 2. Token-boundary match: check if a full gold alias occurs in norm
            for alias_norm in sorted_aliases:
                pattern = r"(?:\b|^)" + re.escape(alias_norm) + r"(?:\b|$)"
                if re.search(pattern, norm):
                    matched_canon = alias_to_canonical[alias_norm]
                    break

        if matched_canon:
            recalled_gold.add(matched_canon)
            matched_extracted.append({"raw": term_text, "matched": matched_canon})
        else:
            unmatched_extracted.append(term_text)

    tp = len(recalled_gold)
    n_emitted = len(extracted_terms)
    fp = len(unmatched_extracted)
    fn = len(gold_terms) - tp

    precision = (tp / n_emitted) if n_emitted > 0 else 0.0
    recall = (tp / len(gold_terms)) if gold_terms else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "gold_count": len(gold_terms),
        "emitted_count": n_emitted,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "recalled_gold": sorted(list(recalled_gold)),
        "missed_gold": sorted([g["canonical"] for g in gold_terms if g["canonical"] not in recalled_gold]),
        "false_positives": unmatched_extracted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate extracted terms against gold domain list")
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--jsonl", type=Path, nargs="+", help="Path(s) to run_llm or ablation jsonl")
    parser.add_argument("--format", choices=["table", "markdown", "latex"], default="table")
    args = parser.parse_args()

    with args.gold.open(encoding="utf-8") as f:
        gold_data = json.load(f)

    if not args.jsonl:
        print("Please pass --jsonl <path>...")
        return

    # Group extracted terms by (mid, asr, cid)
    records: list[dict] = []
    for path in args.jsonl:
        if not path.exists():
            print(f"Warning: {path} not found")
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                cid = row.get("id")
                if cid not in gold_data:
                    continue
                mid = row.get("model", "").replace("Qwen/", "")
                asr = row.get("asr_model", "gold / none").replace("openai/", "")
                terms = row.get("terms", [])
                res = evaluate_clip(terms, gold_data[cid])
                records.append({
                    "model": mid,
                    "asr": asr,
                    "clip": cid,
                    "lang": "TR" if cid.startswith("TR") else "EN",
                    **res
                })

    if args.format == "table":
        print(f"{'Model':<16} {'ASR':<25} {'Clip':<6} {'Gold':<5} {'Emitted':<8} {'TP':<4} {'P (%)':<7} {'R (%)':<7} {'F1 (%)':<7}")
        print("-" * 90)
        for r in records:
            print(
                f"{r['model']:<16} {r['asr']:<25} {r['clip']:<6} {r['gold_count']:<5} {r['emitted_count']:<8} "
                f"{r['tp']:<4} {r['precision']*100:<7.1f} {r['recall']*100:<7.1f} {r['f1']*100:<7.1f}"
            )
    elif args.format == "markdown":
        print("| Model | ASR Input | Clip | Gold | Emitted | TP | P (%) | R (%) | F1 (%) |")
        print("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for r in records:
            print(
                f"| {r['model']} | {r['asr']} | {r['clip']} | {r['gold_count']} | {r['emitted_count']} | "
                f"{r['tp']} | {r['precision']*100:.1f} | {r['recall']*100:.1f} | {r['f1']*100:.1f} |"
            )
    elif args.format == "latex":
        for r in records:
            print(
                f"{r['model']} & {r['asr']} & {r['clip']} & {r['gold_count']} & {r['emitted_count']} & "
                f"{r['tp']} & {r['precision']*100:.1f}\\% & {r['recall']*100:.1f}\\% & {r['f1']*100:.1f}\\% \\\\"
            )


if __name__ == "__main__":
    main()
