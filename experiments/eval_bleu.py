"""
eval_bleu.py: Compute BLEU scores for LLM-extracted explanations against Oxford English Dictionary references.
Supports reading from:
  1. JSONL files (e.g. outputs/ablation-all.jsonl, outputs/llm-*.jsonl)
  2. Text output directories (e.g. outputs/pilot_baseline/semantic_outputs/en)
"""

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
import sacrebleu


def normalize_term(t: str) -> str:
    if t is None:
        return ""
    t = unicodedata.normalize("NFKC", t)
    t = t.strip().replace("**", "").replace("’", "'").replace("`", "'")
    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"\s*\([^)]*\)\s*$", "", t).strip()
    t = re.sub(r"^[\s\.\,\:\;\-\–\—]+|[\s\.\,\:\;\!\?\-\–\—]+$", "", t)
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_oxford_refs(ref_path: Path) -> dict:
    raw = json.loads(ref_path.read_text(encoding="utf-8"))
    return {normalize_term(k): v.strip() for k, v in raw.items()}


def parse_terms_from_text(text: str) -> dict:
    terms = {}
    current_term = None
    current_lines = []
    header_re = re.compile(r"^(?:\s*(?:\d+[\.\)]|\-|\*)\s*)?(.+?)\s*[:\-–]\s*(.*)$")

    def flush():
        nonlocal current_term, current_lines
        if current_term:
            expl = " ".join([ln.strip() for ln in current_lines if ln.strip()]).strip()
            if expl:
                terms[current_term] = expl
        current_term = None
        current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = header_re.match(line)
        if m:
            flush()
            raw_term = m.group(1).strip()
            first_expl = m.group(2).strip()
            term = normalize_term(raw_term)
            if term:
                current_term = term
                if first_expl:
                    current_lines.append(first_expl)
            continue
        if current_term:
            current_lines.append(line)
    flush()
    return terms


def evaluate_jsonl(jsonl_path: Path, refs: dict, lang_filter: str = "en"):
    pairs = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("lang") != lang_filter:
                continue
            model = row.get("model", "unknown")
            asr_model = row.get("asr_model", "unknown")
            clip_id = row.get("id", "unknown")
            for item in row.get("terms", []):
                term_norm = normalize_term(item.get("term", ""))
                expl = (item.get("explanation") or "").strip()
                if term_norm in refs and expl:
                    pairs.append({
                        "model": model,
                        "asr_model": asr_model,
                        "clip_id": clip_id,
                        "term": term_norm,
                        "reference": refs[term_norm],
                        "candidate": expl
                    })
    return pairs


def compute_bleu_summary(pairs: list) -> list:
    groups = {}
    for p in pairs:
        key = (p["model"], p.get("asr_model", ""), p["clip_id"])
        groups.setdefault(key, []).append(p)

    summary = []
    for (model, asr_model, clip_id), items in sorted(groups.items()):
        refs = [it["reference"] for it in items]
        hyps = [it["candidate"] for it in items]
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        summary.append({
            "model": model,
            "asr_model": asr_model,
            "clip_id": clip_id,
            "n_matched_terms": len(items),
            "bleu": round(bleu.score, 2)
        })
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate BLEU on LLM explanations using Oxford references.")
    parser.add_argument("--refs", default="data/oxford_refs_en.json", help="Path to oxford_refs_en.json")
    parser.add_argument("--jsonl", help="Path to JSONL with extracted terms (e.g. outputs/ablation-all.jsonl)")
    parser.add_argument("--out", default=None, help="Output CSV path for BLEU scores")
    args = parser.parse_args()

    refs_path = Path(args.refs)
    if not refs_path.is_absolute():
        base = Path(__file__).resolve().parent
        refs_path = base / args.refs

    if not refs_path.exists():
        raise FileNotFoundError(f"References not found: {refs_path}")

    refs = load_oxford_refs(refs_path)
    print(f"Loaded {len(refs)} reference terms from {refs_path.name}")

    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        if not jsonl_path.is_absolute():
            jsonl_path = Path(__file__).resolve().parent / args.jsonl
        pairs = evaluate_jsonl(jsonl_path, refs)
        print(f"Matched {len(pairs)} term-explanation pairs against Oxford references.")

        summary = compute_bleu_summary(pairs)
        print("\n--- BLEU Summary ---")
        for row in summary:
            print(f"Model: {row['model']} | ASR: {row['asr_model']} | Clip: {row['clip_id']} | Matched: {row['n_matched_terms']} | BLEU: {row['bleu']}")

        if args.out:
            out_path = Path(args.out)
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["model", "asr_model", "clip_id", "n_matched_terms", "bleu"])
                writer.writeheader()
                writer.writerows(summary)
            print(f"\nSaved BLEU summary to: {out_path}")


if __name__ == "__main__":
    main()
