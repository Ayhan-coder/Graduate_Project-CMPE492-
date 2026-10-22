"""summarize_interventions.py: rebuild the acoustic intervention ladder from artifacts.

Stages (all Whisper Large-v3, same 7 gold clips, same host):
  (1) baseline          : outputs/asr-whisper-combined.rescored.jsonl
  (2) + energy VAD      : outputs/asr-vad-large-v3.jsonl
  (3) + VAD + biasing   : outputs/asr-vad-bias-large-v3.jsonl
  (4) + post-correction : outputs/post_correction.jsonl  (corrects stage 3 hypotheses)

Every number is recomputed from a JSONL on disk; nothing is typed by hand.
Terminology is reported as *mention coverage* over the closed gold list, never as
extraction F1, because a transcript is not an emitted term list. Biasing injects the
gold canonical terms themselves, so its coverage row is leakage and is labelled as such.

Usage:
    python summarize_interventions.py [--out outputs/interventions_summary.json]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"

from eval_gold_terms import normalize_term  # noqa: E402
from gold_io import resolve_reference  # noqa: E402
from metrics import pair_metrics  # noqa: E402
from run_asr import load_manifest  # noqa: E402

MODEL = "openai/whisper-large-v3"


def load_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error"):
            continue
        if r.get("model") and r["model"] != MODEL and "source_asr" not in r:
            continue
        rows[r["id"]] = r
    return rows


def mention_coverage(text: str, entries: Sequence[dict]) -> float:
    """Share of gold concepts mentioned in a transcript (canonical or alias)."""
    if not entries:
        return float("nan")
    norm = normalize_term(text or "")
    hits = 0
    for entry in entries:
        variants = [entry.get("canonical", ""), *entry.get("aliases", [])]
        for variant in variants:
            alias = normalize_term(variant or "")
            if alias and re.search(r"(?:\b|^)" + re.escape(alias) + r"(?:\b|$)", norm):
                hits += 1
                break
    return hits / len(entries)


def repetition_rate(text: str, n: int = 8) -> float:
    """Fraction of n-word spans that are duplicates -- a looping detector."""
    toks = (text or "").split()
    if len(toks) < n * 2:
        return 0.0
    spans = [" ".join(toks[i:i + n]).lower() for i in range(len(toks) - n + 1)]
    return round(1.0 - len(set(spans)) / len(spans), 4)


def gold_entries(clip_id: str) -> list[dict]:
    path = ROOT / "data" / "gold_terms.json"
    if not path.exists():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")).get(clip_id, {}).get("terms", []))


def _mean(values: list, ndigits: int) -> float | None:
    """Macro mean that reports absence instead of crashing on an empty cell."""
    vals = [v for v in values if v is not None]
    return round(statistics.mean(vals), ndigits) if vals else None


def summarise(stage: str, rows: dict[str, dict], texts: dict[str, str], refs: dict[str, str],
              langs: dict[str, str], coverages: dict[str, float],
              wer_key: str = "wer_norm", cer_key: str = "cer_norm") -> dict:
    if not rows:
        return {"stage": stage, "missing": True}
    per_lang: dict[str, dict] = {}
    for lang in ("tr", "en"):
        ids = [cid for cid in rows if langs.get(cid) == lang]
        if not ids:
            continue
        wers = [rows[cid].get(wer_key) for cid in ids]
        cers = [rows[cid].get(cer_key) for cid in ids]
        cov = [coverages[cid] for cid in ids if cid in coverages]
        rep = [repetition_rate(texts[cid]) for cid in ids if cid in texts]
        corr_lat = [round(sum(rows[cid].get("correction_latency_s") or []), 2) for cid in ids]
        corr_lat = [c for c in corr_lat if c]
        per_lang[lang] = {
            "clips": ids,
            "wer_norm_macro": _mean(wers, 5),
            "cer_norm_macro": _mean(cers, 5),
            "gold_mention_coverage": _mean(cov, 4),
            "repetition_rate": _mean(rep, 4),
            "latency_s_mean": _mean([rows[cid].get("latency_s") for cid in ids], 2),
            "correction_latency_s_sum": round(sum(corr_lat), 2) or None,
        }
    return {"stage": stage, "per_lang": per_lang,
            "per_clip": {cid: {
                "wer_norm": rows[cid].get(wer_key),
                "cer_norm": rows[cid].get(cer_key),
                "latency_s": rows[cid].get("latency_s")
                if rows[cid].get("latency_s") is not None
                else (round(sum(rows[cid].get("correction_latency_s") or []), 2) or None),
                "gold_mention_coverage": round(coverages[cid], 4) if cid in coverages else None,
                "repetition_rate": repetition_rate(texts[cid]) if cid in texts else None,
            } for cid in rows}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=OUT_DIR / "asr-whisper-combined.rescored.jsonl")
    parser.add_argument("--vad", type=Path, default=OUT_DIR / "asr-vad-large-v3.jsonl")
    parser.add_argument("--bias", type=Path, default=OUT_DIR / "asr-vad-bias-large-v3.jsonl")
    parser.add_argument("--correction", type=Path, default=OUT_DIR / "post_correction.jsonl")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "interventions_summary.json")
    args = parser.parse_args()

    manifest = {r["id"]: r for r in load_manifest(ROOT / "data" / "manifest.jsonl")}
    ids = [cid for cid, row in manifest.items() if row.get("lang") in ("tr", "en")]
    refs = {cid: resolve_reference(manifest[cid]) for cid in ids}
    langs = {cid: manifest[cid]["lang"] for cid in ids}

    base_all = load_rows(args.baseline)
    vad_all = load_rows(args.vad)
    bias_all = load_rows(args.bias)
    corr_all = load_rows(args.correction)

    stages = [
        ("(1) baseline (no VAD)", base_all, {}),
        ("(2) + energy VAD", vad_all, {}),
        ("(3) + VAD + glossary biasing", bias_all, {}),
    ]
    summary = []
    for name, rows, _ in stages:
        rows = {cid: rows[cid] for cid in rows if cid in rows and langs.get(cid)}
        texts = {cid: rows[cid].get("hypothesis", "") for cid in rows}
        cov = {cid: mention_coverage(texts[cid], gold_entries(cid)) for cid in texts if gold_entries(cid)}
        summary.append(summarise(name, rows, texts, refs, langs, cov))

    # Stage 4 corrects the stage-3 hypotheses; WER fields are raw_/corr_ prefixed there.
    corr_rows = {cid: r for cid, r in corr_all.items() if langs.get(cid)}
    if corr_rows:
        texts = {cid: r.get("corrected_text", "") for cid, r in corr_rows.items()}
        cov = {cid: mention_coverage(texts[cid], gold_entries(cid)) for cid in texts if gold_entries(cid)}
        entry = summarise("(4) + LLM post-correction", corr_rows, texts, refs, langs, cov,
                          wer_key="corr_wer_norm", cer_key="corr_cer_norm")
        entry["stage_of_input"] = "(3) + VAD + glossary biasing"
        summary.append(entry)
    else:
        summary.append({"stage": "(4) + LLM post-correction", "missing": True})

    payload = {
        "model": MODEL,
        "clips": sorted(langs, key=lambda c: (langs[c], c)),
        "note": ("Terminology column is mention coverage over the closed gold list "
                 "(canonical + aliases, token-boundary match), not extraction F1. "
                 "Biasing injects the gold canonical terms via prompt_ids, so its "
                 "coverage is leakage and only its WER is interpretable."),
        "stages": summary,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Written: {args.out}\n")
    for s in summary:
        if s.get("missing"):
            print(f"{s['stage']}: artifact missing")
            continue
        for lang, d in s["per_lang"].items():
            def fmt(v, nd=4):
                return "  ---  " if v is None else f"{v:.{nd}f}"
            print(f"{s['stage']:34s} {lang.upper()} "
                  f"WER {fmt(d['wer_norm_macro'])}  CER {fmt(d['cer_norm_macro'])}  "
                  f"rep {fmt(d['repetition_rate'])}  cov {fmt(d['gold_mention_coverage'])}  "
                  f"lat {fmt(d['latency_s_mean'], 1)}s  corr {fmt(d['correction_latency_s_sum'], 1)}s")


if __name__ == "__main__":
    main()
