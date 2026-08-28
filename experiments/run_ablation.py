"""Feed every ASR hypothesis into one LLM (cross-layer ablation).

Input: JSONL from run_asr.py (fields: id, lang, model, hypothesis).
The LLM and the prompt are the same for every ASR system.

Usage:
    python run_ablation.py --asr-outputs outputs/asr-TIMESTAMP.jsonl --model gpt-4o-mini
    python run_ablation.py --asr-outputs outputs/asr.jsonl --model gpt-4o-mini --only-models openai/whisper-medium facebook/wav2vec2-large-xlsr-53
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from run_llm import PROMPT, generate_hf, generate_openai, parse_terms
from fetch_models import load_registry

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def lang_name(code: str | None) -> str:
    c = (code or "").lower()
    if c.startswith("tr"):
        return "Turkish"
    if c.startswith("en"):
        return "English"
    return code or "English"


def llm_entry(model_id: str) -> dict:
    reg = load_registry()
    for m in reg["llm_selected"]:
        if m["id"] == model_id:
            return m
    raise SystemExit(f"{model_id} is not in models.json llm_selected")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-outputs", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="one LLM; default gpt-4o-mini (paper ablation plan)")
    parser.add_argument("--only-models", nargs="*", default=[],
                        help="restrict to these ASR model ids")
    args = parser.parse_args()

    llm = llm_entry(args.model)
    rows = load_jsonl(args.asr_outputs)
    wanted = set(args.only_models) if args.only_models else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"ablation-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            if row.get("skipped") or not row.get("hypothesis"):
                continue
            if wanted is not None and row.get("model") not in wanted:
                continue
            prompt = PROMPT.format(LANG=lang_name(row.get("lang")), TRANSCRIPT=row["hypothesis"])
            t0 = time.perf_counter()
            if llm.get("provider") == "openai_api":
                text = generate_openai(llm["id"], prompt)
            else:
                text = generate_hf(llm["id"], prompt)
            elapsed = time.perf_counter() - t0
            terms = parse_terms(text)
            rec = {
                "id": row.get("id"),
                "lang": row.get("lang"),
                "asr_model": row.get("model"),
                "asr_wer_norm": row.get("wer_norm"),
                "model": llm["id"],
                "latency_s": elapsed,
                "n_terms": len(terms),
                "terms": terms,
                "raw": text,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            print(f"  {row.get('id')} / {row.get('model')}: {len(terms)} terms in {elapsed:.1f}s")
    print(f"wrote {out_path} ({n} rows)")
    print("Next: label terms (annotation/PROTOCOL.md), then python report.py --ablation", out_path.name)


if __name__ == "__main__":
    main()
