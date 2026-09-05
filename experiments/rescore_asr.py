"""Recompute WER/CER on an existing ASR JSONL against current gold.

Does not re-run models. Writes a new JSONL (or overwrites --output).

Usage:
    python rescore_asr.py --asr outputs/asr-20260901-large-v3.jsonl
    python rescore_asr.py --asr outputs/asr-whisper-combined.jsonl --output outputs/asr-whisper-combined.rescored.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gold_io import resolve_reference
from metrics import pair_metrics
from run_asr import load_manifest

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.jsonl")
    parser.add_argument("--output", type=Path, help="single output when one --asr file is given")
    args = parser.parse_args()

    manifest = {row["id"]: row for row in load_manifest(args.manifest)}
    for src in args.asr:
        out = args.output if args.output and len(args.asr) == 1 else src.with_name(src.stem + ".rescored.jsonl")
        rows = []
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            clip = manifest.get(rec.get("id"), {})
            rec["confirmed"] = bool(clip.get("confirmed"))
            rec["reference_source"] = clip.get("reference_source") or rec.get("reference_source") or "missing"
            hyp = rec.get("hypothesis") or ""
            ref = resolve_reference(clip)
            for key in ("wer_raw", "cer_raw", "wer_norm", "cer_norm"):
                rec.pop(key, None)
            if ref and hyp and not rec.get("error") and not rec.get("skipped"):
                rec.update(pair_metrics(ref, hyp, lang=rec.get("lang") or clip.get("lang")))
            rows.append(rec)
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print("wrote", out, "n=", len(rows))


if __name__ == "__main__":
    main()
