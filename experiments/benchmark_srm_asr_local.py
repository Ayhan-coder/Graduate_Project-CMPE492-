"""Measure SRM candidates (whisper-tiny/base/small, Qwen3-ASR-0.6B) on the gold clips.

run_asr.py only accepts ids listed in models.json, so this driver imports it and
calls run_one() directly with the same decoding config, metrics (pair_metrics ->
wer_norm/cer_norm/rtf) and manifest. Only the model list is new; nothing about the
scoring path changes. Peak VRAM is torch's peak allocated over the run.

Usage:
    python benchmark_srm_asr_local.py --models tiny base small qwen3asr
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_asr as R  # noqa: E402

OUT_DIR = ROOT / "outputs"
MANIFEST = ROOT / "data" / "manifest.jsonl"
IDS = ["TR-1", "TR-2", "TR-3", "EN-1", "EN-2", "EN-3"]

SRM = {
    "tiny": {"id": "openai/whisper-tiny", "backend": "whisper", "subset": "multi", "params_m": 39.0},
    "base": {"id": "openai/whisper-base", "backend": "whisper", "subset": "multi", "params_m": 74.0},
    "small": {"id": "openai/whisper-small", "backend": "whisper", "subset": "multi", "params_m": 244.0},
    "qwen3asr": {"id": "Qwen/Qwen3-ASR-0.6B-hf", "backend": "qwen3_asr", "subset": "multi", "params_m": 620.0},
}


def decoding_args() -> Namespace:
    return Namespace(
        chunk_s=30.0,
        language_detect=False,
        condition_on_prev=False,
        use_vad=False,
        prompt_glossary=False,
        prompt_text=None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="SRM ASR measurement on the gold manifest.")
    parser.add_argument("--models", nargs="+", default=list(SRM), choices=list(SRM))
    parser.add_argument("--ids", nargs="+", default=IDS)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "srm_asr_measured.json")
    args = parser.parse_args()

    rows = [r for r in R.load_manifest(MANIFEST) if r.get("id") in set(args.ids)]
    if not rows:
        raise SystemExit("no manifest rows matched")
    dec = decoding_args()

    report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "clip_ids": [r["id"] for r in rows],
              "decoding": R.decoding_config(dec), "results": []}
    if args.out.exists():
        report["results"] = json.loads(args.out.read_text(encoding="utf-8")).get("results", [])

    for key in args.models:
        entry = SRM[key]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        records, t0 = [], time.time()
        for row in rows:
            try:
                rec = R.run_one(dict(entry), row, dec)
            except Exception as exc:
                rec = {"id": row["id"], "model": entry["id"], "error": f"{type(exc).__name__}: {exc}"[:300]}
            records.append(rec)

        ok = [r for r in records if r.get("wer_norm") is not None]
        dur = sum(r.get("audio_duration_s") or 0 for r in ok)
        row_out = {
            "key": key,
            "model": entry["id"],
            "params_m": entry["params_m"],
            "status": "measured" if ok else "failed",
            "clips_scored": len(ok),
            "clips_total": len(records),
            "mean_wer_norm": round(sum(r["wer_norm"] for r in ok) / len(ok), 4) if ok else None,
            "mean_cer_norm": round(sum(r["cer_norm"] for r in ok) / len(ok), 4) if ok else None,
            "mean_rtf": round(sum(r["rtf"] for r in ok) / len(ok), 4) if ok else None,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
            "wall_seconds": round(time.time() - t0, 1),
            "audio_seconds": round(dur, 1),
            "decoding": R.decoding_config(dec),
            "per_clip": [{k: r.get(k) for k in ("id", "lang", "wer_norm", "cer_norm", "rtf", "error")} for r in records],
        }
        report["results"] = [r for r in report["results"] if r.get("model") != entry["id"]] + [row_out]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: row_out[k] for k in ("model", "status", "clips_scored", "mean_wer_norm",
                                                  "mean_cer_norm", "mean_rtf", "peak_vram_gb")},
                         ensure_ascii=False), flush=True)

    print("\n| model | WER(norm) | CER(norm) | RTF | peak VRAM | clips |")
    for r in report["results"]:
        print(f"| {r['model']} | {r['mean_wer_norm']} | {r['mean_cer_norm']} | {r['mean_rtf']} | "
              f"{r['peak_vram_gb']} GB | {r['clips_scored']}/{r['clips_total']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
