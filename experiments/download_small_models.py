"""Fetch the small-model checkpoints needed by benchmark_small_models_local.py.

Weights land in experiments/hf_cache so fetch_models.resolve_hub_cache finds them
and nothing is pulled twice. Only safetensors + tokenizer/config files are fetched;
the duplicate *.bin shards are ignored.

Usage:
    python download_small_models.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "hf_cache"

ALLOW = [
    "*.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "*.txt",
    "*.model",
    "*.tiktoken",
]
IGNORE = ["*.bin", "*.h5", "*.msgpack", "*.onnx", "*.pth", "*.gguf", "*.pt", "original/*"]

REPOS = [
    ("openai/whisper-tiny", "SRM (ASR)"),
    ("openai/whisper-base", "SRM (ASR)"),
    ("openai/whisper-small", "SRM (ASR)"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "SLM (local NF4 baseline)"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "SLM (local NF4 baseline)"),
    ("Qwen/Qwen2.5-3B-Instruct", "SLM"),
    # meta-llama/Llama-3.2-1B-Instruct returns 401 without a HF token; the unsloth
    # mirror serves the identical weights publicly. Recorded as a mirror in the report.
    ("unsloth/Llama-3.2-1B-Instruct", "SLM (mirror of meta-llama/Llama-3.2-1B-Instruct)"),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "SLM"),
]


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    results = []
    for repo, role in REPOS:
        t0 = time.time()
        rec = {"repo": repo, "role": role}
        try:
            path = snapshot_download(
                repo_id=repo,
                cache_dir=str(CACHE),
                allow_patterns=ALLOW,
                ignore_patterns=IGNORE,
                max_workers=4,
            )
            size_gb = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) / 1e9
            rec.update(status="ok", path=path, size_gb=round(size_gb, 2), seconds=round(time.time() - t0, 1))
        except Exception as exc:  # gated repo, offline, network error
            rec.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:400])
        results.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\n{len(ok)}/{len(results)} repos ready in {CACHE}", flush=True)
    (ROOT / "outputs" / "small_models_download.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
