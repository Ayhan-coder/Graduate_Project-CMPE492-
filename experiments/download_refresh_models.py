"""Robust sequential downloader for 2026 refresh ASR models.

Downloads only the necessary model weights and configs without concurrent
requests to avoid HF Hub HTTP 429 rate limiting.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "hf_cache"

MODELS = {
    "selimc/whisper-large-v3-turbo-turkish": [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "normalizer.json",
        "added_tokens.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "model.safetensors",
    ],
    "Qwen/Qwen3-ASR-0.6B-hf": [
        "config.json",
        "generation_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "model.safetensors",
    ],
    "Qwen/Qwen3-ASR-1.7B-hf": [
        "config.json",
        "generation_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "model.safetensors",
    ],
    "nvidia/nemotron-3.5-asr-streaming-0.6b": [
        "config.json",
        "generation_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors",
    ],
}


def download_model(repo_id: str, files: list[str], max_retries: int = 5):
    print(f"\n=======================================================")
    print(f" Downloading: {repo_id}")
    print(f" Cache dir  : {CACHE_DIR}")
    print(f"=======================================================")

    for i, fname in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {fname} ...", flush=True)
        success = False
        for attempt in range(1, max_retries + 1):
            try:
                dest = hf_hub_download(
                    repo_id=repo_id,
                    filename=fname,
                    cache_dir=str(CACHE_DIR),
                )
                print(f"    OK -> {Path(dest).name} ({Path(dest).stat().st_size / (1024*1024):.2f} MB)", flush=True)
                success = True
                break
            except Exception as e:
                wait_s = min(30, 2 ** attempt)
                print(f"    Attempt {attempt}/{max_retries} failed: {e}. Retrying in {wait_s}s...", flush=True)
                time.sleep(wait_s)
        if not success:
            raise SystemExit(f"Failed to download {fname} from {repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    targets = list(MODELS.keys()) if args.model == "all" else [args.model]
    for mid in targets:
        download_model(mid, MODELS[mid])
    print("\nAll requested models downloaded successfully!")


if __name__ == "__main__":
    main()
