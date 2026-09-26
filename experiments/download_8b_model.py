"""Downloader for Qwen3-8B (8 Billion parameter LLM) terminology extractor.

Downloads the necessary safetensors shards, tokenizer, and configuration files
into experiments/hf_cache/ with retry logic to avoid connection drops.

Usage:
    python experiments/download_8b_model.py
    python experiments/download_8b_model.py --model Qwen/Qwen3.5-9B
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "hf_cache"

MODELS = {
    "Qwen/Qwen3-8B": [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
        "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
    ],
    "Qwen/Qwen3.5-9B": [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "preprocessor_config.json",
        "model.safetensors.index.json",
        "model.safetensors-00001-of-00004.safetensors",
        "model.safetensors-00002-of-00004.safetensors",
        "model.safetensors-00003-of-00004.safetensors",
        "model.safetensors-00004-of-00004.safetensors",
    ],
    "google/gemma-4-E4B-it": [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "processor_config.json",
        "model.safetensors",
    ],
    "meta-llama/Llama-3.1-8B-Instruct": [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "model.safetensors.index.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    ],
    "unsloth/Meta-Llama-3.1-8B-Instruct": [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "model.safetensors.index.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    ],
    "meta-llama/Llama-3.3-8B-Instruct": [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "model.safetensors.index.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    ],
}


def download_model(repo_id: str, files: list[str], max_retries: int = 5):
    print(f"\n=======================================================")
    print(f" Downloading 8B/9B Extractor: {repo_id}")
    print(f" Target cache directory     : {CACHE_DIR}")
    print(f" Total files to download    : {len(files)}")
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
                sz_mb = Path(dest).stat().st_size / (1024 * 1024)
                print(f"    OK -> {Path(dest).name} ({sz_mb:.2f} MB)", flush=True)
                success = True
                break
            except Exception as e:
                wait_s = min(30, 2 ** attempt)
                print(f"    Attempt {attempt}/{max_retries} failed: {e}. Retrying in {wait_s}s...", flush=True)
                time.sleep(wait_s)
        if not success:
            raise SystemExit(f"Failed to download {fname} from {repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Download 8B/9B LLM weights for terminology extraction")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="Qwen/Qwen3-8B",
        help="Which model to download (default: Qwen/Qwen3-8B)",
    )
    args = parser.parse_args()

    download_model(args.model, MODELS[args.model])
    print(f"\nModel {args.model} downloaded successfully and ready for GPU extraction!")


if __name__ == "__main__":
    main()
