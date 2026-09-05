"""Download selected Hugging Face checkpoints for the paper refresh.

By default only configs / tokenizers are fetched (registry refreshed 2026-09-04).
Pass --weights to pull the full models into experiments/hf_cache/.

Usage:
    python fetch_models.py
    python fetch_models.py --weights --only openai/whisper-large-v3-turbo
    python fetch_models.py --list
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "models.json"
SNAP_DIR = ROOT / "hf_snapshots"
CACHE_DIR = ROOT / "hf_cache"


def hub_slug(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def default_hub_cache() -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub"


def resolve_hub_cache(model_id: str | None = None) -> Path:
    """Prefer the kit cache, then the user Hugging Face hub cache.

    fetch_models.py --weights writes into experiments/hf_cache.
    Earlier Whisper Medium runs landed in ~/.cache/huggingface/hub.
    from_pretrained must see the same directory or it re-downloads.
    """
    kit = CACHE_DIR
    default = default_hub_cache()
    if model_id:
        slug = hub_slug(model_id)
        if (kit / slug).exists():
            return kit
        if (default / slug).exists():
            return default
    kit.mkdir(parents=True, exist_ok=True)
    return kit
CONFIG_FILES = (
    "config.json",
    "preprocessor_config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "README.md",
)


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def selected_ids(reg: dict, include_llm: bool = True) -> list[dict]:
    items = list(reg["asr_selected"])
    if include_llm:
        items.extend(m for m in reg["llm_selected"] if m.get("provider") == "huggingface")
    return items


def cmd_list(reg: dict) -> None:
    print(f"Fetched at {reg['fetched_at']}")
    print("\nSelected ASR")
    for m in reg["asr_selected"]:
        print(f"  [{m['role']}] {m['id']}  rev={m['revision'][:10]}  langs={m['langs']}")
    print("\nSelected LLM")
    for m in reg["llm_selected"]:
        print(f"  [{m['role']}] {m['id']}  provider={m['provider']}")
    print("\nRejected (incompatible with TR+EN off-the-shelf)")
    for m in reg["rejected"]:
        print(f"  {m['id']}\n      {m['reason']}")


def fetch_configs(models: list[dict]) -> None:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    for m in models:
        dest = SNAP_DIR / m["id"].replace("/", "--")
        dest.mkdir(parents=True, exist_ok=True)
        got = []
        for fname in CONFIG_FILES:
            try:
                hf_hub_download(
                    repo_id=m["id"],
                    filename=fname,
                    revision=m.get("revision"),
                    local_dir=str(dest),
                )
                got.append(fname)
            except Exception:
                continue
        print(f"configs  {m['id']}: {got}")


WEIGHT_IGNORE = (
    "*.fp32*",
    "flax_model*",
    "tf_model*",
    "pytorch_model.bin*",
    "*.h5",
    "*.msgpack",
)


def fetch_weights(models: list[dict], retries: int = 6) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for m in models:
        print(f"weights  {m['id']} ...")
        last = None
        for attempt in range(1, retries + 1):
            try:
                snapshot_download(
                    repo_id=m["id"],
                    revision=m.get("revision"),
                    cache_dir=str(CACHE_DIR),
                    ignore_patterns=WEIGHT_IGNORE,
                )
                last = None
                break
            except Exception as exc:
                last = exc
                wait = min(60, 2 ** attempt)
                print(f"         attempt {attempt}/{retries} failed: {exc}")
                print(f"         retry in {wait}s")
                time.sleep(wait)
        if last is not None:
            raise last
        print("         done")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", action="store_true", help="download full model weights")
    parser.add_argument("--only", action="append", default=[], help="restrict to these repo ids")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    reg = load_registry()
    if args.list:
        cmd_list(reg)
        return

    models = selected_ids(reg)
    if args.only:
        wanted = set(args.only)
        models = [m for m in models if m["id"] in wanted]
        if not models:
            raise SystemExit(f"no selected model matched --only {sorted(wanted)}")

    fetch_configs(models)
    if args.weights:
        fetch_weights(models)
    else:
        print("\nConfigs only. Re-run with --weights to download checkpoints into hf_cache/.")


if __name__ == "__main__":
    main()
