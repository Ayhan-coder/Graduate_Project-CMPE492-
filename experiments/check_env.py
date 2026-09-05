"""Print the decoding/hardware block used in the paper appendix."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "none (PyTorch CPU or no CUDA build)"


def main() -> None:
    try:
        import torch
        torch_ver = f"{torch.__version__}  cuda={torch.cuda.is_available()}"
    except Exception as exc:
        torch_ver = f"not importable ({exc})"

    rec = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "torch": torch_ver,
        "gpu": gpu_name(),
        "openai_key": bool(os.environ.get("OPENAI_API_KEY")),
        "manifest_present": (ROOT / "data" / "manifest.jsonl").exists(),
        "default_decoding": {
            "task": "transcribe",
            "language": "forced tr|en when clip.lang is set",
            "chunk_length_s": 30,
            "longform": "sequential_generate",
            "return_timestamps": True,
            "condition_on_prev_tokens": False,
        },
    }
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
