"""measure_local_vram.py -- peak GPU memory actually occupied by the local 4-bit LLM.

Every VRAM figure published for the local checkpoints was previously an estimate. This
probe loads the real checkpoint through the same path the runners use
(run_llm.get_or_load_llm -> BitsAndBytes nf4 + device_map="auto"), runs one real
extraction call per clip, and records torch's peak allocated/reserved bytes.
Result: outputs/local_vram_probe.json
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3.5-4B"


def nvidia_smi_used_mi() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def main() -> None:
    import run_llm

    clips = [json.loads(l) for l in (HERE / "data" / "transcripts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    selected = [c for c in clips if c["id"] in ("TR-1", "EN-1")]

    torch.cuda.reset_peak_memory_stats()
    base = {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
    }

    records = []
    tok, model = run_llm.get_or_load_llm(MODEL_ID)
    after_load = {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "device_map": getattr(model, "hf_device_map", None),
        "nvsmi_used_mi": nvidia_smi_used_mi(),
    }

    for clip in selected:
        torch.cuda.reset_peak_memory_stats()
        prompt = run_llm.PROMPT.format(LANG=clip["lang"], TRANSCRIPT=clip["transcript"])
        started = time.time()
        text = run_llm.generate_hf(MODEL_ID, prompt)
        records.append({
            "clip": clip["id"],
            "asr_model": clip.get("asr_model"),
            "prompt_chars": len(prompt),
            "out_chars": len(text),
            "latency_s": round(time.time() - started, 2),
            "peak_allocated_mi": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "peak_reserved_mi": round(torch.cuda.max_memory_reserved() / 2**20, 1),
            "nvsmi_used_mi": nvidia_smi_used_mi(),
        })

    payload = {
        "model_id": MODEL_ID,
        "quantization": "nf4 double-quant, bf16 compute, device_map=auto",
        "host": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_mi": round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1),
        },
        "before_load": {k: round(v / 2**20, 1) for k, v in base.items()},
        "after_load": {
            **{k: (round(v / 2**20, 1) if isinstance(v, int) else v)
               for k, v in after_load.items() if k != "device_map"},
            "weights_on_gpu_mi": round(sum(
                p.detach().numel() * p.element_size()
                for p in model.parameters() if p.device.type == "cuda") / 2**20, 1),
            "device_map": after_load["device_map"],
        },
        "runs": records,
        "note": "peak_* is torch's view of its own allocator; nvsmi_used_mi is total device occupancy including context/other processes.",
    }

    out = HERE / "outputs" / "local_vram_probe.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
