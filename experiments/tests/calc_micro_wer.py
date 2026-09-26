import json
from pathlib import Path
from collections import defaultdict

models_runs = {
    "Whisper Medium": "outputs/asr-whisper-combined.rescored.jsonl",
    "Whisper Large-v3": "outputs/asr-whisper-combined.rescored.jsonl",
    "Whisper Turbo": "outputs/asr-whisper-combined.rescored.jsonl",
    "Qwen3-ASR 0.6B": "outputs/asr-qwen3-0.6b.rescored.jsonl",
    "Qwen3-ASR 1.7B": "outputs/asr-qwen3-1.7b.rescored.jsonl",
    "Whisper Turbo TR": "outputs/asr-selimc-turbo.rescored.jsonl",
}

print(f"{'Model':<18} {'Lang':<5} {'Macro WER':<12} {'Micro WER (W-wght)':<20} {'Diff (%)':<10}")
print("-" * 65)

for model_name, path_str in models_runs.items():
    p = Path(path_str)
    if not p.exists():
        continue
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    
    # Filter rows matching this model
    m_rows = [r for r in rows if model_name.lower() in (r.get("model") or "").lower() or (model_name == "Whisper Turbo" and "turbo" in (r.get("model") or "").lower() and "selimc" not in (r.get("model") or "").lower())]
    if not m_rows:
        # try matching by id substring
        pass

