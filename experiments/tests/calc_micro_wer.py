import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load manifest for word counts and clip languages
manifest_file = ROOT / "data" / "manifest.jsonl"
clip_words = {}
clip_lang = {}
if manifest_file.exists():
    for line in manifest_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            cid = r["id"]
            clip_words[cid] = len(r.get("reference", "").split())
            clip_lang[cid] = r.get("lang", "unk")

models_runs = {
    "Whisper Medium": ROOT / "outputs" / "asr-whisper-combined.rescored.jsonl",
    "Whisper Large-v3": ROOT / "outputs" / "asr-whisper-combined.rescored.jsonl",
    "Whisper Turbo": ROOT / "outputs" / "asr-whisper-combined.rescored.jsonl",
    "Whisper Turbo TR": ROOT / "outputs" / "asr-selimc-turbo.rescored.jsonl",
    "Qwen3-ASR 0.6B": ROOT / "outputs" / "asr-qwen3-0.6b.rescored.jsonl",
    "Qwen3-ASR 1.7B": ROOT / "outputs" / "asr-qwen3-1.7b.rescored.jsonl",
    "Nemotron 3.5 ASR": ROOT / "outputs" / "asr-nemotron.rescored.jsonl",
}

print(f"{'Model':<20} {'Lang':<5} {'Macro WER':<12} {'Micro WER (W-wght)':<20} {'Diff (%)':<10}")
print("-" * 72)

for model_name, p in models_runs.items():
    if not p.exists():
        continue
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    
    # Filter rows matching this model
    m_rows = []
    for r in rows:
        mod = (r.get("model") or "").lower()
        if model_name == "Whisper Turbo":
            if "turbo" in mod and "selimc" not in mod:
                m_rows.append(r)
        elif model_name == "Whisper Turbo TR":
            if "selimc" in mod or "turkish" in mod:
                m_rows.append(r)
        elif model_name == "Whisper Medium":
            if "medium" in mod:
                m_rows.append(r)
        elif model_name == "Whisper Large-v3":
            if "large-v3" in mod and "turbo" not in mod:
                m_rows.append(r)
        elif model_name == "Qwen3-ASR 0.6B":
            if "0.6b" in mod:
                m_rows.append(r)
        elif model_name == "Qwen3-ASR 1.7B":
            if "1.7b" in mod:
                m_rows.append(r)
        elif model_name == "Nemotron 3.5 ASR":
            if "nemotron" in mod:
                m_rows.append(r)

    # Group by language
    by_lang = defaultdict(list)
    for r in m_rows:
        cid = r.get("id")
        wer = r.get("wer_norm")
        if wer is not None and cid:
            lang = clip_lang.get(cid, r.get("lang", "unk"))
            by_lang[lang].append((cid, wer))

    for lang in sorted(by_lang.keys()):
        items = by_lang[lang]
        wers = [w for _, w in items]
        words = [clip_words.get(c, 1000) for c, _ in items]
        macro = sum(wers) / len(wers) if wers else 0.0
        micro = sum(w * cwords for (_, w), cwords in zip(items, words)) / sum(words) if sum(words) > 0 else 0.0
        diff = ((micro - macro) / macro * 100.0) if macro > 0 else 0.0
        print(f"{model_name:<20} {lang.upper():<5} {macro:<12.3f} {micro:<20.3f} {diff:+.2f}%")

print("-" * 72)

