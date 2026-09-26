import json
import statistics
from pathlib import Path
import numpy as np

# Manifest word counts and durations
manifest_file = Path("data/manifest.jsonl")
m_rows = [json.loads(line) for line in manifest_file.read_text(encoding="utf-8").splitlines() if line.strip()]
clip_info = {r["id"]: {"words": len(r["reference"].split()), "duration": float(r["duration_s"]), "lang": r["lang"]} for r in m_rows}

# Load all rescored ASR files
asr_files = [
    Path("outputs/asr-whisper-combined.rescored.jsonl"),
    Path("outputs/asr-qwen3-0.6b.rescored.jsonl"),
    Path("outputs/asr-qwen3-1.7b.rescored.jsonl"),
    Path("outputs/asr-selimc-turbo.rescored.jsonl"),
    Path("outputs/asr-nemotron.rescored.jsonl"),
]

all_rows = []
for p in asr_files:
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                all_rows.append(json.loads(line))

# Group by (model, lang)
groups = {}
for r in all_rows:
    cid = r.get("id")
    model = r.get("model")
    wer = r.get("wer_norm")
    if not cid or not model or wer is None or cid not in clip_info:
        continue
    lang = clip_info[cid]["lang"]
    groups.setdefault((model, lang), {})[cid] = wer

np.random.seed(42)

print(f"{'Model':<38} {'Lang':<5} {'n':<3} {'Macro WER':<11} {'Word-Wght WER':<15} {'Dur-Wght WER':<14} {'95% CI (Bootstrap)':<22}")
print("=" * 115)

for (model, lang), clips in sorted(groups.items()):
    wers = list(clips.values())
    cids = list(clips.keys())
    n = len(wers)
    macro = statistics.mean(wers)
    
    words = [clip_info[c]["words"] for c in cids]
    durs = [clip_info[c]["duration"] for c in cids]
    
    word_wght = sum(w * clip_info[c]["words"] for c, w in clips.items()) / sum(words)
    dur_wght = sum(w * clip_info[c]["duration"] for c, w in clips.items()) / sum(durs)
    
    # Bootstrap 95% CI for macro WER (10,000 resamples)
    boot_means = []
    w_arr = np.array(wers)
    for _ in range(10000):
        sample = np.random.choice(w_arr, size=n, replace=True)
        boot_means.append(sample.mean())
    ci_low = np.percentile(boot_means, 2.5)
    ci_high = np.percentile(boot_means, 97.5)
    
    ci_str = f"[{ci_low:.3f}, {ci_high:.3f}]"
    print(f"{model:<38} {lang:<5} {n:<3} {macro:<11.3f} {word_wght:<15.3f} {dur_wght:<14.3f} {ci_str:<22}")

print("=" * 115)
