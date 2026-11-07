import json
import glob
from scipy import stats
import numpy as np

# Load per-clip results for each model
model_files = {
    "Whisper Large-v3": "experiments/outputs/asr-20260901-large-v3.meta.json",
    "Whisper Large-v3 Turbo": "experiments/outputs/asr-20260903-turbo.meta.json",
    "Qwen3-ASR 0.6B": "experiments/outputs/asr-qwen3-0.6b.meta.json",
    "Qwen3-ASR 1.7B": "experiments/outputs/asr-qwen3-1.7b.meta.json",
    "Nemotron 3.5": "experiments/outputs/asr-nemotron.meta.json",
    "Whisper Turkish Turbo": "experiments/outputs/asr-selimc-turbo.meta.json",
}

clip_wers = {}
for mname, fpath in model_files.items():
    try:
        with open(fpath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        clip_wers[mname] = {}
        for r in data.get("results", []):
            cid = r.get("id") or r.get("clip_id")
            w = r.get("wer_norm", r.get("wer", 0))
            clip_wers[mname][cid] = w
    except Exception as e:
        print(f"Error loading {fpath}: {e}")

print("Clips mapped for models:")
for m, c in clip_wers.items():
    print(f"  {m}: {len(c)} clips -> {sorted(list(c.keys()))}")

# Paired comparisons across all common 7 clips (TR-1..3, EN-1..4)
common_7 = ["TR-1", "TR-2", "TR-3", "EN-1", "EN-2", "EN-3", "EN-4"]

base_model = "Whisper Large-v3 Turbo"
base_vals = [clip_wers[base_model].get(c) for c in common_7]

print(f"\nPaired Tests against baseline: {base_model}")
print("-" * 75)
for other_model, other_clips in clip_wers.items():
    if other_model == base_model:
        continue
    other_vals = [other_clips.get(c) for c in common_7 if other_clips.get(c) is not None]
    if len(other_vals) == len(common_7):
        # Wilcoxon signed-rank test
        diff = np.array(base_vals) - np.array(other_vals)
        try:
            w_stat, p_val = stats.wilcoxon(base_vals, other_vals)
        except Exception:
            w_stat, p_val = np.nan, 1.0
        
        # Paired t-test as well
        t_stat, t_pval = stats.ttest_rel(base_vals, other_vals)
        
        # Distinguishable or indistinguishable?
        dist = "AYRIŞIYOR (Distinguishable, p < 0.05)" if p_val < 0.05 or t_pval < 0.05 else "AYRIŞMIYOR (Indistinguishable, p >= 0.05)"
        print(f"{other_model:<25} | Mean Diff: {np.mean(diff):+.4f} | Wilcoxon p={p_val:.4f} | t-test p={t_pval:.4f} | {dist}")

