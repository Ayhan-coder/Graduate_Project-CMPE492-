import json
import numpy as np
from scipy import stats

files_to_read = {
    "Whisper Large-v3": "experiments/outputs/asr-20260901-large-v3.jsonl",
    "Whisper Large-v3 Turbo": "experiments/outputs/asr-20260903-turbo.jsonl",
    "Qwen3-ASR 0.6B": "experiments/outputs/asr-qwen3-0.6b.rescored.jsonl",
    "Qwen3-ASR 1.7B": "experiments/outputs/asr-qwen3-1.7b.rescored.jsonl",
    "Nemotron 3.5": "experiments/outputs/asr-nemotron.rescored.jsonl",
    "Whisper Turkish Turbo": "experiments/outputs/asr-selimc-turbo.rescored.jsonl",
}

clip_wers = {}
for mname, fpath in files_to_read.items():
    clip_wers[mname] = {}
    with open(fpath, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            cid = d.get("id") or d.get("clip_id")
            w = d.get("wer_norm", d.get("wer", 0))
            if cid:
                clip_wers[mname][cid] = w

clips_7 = ["TR-1", "TR-2", "TR-3", "EN-1", "EN-2", "EN-3", "EN-4"]
turbo_vals = [clip_wers["Whisper Large-v3 Turbo"][c] for c in clips_7]

print("=== STATISTICAL COMPARISONS AGAINST WHISPER LARGE-V3 TURBO ===")
for mname in files_to_read:
    if mname in ["Whisper Large-v3 Turbo", "Whisper Turkish Turbo"]:
        continue
    other_vals = [clip_wers[mname][c] for c in clips_7]
    diff = np.array(other_vals) - np.array(turbo_vals)
    
    # Wilcoxon signed-rank test
    # Zero differences handling with wilcoxon
    res_wilc = stats.wilcoxon(other_vals, turbo_vals)
    t_res = stats.ttest_rel(other_vals, turbo_vals)
    
    print(f"\n{mname} vs Whisper Turbo:")
    print(f"  Mean WER: {np.mean(other_vals):.4f} vs {np.mean(turbo_vals):.4f} (Diff: {np.mean(diff):+.4f})")
    print(f"  Wilcoxon p-value: {res_wilc.pvalue:.4f} (statistic: {res_wilc.statistic})")
    print(f"  Paired t-test p-value: {t_res.pvalue:.4f}")
    if res_wilc.pvalue < 0.05 or t_res.pvalue < 0.05:
        print("  -> İstatistiksel olarak AYRIŞAN (Statistically Distinguishable)")
    else:
        print("  -> İstatistiksel olarak AYRIŞMAYAN (Statistically Indistinguishable)")

