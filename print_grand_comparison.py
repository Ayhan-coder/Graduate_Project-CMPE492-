import json

kumru = json.load(open("experiments/outputs/eduterm_kumru_2b_cefr_report.json", "r", encoding="utf-8"))
trendyol = json.load(open("experiments/outputs/eduterm_trendyol_8b_cefr_report.json", "r", encoding="utf-8"))
qwen14b = json.load(open("experiments/outputs/cross_family_comprehensive_benchmark.json", "r", encoding="utf-8"))

print(f"{'Model Checkpoint':<32} | {'VRAM':<9} | {'Eğitim':<8} | {'JSON':<7} | {'Precision':<9} | {'Recall':<8} | {'Macro F1':<8}")
print("-" * 95)

k_sft = kumru["post_sft_evaluation"]["metrics"]
t_sft = trendyol["post_sft_evaluation"]["metrics"]
q_sft = qwen14b["EduTerm-14B-TR (Full CEFR SFT)"]

print(f"{'Kumru-2B (EduTerm-2B SFT)':<32} | {kumru['peak_vram_gb']:>5.2f} GB | {'54.5 dk':<8} | {k_sft['json_schema_validity']*100:>5.1f}% | {k_sft['macro_precision']*100:>7.1f}% | {k_sft['macro_recall']*100:>6.1f}% | {k_sft['macro_f1']*100:>6.1f}%")
print(f"{'Trendyol-LLM-8B (EduTerm SFT)':<32} | {trendyol['peak_vram_gb']:>5.2f} GB | {'3.14 sa':<8} | {t_sft['json_schema_validity']*100:>5.1f}% | {t_sft['macro_precision']*100:>7.1f}% | {t_sft['macro_recall']*100:>6.1f}% | {t_sft['macro_f1']*100:>6.1f}%")
print(f"{'EduTerm-14B-TR (Full CEFR SFT)':<32} | {'24.95 GB':<9} | {'6.6 sa':<8} | {q_sft['json_schema_validity']*100:>5.1f}% | {q_sft['macro_precision']*100:>7.1f}% | {q_sft['macro_recall']*100:>6.1f}% | {q_sft['macro_f1']*100:>6.1f}%")

