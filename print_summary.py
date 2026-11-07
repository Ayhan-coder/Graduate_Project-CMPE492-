import json

d = json.load(open("experiments/outputs/cross_family_comprehensive_benchmark.json", "r", encoding="utf-8"))
print(f"{'Model Mimarisi':<32} | {'VRAM':<8} | {'JSON':<7} | {'Prec (%)':<9} | {'Rec (%)':<8} | {'F1 (%)':<7} | {'Bloom (%)':<9}")
print("-" * 95)
for k, v in d.items():
    print(f"{k:<32} | {v['peak_vram_gb']:>5.2f} GB | {v['json_schema_validity']*100:>5.1f}% | {v['macro_precision']*100:>7.1f}% | {v['macro_recall']*100:>6.1f}% | {v['macro_f1']*100:>5.1f}% | {v['bloom_accuracy']*100:>7.1f}%")

