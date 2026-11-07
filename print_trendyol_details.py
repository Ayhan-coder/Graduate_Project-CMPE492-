import json

d = json.load(open("experiments/outputs/eduterm_trendyol_8b_cefr_report.json", "r", encoding="utf-8"))
for b in d["post_sft_evaluation"]["benchmarks"]:
    print(f"[{b['id']}] {b['domain']}: Valid JSON={b['json_valid']} | Precision={b['precision']} | Recall={b['recall']} | Terms={b['extracted_terms']}")

